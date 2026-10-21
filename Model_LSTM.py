"""
Combined ES-RNN model: wires the ES layer (es_layer.py) together with a
plain LSTM, and contains the batch-gathering glue needed to connect what
ESRNNWindowDataset (windowing.py) returns to what ESLayer.run_recursion()
expects.

v1 scope: a plain (non-dilated, non-attentive, non-residual) 1-layer LSTM.
This gets the full pipeline (ES recursion -> normalize -> LSTM -> adapter
-> denormalize) training correctly end to end first; the paper's dilated/
residual/attentive variants can replace this LSTM later without touching
the ES layer or this glue code.

Why this file recomputes the ES recursion from the START of each series'
(truncated) history on every forward call, instead of using only the
window's own 12 input points: the paper's l_t/s_t are a continuous
recursion over the whole series (section 2.2.3), so a window at position
50 depends on everything before it, not just its own local slice. Since
the recursion is recomputed fresh every forward call using whatever the
current alpha/beta happen to be, there's no staleness issue with ordinary
mini-batch SGD -- see the ES-layer design discussion this came out of.
"""

import numpy as np
import torch
import torch.nn as nn

from Data_LSTM import M4Series
from ES_Layer_LSTM import ESLayer, MONTHLY_SEASON_LENGTH
from windowing_LSTM import truncate_series, MONTHLY_MAX_LENGTH, MONTHLY_INPUT_WINDOW, MONTHLY_OUTPUT_WINDOW


class ESRNN(nn.Module):
    def __init__(
        self,
        series_list: list[M4Series],
        input_window: int = MONTHLY_INPUT_WINDOW,
        output_window: int = MONTHLY_OUTPUT_WINDOW,
        max_length: int = MONTHLY_MAX_LENGTH,
        season_length: int = MONTHLY_SEASON_LENGTH,
        hidden_size: int = 50,  # paper's Table 1 "State size of LSTMs" for Monthly
        domain_size: int = 6,
    ):
        super().__init__()
        self.input_window = input_window
        self.output_window = output_window
        self.max_length = max_length

        num_series = len(series_list)

        # Build one padded tensor holding every series' truncated history,
        # so the model can gather any series' full values by series_idx at
        # forward time (this is what lets the ES recursion run from
        # position 0, not just from the window's own start). Padding uses
        # each series' own last real value -- the exact pad value doesn't
        # matter numerically since enumerate_windows (windowing.py) only
        # ever creates windows that stay fully within each series' real
        # (unpadded) length, so padded positions are never read by any
        # window's level/seasonality lookup.
        full_values = np.zeros((num_series, max_length), dtype=np.float32)
        valid_mask = np.zeros((num_series, max_length), dtype=bool)
        for i, s in enumerate(series_list):
            truncated = truncate_series(s.train, max_length)
            L = len(truncated)
            full_values[i, :L] = truncated
            full_values[i, L:] = truncated[-1] if L > 0 else 1.0
            valid_mask[i, :L] = True

        self.register_buffer("full_values", torch.tensor(full_values))
        self.register_buffer("valid_mask", torch.tensor(valid_mask))

        # Domain vectors stored per-series too (training gets these from the
        # Dataset per-batch; forecast() below needs all of them at once,
        # since real inference isn't driven by the windowed Dataset at all).
        # persistent=False: this is deterministic data rebuilt identically
        # every time the model is constructed from series_list, not a
        # learned value -- excluding it from state_dict means it's never
        # saved/loaded, so adding it here doesn't break loading a
        # checkpoint saved before this field existed.
        domains = np.stack([s.domain for s in series_list]).astype(np.float32)
        self.register_buffer("domains", torch.tensor(domains), persistent=False)

        self.es_layer = ESLayer(num_series, season_length=season_length)

        # LSTM input per timestep: the normalized value (1) + the domain
        # one-hot vector (domain_size), concatenated.
        self.lstm = nn.LSTM(
            input_size=1 + domain_size, hidden_size=hidden_size, num_layers=1, batch_first=True
        )
        self.adapter = nn.Linear(hidden_size, output_window)

    def forward(self, batch: dict) -> dict:
        """batch is exactly what ESRNNWindowDataset's collated output looks
        like: 'input' (B, input_window), 'target' (B, output_window),
        'series_idx' (B,), 'start' (B,), 'domain' (B, domain_size)."""
        series_idx = batch["series_idx"]
        start = batch["start"]
        domain = batch["domain"]
        raw_input = batch["input"]
        raw_target = batch["target"]

        values = self.full_values[series_idx]  # (B, max_length)
        mask = self.valid_mask[series_idx]  # (B, max_length)

        l_all, s_all = self.es_layer.run_recursion(series_idx, values, mask)

        # Level used for normalization is the level at the LAST input
        # position (paper: "l_t ... last known data point"), i.e. at
        # index start + input_window - 1.
        level_idx = start + self.input_window - 1
        level = l_all.gather(1, level_idx.unsqueeze(1)).squeeze(1)  # (B,)

        # Seasonality slice spans the whole window (input + output), so we
        # can deseasonalize the input and reseasonalize the output using
        # one gathered block.
        span = self.input_window + self.output_window
        offsets = torch.arange(span, device=start.device)
        idx_matrix = start.unsqueeze(1) + offsets.unsqueeze(0)  # (B, span)
        seasonality = torch.gather(s_all, 1, idx_matrix)  # (B, span)
        seasonality_input = seasonality[:, : self.input_window]
        seasonality_output = seasonality[:, self.input_window :]

        # Level across the same window span, for the level-wiggliness
        # penalty (section 3.1.3). The paper computes this penalty over a
        # series' whole level sequence; here it's scoped to just this
        # window's span (input+output) as a simplification, since that's
        # what's already available per training example in this mini-batch
        # setup without extra full-series bookkeeping.
        level_span = torch.gather(l_all, 1, idx_matrix)  # (B, span)

        x_norm = self.es_layer.normalize(raw_input, level, seasonality_input)  # (B, input_window)

        domain_expanded = domain.unsqueeze(1).expand(-1, self.input_window, -1)
        lstm_input = torch.cat([x_norm.unsqueeze(-1), domain_expanded], dim=-1)

        lstm_out, _ = self.lstm(lstm_input)
        last_hidden = lstm_out[:, -1, :]
        nn_output = self.adapter(last_hidden)  # (B, output_window), log-space

        # Two representations are returned so the loss function (next
        # piece) can pick whichever space the paper actually trains in:
        # nn_output vs normalized_target (log-space, per section 3.1.1's
        # "the inputs to the NN are already deseasonalized and normalized,
        # so the loss doesn't need to include normalization"), or forecast
        # vs raw_target (real-scale, useful for evaluation either way).
        normalized_target = self.es_layer.normalize(raw_target, level, seasonality_output)
        forecast = self.es_layer.denormalize(nn_output, level, seasonality_output)

        return {
            "nn_output": nn_output,
            "normalized_target": normalized_target,
            "forecast": forecast,
            "level_span": level_span,
            "raw_target": raw_target,
        }

    @torch.no_grad()
    def forecast(self) -> torch.Tensor:
        """Generate genuine output_window-step-ahead forecasts for every
        series, using each series' own most recent input_window real
        points -- unlike forward(), which is only valid during training
        (its "future" window is actually already-known history).

        Seasonality for the forecast horizon can't come from real future
        data the way it does during training. Since input_window ==
        season_length (12 == 12) here, the seasonality values used to
        deseasonalize the last input_window known points are exactly one
        full seasonal cycle -- this tiles that same cycle across the
        forecast horizon, which is standard Holt-Winters practice (reuse
        the most recently estimated seasonal cycle), not an approximation
        specific to this implementation.
        """
        assert self.input_window % self.es_layer.season_length == 0 or self.es_layer.season_length % self.input_window == 0, (
            "forecast()'s seasonality-tiling logic assumes input_window and "
            "season_length divide evenly into each other"
        )

        num_series = self.full_values.shape[0]
        device = self.full_values.device
        series_idx = torch.arange(num_series, device=device)
        values = self.full_values
        mask = self.valid_mask

        l_all, s_all = self.es_layer.run_recursion(series_idx, values, mask)

        valid_lengths = mask.sum(dim=1)  # (N,) each series' real (unpadded) length
        start = valid_lengths - self.input_window  # last available input window
        level_idx = valid_lengths - 1  # last known data point
        level = l_all.gather(1, level_idx.unsqueeze(1)).squeeze(1)  # (N,)

        offsets = torch.arange(self.input_window, device=device)
        input_idx = start.unsqueeze(1) + offsets.unsqueeze(0)  # (N, input_window)
        seasonality_input = torch.gather(s_all, 1, input_idx)  # (N, input_window)
        raw_input = torch.gather(values, 1, input_idx)  # (N, input_window)

        x_norm = self.es_layer.normalize(raw_input, level, seasonality_input)

        domain_expanded = self.domains.unsqueeze(1).expand(-1, self.input_window, -1)
        lstm_input = torch.cat([x_norm.unsqueeze(-1), domain_expanded], dim=-1)

        lstm_out, _ = self.lstm(lstm_input)
        last_hidden = lstm_out[:, -1, :]
        nn_output = self.adapter(last_hidden)  # (N, output_window)

        repeats = (self.output_window + self.input_window - 1) // self.input_window
        seasonality_output = seasonality_input.repeat(1, repeats)[:, : self.output_window]

        return self.es_layer.denormalize(nn_output, level, seasonality_output)  # (N, output_window)