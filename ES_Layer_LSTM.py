"""
Exponential Smoothing layer for ES-RNN (single-seasonality case, e.g.
Monthly data with season_length=12).

Holds the per-series learnable parameters (alpha, beta, initial
seasonality) and implements:
    - the ES recursion (paper eq. 2) that produces l_t and s_t for every
      timestep of a series, given its current parameter values
    - normalize(): turn raw values into the log-squashed, deseasonalized
      input the LSTM expects (paper's X_i formula, Figure 2)
    - denormalize(): turn the LSTM's raw output back into real-scale
      forecasts (paper's Y_hat formula, eq. 5)

This file does NOT do any batch-gathering from the windowed Dataset (i.e.
turning a batch of (series_idx, window_start) pairs into the padded
per-series value arrays this layer's run_recursion() expects). That glue
belongs in the combined ESRNN model, which calls into this layer -- this
file is just the ES math and the per-series parameters.

Recursion is computed fresh on every call (no caching across calls), using
whatever the current parameter values are -- this is what makes it safe to
call once per mini-batch during normal PyTorch training without any
staleness between an optimizer step and the next batch's forward pass.
"""

import torch
import torch.nn as nn

MONTHLY_SEASON_LENGTH = 12


class ESLayer(nn.Module):
    def __init__(self, num_series: int, season_length: int = MONTHLY_SEASON_LENGTH):
        super().__init__()
        self.num_series = num_series
        self.season_length = season_length

        # Local constants (paper's terminology): fixed per series, don't
        # change as we step through time, but ARE updated by the optimizer
        # across training steps, same as the LSTM's global weights.
        self.alpha_raw = nn.Parameter(torch.zeros(num_series))
        self.beta_raw = nn.Parameter(torch.zeros(num_series))
        self.init_seasonality_raw = nn.Parameter(torch.zeros(num_series, season_length))

    def alpha(self, series_idx: torch.Tensor) -> torch.Tensor:
        """sigmoid() keeps alpha in (0,1), per the paper."""
        return torch.sigmoid(self.alpha_raw[series_idx])

    def beta(self, series_idx: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.beta_raw[series_idx])

    def initial_seasonality(self, series_idx: torch.Tensor) -> torch.Tensor:
        """exp() keeps seasonality components positive, per the paper."""
        return torch.exp(self.init_seasonality_raw[series_idx])

    def run_recursion(
        self,
        series_idx: torch.Tensor,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the single-seasonality ES recursion (paper eq. 2) forward
        from t=0 through T-1, for a batch of series.

        Args:
            series_idx: (B,) long tensor, which series each row belongs to.
            values: (B, T) float tensor, values[b, t] = y_t for series b.
                Positions beyond that series' actual length should be
                padded (padding value doesn't matter numerically as long
                as it's nonzero and finite -- those positions are masked
                out and never used downstream).
            valid_mask: (B, T) bool tensor, True where values[b, t] is real
                data, False where it's padding. Currently unused inside
                the recursion itself (padding values just produce l_t/s_t
                values at padded positions that nobody will index into
                later) but kept in the signature so callers are forced to
                think about which positions are real, and so a stricter
                future version can mask more carefully if needed.

        Returns:
            l_all: (B, T) float tensor, l_all[b, t] = level at time t.
            s_all: (B, T) float tensor, s_all[b, t] = seasonality at time t.

        Note on l's initial condition: the paper describes initial
        seasonality as an explicit learned per-series parameter, but
        doesn't give the initial level's seed value. This implementation
        seeds l_{-1} with the series' own first value (values[:, 0]) as a
        simple, scale-appropriate starting point -- a common practical
        choice, not something stated explicitly in the paper.
        """
        B, T = values.shape
        K = self.season_length

        alpha = self.alpha(series_idx)  # (B,)
        beta = self.beta(series_idx)  # (B,)
        s_init = self.initial_seasonality(series_idx)  # (B, K) -> s_0..s_{K-1}

        l_all = torch.zeros(B, T, dtype=values.dtype, device=values.device)
        s_all = torch.zeros(B, T, dtype=values.dtype, device=values.device)

        # s_buffer holds s_t for t values we've produced so far, indexed by
        # t directly (grown as we go) so that at step t we can look up
        # s_t -- either from the learned initial values (t < K) or from a
        # value computed K steps earlier in this same loop.
        s_buffer = [s_init[:, t] for t in range(K)]

        l_prev = values[:, 0]  # seed, see docstring note above

        for t in range(T):
            s_t = s_buffer[t]
            y_t = values[:, t]

            l_t = alpha * (y_t / s_t) + (1 - alpha) * l_prev
            s_next = beta * (y_t / l_t) + (1 - beta) * s_t  # this is s_{t+K}

            l_all[:, t] = l_t
            s_all[:, t] = s_t
            s_buffer.append(s_next)

            l_prev = l_t

        return l_all, s_all

    def normalize(
        self, values: torch.Tensor, level: torch.Tensor, seasonality: torch.Tensor
    ) -> torch.Tensor:
        """Paper's X_i formula (Figure 2): deseasonalize, normalize by
        level, then log-squash. `level` is a scalar per row (the level at
        the last input-window position); `seasonality` is per-position
        (matching the shape of `values`)."""
        return torch.log(values / level.unsqueeze(-1) / seasonality)

    def denormalize(
        self, nn_output: torch.Tensor, level: torch.Tensor, seasonality: torch.Tensor
    ) -> torch.Tensor:
        """Paper's Y_hat formula (eq. 5): exp, then reseasonalize and
        re-scale by level. Shapes mirror normalize()."""
        return torch.exp(nn_output) * level.unsqueeze(-1) * seasonality