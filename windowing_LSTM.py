"""
Rolling-window Dataset for ES-RNN training on M4 Monthly data.

Turns a list of M4Series (full per-series history) into a PyTorch Dataset
of (input_window, target_window, series_idx, domain) examples, one per
valid window position per series.

Per the paper (Smyl 2019), for Monthly (PF model):
    - Max length = 272: series longer than this are truncated to their
      most recent 272 points before windowing (section 3.3).
    - Input window >= one seasonal period (12 for monthly).
    - Output window = forecast horizon (18 for monthly).
    - Windows roll across the (truncated) series, per Figure 1.

This module does NOT do any ES normalization, deseasonalization, or log
squashing -- it only returns raw value slices. That's deliberate: the ES
parameters (alpha, beta, initial seasonality) are learnable and need to
participate in backprop, so the normalization math has to live inside the
model's forward pass, not here. What this Dataset hands back is exactly
what the model needs to do that: the raw input/target slices, which series
they came from (so the model can look up that series' ES parameters), and
that series' one-hot domain vector.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from Data_LSTM import M4Series

MONTHLY_MAX_LENGTH = 272
MONTHLY_INPUT_WINDOW = 12
MONTHLY_OUTPUT_WINDOW = 18


def truncate_series(train: np.ndarray, max_length: int) -> np.ndarray:
    """Keep only the most recent max_length points of a series' training
    history. Series shorter than max_length are returned unchanged."""
    if len(train) <= max_length:
        return train
    return train[-max_length:]


def enumerate_windows(series_length: int, input_window: int, output_window: int) -> list[int]:
    """Return all valid window start positions for a series of this length.
    A window at start t uses train[t : t+input_window] as input and
    train[t+input_window : t+input_window+output_window] as target, so t
    must satisfy t + input_window + output_window <= series_length.
    Returns an empty list if the series is too short for even one window."""
    last_valid_start = series_length - input_window - output_window
    if last_valid_start < 0:
        return []
    return list(range(last_valid_start + 1))


class ESRNNWindowDataset(Dataset):
    """One example = one (input_window, target_window) slice from one
    series. series_idx identifies which series the window came from, so
    the model can index into that series' ES parameters (alpha, beta,
    initial seasonality)."""

    def __init__(
        self,
        series_list: list[M4Series],
        input_window: int = MONTHLY_INPUT_WINDOW,
        output_window: int = MONTHLY_OUTPUT_WINDOW,
        max_length: int = MONTHLY_MAX_LENGTH,
    ):
        self.input_window = input_window
        self.output_window = output_window

        # Truncate each series up front and keep the truncated arrays,
        # since __getitem__ needs to slice the same (truncated) version
        # that enumerate_windows computed positions against.
        self.truncated_train = [
            truncate_series(s.train, max_length) for s in series_list
        ]
        self.domains = [s.domain for s in series_list]
        self.series_ids = [s.series_id for s in series_list]

        # Flat index: one (series_idx, start) entry per valid window,
        # across all series. This is what makes __len__/__getitem__ treat
        # "all windows from all series" as one flat, indexable dataset.
        self.index: list[tuple[int, int]] = []
        for series_idx, train in enumerate(self.truncated_train):
            starts = enumerate_windows(len(train), input_window, output_window)
            for start in starts:
                self.index.append((series_idx, start))

        if not self.index:
            raise ValueError(
                "no valid windows found in any series -- check that series "
                "are long enough for input_window + output_window, or that "
                "min-length filtering was applied before building this dataset"
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        series_idx, start = self.index[i]
        train = self.truncated_train[series_idx]

        input_slice = train[start : start + self.input_window]
        target_slice = train[
            start + self.input_window : start + self.input_window + self.output_window
        ]

        domain = self.domains[series_idx]
        if domain is None:
            raise ValueError(
                f"series {self.series_ids[series_idx]} has no domain vector -- "
                f"call attach_domains() on series_list before building this dataset"
            )

        return {
            "input": torch.tensor(input_slice, dtype=torch.float32),
            "target": torch.tensor(target_slice, dtype=torch.float32),
            "series_idx": torch.tensor(series_idx, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
            "domain": torch.tensor(domain, dtype=torch.float32),
        }