"""
Data loading for the M4 Monthly dataset.

Expects two files in M4's standard format:
    Monthly-train.csv
    Monthly-test.csv

Format (M4 competition standard):
    - One row per series.
    - First column "V1" is the series id (e.g. "M1", "M2", ...).
    - Remaining columns "V2", "V3", ... are the series values, in order.
    - Rows are RAGGED: shorter series have trailing empty cells (NaN when
      read with pandas). Monthly-train.csv holds all history EXCEPT the
      final horizon points; Monthly-test.csv holds exactly those held-out
      points (18 columns for Monthly, i.e. V2..V19).

This module turns that ragged CSV into a list of per-series numpy arrays,
which is the natural representation for ES-RNN (every series has its own
ES parameters and its own length, so there's no benefit to packing into a
single rectangular matrix).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

MONTHLY_HORIZON = 18
MONTHLY_SEASON_LENGTH = 12

# Fixed, sorted so one-hot index is stable across runs. These are M4's six
# standard categories (M4-info.csv's "category" column).
M4_CATEGORIES = ["Demographic", "Finance", "Industry", "Macro", "Micro", "Other"]


@dataclass
class M4Series:
    series_id: str
    train: np.ndarray  # 1D float array, full training history
    test: np.ndarray  # 1D float array, length == MONTHLY_HORIZON
    domain: np.ndarray | None = None  # 1D one-hot float array, length == len(M4_CATEGORIES)


def load_m4_monthly(train_path: str, test_path: str) -> list[M4Series]:
    """Load Monthly-train.csv and Monthly-test.csv and zip them into a list
    of M4Series, one per series id. Raises if a series id is missing from
    either file, since that would silently misalign train and test."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_ids = train_df.iloc[:, 0].astype(str)
    test_ids = test_df.iloc[:, 0].astype(str)

    if set(train_ids) != set(test_ids):
        only_train = set(train_ids) - set(test_ids)
        only_test = set(test_ids) - set(train_ids)
        raise ValueError(
            f"train/test series id mismatch: "
            f"{len(only_train)} ids only in train, {len(only_test)} only in test "
            f"(e.g. {list(only_train)[:3]} / {list(only_test)[:3]})"
        )

    # Build a lookup from series id -> row index in test_df, then pull the
    # whole numeric block out as a single numpy array. Doing this once up
    # front (instead of calling .loc[series_id] per row in a loop) avoids
    # pandas' per-call index-alignment overhead, which is what made the
    # original version slow at 48,000 rows.
    test_id_to_row = {sid: i for i, sid in enumerate(test_ids)}
    train_values = train_df.iloc[:, 1:].to_numpy(dtype=float)
    test_values = test_df.iloc[:, 1:].to_numpy(dtype=float)

    series_list = []
    for i, series_id in enumerate(train_ids):
        train_vals = train_values[i]
        train_vals = train_vals[~np.isnan(train_vals)]

        test_row_idx = test_id_to_row[series_id]
        test_vals = test_values[test_row_idx]
        test_vals = test_vals[~np.isnan(test_vals)]

        if len(test_vals) != MONTHLY_HORIZON:
            raise ValueError(
                f"series {series_id}: expected {MONTHLY_HORIZON} test points, "
                f"got {len(test_vals)}"
            )

        series_list.append(
            M4Series(series_id=series_id, train=train_vals, test=test_vals)
        )

    return series_list


def attach_domains(series_list: list[M4Series], info_path: str) -> list[M4Series]:
    """Read M4-info.csv and attach a one-hot domain/category vector to each
    series in series_list, matched by series id. Expects M4-info.csv to have
    an "M4id" column (series id, matching M4Series.series_id) and a
    "category" column (one of M4_CATEGORIES).

    Returns a NEW list of M4Series (domain field populated) rather than
    mutating series_list in place, since M4Series is a dataclass and callers
    may not expect in-place mutation."""
    info_df = pd.read_csv(info_path)

    if "M4id" not in info_df.columns or "category" not in info_df.columns:
        raise ValueError(
            f"expected columns 'M4id' and 'category' in {info_path}, "
            f"got {list(info_df.columns)}"
        )

    id_to_category = dict(zip(info_df["M4id"].astype(str), info_df["category"]))

    unknown_categories = set(id_to_category.values()) - set(M4_CATEGORIES)
    if unknown_categories:
        raise ValueError(
            f"found categories not in M4_CATEGORIES: {unknown_categories}. "
            f"Update M4_CATEGORIES to include them."
        )

    category_to_onehot = {}
    for i, cat in enumerate(M4_CATEGORIES):
        vec = np.zeros(len(M4_CATEGORIES), dtype=float)
        vec[i] = 1.0
        category_to_onehot[cat] = vec

    updated = []
    missing_ids = []
    for s in series_list:
        if s.series_id not in id_to_category:
            missing_ids.append(s.series_id)
            continue
        category = id_to_category[s.series_id]
        domain_vec = category_to_onehot[category]
        updated.append(
            M4Series(
                series_id=s.series_id,
                train=s.train,
                test=s.test,
                domain=domain_vec,
            )
        )

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} series ids not found in {info_path}, "
            f"e.g. {missing_ids[:3]}"
        )

    return updated


def filter_min_length(
    series_list: list[M4Series], min_length: int
) -> list[M4Series]:
    """Drop series whose training history is too short to build even one
    input window (input window should cover >= one full seasonal period,
    i.e. >= MONTHLY_SEASON_LENGTH points, plus some margin for backtesting)."""
    return [s for s in series_list if len(s.train) >= min_length]


def series_length_stats(series_list: list[M4Series]) -> dict:
    """Return min/max/mean training length across all series, so you can
    see length distribution and decide a sensible min_length cutoff."""
    lengths = [len(s.train) for s in series_list]
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
    }


data = load_m4_monthly("Monthly-train.csv", "Monthly-test.csv")
print(series_length_stats(data))

data = attach_domains(data, "m4_info.csv")
print(f"attached domains to {len(data)} series")
print(f"example: {data[0].series_id} -> domain={data[0].domain}")