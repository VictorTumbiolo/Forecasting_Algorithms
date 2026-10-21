"""
Compares the trained baseline LSTM ES-RNN against the official M4 Monthly results
(Evaluation and Ranks.xlsx, sheet "Point Forecasts-Frequency", Monthly columns).

Computes sMAPE, MASE and OWA the way M4 defines them, prints a side-by-side table,
and saves everything so the (slow) forecast never has to be recomputed.

Run from inside Baseline/ (relative CSV paths, and Data_LSTM loads data on import).

Definitions (M4):
    sMAPE = (200/h) * sum(|y - yhat| / (|y| + |yhat|))                  per series, then averaged
    MASE  = mean|y - yhat| / mean|y_t - y_{t-m}|  (m = 12, over the full in-sample train)  per series, then averaged
    OWA   = (sMAPE / sMAPE_Naive2 + MASE / MASE_Naive2) / 2
"""

import numpy as np
import torch

from Data_LSTM import load_m4_monthly, attach_domains
from Model_LSTM import ESRNN

CHECKPOINT = "esrnn_monthly.pt"
SEASON = 12

# Official M4 Monthly point-forecast results, from Evaluation and Ranks.xlsx.
NAIVE2_SMAPE = 14.427
NAIVE2_MASE = 1.063
REFERENCE = [
    # name,                   sMAPE,  MASE
    ("ES-RNN (Smyl, ID 118)", 12.126, 0.884),
    ("Theta",                 13.002, 0.970),
    ("Comb",                  13.434, 0.966),
    ("Naive2",                14.427, 1.063),
]


def owa(smape, mase):
    return (smape / NAIVE2_SMAPE + mase / NAIVE2_MASE) / 2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

series_list = load_m4_monthly("Monthly-train.csv", "Monthly-test.csv")
series_list = attach_domains(series_list, "m4_info.csv")

model = ESRNN(series_list).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

forecasts = model.forecast().cpu().numpy()  # (N, 18), real-scale, same order as series_list
actuals = np.stack([s.test for s in series_list])  # (N, 18)

# --- sMAPE ---
epsilon = 1e-8
smape_per_series = (
    200.0 * np.abs(actuals - forecasts) / (np.abs(actuals) + np.abs(forecasts) + epsilon)
).mean(axis=1)

# --- MASE (scale = in-sample seasonal-naive MAE over the full, untruncated train series) ---
mase_per_series = np.empty(len(series_list))
for i, s in enumerate(series_list):
    scale = np.mean(np.abs(s.train[SEASON:] - s.train[:-SEASON]))
    mase_per_series[i] = np.mean(np.abs(actuals[i] - forecasts[i])) / max(scale, epsilon)

smape = smape_per_series.mean()
mase = mase_per_series.mean()
owa_score = owa(smape, mase)

# --- table ---
rows = REFERENCE + [("This baseline (LSTM)", smape, mase)]
rows.sort(key=lambda r: owa(r[1], r[2]))

lines = []
lines.append(f"{'Method':<26}{'sMAPE':>8}{'MASE':>8}{'OWA':>8}")
lines.append("-" * 50)
for name, sm, ma in rows:
    marker = "  <--" if name.startswith("This baseline") else ""
    lines.append(f"{name:<26}{sm:>8.3f}{ma:>8.3f}{owa(sm, ma):>8.3f}{marker}")
lines.append("")
lines.append(f"series evaluated: {len(series_list)}   checkpoint: {CHECKPOINT}")
lines.append(f"baseline sMAPE per series -> best {smape_per_series.min():.2f}, "
             f"median {np.median(smape_per_series):.2f}, worst {smape_per_series.max():.2f}")
lines.append(f"gap to ES-RNN (Smyl): sMAPE {smape - 12.126:+.3f}, MASE {mase - 0.884:+.3f}, "
             f"OWA {owa_score - owa(12.126, 0.884):+.3f}")
report = "\n".join(lines)
print("\n" + report)

# --- save everything ---
np.save("compare_forecasts_LSTM.npy", forecasts)
np.savetxt(
    "compare_per_series_LSTM.csv",
    np.column_stack([[s.series_id for s in series_list], smape_per_series, mase_per_series]),
    fmt="%s", delimiter=",", header="series_id,smape,mase", comments="",
)
with open("compare_results_LSTM.txt", "w") as f:
    f.write(report + "\n")
print("\nsaved compare_results_LSTM.txt, compare_per_series_LSTM.csv, compare_forecasts_LSTM.npy")
