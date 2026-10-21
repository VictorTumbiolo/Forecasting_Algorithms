"""
Evaluates the trained ES-RNN model against M4's real held-out Monthly test
points (the actual 18-step-ahead future, from Monthly-test.csv), using
sMAPE -- the same metric family the M4 competition itself used (paper
section 3.1.1 references Makridakis et al. 2018b).

sMAPE formula (M4's standard definition):
    sMAPE = (200/h) * sum(|y_t - yhat_t| / (|y_t| + |yhat_t|))
"""

import numpy as np
import torch

from Data import load_m4_monthly, attach_domains
from Model_LSTM import ESRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

series_list = load_m4_monthly("Monthly-train.csv", "Monthly-test.csv")
series_list = attach_domains(series_list, "m4_info.csv")

model = ESRNN(series_list).to(device)
model.load_state_dict(torch.load("esrnn_monthly.pt", map_location=device))
model.eval()

forecasts = model.forecast()  # (N, 18), real-scale, same series order as series_list
forecasts = forecasts.cpu().numpy()

actuals = np.stack([s.test for s in series_list])  # (N, 18)

epsilon = 1e-8  # guards against the rare exact-zero case in both y_true and y_pred
smape_per_point = 200.0 * np.abs(actuals - forecasts) / (np.abs(actuals) + np.abs(forecasts) + epsilon)
smape_per_series = smape_per_point.mean(axis=1)  # (N,)

overall_smape = smape_per_series.mean()
print(f"overall sMAPE across {len(series_list)} series: {overall_smape:.2f}%")

print(f"best series sMAPE: {smape_per_series.min():.2f}%")
print(f"worst series sMAPE: {smape_per_series.max():.2f}%")
print(f"median series sMAPE: {np.median(smape_per_series):.2f}%")

worst_idx = smape_per_series.argmax()
best_idx = smape_per_series.argmin()
print(f"\nworst series: {series_list[worst_idx].series_id}")
print(f"  actual:   {actuals[worst_idx]}")
print(f"  forecast: {forecasts[worst_idx]}")

print(f"\nbest series: {series_list[best_idx].series_id}")
print(f"  actual:   {actuals[best_idx]}")
print(f"  forecast: {forecasts[best_idx]}")