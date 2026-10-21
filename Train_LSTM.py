"""
Training loop for ES-RNN on M4 Monthly.

Hyperparameters from the paper's Table 1 (Monthly, PF model):
    epochs = 10
    learning rate = 5e-4

Batch size (1024) and optimizer (Adam) are NOT specified in the paper --
Table 1 doesn't give a batch size at all, and "Stochastic Gradient Descent"
is used generically; Adam is the modern practical default and is used here
instead of trying to replicate the original Dynet implementation's exact
SGD variant.

One optimizer covers every parameter in the model -- both the LSTM/adapter
weights (global constants) and the ES layer's per-series alpha/beta/initial
seasonality (local constants) -- since model.parameters() recurses into the
es_layer submodule automatically. This is what makes the "partly global,
partly time-series-specific" joint training (paper section 2.2.5) work with
a completely standard PyTorch training loop.

This script trains only -- it does not evaluate against Monthly-test.csv.
Generating an actual 18-step-ahead forecast at inference time needs a
different procedure (run the ES recursion over the full known series, then
extrapolate seasonality forward since future actuals aren't available the
way they are during training) and belongs in a separate script.
"""

import time

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from Data_LSTM import load_m4_monthly, attach_domains
from windowing_LSTM import ESRNNWindowDataset
from Model_LSTM import ESRNN
from Loss_LSTM import esrnn_loss

EPOCHS = 1  # paper Table 1, Monthly PF
LEARNING_RATE = 5e-4  # paper Table 1, Monthly PF
BATCH_SIZE = 8192 # not specified by the paper -- our own choice

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

series_list = load_m4_monthly("Monthly-train.csv", "Monthly-test.csv")
series_list = attach_domains(series_list, "m4_info.csv")

dataset = ESRNNWindowDataset(series_list)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ESRNN(series_list).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

PRINT_EVERY = 50  # print a progress line every N batches, within each epoch

num_batches_per_epoch = len(loader)
print(f"training on {len(dataset)} windows, {num_batches_per_epoch} batches/epoch, {EPOCHS} epochs")

epoch_total_history = []
epoch_pinball_history = []
epoch_wiggliness_history = []

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    running_total = 0.0
    running_pinball = 0.0
    running_wiggliness = 0.0
    num_batches = 0

    for batch in loader:
        batch_start = time.time()
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        output = model(batch)
        losses = esrnn_loss(output)
        losses["total"].backward()
        optimizer.step()

        running_total += losses["total"].item()
        running_pinball += losses["pinball"].item()
        running_wiggliness += losses["wiggliness"].item()
        num_batches += 1

        if num_batches % PRINT_EVERY == 0:
            elapsed = time.time() - epoch_start
            batch_time = time.time() - batch_start
            avg_batch_time = elapsed / num_batches
            eta_epoch = avg_batch_time * (num_batches_per_epoch - num_batches)
            print(
                f"  epoch {epoch} batch {num_batches}/{num_batches_per_epoch} | "
                f"total={losses['total'].item():.4f} | "
                f"{batch_time*1000:.0f}ms/batch (avg {avg_batch_time*1000:.0f}ms) | "
                f"ETA this epoch: {eta_epoch/60:.1f} min"
            )

    epoch_time = time.time() - epoch_start
    avg_total = running_total / num_batches
    avg_pinball = running_pinball / num_batches
    avg_wiggliness = running_wiggliness / num_batches

    epoch_total_history.append(avg_total)
    epoch_pinball_history.append(avg_pinball)
    epoch_wiggliness_history.append(avg_wiggliness)

    print(
        f"epoch {epoch}/{EPOCHS} | "
        f"total={avg_total:.4f} | "
        f"pinball={avg_pinball:.4f} | "
        f"wiggliness={avg_wiggliness:.4f} | "
        f"{epoch_time:.1f}s"
    )

torch.save(model.state_dict(), "esrnn_monthly_TEST.pt")
print("saved model to esrnn_monthly.pt")

epochs_x = list(range(1, EPOCHS + 1))
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(epochs_x, epoch_total_history, marker="o")
axes[0].set_title("Total loss")
axes[0].set_xlabel("epoch")

axes[1].plot(epochs_x, epoch_pinball_history, marker="o", color="tab:orange")
axes[1].set_title("Pinball loss")
axes[1].set_xlabel("epoch")

axes[2].plot(epochs_x, epoch_wiggliness_history, marker="o", color="tab:green")
axes[2].set_title("Level-wiggliness penalty")
axes[2].set_xlabel("epoch")

plt.tight_layout()
plt.savefig("loss_curve_TEST.png")
print("saved loss curve to loss_curve.png")