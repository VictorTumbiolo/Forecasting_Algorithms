"""
Loss for ES-RNN: pinball loss (paper eq. 7) plus the level-wiggliness
penalty (section 3.1.3, "LVP").

Trained in log/normalized space: pinball_loss(normalized_target, nn_output),
not on real-scale forecast vs raw_target. This follows section 3.1.1 -- "the
inputs to the NN are already deseasonalized and normalized in this system,
[so] the training loss function does not need to include normalization" --
i.e. the comparison happens in the same space the NN already operates in.

LVP scope note: the paper computes the wiggliness penalty over a series'
whole level sequence. Here it's computed over the window's own level span
(what model.py's forward() returns as 'level_span') as a simplification --
see model.py's comment on that field for why.
"""

import torch

MONTHLY_TRAINING_PERCENTILE = 0.49  # paper Table 1, Monthly PF: "Training percentile=49"
MONTHLY_LVP = 50.0  # paper Table 1, Monthly PF: "LVP=50"


def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float) -> torch.Tensor:
    """Paper eq. 7. Asymmetric: under-forecasting and over-forecasting are
    penalized differently depending on tau. Returns a scalar (mean over all
    elements)."""
    diff = y_true - y_pred
    loss = torch.where(diff >= 0, tau * diff, (tau - 1) * diff)
    return loss.mean()


def level_wiggliness_penalty(level_span: torch.Tensor) -> torch.Tensor:
    """Paper section 3.1.3: second differences of log(l_{t+1}/l_t), squared
    and averaged. level_span is (B, span); returns a scalar."""
    log_ratios = torch.log(level_span[:, 1:] / level_span[:, :-1])  # (B, span-1)
    second_diffs = log_ratios[:, 1:] - log_ratios[:, :-1]  # (B, span-2)
    return (second_diffs**2).mean()


def esrnn_loss(
    model_output: dict,
    tau: float = MONTHLY_TRAINING_PERCENTILE,
    lvp_weight: float = MONTHLY_LVP,
) -> dict:
    """Combines pinball loss (in log/normalized space) with the level-
    wiggliness penalty. Returns a dict with the total loss plus the two
    components separately, so you can log/inspect them independently
    during training."""
    pinball = pinball_loss(
        model_output["normalized_target"], model_output["nn_output"], tau
    )
    wiggliness = level_wiggliness_penalty(model_output["level_span"])
    total = pinball + lvp_weight * wiggliness

    return {
        "total": total,
        "pinball": pinball,
        "wiggliness": wiggliness,
    }