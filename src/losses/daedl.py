"""DAEDL loss: standard EDL MSE + fixed-weight KL (Yoon et al., ICML 2024).

Official code uses a FIXED regularisation weight (reg_param=5e-2) with no
epoch annealing — unlike edl_fixed which linearly anneals from 0.  The head
(DAEDLHead) already applies density weighting to the logits before computing
alpha, so this loss operates on the density-weighted alpha transparently.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.contracts.schemas import LOSS_SCHEMA_VERSION
from src.losses.edl_fixed import _edl_fit_per_sample, _kl_dirichlet_to_uniform_per_sample
from src.registry.losses import LOSS_REGISTRY


@LOSS_REGISTRY.register("daedl")
class DAEDLLoss(nn.Module):
    """DAEDL loss — EDL MSE + fixed KL (no annealing).

    Args:
        lam: KL regularisation weight. Official default: 5e-2.
    """

    def __init__(self, lam: float = 5e-2) -> None:
        super().__init__()
        self.lam = float(lam)

    def forward(self, alpha: Tensor, target: Tensor, **kwargs: Any) -> Dict[str, Any]:
        fit_ps = _edl_fit_per_sample(alpha, target)
        fit = fit_ps.mean()

        y = F.one_hot(target, num_classes=alpha.size(1)).float()
        alpha_hat = alpha * (1.0 - y) + y
        kl_ps = _kl_dirichlet_to_uniform_per_sample(alpha_hat)
        reg = kl_ps.mean()

        total = fit + self.lam * reg

        return {
            "total": total,
            "fit": fit,
            "reg": reg,
            "aux": {
                "lambda_mean": float(self.lam),
                "lambda_min": float(self.lam),
                "lambda_max": float(self.lam),
                "lambda_std": 0.0,
                "info": float("nan"),
                "info_std": float("nan"),
                "fisher_trace": float("nan"),
            },
            "schema_version": LOSS_SCHEMA_VERSION,
        }
