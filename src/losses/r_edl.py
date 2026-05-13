"""R-EDL loss (Chen et al., ICLR 2024).

Drops the KL-to-uniform regularizer and replaces the K-fixed per-class prior
with a single hyperparameter `lambda_prior` (paper default 0.1).
"""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor, nn

from src.contracts.schemas import LOSS_SCHEMA_VERSION
from src.losses.edl_fixed import _edl_fit_per_sample
from src.registry.losses import LOSS_REGISTRY


@LOSS_REGISTRY.register("r_edl")
class REDLLoss(nn.Module):
    """R-EDL baseline with matched pipeline.

    Head still emits alpha = evidence + 1; we re-base to
    alpha_redl = evidence + lambda_prior internally so the loss plugs into the
    same backbone / head / optimizer as FI-EDL, I-EDL, and EDL-fixed.
    """

    def __init__(self, lambda_prior: float = 0.1) -> None:
        super().__init__()
        if lambda_prior <= 0.0:
            raise ValueError(
                f"lambda_prior must be positive (R-EDL assumes a proper Dirichlet), "
                f"got {lambda_prior!r}"
            )
        self.lambda_prior = float(lambda_prior)

    def _rebase_alpha(self, alpha: Tensor) -> Tensor:
        evidence = (alpha - 1.0).clamp_min(0.0)
        return evidence + self.lambda_prior

    def forward(self, alpha: Tensor, target: Tensor, **kwargs: Any) -> Dict[str, Any]:
        alpha_redl = self._rebase_alpha(alpha)
        fit_ps = _edl_fit_per_sample(alpha_redl, target)
        fit = fit_ps.mean()
        reg = torch.zeros((), device=alpha.device, dtype=fit.dtype)
        total = fit
        evidence = alpha_redl - self.lambda_prior
        alpha0 = alpha_redl.sum(dim=1)
        return {
            "total": total,
            "fit": fit,
            "reg": reg,
            "aux": {
                "lambda_mean": 0.0,
                "lambda_min": 0.0,
                "lambda_max": 0.0,
                "lambda_std": 0.0,
                "lambda_prior": float(self.lambda_prior),
                "alpha0_mean": float(alpha0.mean().detach().item()),
                "alpha0_std": float(alpha0.std(unbiased=False).detach().item()),
                "evidence_mean": float(evidence.mean().detach().item()),
                "evidence_std": float(evidence.std(unbiased=False).detach().item()),
                "info": float("nan"),
                "info_std": float("nan"),
                "fisher_trace": float("nan"),
                "kl_enabled": 0.0,
            },
            "schema_version": LOSS_SCHEMA_VERSION,
        }
