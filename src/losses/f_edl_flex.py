"""F-EDL loss: Flexible Dirichlet expected MSE + Brier score on p.

Reference:
    Taeseong Yoon, Heeyoung Kim.
    "Uncertainty Estimation by Flexible Evidential Deep Learning."
    NeurIPS 2025.

The F-EDL head predicts (α, p, τ) and passes α_eff = α + τ·p as ``alpha``.
This loss computes (per sample):

    L = E_{π~FD(α,p,τ)}[‖y − π‖²]  +  ‖y − p‖²

Using closed-form FD moments (Appendix A of paper):

    μ_k   = α_eff_k / α0_eff                              (FD mean)
    Var[π_k] = μ_k(1−μ_k)/(α0_eff+1)
             + τ²·p_k(1−p_k) / (α0_eff·(α0_eff+1))      (FD variance)

where α0_eff = Σ_k α_eff_k = Σ_k α_raw_k + τ.

The first term expands to Σ_k[(y_k−μ_k)² + Var[π_k]], the second term is the
Brier score on the allocation probabilities p (input-dependent calibration).
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.contracts.schemas import LOSS_SCHEMA_VERSION
from src.registry.losses import LOSS_REGISTRY


@LOSS_REGISTRY.register("f_edl")
class FEDLFlexLoss(nn.Module):
    """F-EDL Flexible Dirichlet loss — no tuneable hyperparameters."""

    def forward(self, alpha: Tensor, target: Tensor, **kwargs: Any) -> Dict[str, Any]:
        p = kwargs.get("p")
        tau = kwargs.get("tau")
        if p is None or tau is None:
            raise ValueError(
                "f_edl loss requires 'p' and 'tau' from FEDLHead. "
                "Ensure experiment config has model.head=f_edl."
            )

        # alpha = α_eff = α_raw + τ·p  [B, K]
        # tau: [B, 1]
        alpha0 = alpha.sum(dim=1, keepdim=True)  # α0_eff  [B, 1]
        mu = alpha / alpha0                        # E[π_k]  [B, K]

        # FD variance decomposition
        denom_a = alpha0 + 1.0                     # α0+1
        denom_b = alpha0 * (alpha0 + 1.0)          # α0(α0+1)
        var_fd = mu * (1.0 - mu) / denom_a + tau.pow(2) * p * (1.0 - p) / denom_b

        y = F.one_hot(target, num_classes=alpha.size(1)).float()

        # E_FD[‖y − π‖²] = Σ_k[(y_k − μ_k)² + Var[π_k]]
        fit = ((y - mu).pow(2) + var_fd).sum(dim=1).mean()

        # Brier regularisation on allocation probs (promotes input-dependent calibration)
        reg = ((y - p).pow(2)).sum(dim=1).mean()

        total = fit + reg

        tau_scalar = tau.squeeze(1)  # [B] for std computation
        return {
            "total": total,
            "fit": fit,
            "reg": reg,
            "aux": {
                "lambda_mean": 1.0,
                "lambda_min": 1.0,
                "lambda_max": 1.0,
                "lambda_std": 0.0,
                "tau_mean": float(tau_scalar.mean().detach().item()),
                "tau_std": float(tau_scalar.std(unbiased=False).detach().item()),
                "alpha0_mean": float(alpha0.mean().detach().item()),
                "alpha0_std": float(alpha0.std(unbiased=False).detach().item()),
                "info": float(alpha0.mean().detach().item()),
                "info_std": float(alpha0.std(unbiased=False).detach().item()),
                "fisher_trace": float("nan"),
            },
            "schema_version": LOSS_SCHEMA_VERSION,
        }
