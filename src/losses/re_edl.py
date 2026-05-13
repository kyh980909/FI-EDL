"""Re-EDL loss (Chen et al., arXiv 2024 — extension of ICLR'24 R-EDL).

Re-EDL goes one step further than R-EDL by also dropping the
variance-minimization term ``L_var = sum_k p_k (1 - p_k) / (alpha_0 + 1)``
from the squared-error fit, leaving only the expected squared error
``sum_k (y_k - p_k)^2`` evaluated at the rebased Dirichlet mean.

Loss form (per-sample)::

    alpha_re = evidence + lambda_prior        # rebase, same as R-EDL
    p_re     = alpha_re / sum_k alpha_re_k
    fit      = sum_k (y_k - p_re_k)^2          # NO variance term
    total    = fit                              # NO KL term

This isolates the third "nonessential" setting Re-EDL identifies: the
variance-minimization term is shown to push posteriors toward Dirac deltas
and worsen overconfidence.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.contracts.schemas import LOSS_SCHEMA_VERSION
from src.registry.losses import LOSS_REGISTRY


def _re_edl_fit_per_sample(alpha: Tensor, target: Tensor) -> Tensor:
    s = alpha.sum(dim=1, keepdim=True)
    p = alpha / s
    y = F.one_hot(target, num_classes=alpha.size(1)).float()
    err = (y - p).pow(2)
    return err.sum(dim=1)


@LOSS_REGISTRY.register("re_edl")
class ReEDLLoss(nn.Module):
    """Re-EDL baseline with matched pipeline.

    Same alpha rebasing as R-EDL (evidence + lambda_prior), but drops the
    variance-minimization term in addition to the KL regularizer.
    """

    def __init__(self, lambda_prior: float = 0.1) -> None:
        super().__init__()
        if lambda_prior <= 0.0:
            raise ValueError(
                f"lambda_prior must be positive (Re-EDL assumes a proper Dirichlet), "
                f"got {lambda_prior!r}"
            )
        self.lambda_prior = float(lambda_prior)

    def _rebase_alpha(self, alpha: Tensor) -> Tensor:
        evidence = (alpha - 1.0).clamp_min(0.0)
        return evidence + self.lambda_prior

    def forward(self, alpha: Tensor, target: Tensor, **kwargs: Any) -> Dict[str, Any]:
        alpha_re = self._rebase_alpha(alpha)
        fit_ps = _re_edl_fit_per_sample(alpha_re, target)
        fit = fit_ps.mean()
        reg = torch.zeros((), device=alpha.device, dtype=fit.dtype)
        total = fit
        evidence = alpha_re - self.lambda_prior
        alpha0 = alpha_re.sum(dim=1)
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
