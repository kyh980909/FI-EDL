"""F-EDL classification head (Yoon et al., NeurIPS 2025).

Three sub-heads predict parameters of a Flexible Dirichlet (FD) distribution:

  α = exp(g_α(z))              concentration parameters  [B, K]
  p = softmax(g_p(z))          allocation probabilities  [B, K]
  τ = softplus(g_τ(z)) + ε    dispersion scalar          [B, 1]

Spectral normalization is applied to g_α and g_p following the paper.

The output ``alpha`` is the effective Dirichlet precision α_eff = α + τ·p,
which lets existing score functions (vacuity = K/α0_eff, alpha0 = α0_eff)
work without modification.

Reference:
    Taeseong Yoon, Heeyoung Kim.
    "Uncertainty Estimation by Flexible Evidential Deep Learning."
    NeurIPS 2025.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import spectral_norm

from src.registry.heads import HEAD_REGISTRY


@HEAD_REGISTRY.register("f_edl")
class FEDLHead(nn.Module):
    """Three-head F-EDL: predicts (α, p, τ) for the Flexible Dirichlet.

    Args:
        in_dim: Backbone output dimension.
        num_classes: Number of target classes K.
        evidence_fn: Activation for α head — 'exp' (default, per paper) or 'softplus'.
        use_spectral_norm: Apply spectral normalisation to α and p heads (per paper).
        tau_min: Minimum value added to τ after softplus for numerical stability.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        evidence_fn: str = "exp",
        use_spectral_norm: bool = True,
        tau_min: float = 1e-4,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.evidence_fn = str(evidence_fn).lower()
        self.tau_min = float(tau_min)

        fc_alpha = nn.Linear(in_dim, num_classes)
        fc_p = nn.Linear(in_dim, num_classes)
        fc_tau = nn.Linear(in_dim, 1)

        if use_spectral_norm:
            self.fc_alpha: nn.Module = spectral_norm(fc_alpha)
            self.fc_p: nn.Module = spectral_norm(fc_p)
        else:
            self.fc_alpha = fc_alpha
            self.fc_p = fc_p
        self.fc_tau = fc_tau

    def _alpha_activation(self, logits: Tensor) -> Tensor:
        if self.evidence_fn == "softplus":
            return F.softplus(logits)
        return torch.exp(logits)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        logits = self.fc_alpha(features)
        alpha_raw = self._alpha_activation(logits)            # [B, K], positive
        p = F.softmax(self.fc_p(features), dim=1)             # [B, K], on simplex
        tau = F.softplus(self.fc_tau(features)) + self.tau_min  # [B, 1], positive

        # Effective Dirichlet params exposed as `alpha` for downstream scores
        # α_eff_k = α_raw_k + τ · p_k  →  α0_eff = Σα_raw + τ
        alpha_eff = alpha_raw + tau * p                        # [B, K]
        probs = alpha_eff / alpha_eff.sum(dim=1, keepdim=True)

        return {
            "logits": logits,
            "evidence": alpha_raw,  # raw concentration (for auxiliary logging)
            "alpha": alpha_eff,     # effective precision (consumed by scores + loss)
            "p": p,
            "tau": tau,
            "probs": probs,
        }
