"""F-EDL classification head (Yoon et al., NeurIPS 2025).

Three sub-heads predict parameters of a Flexible Dirichlet (FD) distribution:

  α = exp(g_α(z))              concentration parameters  [B, K]
  p = softmax(g_p(z))          allocation probabilities  [B, K]
  τ = softplus(g_τ(z)) + ε    dispersion scalar          [B, 1]

Per Algorithm 1 of the paper, spectral normalisation (SN) is applied only to
g_α (φ₁) — NOT to g_p (φ₂) or g_τ (φ₃). SN on the backbone (θ) is handled
separately via ``model.backbone_spectral_norm: true`` in the Hydra config.

The p and τ sub-heads are shallow MLPs (L layers, H hidden units) per
Appendix E.2: L=2, H=256 for CIFAR-10/CIFAR-100; L=1, H=64 for DMNIST-like
settings. These are controlled via ``head_num_layers`` and ``head_hidden_dim``.

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


def _build_mlp(in_dim: int, out_dim: int, num_layers: int, hidden_dim: int) -> nn.Module:
    """Build a shallow MLP: (num_layers-1) hidden ReLU layers + linear output."""
    if num_layers <= 1:
        return nn.Linear(in_dim, out_dim)
    layers: list[nn.Module] = []
    for i in range(num_layers - 1):
        d_in = in_dim if i == 0 else hidden_dim
        layers += [nn.Linear(d_in, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


@HEAD_REGISTRY.register("f_edl")
class FEDLHead(nn.Module):
    """Three-head F-EDL: predicts (α, p, τ) for the Flexible Dirichlet.

    Args:
        in_dim: Backbone output dimension.
        num_classes: Number of target classes K.
        evidence_fn: Activation for α — 'exp' (default, per paper) or 'softplus'.
        head_num_layers: MLP depth for p and τ sub-heads. Paper: 2 (CIFAR-10/100),
            1 (DMNIST). Default 1 (safe for any dataset).
        head_hidden_dim: Hidden units per MLP layer (only used when num_layers>1).
            Paper: 256 (CIFAR-10/100), 64 (DMNIST). Default 256.
        tau_min: Minimum value added to τ after softplus for numerical stability.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        evidence_fn: str = "exp",
        head_num_layers: int = 1,
        head_hidden_dim: int = 256,
        tau_min: float = 1e-4,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.evidence_fn = str(evidence_fn).lower()
        self.tau_min = float(tau_min)

        # φ₁: α head — SN applied per Algorithm 1 of the paper
        self.fc_alpha: nn.Module = spectral_norm(nn.Linear(in_dim, num_classes))

        # φ₂: p head — shallow MLP, NO spectral norm (paper only specifies SN on θ and φ₁)
        self.fc_p = _build_mlp(in_dim, num_classes, head_num_layers, head_hidden_dim)

        # φ₃: τ head — shallow MLP, no spectral norm
        self.fc_tau = _build_mlp(in_dim, 1, head_num_layers, head_hidden_dim)

    def _alpha_activation(self, logits: Tensor) -> Tensor:
        if self.evidence_fn == "softplus":
            return F.softplus(logits)
        return torch.exp(logits)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        logits = self.fc_alpha(features)
        alpha_raw = self._alpha_activation(logits)            # [B, K], positive
        p = F.softmax(self.fc_p(features), dim=1)             # [B, K], on simplex
        tau = F.softplus(self.fc_tau(features)) + self.tau_min  # [B, 1], positive

        # Effective Dirichlet params: α_eff = α_raw + τ·p
        # α0_eff = Σα_raw + τ  →  used by score functions as the precision
        alpha_eff = alpha_raw + tau * p                        # [B, K]
        probs = alpha_eff / alpha_eff.sum(dim=1, keepdim=True)

        return {
            "logits": logits,
            "evidence": alpha_raw,  # raw concentration (auxiliary logging)
            "alpha": alpha_eff,     # effective precision (scores + loss)
            "p": p,
            "tau": tau,
            "probs": probs,
        }
