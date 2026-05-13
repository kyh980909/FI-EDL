"""DAEDL classification head: Distribution-Aware EDL (Yoon et al., ICML 2024).

Key design (per paper description):
  1. Spectral normalisation on the evidence projection layer.
  2. Learnable class prototypes in feature space (K × D matrix).
  3. Density weight: softmax(−dist²/T) — the class whose prototype is
     nearest to the current feature receives a larger evidence boost.
  4. evidence_scaled_k = evidence_k · (1 + γ · density_k)
  5. alpha_k = evidence_scaled_k + 1

The density-weighted evidence is then passed to the standard EDL loss
(``daedl`` in the loss registry).

NOTE: This is a self-contained approximation for matched-pipeline comparison.
      The exact DAEDL algorithm uses feature-space density estimation
      (normalising flow or similar); this implementation uses learnable
      class prototypes with RBF density as a principled substitute.
      Validate against the official code when available.

Reference:
    Taeseong Yoon et al.
    "Distribution-Aware Evidential Deep Learning."
    ICML 2024.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import spectral_norm

from src.registry.heads import HEAD_REGISTRY


@HEAD_REGISTRY.register("daedl")
class DAEDLHead(nn.Module):
    """Spectral-norm EDL head with prototype-based density scaling.

    Args:
        in_dim: Backbone output dimension D.
        num_classes: Number of target classes K.
        evidence_fn: Base evidence activation — 'softplus' (default) or 'exp'.
        density_gamma: Evidence amplification strength γ for density scaling.
        density_temp: Temperature T of the softmax distance kernel.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        evidence_fn: str = "softplus",
        density_gamma: float = 1.0,
        density_temp: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.evidence_fn = str(evidence_fn).lower()
        self.density_gamma = float(density_gamma)
        self.density_temp = max(float(density_temp), 1e-6)

        # Spectral norm on the classification head (Lipschitz smoothness)
        self.fc = spectral_norm(nn.Linear(in_dim, num_classes))

        # Learnable class prototypes in feature space [K, D]
        self.prototypes = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.normal_(self.prototypes, std=0.01)

    def _base_evidence(self, logits: Tensor) -> Tensor:
        if self.evidence_fn == "exp":
            return torch.exp(logits)
        return F.softplus(logits)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        logits = self.fc(features)                      # [B, K]
        evidence = self._base_evidence(logits)          # [B, K], positive

        # Pairwise squared distances: features [B, D], prototypes [K, D] → [B, K]
        diff = features.unsqueeze(1) - self.prototypes.unsqueeze(0)  # [B, K, D]
        sq_dist = diff.pow(2).sum(dim=2)                              # [B, K]

        # Soft class assignment via RBF kernel (nearest prototype = highest weight)
        density = F.softmax(-sq_dist / self.density_temp, dim=1)     # [B, K]

        # Density-amplified evidence and Dirichlet alpha
        evidence_scaled = evidence * (1.0 + self.density_gamma * density)
        alpha = evidence_scaled + 1.0
        probs = alpha / alpha.sum(dim=1, keepdim=True)

        return {
            "logits": logits,
            "evidence": evidence_scaled,
            "alpha": alpha,
            "probs": probs,
        }
