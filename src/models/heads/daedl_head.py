"""DAEDL classification head: Density-Aware Evidential Deep Learning.

Reference:
    Taeseong Yoon, Heeyoung Kim.
    "Uncertainty Estimation by Density Aware Evidential Deep Learning."
    ICML 2024.  https://github.com/TaeseongYoon/DAEDL

Algorithm (two-phase):
  Phase 1 — Training (GMM not yet fitted):
    alpha = 1e-6 + exp(spectral_norm_fc(features))   (density-free)

  Phase 2 — Post-training GMM fitting (via fit_gmm()):
    For each class c, fit MultivariateNormal(μ_c, Σ_c) on backbone features.
    Record training-set density [min, max] for normalisation.

  Inference (after fit_gmm):
    log_d_c = -½ (z − μ_c)ᵀ Σ_c⁻¹ (z − μ_c)           # per-class Mahalanobis
    d = max_c(log_d_c)                                   # best-class density
    d_norm = clamp((d − d_min) / (d_max − d_min), 0, 1) # normalise to [0,1]
    logit_w  = logit × d_norm                            # density-weighted logits
    alpha = 1e-6 + exp(logit_w)

GMM parameters are stored as registered buffers so they survive
checkpoint save/load automatically.  The training script (src/train.py)
calls fit_gmm() after trainer.fit() and re-saves the checkpoint.
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
    """DAEDL head: spectral-norm linear classifier + post-training GMM density.

    Args:
        in_dim: Backbone output dimension D.
        num_classes: Number of classes K.
        evidence_fn: Ignored (DAEDL always uses exp activation per official code).
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        evidence_fn: str = "exp",  # kept for interface compatibility; always exp
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_dim = in_dim

        # Spectral-norm classification head (per official codebase)
        self.fc = spectral_norm(nn.Linear(in_dim, num_classes))

        # GMM parameters — populated by fit_gmm(), persisted in checkpoint
        self.register_buffer("gmm_means", torch.zeros(num_classes, in_dim))
        self.register_buffer(
            "gmm_covs_inv",
            torch.eye(in_dim).unsqueeze(0).expand(num_classes, -1, -1).clone(),
        )
        self.register_buffer("gmm_log_dens_min", torch.tensor(0.0))
        self.register_buffer("gmm_log_dens_max", torch.tensor(1.0))
        self.register_buffer("gmm_fitted", torch.zeros(1, dtype=torch.bool))

    # ------------------------------------------------------------------
    # GMM fitting (called once after training, before eval)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit_gmm(self, features: Tensor, labels: Tensor) -> None:
        """Fit class-conditional Gaussians on backbone features.

        Args:
            features: [N, D] backbone outputs from training set.
            labels:   [N]    integer class labels.
        """
        device = features.device
        D = self.in_dim
        means = torch.zeros(self.num_classes, D, device=device)
        covs_inv = torch.eye(D, device=device).unsqueeze(0).expand(self.num_classes, -1, -1).clone()

        for c in range(self.num_classes):
            mask = labels == c
            n_c = mask.sum().item()
            if n_c < 2:
                continue
            feats_c = features[mask].float()
            mean_c = feats_c.mean(0)
            means[c] = mean_c
            centered = feats_c - mean_c.unsqueeze(0)
            cov_c = (centered.T @ centered) / (n_c - 1)
            cov_c = cov_c + 1e-6 * torch.eye(D, device=device)
            try:
                covs_inv[c] = torch.linalg.inv(cov_c)
            except RuntimeError:
                pass  # leave identity if singular

        self.gmm_means.copy_(means)
        self.gmm_covs_inv.copy_(covs_inv)

        # Compute per-sample best-class log-density on training set,
        # then record [min, max] for normalisation at inference time.
        log_dens_all = self._mahalanobis_log_dens(features.float())  # [N, K]
        best = log_dens_all.max(dim=1).values                         # [N]
        self.gmm_log_dens_min.fill_(float(best.min().item()))
        self.gmm_log_dens_max.fill_(float(best.max().item()))
        self.gmm_fitted.fill_(True)

    # ------------------------------------------------------------------
    # Density helpers
    # ------------------------------------------------------------------

    def _mahalanobis_log_dens(self, features: Tensor) -> Tensor:
        """Return per-class Mahalanobis log-density (up to constant). [B, K]"""
        # diff: [B, K, D]
        diff = features.unsqueeze(1) - self.gmm_means.unsqueeze(0)
        # v = Σ_c⁻¹ @ diff^T  per class:  [B, K, D, 1] = [K, D, D] @ [B, K, D, 1]
        v = (self.gmm_covs_inv.unsqueeze(0) @ diff.unsqueeze(-1)).squeeze(-1)  # [B, K, D]
        return -0.5 * (diff * v).sum(dim=-1)  # [B, K]

    def _density_score(self, features: Tensor) -> Tensor:
        """Normalised density scalar ∈ [0, 1] per sample. Shape [B, 1]."""
        log_dens = self._mahalanobis_log_dens(features)  # [B, K]
        best = log_dens.max(dim=1, keepdim=True).values   # [B, 1]
        span = (self.gmm_log_dens_max - self.gmm_log_dens_min).clamp_min(1e-8)
        return ((best - self.gmm_log_dens_min) / span).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        logits = self.fc(features)  # [B, K]

        if self.gmm_fitted.item():
            # Density-weighted logits — only active after fit_gmm()
            density = self._density_score(features.detach())  # [B, 1]
            logits = logits * density

        # alpha = 1e-6 + exp(logits)  — per official DAEDL codebase
        alpha = 1e-6 + torch.exp(logits)
        probs = alpha / alpha.sum(dim=1, keepdim=True)

        return {
            "logits": logits,
            "evidence": alpha,
            "alpha": alpha,
            "probs": probs,
        }
