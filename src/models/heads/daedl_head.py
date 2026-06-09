"""DAEDL classification head: Density-Aware Evidential Deep Learning.

Reference:
    Taeseong Yoon, Heeyoung Kim.
    "Uncertainty Estimation by Density Aware Evidential Deep Learning."
    ICML 2024.  https://github.com/TaeseongYoon/DAEDL

Algorithm (two-phase):
  Phase 1 — Training (GMM not yet fitted):
    alpha = 1e-6 + exp(spectral_norm_fc(features))          (density-free)

  Phase 2 — Post-training GMM fitting (via fit_gmm()):
    For each class c, fit MultivariateNormal(μ_c, Σ_c) on backbone features
    using double precision and progressive jitter (matching official code).
    Store lower-Cholesky factor scale_tril for numerically stable log-prob.

  Inference (after fit_gmm):
    log p(z|c)  = MVN.log_prob via stored Cholesky          # per-class log-density
    log p(z)    = logsumexp_c log p(z|c)                    # log mixture density
    d_norm      = clamp((log p(z) - d_min) / (d_max - d_min), 0, 1)
    logit_w     = logit × d_norm
    alpha       = 1e-6 + exp(logit_w)

Cross-check notes vs official density_estimation.py:
  - Uses torch.double for GMM fitting (official: get_embeddings returns torch.double)
  - Uses progressive jitter from [0, tiny, 1e-307, ..., 0.1] (official: JITTERS list)
  - Uses MultivariateNormal internally → Cholesky stored as scale_tril buffer
  - Mixture density = logsumexp over classes (official: torch.logsumexp(log_probs, dim=-1))

GMM parameters are stored as registered buffers so they survive checkpoint
save/load. src/train.py calls fit_gmm() after trainer.fit() and re-saves.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import spectral_norm

from src.registry.heads import HEAD_REGISTRY

# Progressive jitter values matching official DAEDL JITTERS list
_JITTERS = [0.0, torch.finfo(torch.double).tiny] + [10 ** e for e in range(-307, 0, 1)]


@HEAD_REGISTRY.register("daedl")
class DAEDLHead(nn.Module):
    """DAEDL head: spectral-norm linear + post-training class-conditional GMM density.

    Args:
        in_dim: Backbone output dimension D.
        num_classes: Number of classes K.
        evidence_fn: Ignored (DAEDL always uses exp activation per official code).
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        evidence_fn: str = "exp",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_dim = in_dim

        self.fc = spectral_norm(nn.Linear(in_dim, num_classes))

        # GMM buffers — populated by fit_gmm(), persisted through checkpoint
        self.register_buffer("gmm_means", torch.zeros(num_classes, in_dim))
        # Lower-Cholesky of covariance per class [K, D, D]
        self.register_buffer(
            "gmm_scale_trils",
            torch.eye(in_dim).unsqueeze(0).expand(num_classes, -1, -1).clone(),
        )
        self.register_buffer("gmm_log_dens_min", torch.tensor(0.0))
        self.register_buffer("gmm_log_dens_max", torch.tensor(1.0))
        self.register_buffer("gmm_fitted", torch.zeros(1, dtype=torch.bool))

    # ------------------------------------------------------------------
    # GMM fitting — called once post-training from src/train.py
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit_gmm(self, features: Tensor, labels: Tensor) -> None:
        """Fit class-conditional Gaussians on backbone features.

        Uses double precision and progressive jitter to match the official
        DAEDL density_estimation.py (gmm_fit / centered_cov_torch).

        Args:
            features: [N, D] float backbone outputs from training set.
            labels:   [N]    integer class labels.
        """
        device = features.device
        D = self.in_dim

        # Compute per-class mean + Bessel-corrected covariance in double precision
        feats_d = features.double()
        means_d = torch.zeros(self.num_classes, D, dtype=torch.double, device=device)
        scale_trils = torch.eye(D, device=device).unsqueeze(0).expand(
            self.num_classes, -1, -1
        ).clone().float()

        for c in range(self.num_classes):
            mask = labels == c
            n_c = int(mask.sum().item())
            if n_c < 2:
                continue
            fc = feats_d[mask]
            mu_c = fc.mean(0)
            means_d[c] = mu_c
            centered = fc - mu_c.unsqueeze(0)                      # [n_c, D]
            cov_c = centered.T.mm(centered) / (n_c - 1)            # Bessel correction

            # Progressive jitter: increase until Cholesky succeeds with a
            # numerically usable diagonal. Rank-deficient covariance (e.g. dead
            # ReLU dims) gives tiny (~1e-45) or zero diagonals; the resulting
            # triangular solve produces NaN/inf maha distances, which leak NaN
            # into log densities downstream. Require ≥ MIN_DIAG.
            MIN_DIAG = 1e-3
            for jitter in _JITTERS:
                try:
                    jitter_mat = jitter * torch.eye(D, dtype=torch.double, device=device)
                    dist = torch.distributions.MultivariateNormal(
                        loc=mu_c,
                        covariance_matrix=cov_c + jitter_mat,
                    )
                    L = dist.scale_tril.float()
                    if L.diagonal().min().item() >= MIN_DIAG:
                        scale_trils[c] = L
                        break
                except (RuntimeError, ValueError):
                    continue

        self.gmm_means.copy_(means_d.float())
        self.gmm_scale_trils.copy_(scale_trils)

        # Record training log-mixture-density [min, max] for normalisation
        log_dens = self._log_mixture_density(features.float())  # [N]
        self.gmm_log_dens_min.fill_(float(log_dens.min().item()))
        self.gmm_log_dens_max.fill_(float(log_dens.max().item()))
        self.gmm_fitted.fill_(True)

    # ------------------------------------------------------------------
    # Density helpers
    # ------------------------------------------------------------------

    def _log_prob_per_class(self, features: Tensor) -> Tensor:
        """Per-class MVN log-probability via stored Cholesky. Returns [B, K].

        Matches official gmm_forward → gaussians_model.log_prob(z[:, None, :]).
        Formula: -0.5 * [D*log(2π) + 2*log|L_c| + ||L_c^{-1}(z - μ_c)||²]
        """
        B, D = features.shape
        K = self.num_classes

        # Solve per class to avoid materialising [B, K, D, D] (≈125 GB at B=48k, D=256).
        maha = torch.empty(B, K, device=features.device, dtype=features.dtype)
        for c in range(K):
            diff_c = features - self.gmm_means[c]                              # [B, D]
            v_c = torch.linalg.solve_triangular(
                self.gmm_scale_trils[c], diff_c.t(), upper=False
            ).t()                                                              # [B, D]
            maha[:, c] = v_c.pow(2).sum(dim=-1)

        # log|Σ_c| = 2 * Σ_d log L_c[d,d]
        log_dets = (
            self.gmm_scale_trils.diagonal(dim1=-2, dim2=-1)           # [K, D]
            .log()
            .sum(dim=-1)                                               # [K]
        )
        log_2pi = D * math.log(2 * math.pi)

        return -0.5 * (log_2pi + 2.0 * log_dets.unsqueeze(0) + maha)  # [B, K]

    def _log_mixture_density(self, features: Tensor) -> Tensor:
        """Log mixture density log p(z) = logsumexp_c log p(z|c). Returns [B]."""
        return torch.logsumexp(self._log_prob_per_class(features), dim=1)

    def _density_score(self, features: Tensor) -> Tensor:
        """Normalised mixture density ∈ [0, 1] per sample. Shape [B, 1].

        Matches official ood_detection.py normalisation:
            d_norm = (d - train_min) / (train_max - train_min)
        """
        log_d = self._log_mixture_density(features).unsqueeze(1)  # [B, 1]
        span = (self.gmm_log_dens_max - self.gmm_log_dens_min).clamp_min(1e-8)
        return ((log_d - self.gmm_log_dens_min) / span).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        logits = self.fc(features)          # [B, K]

        if self.gmm_fitted.item():
            # Density-weighted logits: logit_w = logit × d_norm
            density = self._density_score(features.detach())  # [B, 1]
            logits = logits * density

        # alpha = 1e-6 + exp(logits)  — per official train_daedl()
        alpha = 1e-6 + torch.exp(logits)
        probs = alpha / alpha.sum(dim=1, keepdim=True)

        return {
            "logits": logits,
            "evidence": alpha,
            "alpha": alpha,
            "probs": probs,
        }
