"""FI-EDL loss: Fisher Information–gated adaptive KL weight (this paper).

The per-sample weight is

    lambda(v_info) = beta * exp(-gamma * norm(v_info))

with v_info ∈ {trace(I(alpha)), alpha_0, (alpha·y)_t} selected via `info_type`.
The signal is detached by default so the adaptive weight does not receive
gradients — only the KL term does.

`signal_norm` controls the *scale* the gate operates on. With the raw Fisher
trace (≈ O(K) ≈ 14.5 for K=10), the default gamma=1 makes exp(-gamma·trace)
≈ 5e-7, i.e. the gate saturates OFF and no KL regularisation is applied.
Normalising the control signal restores a usable operating range so gamma
actually modulates per-sample regularisation:
  - "none"    : raw signal (legacy; gate saturated for Fisher trace).
  - "div_k"   : signal / K  → O(1) absolute scale.
  - "batch_z" : (signal - mean) / std within the batch → relative gating,
                dataset-agnostic, lambda centred at beta (recommended).
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.contracts.schemas import LOSS_SCHEMA_VERSION
from src.losses.edl_fixed import _edl_fit_per_sample, _kl_dirichlet_to_uniform_per_sample
from src.registry.losses import LOSS_REGISTRY


def _fisher_trace(alpha: Tensor) -> Tensor:
    """Trace of the Dirichlet Fisher information via trigamma diagonal terms."""
    trigamma_alpha = torch.polygamma(1, alpha)
    trigamma_sum = torch.polygamma(1, alpha.sum(dim=1, keepdim=True))
    return (trigamma_alpha - trigamma_sum).sum(dim=1)


@LOSS_REGISTRY.register("fi_edl")
class FIEDLLoss(nn.Module):
    def __init__(
        self,
        beta: float = 1.0,
        gamma: float = 1.0,
        info_type: str = "fisher",
        gate_type: str = "exp",
        detach_weight: bool = True,
        signal_norm: str = "none",
        anneal_epochs: float = 10.0,
        kl_target_mask: bool = False,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.info_type = str(info_type)
        self.gate_type = str(gate_type)
        self.detach_weight = bool(detach_weight)
        self.signal_norm = str(signal_norm)
        # KL warmup: ramp the regularisation contribution 0→1 over the first
        # `anneal_epochs` epochs. Standard EDL practice — applying full KL from
        # epoch 0 collapses the Dirichlet to uniform before the backbone learns
        # features (val/acc stuck at chance). Harmless for the legacy gate-off
        # default (signal_norm=none → lambda≈5e-7, so the ramped term stays ≈0).
        self.anneal_epochs = float(anneal_epochs)
        # If true, compute the KL on alpha_hat = alpha·(1-y) + y so the
        # target-class evidence is replaced by 1 and excluded from the
        # uniform-prior penalty (standard EDL practice). Without masking,
        # the full-alpha KL drives the target-class probability to 1/K →
        # severe underconfidence (acc 90% / conf 10% / ECE ~70%) when the
        # adaptive gate restores meaningful KL strength.
        self.kl_target_mask = bool(kl_target_mask)

    def _control_signal(self, alpha: Tensor, target: Tensor) -> Tensor:
        info_type = self.info_type.lower()
        if info_type == "fisher":
            return _fisher_trace(alpha)
        if info_type == "alpha0":
            return alpha.sum(dim=1)
        if info_type == "target_alpha":
            y = F.one_hot(target, num_classes=alpha.size(1)).float()
            return (alpha * y).sum(dim=1)
        raise ValueError(f"Unsupported info_type: {self.info_type}")

    def _normalize_signal(self, signal: Tensor, num_classes: int) -> Tensor:
        norm = self.signal_norm.lower()
        if norm == "none":
            return signal
        if norm == "div_k":
            return signal / float(num_classes)
        if norm == "batch_z":
            mean = signal.mean()
            std = signal.std(unbiased=False).clamp_min(1e-6)
            # Clamp the z-score so the exp gate does not blow up on batch
            # outliers (exp(-gamma·z) is unbounded as z → -∞).
            return ((signal - mean) / std).clamp(-3.0, 3.0)
        raise ValueError(f"Unsupported signal_norm: {self.signal_norm}")

    def _lambda_weight(self, signal: Tensor) -> Tensor:
        gate_type = self.gate_type.lower()
        if gate_type == "exp":
            return self.beta * torch.exp(-self.gamma * signal)
        if gate_type == "constant":
            return torch.full_like(signal, fill_value=self.beta)
        raise ValueError(f"Unsupported gate_type: {self.gate_type}")

    def forward(self, alpha: Tensor, target: Tensor, **kwargs: Any) -> Dict[str, Any]:
        fit_ps = _edl_fit_per_sample(alpha, target)
        if self.kl_target_mask:
            y = F.one_hot(target, num_classes=alpha.size(1)).float()
            alpha_for_kl = alpha * (1.0 - y) + y
        else:
            alpha_for_kl = alpha
        kl_ps = _kl_dirichlet_to_uniform_per_sample(alpha_for_kl)
        fit = fit_ps.mean()
        reg = kl_ps.mean()
        signal = self._control_signal(alpha=alpha, target=target)
        if self.detach_weight:
            signal = signal.detach()
        signal_n = self._normalize_signal(signal, num_classes=alpha.size(1))
        lam = self._lambda_weight(signal_n)
        epoch = float(kwargs.get("epoch", 0.0))
        warmup = min(1.0, max(epoch, 0.0) / max(self.anneal_epochs, 1e-6))
        total = fit + warmup * (lam * kl_ps).mean()

        fisher_mean = float(_fisher_trace(alpha).mean().detach().item())
        return {
            "total": total,
            "fit": fit,
            "reg": reg,
            "aux": {
                "lambda_mean": float(lam.mean().item()),
                "lambda_min": float(lam.min().item()),
                "lambda_max": float(lam.max().item()),
                "lambda_std": float(lam.std(unbiased=False).item()),
                "info": float(signal.mean().detach().item()),
                "info_std": float(signal.std(unbiased=False).detach().item()),
                "info_norm": float(signal_n.mean().detach().item()),
                "warmup": float(warmup),
                "fisher_trace": fisher_mean,
                "detach_weight": float(self.detach_weight),
            },
            "schema_version": LOSS_SCHEMA_VERSION,
        }
