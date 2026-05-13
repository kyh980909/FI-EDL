"""EDL classification head.

    alpha = evidence(logits) + 1
    probs = alpha / alpha.sum(dim=-1)

Evidence activation defaults to `softplus` to match the I-EDL reference
codebase's `FisherEvidentialN.forward` (the supplied configs all use
`clf_type: "softplus"`). `exp` is supported as an opt-in alternative.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.registry.heads import HEAD_REGISTRY


_SUPPORTED_EVIDENCE_FNS = ("softplus", "exp")


@HEAD_REGISTRY.register("edl")
class EDLHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, evidence_fn: str = "softplus") -> None:
        super().__init__()
        evidence_fn = str(evidence_fn).lower()
        if evidence_fn not in _SUPPORTED_EVIDENCE_FNS:
            raise ValueError(
                f"Unsupported evidence_fn={evidence_fn!r}. Supported: {_SUPPORTED_EVIDENCE_FNS}."
            )
        self.fc = nn.Linear(in_dim, num_classes)
        self.evidence_fn = evidence_fn

    def _evidence(self, logits: Tensor) -> Tensor:
        if self.evidence_fn == "exp":
            return torch.exp(logits)
        return F.softplus(logits)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        logits = self.fc(features)
        evidence = self._evidence(logits)
        alpha = evidence + 1.0
        probs = alpha / alpha.sum(dim=1, keepdim=True)
        return {"logits": logits, "evidence": evidence, "alpha": alpha, "probs": probs}
