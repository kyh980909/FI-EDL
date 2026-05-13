"""DAEDL loss: density-aware evidential deep learning (Yoon et al., ICML 2024).

The density scaling is performed inside DAEDLHead; this loss applies the
standard EDL MSE + annealed KL objective to the density-weighted α.
Separating it from edl_fixed allows clear per-method experiment tracking.
"""
from __future__ import annotations

from src.losses.edl_fixed import EDLFixedLoss
from src.registry.losses import LOSS_REGISTRY


@LOSS_REGISTRY.register("daedl")
class DAEDLLoss(EDLFixedLoss):
    """DAEDL loss — EDL objective applied to density-weighted evidence from DAEDLHead."""
