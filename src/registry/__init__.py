"""Registry root.

Importing this package as a side effect imports every component module so their
`@..._REGISTRY.register(...)` decorators populate the name tables. Do not
remove or "clean up" the side-effect imports below — the registry contents
depend on them.
"""
# Backbones
import src.models.backbones.convnet  # noqa: F401
import src.models.backbones.resnet18  # noqa: F401

# Heads
import src.models.heads.edl_head  # noqa: F401

# Losses
import src.losses.edl_fixed  # noqa: F401
import src.losses.fi_edl  # noqa: F401
import src.losses.i_edl  # noqa: F401
import src.losses.r_edl  # noqa: F401
import src.losses.re_edl  # noqa: F401

# Scores
import src.scores.alpha0  # noqa: F401
import src.scores.maxp  # noqa: F401
import src.scores.vacuity  # noqa: F401

from src.registry.backbones import BACKBONE_REGISTRY
from src.registry.heads import HEAD_REGISTRY
from src.registry.losses import LOSS_REGISTRY
from src.registry.scores import SCORE_REGISTRY

__all__ = ["BACKBONE_REGISTRY", "HEAD_REGISTRY", "LOSS_REGISTRY", "SCORE_REGISTRY"]
