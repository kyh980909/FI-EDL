"""Config-side validation for registry bindings.

Called from `src.train.run_train` before the model is built so config typos
surface as a clear ValueError instead of a KeyError deep inside Lightning.
"""
from __future__ import annotations

from omegaconf import DictConfig

from src.registry import BACKBONE_REGISTRY, HEAD_REGISTRY, LOSS_REGISTRY, SCORE_REGISTRY


LOSS_ALLOWED_KEYS = {
    "edl_fixed": {"name", "lambda_value", "anneal_epochs"},
    "fi_edl": {"name", "beta", "gamma", "info_type", "gate_type", "detach_weight"},
    "i_edl": {"name", "lambda_kl", "lambda_logdet", "fisher_c", "kl_anneal_epochs", "anneal_epochs"},
    "r_edl": {"name", "lambda_prior"},
    "re_edl": {"name", "lambda_prior"},
}

LOSS_REQUIRED_KEYS = {
    "edl_fixed": {"name", "lambda_value", "anneal_epochs"},
    "fi_edl": {"name", "beta", "gamma"},
    "i_edl": {"name", "lambda_kl", "lambda_logdet", "fisher_c", "kl_anneal_epochs"},
    "r_edl": {"name", "lambda_prior"},
    "re_edl": {"name", "lambda_prior"},
}


def validate_registry_bindings(cfg: DictConfig) -> None:
    BACKBONE_REGISTRY.get(cfg.model.backbone)
    HEAD_REGISTRY.get(cfg.model.head)
    LOSS_REGISTRY.get(cfg.loss.name)
    SCORE_REGISTRY.get(cfg.score.name)

    loss_name = str(cfg.loss.name)
    allowed = LOSS_ALLOWED_KEYS.get(loss_name)
    required = LOSS_REQUIRED_KEYS.get(loss_name)
    if allowed is None or required is None:
        raise ValueError(f"Missing validator spec for loss '{loss_name}'")

    present = set(cfg.loss.keys())
    extra = sorted(present - allowed)
    missing = sorted(required - present)
    if extra:
        raise ValueError(f"Unused loss config keys for '{loss_name}': {', '.join(extra)}")
    if missing:
        raise ValueError(f"Missing required loss config keys for '{loss_name}': {', '.join(missing)}")
