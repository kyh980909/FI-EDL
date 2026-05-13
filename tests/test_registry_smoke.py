from __future__ import annotations

import pytest

import src.registry  # noqa: F401  populates registries
from src.registry.backbones import BACKBONE_REGISTRY
from src.registry.core import RegistryError
from src.registry.heads import HEAD_REGISTRY
from src.registry.losses import LOSS_REGISTRY
from src.registry.scores import SCORE_REGISTRY


def test_expected_backbones_registered():
    assert set(BACKBONE_REGISTRY.keys()) == {"convnet", "resnet18"}


def test_expected_heads_registered():
    assert set(HEAD_REGISTRY.keys()) == {"edl"}


def test_expected_losses_registered():
    assert set(LOSS_REGISTRY.keys()) == {"edl_fixed", "fi_edl", "i_edl"}


def test_expected_scores_registered():
    assert set(SCORE_REGISTRY.keys()) == {"maxp", "alpha0", "vacuity"}


def test_duplicate_registration_raises():
    with pytest.raises(RegistryError):

        @LOSS_REGISTRY.register("edl_fixed")
        class _Dup:
            pass


def test_unknown_key_raises():
    with pytest.raises(RegistryError):
        LOSS_REGISTRY.get("does_not_exist")
