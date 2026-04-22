from __future__ import annotations

import pytest
from hydra import compose, initialize

from src.registry.validators import validate_registry_bindings


EXPERIMENTS = ("edl_l1", "edl_l01", "edl_l0001", "fi_edl", "i_edl")
DATASETS = ("mnist", "cifar10")
BACKBONES = ("convnet", "resnet18")


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_experiment_config_loads(exp):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=[f"experiment={exp}"])
        assert cfg.loss.name
        assert cfg.experiment.name == exp


@pytest.mark.parametrize("dataset", DATASETS)
def test_dataset_config_loads(dataset):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=[f"dataset={dataset}"])
        assert cfg.data.id == dataset
        assert list(cfg.data.ood_list)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_backbone_config_loads(backbone):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=[f"backbone={backbone}"])
        assert cfg.model.backbone == backbone


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_validator_passes_for_default_stack(exp):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=[f"experiment={exp}"])
        validate_registry_bindings(cfg)
