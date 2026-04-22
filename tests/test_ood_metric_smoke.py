from __future__ import annotations

import numpy as np

from src.metrics.ood_metrics import (
    aupr,
    aurc_from_confidence,
    auroc_and_fpr95,
    binary_auroc_and_fpr95,
    multiclass_ece,
    multiclass_nll,
    reliability_bins,
)


def test_auroc_perfect_separation():
    rng = np.random.default_rng(0)
    id_scores = rng.uniform(0.0, 0.1, size=100)
    ood_scores = rng.uniform(0.9, 1.0, size=100)
    auroc, fpr95, _ = auroc_and_fpr95(id_scores=id_scores, ood_scores=ood_scores)
    assert auroc == 1.0
    assert fpr95 == 0.0


def test_aupr_perfect_separation():
    rng = np.random.default_rng(0)
    id_scores = rng.uniform(0.0, 0.1, size=100)
    ood_scores = rng.uniform(0.9, 1.0, size=100)
    assert aupr(id_scores=id_scores, ood_scores=ood_scores) == 1.0


def test_multiclass_ece_perfect_calibration():
    probs = np.zeros((5, 3), dtype=np.float32)
    probs[np.arange(5), [0, 1, 2, 0, 1]] = 1.0
    labels = np.array([0, 1, 2, 0, 1])
    assert multiclass_ece(probs, labels, n_bins=5) == 0.0


def test_multiclass_nll_non_negative():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    labels = np.array([0, 1])
    assert multiclass_nll(probs, labels) > 0.0


def test_reliability_bins_shapes():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(alpha=[1.0, 1.0, 1.0], size=50)
    labels = rng.integers(0, 3, size=50)
    bins = reliability_bins(probs=probs, labels=labels, n_bins=10)
    assert bins["bin_centers"].shape == (10,)
    assert bins["accuracies"].shape == (10,)


def test_aurc_monotone_expectation():
    # A perfect confidence selector should have low AURC.
    confidence = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    correct = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    risk = aurc_from_confidence(confidence, correct)
    assert 0.0 <= risk <= 1.0


def test_binary_auroc_handles_single_class():
    auroc, fpr95, meta = binary_auroc_and_fpr95(
        np.zeros(10, dtype=np.int32), np.linspace(0, 1, 10)
    )
    assert np.isnan(auroc)
    assert np.isnan(fpr95)
    assert np.isnan(meta["threshold"])
