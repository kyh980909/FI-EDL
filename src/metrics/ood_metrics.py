"""OOD-detection and calibration metrics (numpy/sklearn)."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def auroc_and_fpr95(
    id_scores: np.ndarray, ood_scores: np.ndarray
) -> Tuple[float, float, dict]:
    """OOD AUROC and FPR at TPR=0.95 treating OOD as the positive class."""
    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    auroc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, thr = roc_curve(y_true, y_score)
    idx = int(np.argmin(np.abs(tpr - 0.95)))
    fpr95 = float(fpr[idx])
    meta = {"threshold": float(thr[idx]), "tpr": float(tpr[idx]), "fpr": float(fpr[idx])}
    return auroc, fpr95, meta


def aupr(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    return float(average_precision_score(y_true, y_score))


def binary_auroc_and_fpr95(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[float, float, dict]:
    """AUROC / FPR95 on arbitrary binary labels (e.g. correct-vs-wrong)."""
    labels = np.asarray(y_true).astype(np.int32)
    scores = np.asarray(y_score, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must be 1D arrays of identical length")
    if labels.min() == labels.max():
        return (
            float("nan"),
            float("nan"),
            {"threshold": float("nan"), "tpr": float("nan"), "fpr": float("nan")},
        )
    auroc = float(roc_auc_score(labels, scores))
    fpr, tpr, thr = roc_curve(labels, scores)
    idx = int(np.argmin(np.abs(tpr - 0.95)))
    meta = {"threshold": float(thr[idx]), "tpr": float(tpr[idx]), "fpr": float(fpr[idx])}
    return auroc, float(fpr[idx]), meta


def multiclass_nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs[np.arange(labels.shape[0]), labels], eps, 1.0)
    return float(-np.log(p).mean())


def multiclass_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        acc_bin = float(correct[mask].mean())
        conf_bin = float(conf[mask].mean())
        ece += float(mask.mean()) * abs(acc_bin - conf_bin)
    return float(ece)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Dict[str, Any]:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    counts = np.zeros(n_bins, dtype=np.int64)
    accuracies = np.full(n_bins, np.nan, dtype=np.float64)
    confidences = np.full(n_bins, np.nan, dtype=np.float64)
    gaps = np.full(n_bins, np.nan, dtype=np.float64)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        counts[i] = int(mask.sum())
        if not np.any(mask):
            continue
        accuracies[i] = float(correct[mask].mean())
        confidences[i] = float(conf[mask].mean())
        gaps[i] = float(accuracies[i] - confidences[i])
    return {
        "bin_edges": bins,
        "bin_centers": centers,
        "counts": counts,
        "accuracies": accuracies,
        "confidences": confidences,
        "gaps": gaps,
    }


def aurc_from_confidence(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve using confidence as the selector."""
    order = np.argsort(-confidence)
    sorted_correct = correct[order].astype(np.float32)
    risks = [float(1.0 - sorted_correct[:k].mean()) for k in range(1, sorted_correct.shape[0] + 1)]
    return float(np.mean(risks))
