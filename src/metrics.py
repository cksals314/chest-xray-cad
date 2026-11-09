"""Evaluation metrics for 14-label multi-label classification."""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width-bin Expected Calibration Error for one binary label."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)

    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if y_true.size == 0:
        return float("nan")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob contains NaN or infinite values")

    y_prob = np.clip(y_prob, 0.0, 1.0)
    # Include probability=1.0 in the final bin.
    bin_ids = np.minimum((y_prob * n_bins).astype(np.int64), n_bins - 1)

    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        bin_accuracy = float(y_true[mask].mean())
        bin_confidence = float(y_prob[mask].mean())
        ece += float(mask.mean()) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    disease_names: Sequence[str],
    n_bins: int = 15,
) -> tuple[dict[str, float], list[dict[str, float | str | int]]]:
    """Compute macro AUROC, macro AUPRC (average precision), and macro ECE.

    AUROC is undefined when a validation label has only one class; that class is
    excluded from the macro AUROC. Average precision and ECE are still computed.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_true.ndim != 2 or y_prob.ndim != 2 or y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must be matching 2-D arrays")
    if y_true.shape[1] != len(disease_names):
        raise ValueError("Number of columns does not match disease_names")
    if not np.isfinite(y_prob).all():
        raise ValueError("Predicted probabilities contain NaN or infinite values")

    per_class_rows: list[dict[str, float | str | int]] = []
    auc_values: list[float] = []
    ap_values: list[float] = []
    ece_values: list[float] = []

    for class_idx, disease in enumerate(disease_names):
        truth = y_true[:, class_idx].astype(np.int64)
        probability = np.clip(y_prob[:, class_idx], 0.0, 1.0)
        positives = int(truth.sum())
        negatives = int(truth.size - positives)

        if np.unique(truth).size == 2:
            auc = float(roc_auc_score(truth, probability))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auprc = float(average_precision_score(truth, probability))
            auc_values.append(auc)
            ap_values.append(auprc)
        else:
            auc = float("nan")
            # AP has a defined convention for all-negative targets (0.0).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auprc = float(average_precision_score(truth, probability))
            ap_values.append(auprc)

        ece = binary_ece(truth, probability, n_bins=n_bins)
        ece_values.append(ece)
        per_class_rows.append(
            {
                "Disease": disease,
                "Positive_Count": positives,
                "Negative_Count": negatives,
                "AUROC": auc,
                "AUPRC": auprc,
                "ECE": ece,
            }
        )

    macro_auc = float(np.mean(auc_values)) if auc_values else float("nan")
    macro_ap = float(np.mean(ap_values)) if ap_values else float("nan")
    macro_ece = float(np.mean(ece_values)) if ece_values else float("nan")

    summary = {
        "AUROC": macro_auc,
        "AUPRC": macro_ap,
        "ECE": macro_ece,
        "AUROC_Valid_Classes": int(len(auc_values)),
    }
    return summary, per_class_rows
