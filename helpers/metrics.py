"""Scores, the validation-chosen decision threshold, and bootstrap uncertainty.

Threshold rule: the threshold is the value that maximizes Youden's J (sensitivity + specificity - 1) on
the validation subset. Three quarters of the molecules are positive, so a threshold chosen for accuracy
would drift towards calling everything penetrating; Youden's J weights the two classes equally. The
threshold is fixed on validation and then applied unchanged to the matching test subset.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)


def choose_threshold(y_true, scores):
    """Threshold maximizing Youden's J on this subset (used on validation only)."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    best = int(np.argmax(tpr - fpr))
    return float(thresholds[best])


def scores_at(y_true, scores, threshold):
    """Every metric the brief asks for, at one fixed threshold."""
    y_true = np.asarray(y_true).astype(int)
    predicted = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "accuracy": (tp + tn) / len(y_true),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
        "n": len(y_true),
        "prevalence": float(y_true.mean()),
    }


def bootstrap_ci(y_true, scores, metric="roc_auc", n_boot=2000, seed=0, threshold=None):
    """Percentile 95% confidence interval over molecules, resampled with replacement."""
    rng = np.random.default_rng(seed)
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:            # a resample with one class has no ROC-AUC
            continue
        t = threshold if threshold is not None else 0.5
        values.append(scores_at(y_true[idx], scores[idx], t)[metric])
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def paired_bootstrap_ci(y_true, scores_a, scores_b, metric="roc_auc", n_boot=2000, seed=0):
    """95% interval for metric(a) − metric(b), both models scored on the same resampled molecules.

    Drawing one resample per iteration and scoring both models on it keeps the comparison paired. Two
    separate intervals, one per model, each carry the variation that comes from the test subset being
    easy or hard; that variation is shared, so it cancels out of a difference and separate intervals
    overstate how uncertain the comparison is.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    differences = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:            # a resample with one class has no ROC-AUC
            continue
        differences.append(scores_at(y_true[idx], a[idx], 0.5)[metric]
                           - scores_at(y_true[idx], b[idx], 0.5)[metric])
    return (float(np.mean(differences)),
            float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5)))


def summarise_seeds(per_seed):
    """Mean and SD across seeds for every metric (per_seed: list of dicts from `scores_at`)."""
    df = pd.DataFrame(per_seed)
    numeric = df.select_dtypes("number")
    return pd.Series({**{f"{c}_mean": numeric[c].mean() for c in numeric},
                      **{f"{c}_sd": numeric[c].std(ddof=1) for c in numeric}})


def curve_points(y_true, scores):
    """ROC and precision-recall curves plus the chance line for PR (the positive prevalence)."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall,
            "prevalence": float(np.mean(y_true))}
