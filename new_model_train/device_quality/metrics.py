import math

import numpy as np
from scipy.special import expit
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


OVERALL_PREFERENCE_SLOPE = 0.693
OVERALL_NOTICEABLE_GAP = 2.0


def _finite_pair(y_true, y_pred):
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(true) & np.isfinite(pred)
    return true[valid], pred[valid]


def _safe_correlation(function, y_true, y_pred):
    if len(y_true) < 2 or np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return np.nan
    return float(function(y_true, y_pred)[0])


def pairwise_ranking_accuracy(y_true, y_pred, minimum_gap=0.0):
    true, pred = _finite_pair(y_true, y_pred)
    correct = []
    for left in range(len(true)):
        for right in range(left + 1, len(true)):
            true_diff = true[left] - true[right]
            if abs(true_diff) <= minimum_gap:
                continue
            pred_diff = pred[left] - pred[right]
            correct.append(float(np.sign(true_diff) == np.sign(pred_diff)))
    return float(np.mean(correct)) if correct else np.nan


def regression_metrics(y_true, y_pred, pairwise_gap=0.0):
    true, pred = _finite_pair(y_true, y_pred)
    if not len(true):
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "pearson": np.nan,
            "spearman": np.nan,
            "kendall_tau": np.nan,
            "pairwise_ranking_accuracy": np.nan,
        }
    return {
        "n": int(len(true)),
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(math.sqrt(mean_squared_error(true, pred))),
        "pearson": _safe_correlation(pearsonr, true, pred),
        "spearman": _safe_correlation(spearmanr, true, pred),
        "kendall_tau": _safe_correlation(kendalltau, true, pred),
        "pairwise_ranking_accuracy": pairwise_ranking_accuracy(true, pred, pairwise_gap),
    }


def overall_pairwise_metrics(y_true, y_pred):
    true, pred = _finite_pair(y_true, y_pred)
    targets = []
    probabilities = []
    hard_correct = []
    for left in range(len(true)):
        for right in range(left + 1, len(true)):
            true_diff = true[left] - true[right]
            if abs(true_diff) < OVERALL_NOTICEABLE_GAP:
                continue
            target = float(expit(OVERALL_PREFERENCE_SLOPE * true_diff))
            probability = float(expit(OVERALL_PREFERENCE_SLOPE * (pred[left] - pred[right])))
            targets.append(target)
            probabilities.append(np.clip(probability, 1e-7, 1.0 - 1e-7))
            hard_correct.append(float(np.sign(true_diff) == np.sign(pred[left] - pred[right])))
    if not targets:
        return {
            "pairwise_ranking_accuracy": np.nan,
            "pairwise_brier_score": np.nan,
            "pairwise_log_loss": np.nan,
            "n_pairs": 0,
        }
    target_array = np.asarray(targets)
    probability_array = np.asarray(probabilities)
    # Soft-target cross entropy; sklearn's log_loss expects hard labels.
    cross_entropy = -np.mean(
        target_array * np.log(probability_array)
        + (1.0 - target_array) * np.log(1.0 - probability_array)
    )
    return {
        "pairwise_ranking_accuracy": float(np.mean(hard_correct)),
        "pairwise_brier_score": float(np.mean(np.square(probability_array - target_array))),
        "pairwise_log_loss": float(cross_entropy),
        "n_pairs": int(len(targets)),
    }


def overall_metrics(y_true, y_pred):
    result = regression_metrics(y_true, y_pred, pairwise_gap=OVERALL_NOTICEABLE_GAP)
    result.update(overall_pairwise_metrics(y_true, y_pred))
    return result


def metric_selection_loss(metrics):
    mae = metrics.get("mae", np.nan)
    spearman = metrics.get("spearman", np.nan)
    pairwise = metrics.get("pairwise_ranking_accuracy", np.nan)
    mae = 100.0 if not np.isfinite(mae) else float(mae)
    spearman = 0.0 if not np.isfinite(spearman) else float(spearman)
    pairwise = 0.5 if not np.isfinite(pairwise) else float(pairwise)
    return mae + 20.0 * (1.0 - spearman) + 10.0 * (1.0 - pairwise)
