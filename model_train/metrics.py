import math

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _safe_correlation(func, y_true, y_pred):
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    result = func(y_true, y_pred)
    return float(result.statistic)


def preference_probability(score_diff, tie_margin=1.0, preference_at_margin=0.8):
    slope = math.log(preference_at_margin / (1.0 - preference_at_margin)) / tie_margin
    value = np.clip(slope * np.asarray(score_diff, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-value))


def pairwise_metrics(df, target_col, prediction_col, group_col=None, tie_margin=1.0, preference_at_margin=0.8):
    hard_correct = []
    tie_correct = []
    brier = []
    log_losses = []

    groups = [(None, df)] if group_col is None else df.groupby(group_col, dropna=False)
    for _, group in groups:
        group = group.dropna(subset=[target_col, prediction_col]).reset_index(drop=True)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                true_diff = float(group.loc[i, target_col] - group.loc[j, target_col])
                pred_diff = float(group.loc[i, prediction_col] - group.loc[j, prediction_col])
                p_true = float(preference_probability(true_diff, tie_margin, preference_at_margin))
                p_pred = float(preference_probability(pred_diff, tie_margin, preference_at_margin))
                brier.append((p_true - p_pred) ** 2)
                p_pred = min(max(p_pred, 1e-8), 1.0 - 1e-8)
                log_losses.append(-(p_true * math.log(p_pred) + (1.0 - p_true) * math.log(1.0 - p_pred)))

                if abs(true_diff) > tie_margin:
                    hard_correct.append(float(np.sign(true_diff) == np.sign(pred_diff)))
                else:
                    tie_correct.append(float(abs(pred_diff) <= tie_margin))

    return {
        "pairwise_accuracy_over_margin": float(np.mean(hard_correct)) if hard_correct else np.nan,
        "num_ordered_pairs": int(len(hard_correct)),
        "tie_accuracy": float(np.mean(tie_correct)) if tie_correct else np.nan,
        "num_tie_pairs": int(len(tie_correct)),
        "pairwise_brier": float(np.mean(brier)) if brier else np.nan,
        "pairwise_log_loss": float(np.mean(log_losses)) if log_losses else np.nan,
    }


def regression_ranking_metrics(df, target_col, prediction_col, group_col=None, tie_margin=1.0, preference_at_margin=0.8):
    valid = df.dropna(subset=[target_col, prediction_col])
    y_true = valid[target_col].to_numpy(dtype=np.float64)
    y_pred = valid[prediction_col].to_numpy(dtype=np.float64)
    result = {
        "num_rows": int(len(valid)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pearson": _safe_correlation(pearsonr, y_true, y_pred),
        "spearman": _safe_correlation(spearmanr, y_true, y_pred),
        "kendall_tau": _safe_correlation(kendalltau, y_true, y_pred),
    }
    result.update(
        pairwise_metrics(
            valid,
            target_col,
            prediction_col,
            group_col=group_col,
            tie_margin=tie_margin,
            preference_at_margin=preference_at_margin,
        )
    )
    return result
