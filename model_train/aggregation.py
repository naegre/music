from pathlib import Path

import numpy as np
import pandas as pd

from .config import ColumnConfig


def _safe_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def _numeric_values(group, column):
    return pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(dtype=np.float64)


def _pooled_mean_variance(group, mean_col, var_col):
    means = pd.to_numeric(group[mean_col], errors="coerce")
    variances = pd.to_numeric(group[var_col], errors="coerce")
    valid = means.notna() & variances.notna()
    means = means[valid].to_numpy(dtype=np.float64)
    variances = np.maximum(variances[valid].to_numpy(dtype=np.float64), 0.0)
    if means.size == 0:
        return np.nan, np.nan
    pooled_mean = float(means.mean())
    pooled_var = float(np.mean(variances + means ** 2) - pooled_mean ** 2)
    return pooled_mean, max(pooled_var, 0.0)


def validate_input_columns(df, columns):
    required = set(columns.audio_group_columns + columns.metric_columns)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")


def aggregate_windows_to_audio(df, columns=None):
    columns = columns or ColumnConfig()
    validate_input_columns(df, columns)
    rows = []

    for keys, group in df.groupby(columns.audio_group_columns, dropna=False, sort=False):
        row = dict(zip(columns.audio_group_columns, keys))
        row["n_windows"] = int(len(group))
        if columns.test_file in group.columns:
            paths = group[columns.test_file].dropna().astype(str).unique().tolist()
            row["test_files"] = "|".join(paths)

        for metric in columns.metric_columns:
            values = _numeric_values(group, metric)
            if values.size == 0:
                for suffix in ("mean", "median", "std", "min", "max", "range"):
                    row[f"{metric}_audio_{suffix}"] = np.nan
                continue
            row[f"{metric}_audio_mean"] = float(values.mean())
            row[f"{metric}_audio_median"] = float(np.median(values))
            row[f"{metric}_audio_std"] = _safe_std(values)
            row[f"{metric}_audio_min"] = float(values.min())
            row[f"{metric}_audio_max"] = float(values.max())
            row[f"{metric}_audio_range"] = float(values.max() - values.min())

        kl_mean, kl_var = _pooled_mean_variance(group, columns.kl_mean, columns.kl_var)
        js_mean, js_var = _pooled_mean_variance(group, columns.js_mean, columns.js_var)
        row["kl_pooled_audio_mean"] = kl_mean
        row["kl_pooled_audio_var"] = kl_var
        row["js_pooled_audio_mean"] = js_mean
        row["js_pooled_audio_var"] = js_var
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_audio_to_device_subdir(audio_df, columns=None):
    columns = columns or ColumnConfig()
    required = set(columns.device_subdir_group_columns + [columns.audio_id])
    missing = sorted(required - set(audio_df.columns))
    if missing:
        raise ValueError(f"Missing audio-level columns: {missing}")

    excluded = set(columns.audio_group_columns + ["test_files", "n_windows"])
    feature_cols = [
        col for col in audio_df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(audio_df[col])
    ]
    rows = []

    for keys, group in audio_df.groupby(columns.device_subdir_group_columns, dropna=False, sort=False):
        row = dict(zip(columns.device_subdir_group_columns, keys))
        row["n_audio"] = int(len(group))
        row["n_windows_total"] = int(group["n_windows"].sum())

        for feature in feature_cols:
            values = pd.to_numeric(group[feature], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if values.size == 0:
                for suffix in ("device_mean", "device_median", "between_audio_std", "standard_error"):
                    row[f"{feature}_{suffix}"] = np.nan
                continue
            std = _safe_std(values)
            row[f"{feature}_device_mean"] = float(values.mean())
            row[f"{feature}_device_median"] = float(np.median(values))
            row[f"{feature}_between_audio_std"] = std
            row[f"{feature}_standard_error"] = float(std / np.sqrt(values.size))
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_metrics(csv_or_df, columns=None):
    columns = columns or ColumnConfig()
    if isinstance(csv_or_df, (str, Path)):
        window_df = pd.read_csv(csv_or_df)
    else:
        window_df = csv_or_df.copy()
    audio_df = aggregate_windows_to_audio(window_df, columns)
    device_subdir_df = aggregate_audio_to_device_subdir(audio_df, columns)
    return audio_df, device_subdir_df
