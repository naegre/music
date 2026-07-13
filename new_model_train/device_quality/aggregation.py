from pathlib import Path

import numpy as np
import pandas as pd

from .config import ColumnConfig
from .normalization import PairedRobustNormalizer
from .validation import validate_and_clean_x_metrics


def _safe_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    return float(np.std(values, ddof=1))


def aggregate_normalized_x(normalized, normalizer, columns=None):
    columns = columns or normalizer.columns
    rows = []
    for (device, subdir), group in normalized.groupby(
        [columns.device, columns.subdir], dropna=False, sort=False
    ):
        row = {columns.device: str(device), columns.subdir: str(subdir)}
        n_x = int(len(group))
        unique_x = int(group[columns.audio_id].astype(str).nunique())
        expected = len(normalizer.expected_x_by_subdir_.get(str(subdir), []))
        row["n_x"] = n_x
        row["n_unique_x"] = unique_x
        row["expected_x"] = expected
        row["coverage_ratio"] = min(1.0, unique_x / expected) if expected else np.nan
        row["n_windows_total"] = int(group[columns.n_windows].sum())
        row["has_multiple_x"] = int(n_x >= 2)
        row["uncertainty_mode"] = "between_and_within" if n_x >= 2 else "within_only"

        fallback_used = False
        for spec in columns.metric_specs:
            values = group[f"{spec.key}_quality"].to_numpy(dtype=np.float64)
            within_se = group[f"{spec.key}_within_se"].to_numpy(dtype=np.float64)
            between_std = _safe_std(values)
            between_se = between_std / np.sqrt(n_x) if n_x >= 2 else np.nan
            within_component = float(np.sum(np.square(within_se)) / (n_x ** 2))
            between_component = float((between_std ** 2) / n_x) if n_x >= 2 else 0.0
            total_se = float(np.sqrt(max(within_component + between_component, 0.0)))

            row[f"{spec.key}_quality_mean"] = float(np.mean(values))
            row[f"{spec.key}_quality_median"] = float(np.median(values))
            row[f"{spec.key}_between_x_std"] = between_std
            row[f"{spec.key}_between_x_se"] = between_se
            row[f"{spec.key}_within_se_mean"] = float(np.mean(within_se))
            row[f"{spec.key}_total_se"] = total_se
            row[f"{spec.key}_stability_quality"] = -between_std if np.isfinite(between_std) else np.nan
            row[f"{spec.key}_uncertainty_quality"] = -total_se
            fallback_used = fallback_used or (
                group[f"{spec.key}_normalization_level"].astype(str) != "subdir_x"
            ).any()

        count_reliability = np.sqrt(min(n_x, 3) / 3.0)
        coverage_reliability = np.sqrt(row["coverage_ratio"]) if np.isfinite(row["coverage_ratio"]) else 0.5
        row["reliability_weight"] = float(np.clip(count_reliability * coverage_reliability, 0.25, 1.0))
        row["normalization_fallback_used"] = int(fallback_used)
        warnings = []
        if n_x == 1:
            warnings.append("single_x_within_only")
        if np.isfinite(row["coverage_ratio"]) and row["coverage_ratio"] < 1.0:
            warnings.append("partial_x_coverage")
        if fallback_used:
            warnings.append("normalization_fallback")
        row["reliability_warning"] = "|".join(warnings)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_transform_aggregate(fit_frame, transform_frame, columns=None):
    columns = columns or ColumnConfig()
    normalizer = PairedRobustNormalizer(columns).fit(fit_frame)
    normalized = normalizer.transform(transform_frame)
    aggregated = aggregate_normalized_x(normalized, normalizer, columns)
    return normalized, aggregated, normalizer


def load_x_metrics(csv_or_frame, columns=None):
    columns = columns or ColumnConfig()
    if isinstance(csv_or_frame, (str, Path)):
        frame = pd.read_csv(csv_or_frame)
    else:
        frame = csv_or_frame.copy()
    return validate_and_clean_x_metrics(frame, columns)
