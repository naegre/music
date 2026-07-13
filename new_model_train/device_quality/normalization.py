from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ColumnConfig


EPS = 1e-8


def _robust_center_scale(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0, "constant"

    center = float(np.median(values))
    mad_scale = 1.4826 * float(np.median(np.abs(values - center)))
    if mad_scale > EPS:
        return center, mad_scale, "mad"

    if values.size >= 2:
        q25, q75 = np.percentile(values, [25, 75])
        iqr_scale = float((q75 - q25) / 1.349)
        if iqr_scale > EPS:
            return center, iqr_scale, "iqr"

        std_scale = float(np.std(values, ddof=1))
        if std_scale > EPS:
            return center, std_scale, "std"
    return center, 1.0, "constant"


def _transform_mean_variance(mean, variance, transform):
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.maximum(np.asarray(variance, dtype=np.float64), 0.0)
    if transform == "log1p":
        transformed = np.log1p(np.maximum(mean, 0.0))
        transformed_variance = variance / np.square(1.0 + np.maximum(mean, 0.0))
        return transformed, transformed_variance
    if transform == "fisher":
        clipped = np.clip(mean, -0.999999, 0.999999)
        transformed = np.arctanh(clipped)
        derivative = 1.0 / np.maximum(1.0 - np.square(clipped), EPS)
        transformed_variance = variance * np.square(derivative)
        return transformed, transformed_variance
    raise ValueError(f"Unknown metric transform: {transform}")


@dataclass
class NormalizationStat:
    center: float
    scale: float
    method: str
    count: int


class PairedRobustNormalizer:
    def __init__(self, columns=None):
        self.columns = columns or ColumnConfig()

    def _transformed_frame(self, frame):
        result = frame.copy()
        n_windows = result[self.columns.n_windows].to_numpy(dtype=np.float64)
        result["n_effective"] = np.maximum(1.0, (n_windows + 4.0) / 5.0)
        for spec in self.columns.metric_specs:
            transformed, transformed_var = _transform_mean_variance(
                result[spec.mean_column].to_numpy(dtype=np.float64),
                result[spec.variance_column].to_numpy(dtype=np.float64),
                spec.transform,
            )
            result[f"__{spec.key}_transformed"] = transformed
            result[f"__{spec.key}_transformed_var"] = transformed_var
        return result

    @staticmethod
    def _make_stat(values):
        center, scale, method = _robust_center_scale(values)
        return NormalizationStat(center, scale, method, int(np.isfinite(values).sum()))

    def fit(self, frame):
        transformed = self._transformed_frame(frame)
        self.exact_stats_ = {}
        self.subdir_stats_ = {}
        self.global_stats_ = {}

        for spec in self.columns.metric_specs:
            value_col = f"__{spec.key}_transformed"
            for (subdir, audio_id), group in transformed.groupby(
                [self.columns.subdir, self.columns.audio_id], dropna=False
            ):
                self.exact_stats_[(str(subdir), str(audio_id), spec.key)] = self._make_stat(
                    group[value_col].to_numpy(dtype=np.float64)
                )
            for subdir, group in transformed.groupby(self.columns.subdir, dropna=False):
                self.subdir_stats_[(str(subdir), spec.key)] = self._make_stat(
                    group[value_col].to_numpy(dtype=np.float64)
                )
            self.global_stats_[spec.key] = self._make_stat(
                transformed[value_col].to_numpy(dtype=np.float64)
            )

        self.expected_x_by_subdir_ = {
            str(subdir): sorted(group[self.columns.audio_id].astype(str).unique().tolist())
            for subdir, group in transformed.groupby(self.columns.subdir, dropna=False)
        }
        self.fit_devices_ = sorted(transformed[self.columns.device].astype(str).unique().tolist())
        return self

    def _lookup(self, subdir, audio_id, metric_key):
        exact_key = (str(subdir), str(audio_id), metric_key)
        if exact_key in self.exact_stats_:
            return self.exact_stats_[exact_key], "subdir_x"
        subdir_key = (str(subdir), metric_key)
        if subdir_key in self.subdir_stats_:
            return self.subdir_stats_[subdir_key], "subdir_fallback"
        return self.global_stats_[metric_key], "global_fallback"

    def transform(self, frame):
        transformed = self._transformed_frame(frame)
        for spec in self.columns.metric_specs:
            values = transformed[f"__{spec.key}_transformed"].to_numpy(dtype=np.float64)
            variances = transformed[f"__{spec.key}_transformed_var"].to_numpy(dtype=np.float64)
            n_effective = transformed["n_effective"].to_numpy(dtype=np.float64)
            qualities = np.empty(len(transformed), dtype=np.float64)
            within_se = np.empty(len(transformed), dtype=np.float64)
            levels = []
            methods = []

            for index, (_, row) in enumerate(transformed.iterrows()):
                stat, level = self._lookup(row[self.columns.subdir], row[self.columns.audio_id], spec.key)
                qualities[index] = spec.direction * (values[index] - stat.center) / stat.scale
                within_se[index] = np.sqrt(max(variances[index], 0.0) / n_effective[index]) / stat.scale
                levels.append(level)
                methods.append(stat.method)

            transformed[f"{spec.key}_quality"] = qualities
            transformed[f"{spec.key}_within_se"] = within_se
            transformed[f"{spec.key}_normalization_level"] = levels
            transformed[f"{spec.key}_normalization_method"] = methods
        return transformed

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)

    def summary_frame(self):
        rows = []
        for (subdir, audio_id, metric), stat in self.exact_stats_.items():
            rows.append(
                {
                    "level": "subdir_x",
                    "subdir": subdir,
                    "x": audio_id,
                    "metric": metric,
                    "center": stat.center,
                    "scale": stat.scale,
                    "method": stat.method,
                    "count": stat.count,
                }
            )
        for (subdir, metric), stat in self.subdir_stats_.items():
            rows.append(
                {
                    "level": "subdir_fallback",
                    "subdir": subdir,
                    "x": "",
                    "metric": metric,
                    "center": stat.center,
                    "scale": stat.scale,
                    "method": stat.method,
                    "count": stat.count,
                }
            )
        return pd.DataFrame(rows)
