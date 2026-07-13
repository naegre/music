from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import nnls


PRIMARY_FEATURES = (
    "l2_quality_mean",
    "l2_quality_median",
    "cosim_quality_mean",
    "cosim_quality_median",
)

STABILITY_FEATURES = (
    "l2_stability_quality",
    "cosim_stability_quality",
    "l2_uncertainty_quality",
    "cosim_uncertainty_quality",
)

AUXILIARY_FEATURES = (
    "kl_quality_mean",
    "kl_quality_median",
    "js_quality_mean",
    "js_quality_median",
    "kl_stability_quality",
    "js_stability_quality",
)


@dataclass(frozen=True)
class FeatureSchema:
    columns: tuple
    add_missing_indicators: bool = False


class FoldNumericTransformer:
    """Median imputation and standardization fitted inside a training fold."""

    def __init__(self, schema):
        self.schema = schema

    def fit(self, frame):
        values = frame.reindex(columns=self.schema.columns).astype(float)
        medians = values.median(axis=0, skipna=True).fillna(0.0)
        filled = values.fillna(medians)
        means = filled.mean(axis=0)
        scales = filled.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        self.medians_ = medians
        self.means_ = means
        self.scales_ = scales
        self.feature_names_ = list(self.schema.columns)
        if self.schema.add_missing_indicators:
            self.feature_names_.extend(f"{name}__missing" for name in self.schema.columns)
        return self

    def transform(self, frame):
        values = frame.reindex(columns=self.schema.columns).astype(float)
        missing = values.isna().astype(float)
        standardized = (values.fillna(self.medians_) - self.means_) / self.scales_
        arrays = [standardized.to_numpy(dtype=np.float64)]
        if self.schema.add_missing_indicators:
            arrays.append(missing.to_numpy(dtype=np.float64))
        return np.concatenate(arrays, axis=1)

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)


class PositiveRidgeRegressor:
    """A small monotonic linear model over already quality-oriented features."""

    def __init__(self, feature_columns, alpha=10.0, add_missing_indicators=False):
        self.feature_columns = tuple(feature_columns)
        self.alpha = float(alpha)
        self.add_missing_indicators = bool(add_missing_indicators)

    def fit(self, frame, target, sample_weight=None):
        schema = FeatureSchema(self.feature_columns, self.add_missing_indicators)
        self.transformer_ = FoldNumericTransformer(schema)
        matrix = self.transformer_.fit_transform(frame)
        target = np.asarray(target, dtype=np.float64)
        if sample_weight is None:
            sample_weight = np.ones(len(target), dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
        sample_weight = sample_weight / max(float(np.mean(sample_weight)), 1e-12)
        x_center = np.average(matrix, axis=0, weights=sample_weight)
        y_center = float(np.average(target, weights=sample_weight))
        centered_x = matrix - x_center
        centered_y = target - y_center
        root_weight = np.sqrt(sample_weight)
        augmented_x = np.vstack(
            [
                centered_x * root_weight[:, None],
                np.sqrt(self.alpha) * np.eye(matrix.shape[1]),
            ]
        )
        augmented_y = np.concatenate([centered_y * root_weight, np.zeros(matrix.shape[1])])
        self.coef_, self.residual_norm_ = nnls(augmented_x, augmented_y)
        self.intercept_ = y_center - float(x_center @ self.coef_)
        return self

    def predict(self, frame):
        matrix = self.transformer_.transform(frame)
        return self.intercept_ + matrix @ self.coef_

    def coefficient_frame(self, model_name):
        rows = [
            {
                "model": model_name,
                "feature": "intercept",
                "coefficient": self.intercept_,
                "constraint": "free",
            }
        ]
        rows.extend(
            {
                "model": model_name,
                "feature": feature,
                "coefficient": float(coefficient),
                "constraint": "nonnegative",
            }
            for feature, coefficient in zip(self.transformer_.feature_names_, self.coef_)
        )
        return pd.DataFrame(rows)
