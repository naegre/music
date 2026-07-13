from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .metrics import OVERALL_PREFERENCE_SLOPE, metric_selection_loss, overall_metrics


@dataclass(frozen=True)
class OverallCandidateSpec:
    alpha: float
    rank_lambda: float

    @property
    def key(self):
        return f"a={self.alpha:g}|rank={self.rank_lambda:g}"


class SoftRankingOverallRegressor:
    """Nonnegative subdir weights with regression and soft pairwise losses."""

    def __init__(self, quality_columns, alpha=1.0, rank_lambda=1.0):
        self.quality_columns = tuple(quality_columns)
        self.alpha = float(alpha)
        self.rank_lambda = float(rank_lambda)

    def _fit_transform(self, frame):
        values = frame.reindex(columns=self.quality_columns).astype(float)
        self.medians_ = values.median(axis=0, skipna=True).fillna(50.0)
        filled = values.fillna(self.medians_)
        self.means_ = filled.mean(axis=0)
        self.scales_ = filled.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        quality = ((filled - self.means_) / self.scales_).to_numpy(dtype=np.float64)
        missing = values.isna().astype(float).to_numpy(dtype=np.float64)
        return np.column_stack([np.ones(len(values)), quality, missing])

    def _transform(self, frame):
        values = frame.reindex(columns=self.quality_columns).astype(float)
        filled = values.fillna(self.medians_)
        quality = ((filled - self.means_) / self.scales_).to_numpy(dtype=np.float64)
        missing = values.isna().astype(float).to_numpy(dtype=np.float64)
        return np.column_stack([np.ones(len(values)), quality, missing])

    def _loss_and_gradient(self, parameters, matrix, target, sample_weight):
        prediction = matrix @ parameters
        target_scale = max(float(np.std(target, ddof=0)), 1.0)
        weight_sum = max(float(np.sum(sample_weight)), 1e-12)
        residual = prediction - target
        regression_loss = float(np.sum(sample_weight * np.square(residual)) / weight_sum)
        regression_loss /= target_scale ** 2
        gradient = 2.0 * matrix.T @ (sample_weight * residual) / weight_sum
        gradient /= target_scale ** 2

        pair_loss = 0.0
        pair_gradient_prediction = np.zeros(len(target), dtype=np.float64)
        pair_count = 0
        for left in range(len(target)):
            for right in range(left + 1, len(target)):
                true_probability = expit(
                    OVERALL_PREFERENCE_SLOPE * (target[left] - target[right])
                )
                predicted_probability = np.clip(
                    expit(OVERALL_PREFERENCE_SLOPE * (prediction[left] - prediction[right])),
                    1e-9,
                    1.0 - 1e-9,
                )
                pair_weight = float(np.sqrt(sample_weight[left] * sample_weight[right]))
                pair_loss += pair_weight * (
                    -true_probability * np.log(predicted_probability)
                    - (1.0 - true_probability) * np.log(1.0 - predicted_probability)
                )
                derivative = (
                    pair_weight
                    * OVERALL_PREFERENCE_SLOPE
                    * (predicted_probability - true_probability)
                )
                pair_gradient_prediction[left] += derivative
                pair_gradient_prediction[right] -= derivative
                pair_count += 1
        if pair_count:
            pair_loss /= pair_count
            pair_gradient = matrix.T @ pair_gradient_prediction / pair_count
        else:
            pair_gradient = np.zeros_like(parameters)

        regularized = parameters.copy()
        regularized[0] = 0.0
        regularization_loss = self.alpha * float(np.sum(np.square(regularized))) / max(len(target), 1)
        regularization_gradient = 2.0 * self.alpha * regularized / max(len(target), 1)
        total_loss = regression_loss + self.rank_lambda * pair_loss + regularization_loss
        total_gradient = gradient + self.rank_lambda * pair_gradient + regularization_gradient
        return total_loss, total_gradient

    def fit(self, frame, target, sample_weight=None):
        target = np.asarray(target, dtype=np.float64)
        if len(target) < 3:
            raise ValueError("The overall model requires at least three labeled devices")
        matrix = self._fit_transform(frame)
        if sample_weight is None:
            sample_weight = np.ones(len(target), dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
        initial = np.zeros(matrix.shape[1], dtype=np.float64)
        initial[0] = float(np.average(target, weights=sample_weight))
        quality_count = len(self.quality_columns)
        bounds = [(None, None)]
        bounds.extend((0.0, None) for _ in range(quality_count))
        bounds.extend((None, None) for _ in range(quality_count))
        result = minimize(
            lambda parameters: self._loss_and_gradient(parameters, matrix, target, sample_weight),
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if not result.success:
            raise RuntimeError(f"Overall optimization failed: {result.message}")
        self.intercept_ = float(result.x[0])
        self.quality_coefficients_ = result.x[1 : 1 + quality_count]
        self.missing_coefficients_ = result.x[1 + quality_count :]
        self.parameters_ = result.x
        self.optimization_result_ = {
            "success": bool(result.success),
            "iterations": int(result.nit),
            "loss": float(result.fun),
        }
        return self

    def predict(self, frame):
        return self._transform(frame) @ self.parameters_

    def coefficient_frame(self):
        rows = [{"feature": "intercept", "coefficient": self.intercept_, "constraint": "free"}]
        for name, coefficient in zip(self.quality_columns, self.quality_coefficients_):
            rows.append(
                {
                    "feature": name,
                    "coefficient": float(coefficient),
                    "constraint": "nonnegative",
                }
            )
        for name, coefficient in zip(self.quality_columns, self.missing_coefficients_):
            rows.append(
                {
                    "feature": f"{name}__missing",
                    "coefficient": float(coefficient),
                    "constraint": "free",
                }
            )
        return pd.DataFrame(rows)


def make_overall_specs(alphas=(0.1, 1.0, 10.0), rank_lambdas=(0.25, 1.0, 4.0)):
    return [
        OverallCandidateSpec(float(alpha), float(rank_lambda))
        for alpha in alphas
        for rank_lambda in rank_lambdas
    ]


def _overall_selection_loss(metrics):
    base = metric_selection_loss(metrics)
    brier = metrics.get("pairwise_brier_score", np.nan)
    pair_log = metrics.get("pairwise_log_loss", np.nan)
    if np.isfinite(brier):
        base += 20.0 * float(brier)
    if np.isfinite(pair_log):
        base += 2.0 * float(pair_log)
    return base


def evaluate_overall_specs(feature_frame, score_frame, quality_columns, specs, device_column="name"):
    scores = score_frame[[device_column, "overall_human_score"]].drop_duplicates(device_column)
    data = scores.merge(feature_frame, on=device_column, how="left", validate="one_to_one")
    data = data.sort_values(device_column).reset_index(drop=True)
    prediction_map = {spec.key: {} for spec in specs}
    for heldout_index in range(len(data)):
        train = data.drop(index=heldout_index)
        heldout = data.iloc[[heldout_index]]
        weights = (
            train["overall_reliability"].fillna(0.5).to_numpy(dtype=np.float64)
            if "overall_reliability" in train
            else None
        )
        for spec in specs:
            model = SoftRankingOverallRegressor(
                quality_columns, alpha=spec.alpha, rank_lambda=spec.rank_lambda
            )
            model.fit(train, train["overall_human_score"], sample_weight=weights)
            prediction_map[spec.key][heldout.iloc[0][device_column]] = float(model.predict(heldout)[0])

    metric_rows = []
    prediction_rows = []
    score_map = data.set_index(device_column)["overall_human_score"].astype(float).to_dict()
    devices = data[device_column].astype(str).tolist()
    for spec in specs:
        predictions = [prediction_map[spec.key][device] for device in devices]
        targets = [score_map[device] for device in devices]
        metrics = overall_metrics(targets, predictions)
        row = {
            "candidate_key": spec.key,
            "alpha": spec.alpha,
            "rank_lambda": spec.rank_lambda,
            **metrics,
        }
        row["selection_loss"] = _overall_selection_loss(metrics)
        metric_rows.append(row)
        for device, target, prediction in zip(devices, targets, predictions):
            prediction_rows.append(
                {
                    device_column: device,
                    "candidate_key": spec.key,
                    "overall_human_score": target,
                    "overall_prediction": prediction,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def select_overall_spec(feature_frame, score_frame, quality_columns, specs, device_column="name"):
    metrics, predictions = evaluate_overall_specs(
        feature_frame, score_frame, quality_columns, specs, device_column
    )
    selected_row = metrics.sort_values(
        ["selection_loss", "alpha", "rank_lambda"], kind="stable"
    ).iloc[0]
    selected = OverallCandidateSpec(float(selected_row["alpha"]), float(selected_row["rank_lambda"]))
    return selected, metrics, predictions


def fit_overall_model(feature_frame, score_frame, quality_columns, spec, device_column="name"):
    scores = score_frame[[device_column, "overall_human_score"]].drop_duplicates(device_column)
    data = scores.merge(feature_frame, on=device_column, how="left", validate="one_to_one")
    weights = (
        data["overall_reliability"].fillna(0.5).to_numpy(dtype=np.float64)
        if "overall_reliability" in data
        else None
    )
    model = SoftRankingOverallRegressor(
        quality_columns, alpha=spec.alpha, rank_lambda=spec.rank_lambda
    )
    model.fit(data, data["overall_human_score"], sample_weight=weights)
    return model
