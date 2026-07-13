from copy import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .aggregation import aggregate_normalized_x
from .config import ColumnConfig
from .features import (
    AUXILIARY_FEATURES,
    PRIMARY_FEATURES,
    STABILITY_FEATURES,
    PositiveRidgeRegressor,
)
from .metrics import metric_selection_loss, regression_metrics
from .normalization import PairedRobustNormalizer
from .targets import PercentileTargetTransformer


ALLOWED_GAMMAS = (0.0, 0.05, 0.10, 0.20)


class SubdirTrainingCache:
    """Per-training-run cache keyed by the complete device fold definition."""

    def __init__(self):
        self.fold_training_data = {}
        self.fold_eval_data = {}
        self.selections = {}
        self.models = {}
        self.primary_oof_residuals = {}


@dataclass(frozen=True)
class SubdirCandidateSpec:
    variant: str
    alpha: float
    gamma: float = 0.0
    auxiliary_alpha: float = 100.0

    @property
    def key(self):
        return f"{self.variant}|a={self.alpha:g}|g={self.gamma:g}|aa={self.auxiliary_alpha:g}"

    @property
    def primary_features(self):
        if self.variant == "primary":
            return PRIMARY_FEATURES
        return PRIMARY_FEATURES + STABILITY_FEATURES

    @property
    def ablation_name(self):
        if self.variant == "primary":
            return "primary_only"
        if self.variant == "stability":
            return "primary_plus_stability"
        return "primary_plus_stability_plus_kl_js"


class FittedSubdirModel:
    def __init__(
        self,
        subdir,
        columns,
        normalizer,
        spec,
        target_transformer,
        primary_model,
        auxiliary_model=None,
    ):
        self.subdir = str(subdir)
        self.columns = columns
        self.normalizer = normalizer
        self.spec = spec
        self.target_transformer = target_transformer
        self.primary_model = primary_model
        self.auxiliary_model = auxiliary_model

    def predict_aggregated(self, aggregated):
        if aggregated.empty:
            return np.empty(0, dtype=np.float64)
        prediction = self.primary_model.predict(aggregated)
        if self.auxiliary_model is not None and self.spec.gamma > 0:
            prediction = prediction + self.spec.gamma * self.auxiliary_model.predict(aggregated)
        return np.clip(prediction, 0.0, 100.0)

    def transform_raw(self, raw_frame):
        subset = raw_frame.loc[
            raw_frame[self.columns.subdir].astype(str) == self.subdir
        ].copy()
        if subset.empty:
            return pd.DataFrame()
        normalized = self.normalizer.transform(subset)
        return aggregate_normalized_x(normalized, self.normalizer, self.columns)

    def predict_raw(self, raw_frame):
        aggregated = self.transform_raw(raw_frame)
        if aggregated.empty:
            return aggregated
        aggregated = aggregated.copy()
        aggregated["predicted_percentile"] = self.predict_aggregated(aggregated)
        return aggregated

    def coefficient_frame(self):
        frames = [self.primary_model.coefficient_frame(f"subdir:{self.subdir}:primary")]
        if self.auxiliary_model is not None:
            frames.append(self.auxiliary_model.coefficient_frame(f"subdir:{self.subdir}:auxiliary"))
        result = pd.concat(frames, ignore_index=True)
        result.insert(0, "subdir", self.subdir)
        result["variant"] = self.spec.variant
        result["alpha"] = self.spec.alpha
        result["gamma"] = self.spec.gamma
        is_auxiliary = result["model"].str.endswith(":auxiliary")
        result["effective_coefficient"] = result["coefficient"]
        result.loc[is_auxiliary, "effective_coefficient"] *= self.spec.gamma
        return result


def make_candidate_specs(alphas=(1.0, 10.0, 100.0), gammas=ALLOWED_GAMMAS, auxiliary_alpha=100.0):
    invalid = sorted(set(float(value) for value in gammas) - set(ALLOWED_GAMMAS))
    if invalid:
        raise ValueError(f"gamma values must be chosen from {ALLOWED_GAMMAS}; got {invalid}")
    specs = []
    for alpha in alphas:
        specs.append(SubdirCandidateSpec("primary", float(alpha), 0.0, auxiliary_alpha))
        specs.append(SubdirCandidateSpec("stability", float(alpha), 0.0, auxiliary_alpha))
        for gamma in gammas:
            specs.append(SubdirCandidateSpec("full", float(alpha), float(gamma), auxiliary_alpha))
    return specs


def _available_devices(raw_frame, label_frame, subdir, columns):
    raw_devices = set(
        raw_frame.loc[
            raw_frame[columns.subdir].astype(str) == str(subdir), columns.device
        ].astype(str)
    )
    label_devices = set(
        label_frame.loc[
            label_frame[columns.subdir].astype(str) == str(subdir), columns.device
        ].astype(str)
    )
    return sorted(raw_devices & label_devices)


def _prepare_fold(
    raw_frame,
    subdir,
    train_devices,
    eval_devices,
    columns,
    reference_frame=None,
    training_cache=None,
):
    train_key = (
        str(subdir),
        tuple(sorted(str(value) for value in train_devices)),
    )
    subdir_mask = raw_frame[columns.subdir].astype(str) == str(subdir)
    if reference_frame is None and training_cache is not None:
        if train_key not in training_cache.fold_training_data:
            reference_mask = subdir_mask & raw_frame[columns.device].astype(str).isin(train_devices)
            reference = raw_frame.loc[reference_mask].copy()
            if reference.empty:
                raise ValueError(f"No normalization reference rows for subdir {subdir!r}")
            normalizer = PairedRobustNormalizer(columns).fit(reference)
            normalized_train = normalizer.transform(reference)
            train_aggregated = aggregate_normalized_x(
                normalized_train, normalizer, columns
            ).reset_index(drop=True)
            training_cache.fold_training_data[train_key] = (train_aggregated, normalizer)
        train_aggregated, normalizer = training_cache.fold_training_data[train_key]
        eval_key = train_key + (tuple(sorted(str(value) for value in eval_devices)),)
        if eval_key not in training_cache.fold_eval_data:
            eval_mask = subdir_mask & raw_frame[columns.device].astype(str).isin(eval_devices)
            eval_subset = raw_frame.loc[eval_mask].copy()
            if eval_subset.empty:
                eval_aggregated = pd.DataFrame()
            else:
                normalized_eval = normalizer.transform(eval_subset)
                eval_aggregated = aggregate_normalized_x(
                    normalized_eval, normalizer, columns
                ).reset_index(drop=True)
            training_cache.fold_eval_data[eval_key] = eval_aggregated
        return train_aggregated, training_cache.fold_eval_data[eval_key], normalizer

    if reference_frame is None:
        reference_mask = subdir_mask & raw_frame[columns.device].astype(str).isin(train_devices)
        reference = raw_frame.loc[reference_mask].copy()
    else:
        reference = reference_frame.loc[
            reference_frame[columns.subdir].astype(str) == str(subdir)
        ].copy()
    if reference.empty:
        raise ValueError(f"No normalization reference rows for subdir {subdir!r}")

    normalizer = PairedRobustNormalizer(columns).fit(reference)
    requested = set(str(value) for value in train_devices) | set(str(value) for value in eval_devices)
    transform_mask = subdir_mask & raw_frame[columns.device].astype(str).isin(requested)
    transformed = normalizer.transform(raw_frame.loc[transform_mask].copy())
    aggregated = aggregate_normalized_x(transformed, normalizer, columns)
    device_values = aggregated[columns.device].astype(str)
    train_aggregated = aggregated.loc[device_values.isin(train_devices)].reset_index(drop=True)
    eval_aggregated = aggregated.loc[device_values.isin(eval_devices)].reset_index(drop=True)
    return train_aggregated, eval_aggregated, normalizer


def _merge_training_labels(aggregated, label_frame, subdir, columns):
    labels = label_frame.loc[
        label_frame[columns.subdir].astype(str) == str(subdir),
        [columns.device, "human_score"],
    ].copy()
    labels[columns.device] = labels[columns.device].astype(str)
    merged = aggregated.merge(labels, on=columns.device, how="inner", validate="one_to_one")
    if len(merged) < 2:
        raise ValueError(f"Subdir {subdir!r} needs at least two labeled devices in a fold")
    return merged


def _fit_on_aggregated(
    train_aggregated,
    label_frame,
    subdir,
    spec,
    columns,
    normalizer,
    auxiliary_residual_by_device=None,
):
    training = _merge_training_labels(train_aggregated, label_frame, subdir, columns).reset_index(
        drop=True
    )
    target_transformer = PercentileTargetTransformer()
    target = target_transformer.fit_transform(training["human_score"].to_numpy(dtype=np.float64))
    weights = training["reliability_weight"].to_numpy(dtype=np.float64)

    primary = PositiveRidgeRegressor(spec.primary_features, alpha=spec.alpha)
    primary.fit(training, target, sample_weight=weights)
    auxiliary = None
    if spec.variant == "full":
        if auxiliary_residual_by_device is None:
            raise ValueError("Full subdir model requires fold-safe primary OOF residuals")
        residual = np.asarray(
            [
                auxiliary_residual_by_device[str(device)]
                for device in training[columns.device].astype(str)
            ],
            dtype=np.float64,
        )
        auxiliary = PositiveRidgeRegressor(AUXILIARY_FEATURES, alpha=spec.auxiliary_alpha)
        auxiliary.fit(training, residual, sample_weight=weights)

    return FittedSubdirModel(
        subdir,
        columns,
        normalizer,
        spec,
        target_transformer,
        primary,
        auxiliary,
    )


def _fold_safe_primary_oof_residuals(
    raw_frame,
    label_frame,
    subdir,
    train_devices,
    spec,
    columns,
    training_cache=None,
):
    train_devices = tuple(sorted(str(value) for value in train_devices))
    cache_key = (str(subdir), train_devices, float(spec.alpha), tuple(spec.primary_features))
    if (
        training_cache is not None
        and cache_key in training_cache.primary_oof_residuals
    ):
        return training_cache.primary_oof_residuals[cache_key]

    labels = label_frame.loc[
        (label_frame[columns.subdir].astype(str) == str(subdir))
        & label_frame[columns.device].astype(str).isin(train_devices),
        [columns.device, "human_score"],
    ].drop_duplicates(columns.device)
    score_map = labels.set_index(columns.device)["human_score"].astype(float).to_dict()
    target_transformer = PercentileTargetTransformer()
    full_target = target_transformer.fit_transform([score_map[device] for device in train_devices])
    target_map = dict(zip(train_devices, full_target))
    primary_spec = SubdirCandidateSpec(
        "stability", spec.alpha, 0.0, spec.auxiliary_alpha
    )
    residuals = {}
    for heldout in train_devices:
        inner_train_devices = [device for device in train_devices if device != heldout]
        inner_aggregated, eval_aggregated, normalizer = _prepare_fold(
            raw_frame,
            subdir,
            inner_train_devices,
            [heldout],
            columns,
            training_cache=training_cache,
        )
        model_key = (
            str(subdir),
            tuple(sorted(inner_train_devices)),
            f"residual_primary|a={primary_spec.alpha:g}",
        )
        if training_cache is not None and model_key in training_cache.models:
            inner_model = training_cache.models[model_key]
        else:
            inner_model = _fit_on_aggregated(
                inner_aggregated,
                label_frame,
                subdir,
                primary_spec,
                columns,
                normalizer,
            )
            if training_cache is not None:
                training_cache.models[model_key] = inner_model
        residuals[heldout] = target_map[heldout] - float(
            inner_model.predict_aggregated(eval_aggregated)[0]
        )
    if training_cache is not None:
        training_cache.primary_oof_residuals[cache_key] = residuals
    return residuals


def fit_subdir_model(
    raw_frame,
    label_frame,
    subdir,
    train_devices,
    spec,
    columns=None,
    normalization_reference=None,
    training_cache=None,
):
    columns = columns or ColumnConfig()
    model_cache_key = (str(subdir), tuple(sorted(str(value) for value in train_devices)), spec.key)
    if (
        normalization_reference is None
        and training_cache is not None
        and model_cache_key in training_cache.models
    ):
        return training_cache.models[model_cache_key]
    train_aggregated, _, normalizer = _prepare_fold(
        raw_frame,
        subdir,
        train_devices,
        (),
        columns,
        reference_frame=normalization_reference,
        training_cache=training_cache,
    )
    residuals = None
    if spec.variant == "full":
        residuals = _fold_safe_primary_oof_residuals(
            raw_frame,
            label_frame,
            subdir,
            train_devices,
            spec,
            columns,
            training_cache,
        )
    model = _fit_on_aggregated(
        train_aggregated,
        label_frame,
        subdir,
        spec,
        columns,
        normalizer,
        auxiliary_residual_by_device=residuals,
    )
    if normalization_reference is None and training_cache is not None:
        training_cache.models[model_cache_key] = model
    return model


def evaluate_candidate_specs(
    raw_frame,
    label_frame,
    subdir,
    devices,
    specs,
    columns=None,
    training_cache=None,
):
    columns = columns or ColumnConfig()
    devices = sorted(set(str(value) for value in devices))
    labels = label_frame.loc[
        (label_frame[columns.subdir].astype(str) == str(subdir))
        & label_frame[columns.device].astype(str).isin(devices),
        [columns.device, "human_score"],
    ].drop_duplicates(columns.device)
    label_map = labels.set_index(columns.device)["human_score"].astype(float).to_dict()
    target_transformer = PercentileTargetTransformer()
    ordered_scores = np.asarray([label_map[device] for device in devices], dtype=np.float64)
    ordered_targets = target_transformer.fit_transform(ordered_scores)
    true_target = dict(zip(devices, ordered_targets))
    predictions = {spec.key: {} for spec in specs}

    for heldout in devices:
        train_devices = [device for device in devices if device != heldout]
        if len(train_devices) < 2:
            continue
        train_aggregated, eval_aggregated, normalizer = _prepare_fold(
            raw_frame,
            subdir,
            train_devices,
            [heldout],
            columns,
            training_cache=training_cache,
        )
        if eval_aggregated.empty:
            continue
        fitted_cache = {}
        for spec in specs:
            fit_key = (spec.variant, spec.alpha, spec.auxiliary_alpha)
            if fit_key not in fitted_cache:
                persistent_key = (
                    str(subdir),
                    tuple(sorted(train_devices)),
                    f"{spec.variant}|a={spec.alpha:g}|aa={spec.auxiliary_alpha:g}",
                )
                if training_cache is not None and persistent_key in training_cache.models:
                    fitted_cache[fit_key] = training_cache.models[persistent_key]
                else:
                    residuals = None
                    if spec.variant == "full":
                        residuals = _fold_safe_primary_oof_residuals(
                            raw_frame,
                            label_frame,
                            subdir,
                            train_devices,
                            spec,
                            columns,
                            training_cache,
                        )
                    fitted_cache[fit_key] = _fit_on_aggregated(
                        train_aggregated,
                        label_frame,
                        subdir,
                        spec,
                        columns,
                        normalizer,
                        auxiliary_residual_by_device=residuals,
                    )
                    if training_cache is not None:
                        training_cache.models[persistent_key] = fitted_cache[fit_key]
            model = copy(fitted_cache[fit_key])
            model.spec = spec
            predictions[spec.key][heldout] = float(model.predict_aggregated(eval_aggregated)[0])

    metric_rows = []
    prediction_rows = []
    for spec in specs:
        evaluated_devices = [device for device in devices if device in predictions[spec.key]]
        true_values = [true_target[device] for device in evaluated_devices]
        predicted_values = [predictions[spec.key][device] for device in evaluated_devices]
        metrics = regression_metrics(true_values, predicted_values)
        row = {
            "subdir": str(subdir),
            "candidate_key": spec.key,
            "variant": spec.variant,
            "ablation": spec.ablation_name,
            "alpha": spec.alpha,
            "gamma": spec.gamma,
            "auxiliary_alpha": spec.auxiliary_alpha,
            **metrics,
        }
        row["selection_loss"] = metric_selection_loss(metrics)
        metric_rows.append(row)
        for device, true_value, predicted_value in zip(evaluated_devices, true_values, predicted_values):
            prediction_rows.append(
                {
                    "subdir": str(subdir),
                    columns.device: device,
                    "candidate_key": spec.key,
                    "target_percentile": true_value,
                    "predicted_percentile": predicted_value,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def _spec_from_row(row):
    return SubdirCandidateSpec(
        str(row["variant"]),
        float(row["alpha"]),
        float(row["gamma"]),
        float(row["auxiliary_alpha"]),
    )


def _best_row(frame):
    if frame.empty:
        raise ValueError("No candidate model could be evaluated")
    return frame.sort_values(["selection_loss", "alpha", "gamma"], kind="stable").iloc[0]


def select_subdir_spec(
    raw_frame,
    label_frame,
    subdir,
    devices,
    specs,
    columns=None,
    training_cache=None,
):
    columns = columns or ColumnConfig()
    selection_key = (
        str(subdir),
        tuple(sorted(str(value) for value in devices)),
        tuple(spec.key for spec in specs),
    )
    if training_cache is not None and selection_key in training_cache.selections:
        return training_cache.selections[selection_key]
    metrics, predictions = evaluate_candidate_specs(
        raw_frame,
        label_frame,
        subdir,
        devices,
        specs,
        columns,
        training_cache=training_cache,
    )
    primary_row = _best_row(metrics.loc[metrics["variant"] == "primary"])
    stability_row = _best_row(metrics.loc[metrics["variant"] == "stability"])
    baseline_row = _best_row(pd.DataFrame([primary_row, stability_row]))

    nonzero = metrics.loc[(metrics["variant"] == "full") & (metrics["gamma"] > 0)].copy()
    selected_row = baseline_row
    auxiliary_accepted = False
    if not nonzero.empty:
        auxiliary_row = _best_row(nonzero)
        base_spearman = baseline_row["spearman"] if np.isfinite(baseline_row["spearman"]) else 0.0
        aux_spearman = auxiliary_row["spearman"] if np.isfinite(auxiliary_row["spearman"]) else 0.0
        base_rank = (
            baseline_row["pairwise_ranking_accuracy"]
            if np.isfinite(baseline_row["pairwise_ranking_accuracy"])
            else 0.5
        )
        aux_rank = (
            auxiliary_row["pairwise_ranking_accuracy"]
            if np.isfinite(auxiliary_row["pairwise_ranking_accuracy"])
            else 0.5
        )
        no_material_regression = (
            auxiliary_row["mae"] <= baseline_row["mae"] + 0.5
            and aux_spearman >= base_spearman - 0.01
            and aux_rank >= base_rank - 0.01
        )
        meaningful_gain = (
            auxiliary_row["mae"] <= baseline_row["mae"] - 0.5
            or aux_spearman >= base_spearman + 0.01
            or aux_rank >= base_rank + 0.01
        )
        better_composite = auxiliary_row["selection_loss"] < baseline_row["selection_loss"]
        if no_material_regression and meaningful_gain and better_composite:
            selected_row = auxiliary_row
            auxiliary_accepted = True

    selected = _spec_from_row(selected_row)
    best_full = _best_row(metrics.loc[metrics["variant"] == "full"])
    ablation = pd.DataFrame([primary_row, stability_row, best_full]).reset_index(drop=True)
    ablation["selected_for_final"] = ablation["candidate_key"].eq(selected.key).astype(int)
    selection = {
        "selected_spec": selected,
        "auxiliary_accepted": auxiliary_accepted,
        "candidate_metrics": metrics,
        "candidate_predictions": predictions,
        "ablation": ablation,
    }
    if training_cache is not None:
        training_cache.selections[selection_key] = selection
    return selection


def crossfit_fixed_subdir_spec(
    raw_frame,
    label_frame,
    subdir,
    devices,
    spec,
    columns=None,
    training_cache=None,
):
    columns = columns or ColumnConfig()
    devices = sorted(set(str(value) for value in devices))
    rows = []
    for heldout in devices:
        train_devices = [device for device in devices if device != heldout]
        train_aggregated, eval_aggregated, normalizer = _prepare_fold(
            raw_frame,
            subdir,
            train_devices,
            [heldout],
            columns,
            training_cache=training_cache,
        )
        if eval_aggregated.empty:
            continue
        model = _fit_on_aggregated(
            train_aggregated,
            label_frame,
            subdir,
            spec,
            columns,
            normalizer,
            auxiliary_residual_by_device=(
                _fold_safe_primary_oof_residuals(
                    raw_frame,
                    label_frame,
                    subdir,
                    train_devices,
                    spec,
                    columns,
                    training_cache,
                )
                if spec.variant == "full"
                else None
            ),
        )
        prediction = float(model.predict_aggregated(eval_aggregated)[0])
        rows.append({columns.device: heldout, "subdir": str(subdir), "predicted_percentile": prediction})
    return pd.DataFrame(rows)


def train_subdir_with_nested_loodo(
    raw_frame,
    label_frame,
    subdir,
    columns=None,
    alphas=(1.0, 10.0, 100.0),
    gammas=ALLOWED_GAMMAS,
    auxiliary_alpha=100.0,
    training_cache=None,
):
    columns = columns or ColumnConfig()
    devices = _available_devices(raw_frame, label_frame, subdir, columns)
    if len(devices) < 4:
        raise ValueError(
            f"Subdir {subdir!r} has only {len(devices)} labeled devices with CSV data; "
            "nested leave-one-device-out requires at least four"
        )
    specs = make_candidate_specs(alphas, gammas, auxiliary_alpha)
    labels = label_frame.loc[
        (label_frame[columns.subdir].astype(str) == str(subdir))
        & label_frame[columns.device].astype(str).isin(devices),
        [columns.device, "human_score"],
    ].drop_duplicates(columns.device)
    score_map = labels.set_index(columns.device)["human_score"].astype(float).to_dict()
    target_transformer = PercentileTargetTransformer()
    target_values = target_transformer.fit_transform([score_map[device] for device in devices])
    target_map = dict(zip(devices, target_values))

    oof_rows = []
    fold_rows = []
    for heldout in devices:
        outer_train = [device for device in devices if device != heldout]
        selection = select_subdir_spec(
            raw_frame,
            label_frame,
            subdir,
            outer_train,
            specs,
            columns,
            training_cache=training_cache,
        )
        selected_spec = selection["selected_spec"]
        train_aggregated, eval_aggregated, normalizer = _prepare_fold(
            raw_frame,
            subdir,
            outer_train,
            [heldout],
            columns,
            training_cache=training_cache,
        )
        model = _fit_on_aggregated(
            train_aggregated,
            label_frame,
            subdir,
            selected_spec,
            columns,
            normalizer,
            auxiliary_residual_by_device=(
                _fold_safe_primary_oof_residuals(
                    raw_frame,
                    label_frame,
                    subdir,
                    outer_train,
                    selected_spec,
                    columns,
                    training_cache,
                )
                if selected_spec.variant == "full"
                else None
            ),
        )
        prediction = float(model.predict_aggregated(eval_aggregated)[0])
        oof_rows.append(
            {
                columns.device: heldout,
                "subdir": str(subdir),
                "human_score": score_map[heldout],
                "target_percentile": target_map[heldout],
                "predicted_percentile": prediction,
                "selected_variant": selected_spec.variant,
                "selected_alpha": selected_spec.alpha,
                "selected_gamma": selected_spec.gamma,
            }
        )
        fold_rows.append(
            {
                "subdir": str(subdir),
                "heldout_device": heldout,
                "selected_variant": selected_spec.variant,
                "selected_alpha": selected_spec.alpha,
                "selected_gamma": selected_spec.gamma,
                "auxiliary_accepted": int(selection["auxiliary_accepted"]),
            }
        )

    oof = pd.DataFrame(oof_rows)
    oof_metrics = regression_metrics(oof["target_percentile"], oof["predicted_percentile"])
    final_selection = select_subdir_spec(
        raw_frame,
        label_frame,
        subdir,
        devices,
        specs,
        columns,
        training_cache=training_cache,
    )
    final_spec = final_selection["selected_spec"]
    final_model = fit_subdir_model(
        raw_frame,
        label_frame,
        subdir,
        devices,
        final_spec,
        columns,
        normalization_reference=raw_frame,
        training_cache=training_cache,
    )
    return {
        "model": final_model,
        "devices": devices,
        "oof": oof,
        "oof_metrics": oof_metrics,
        "fold_selection": pd.DataFrame(fold_rows),
        "candidate_metrics": final_selection["candidate_metrics"],
        "ablation": final_selection["ablation"],
        "final_selection": {
            "variant": final_spec.variant,
            "alpha": final_spec.alpha,
            "gamma": final_spec.gamma,
            "auxiliary_alpha": final_spec.auxiliary_alpha,
            "auxiliary_accepted": bool(final_selection["auxiliary_accepted"]),
        },
    }
