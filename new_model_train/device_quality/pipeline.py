from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

from .aggregation import aggregate_normalized_x
from .config import ColumnConfig
from .metrics import OVERALL_NOTICEABLE_GAP, OVERALL_PREFERENCE_SLOPE, overall_metrics
from .overall_model import (
    OverallCandidateSpec,
    fit_overall_model,
    make_overall_specs,
    select_overall_spec,
)
from .subdir_model import (
    ALLOWED_GAMMAS,
    SubdirCandidateSpec,
    SubdirTrainingCache,
    _available_devices,
    fit_subdir_model,
    make_candidate_specs,
    select_subdir_spec,
    train_subdir_with_nested_loodo,
)


MODEL_FORMAT_VERSION = 2


def stage2_column(subdir):
    return f"subdir::{subdir}"


@dataclass
class DeviceQualityBundle:
    format_version: int
    columns: ColumnConfig
    subdirs: list
    subdir_models: dict
    overall_model: object
    overall_quality_columns: list
    known_overall_scores: pd.DataFrame
    training_metadata: dict


def _predict_subdir_device(model, raw_frame, device, columns):
    subset = raw_frame.loc[raw_frame[columns.device].astype(str) == str(device)].copy()
    prediction = model.predict_raw(subset)
    if prediction.empty:
        return np.nan, np.nan, "missing_subdir"
    row = prediction.iloc[0]
    return (
        float(row["predicted_percentile"]),
        float(row["reliability_weight"]),
        str(row["reliability_warning"]),
    )


def _fallback_subdir_spec(alphas, auxiliary_alpha):
    return SubdirCandidateSpec("stability", float(max(alphas)), 0.0, float(auxiliary_alpha))


def build_fold_safe_stage2(
    raw_frame,
    subdir_labels,
    subdirs,
    outer_train_devices,
    heldout_device,
    columns,
    subdir_alphas,
    gammas,
    auxiliary_alpha,
    training_cache,
):
    """Build one overall outer fold without using the held-out device anywhere."""
    train_rows = {str(device): {columns.device: str(device)} for device in outer_train_devices}
    heldout_row = {columns.device: str(heldout_device)}
    selection_rows = []
    specs = make_candidate_specs(subdir_alphas, gammas, auxiliary_alpha)

    for subdir in subdirs:
        available = _available_devices(raw_frame, subdir_labels, subdir, columns)
        safe_devices = [device for device in available if device != str(heldout_device)]
        if len(safe_devices) >= 4:
            selection = select_subdir_spec(
                raw_frame,
                subdir_labels,
                subdir,
                safe_devices,
                specs,
                columns,
                training_cache=training_cache,
            )
            selected_spec = selection["selected_spec"]
            auxiliary_accepted = int(selection["auxiliary_accepted"])
        elif len(safe_devices) >= 2:
            selected_spec = _fallback_subdir_spec(subdir_alphas, auxiliary_alpha)
            auxiliary_accepted = 0
        else:
            selected_spec = None
            auxiliary_accepted = 0

        selection_rows.append(
            {
                "outer_heldout_device": str(heldout_device),
                "subdir": str(subdir),
                "n_safe_subdir_devices": len(safe_devices),
                "safe_subdir_devices": "|".join(safe_devices),
                "selected_variant": selected_spec.variant if selected_spec else "unavailable",
                "selected_alpha": selected_spec.alpha if selected_spec else np.nan,
                "selected_gamma": selected_spec.gamma if selected_spec else np.nan,
                "auxiliary_accepted": auxiliary_accepted,
            }
        )
        quality_column = stage2_column(subdir)
        reliability_column = f"{quality_column}__reliability"
        warning_column = f"{quality_column}__warning"
        if selected_spec is None:
            for device in outer_train_devices:
                train_rows[str(device)][quality_column] = np.nan
                train_rows[str(device)][reliability_column] = np.nan
                train_rows[str(device)][warning_column] = "subdir_model_unavailable"
            heldout_row[quality_column] = np.nan
            heldout_row[reliability_column] = np.nan
            heldout_row[warning_column] = "subdir_model_unavailable"
            continue

        model_cache = {}

        def model_for(excluded_device=None):
            fit_devices = tuple(
                device for device in safe_devices if device != str(excluded_device)
            )
            if len(fit_devices) < 2:
                return None
            if fit_devices not in model_cache:
                model_cache[fit_devices] = fit_subdir_model(
                    raw_frame,
                    subdir_labels,
                    subdir,
                    fit_devices,
                    selected_spec,
                    columns,
                    training_cache=training_cache,
                )
            return model_cache[fit_devices]

        for device in outer_train_devices:
            device = str(device)
            # A labeled base-model row is predicted by a model that excludes itself.
            excluded = device if device in safe_devices else None
            model = model_for(excluded)
            if model is None:
                value, reliability, warning = np.nan, np.nan, "insufficient_fold_devices"
            else:
                value, reliability, warning = _predict_subdir_device(
                    model, raw_frame, device, columns
                )
            train_rows[device][quality_column] = value
            train_rows[device][reliability_column] = reliability
            train_rows[device][warning_column] = warning

        heldout_model = model_for(None)
        if heldout_model is None:
            value, reliability, warning = np.nan, np.nan, "insufficient_fold_devices"
        else:
            value, reliability, warning = _predict_subdir_device(
                heldout_model, raw_frame, heldout_device, columns
            )
        heldout_row[quality_column] = value
        heldout_row[reliability_column] = reliability
        heldout_row[warning_column] = warning

    train_frame = pd.DataFrame(list(train_rows.values()))
    heldout_frame = pd.DataFrame([heldout_row])
    quality_columns = [stage2_column(subdir) for subdir in subdirs]
    reliability_columns = [f"{column}__reliability" for column in quality_columns]
    for frame in (train_frame, heldout_frame):
        available_fraction = frame[quality_columns].notna().mean(axis=1)
        reliability = frame[reliability_columns].mean(axis=1, skipna=True).fillna(0.25)
        frame["overall_reliability"] = np.clip(
            available_fraction * reliability, 0.10, 1.0
        )
    return train_frame, heldout_frame, pd.DataFrame(selection_rows)


def _aggregate_candidate_oof(candidate_predictions, score_map, specs, device_column):
    rows = []
    for spec in specs:
        subset = candidate_predictions.loc[
            candidate_predictions["candidate_key"] == spec.key
        ].sort_values(device_column)
        targets = [score_map[str(device)] for device in subset[device_column].astype(str)]
        metrics = overall_metrics(targets, subset["overall_prediction"])
        rows.append(
            {
                "candidate_key": spec.key,
                "alpha": spec.alpha,
                "rank_lambda": spec.rank_lambda,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def train_device_quality(
    raw_frame,
    subdir_labels,
    overall_labels,
    columns=None,
    subdir_alphas=(1.0, 10.0, 100.0),
    gammas=ALLOWED_GAMMAS,
    auxiliary_alpha=100.0,
    overall_alphas=(0.1, 1.0, 10.0),
    rank_lambdas=(0.25, 1.0, 4.0),
):
    columns = columns or ColumnConfig()
    available_subdirs = set(raw_frame[columns.subdir].astype(str))
    labeled_subdirs = set(subdir_labels[columns.subdir].astype(str))
    subdirs = sorted(available_subdirs & labeled_subdirs)
    if not subdirs:
        raise ValueError("No subdir appears in both the CSV and human-score JSON")

    training_cache = SubdirTrainingCache()
    subdir_results = {}
    normalized_frames = []
    aggregated_frames = []
    for subdir in subdirs:
        result = train_subdir_with_nested_loodo(
            raw_frame,
            subdir_labels,
            subdir,
            columns,
            alphas=subdir_alphas,
            gammas=gammas,
            auxiliary_alpha=auxiliary_alpha,
            training_cache=training_cache,
        )
        subdir_results[subdir] = result
        model = result["model"]
        subset = raw_frame.loc[raw_frame[columns.subdir].astype(str) == str(subdir)].copy()
        normalized = model.normalizer.transform(subset)
        aggregated = aggregate_normalized_x(normalized, model.normalizer, columns)
        normalized_frames.append(normalized)
        aggregated_frames.append(aggregated)

    overall_scores = overall_labels.copy()
    overall_scores[columns.device] = overall_scores[columns.device].astype(str)
    raw_devices = set(raw_frame[columns.device].astype(str))
    overall_scores = overall_scores.loc[
        overall_scores[columns.device].isin(raw_devices)
    ].drop_duplicates(columns.device)
    overall_devices = sorted(overall_scores[columns.device].tolist())
    if len(overall_devices) < 5:
        raise ValueError("At least five overall-labeled devices with CSV data are required")
    score_map = overall_scores.set_index(columns.device)["overall_human_score"].astype(float).to_dict()
    quality_columns = [stage2_column(subdir) for subdir in subdirs]
    overall_specs = make_overall_specs(overall_alphas, rank_lambdas)

    overall_oof_rows = []
    stage2_oof_rows = []
    overall_selection_rows = []
    stage2_selection_frames = []
    candidate_oof_rows = []
    for heldout in overall_devices:
        outer_train = [device for device in overall_devices if device != heldout]
        train_features, heldout_features, stage2_selection = build_fold_safe_stage2(
            raw_frame,
            subdir_labels,
            subdirs,
            outer_train,
            heldout,
            columns,
            subdir_alphas,
            gammas,
            auxiliary_alpha,
            training_cache,
        )
        stage2_selection_frames.append(stage2_selection)
        outer_scores = overall_scores.loc[
            overall_scores[columns.device].isin(outer_train)
        ].copy()
        selected_spec, inner_metrics, _ = select_overall_spec(
            train_features,
            outer_scores,
            quality_columns,
            overall_specs,
            columns.device,
        )
        selected_model = fit_overall_model(
            train_features,
            outer_scores,
            quality_columns,
            selected_spec,
            columns.device,
        )
        selected_prediction = float(selected_model.predict(heldout_features)[0])
        overall_oof_rows.append(
            {
                columns.device: heldout,
                "overall_human_score": score_map[heldout],
                "overall_prediction": selected_prediction,
                "selected_alpha": selected_spec.alpha,
                "selected_rank_lambda": selected_spec.rank_lambda,
            }
        )
        overall_selection_rows.append(
            {
                "heldout_device": heldout,
                "selected_alpha": selected_spec.alpha,
                "selected_rank_lambda": selected_spec.rank_lambda,
                "inner_selection_loss": float(
                    inner_metrics.loc[
                        inner_metrics["candidate_key"] == selected_spec.key,
                        "selection_loss",
                    ].iloc[0]
                ),
            }
        )
        stage2_oof_rows.append(heldout_features.iloc[0].to_dict())
        for spec in overall_specs:
            candidate_model = fit_overall_model(
                train_features,
                outer_scores,
                quality_columns,
                spec,
                columns.device,
            )
            candidate_oof_rows.append(
                {
                    columns.device: heldout,
                    "candidate_key": spec.key,
                    "overall_prediction": float(candidate_model.predict(heldout_features)[0]),
                }
            )

    overall_oof = pd.DataFrame(overall_oof_rows).sort_values(columns.device).reset_index(drop=True)
    overall_oof_metrics = overall_metrics(
        overall_oof["overall_human_score"], overall_oof["overall_prediction"]
    )
    stage2_oof = pd.DataFrame(stage2_oof_rows).sort_values(columns.device).reset_index(drop=True)
    candidate_oof = pd.DataFrame(candidate_oof_rows)
    overall_candidate_metrics = _aggregate_candidate_oof(
        candidate_oof, score_map, overall_specs, columns.device
    )

    selected_keys = [
        OverallCandidateSpec(row["selected_alpha"], row["selected_rank_lambda"]).key
        for row in overall_selection_rows
    ]
    modal_key = Counter(selected_keys).most_common(1)[0][0]
    final_overall_spec = next(spec for spec in overall_specs if spec.key == modal_key)
    final_overall_model = fit_overall_model(
        stage2_oof,
        overall_scores,
        quality_columns,
        final_overall_spec,
        columns.device,
    )

    subdir_models = {subdir: result["model"] for subdir, result in subdir_results.items()}
    metadata = {
        "format_version": MODEL_FORMAT_VERSION,
        "subdirs": subdirs,
        "subdir_final_selection": {
            subdir: result["final_selection"] for subdir, result in subdir_results.items()
        },
        "overall_final_spec": {
            "alpha": final_overall_spec.alpha,
            "rank_lambda": final_overall_spec.rank_lambda,
            "selection_rule": "modal nested outer-fold selection",
        },
        "preference_mapping": {
            "slope": OVERALL_PREFERENCE_SLOPE,
            "noticeable_gap": OVERALL_NOTICEABLE_GAP,
        },
        "auxiliary_residual_mode": "leave_one_device_out_primary_residual",
        "leakage_guards": [
            "outer held-out device excluded from every subdir normalization and model fit",
            "subdir gamma selected only inside the corresponding training fold",
            "overall training features are cross-fitted by device",
            "overall imputation and feature scaling are fitted inside each model fold",
            "unlabeled devices enter only the final deployed robust baseline",
        ],
    }
    bundle = DeviceQualityBundle(
        MODEL_FORMAT_VERSION,
        columns,
        subdirs,
        subdir_models,
        final_overall_model,
        quality_columns,
        overall_scores[[columns.device, "overall_human_score"]].copy(),
        metadata,
    )
    return {
        "bundle": bundle,
        "normalized_x": pd.concat(normalized_frames, ignore_index=True),
        "aggregated": pd.concat(aggregated_frames, ignore_index=True),
        "subdir_results": subdir_results,
        "overall_oof": overall_oof,
        "stage2_oof": stage2_oof,
        "overall_metrics": overall_oof_metrics,
        "overall_selection": pd.DataFrame(overall_selection_rows),
        "overall_candidate_metrics": overall_candidate_metrics,
        "overall_stage2_selection": pd.concat(stage2_selection_frames, ignore_index=True),
    }


def predict_device_quality(bundle, raw_frame):
    columns = bundle.columns
    devices = sorted(raw_frame[columns.device].astype(str).unique().tolist())
    subdir_frames = []
    feature_rows = {device: {columns.device: device} for device in devices}
    for subdir in bundle.subdirs:
        model = bundle.subdir_models[subdir]
        predicted = model.predict_raw(raw_frame)
        if not predicted.empty:
            predicted = predicted.copy()
            predicted["selected_variant"] = model.spec.variant
            predicted["selected_gamma"] = model.spec.gamma
            subdir_frames.append(predicted)
            prediction_map = predicted.set_index(columns.device)["predicted_percentile"].to_dict()
            reliability_map = predicted.set_index(columns.device)["reliability_weight"].to_dict()
            warning_map = predicted.set_index(columns.device)["reliability_warning"].to_dict()
        else:
            prediction_map = {}
            reliability_map = {}
            warning_map = {}
        quality_column = stage2_column(subdir)
        for device in devices:
            feature_rows[device][quality_column] = prediction_map.get(device, np.nan)
            feature_rows[device][f"{quality_column}__reliability"] = reliability_map.get(device, np.nan)
            feature_rows[device][f"{quality_column}__warning"] = warning_map.get(
                device, "missing_subdir"
            )

    features = pd.DataFrame(list(feature_rows.values()))
    reliability_columns = [f"{column}__reliability" for column in bundle.overall_quality_columns]
    available_fraction = features[bundle.overall_quality_columns].notna().mean(axis=1)
    mean_reliability = features[reliability_columns].mean(axis=1, skipna=True).fillna(0.25)
    features["overall_reliability"] = np.clip(available_fraction * mean_reliability, 0.10, 1.0)
    overall_prediction = bundle.overall_model.predict(features)
    known = bundle.known_overall_scores.copy()
    known[columns.device] = known[columns.device].astype(str)
    known_map = known.set_index(columns.device)["overall_human_score"].astype(float).to_dict()

    summary_rows = []
    pairwise_rows = []
    for row_index, device in enumerate(devices):
        prediction = float(overall_prediction[row_index])
        comparison = {name: score for name, score in known_map.items() if name != device}
        rank = 1 + sum(score > prediction for score in comparison.values())
        better = sorted(
            ((name, score) for name, score in comparison.items() if score >= prediction + 2.0),
            key=lambda item: item[1],
        )
        worse = sorted(
            ((name, score) for name, score in comparison.items() if score <= prediction - 2.0),
            key=lambda item: item[1],
            reverse=True,
        )
        similar = sorted(
            ((name, score) for name, score in comparison.items() if abs(score - prediction) < 2.0),
            key=lambda item: abs(item[1] - prediction),
        )
        missing_subdirs = [
            subdir
            for subdir in bundle.subdirs
            if not np.isfinite(features.loc[row_index, stage2_column(subdir)])
        ]
        warning_parts = []
        if missing_subdirs:
            warning_parts.append("missing_subdirs=" + ",".join(missing_subdirs))
        if features.loc[row_index, "overall_reliability"] < 0.6:
            warning_parts.append("low_overall_reliability")
        summary_rows.append(
            {
                columns.device: device,
                "overall_prediction": prediction,
                "estimated_rank": int(rank),
                "ranking_population": len(comparison) + 1,
                "nearest_clearly_better": better[0][0] if better else "",
                "nearest_clearly_worse": worse[0][0] if worse else "",
                "approximately_equal_devices": "|".join(name for name, _ in similar),
                "subdir_coverage": 1.0 - len(missing_subdirs) / max(len(bundle.subdirs), 1),
                "overall_reliability": float(features.loc[row_index, "overall_reliability"]),
                "reliability_warning": "|".join(warning_parts),
            }
        )
        for known_device, known_score in sorted(comparison.items()):
            probability = float(
                expit(OVERALL_PREFERENCE_SLOPE * (prediction - known_score))
            )
            pairwise_rows.append(
                {
                    columns.device: device,
                    "known_device": known_device,
                    "known_human_score": known_score,
                    "new_overall_prediction": prediction,
                    "probability_new_better": probability,
                }
            )
    return {
        "subdir_predictions": (
            pd.concat(subdir_frames, ignore_index=True) if subdir_frames else pd.DataFrame()
        ),
        "stage2_features": features,
        "ranking": pd.DataFrame(summary_rows),
        "pairwise_probabilities": pd.DataFrame(pairwise_rows),
    }
