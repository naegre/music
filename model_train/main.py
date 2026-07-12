import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .aggregation import aggregate_metrics
from .config import ColumnConfig
from .features import default_feature_columns
from .labels import load_human_scores
from .metrics import regression_ranking_metrics
from .modeling import (
    DEFAULT_ALPHAS,
    build_overall_table,
    nested_leave_one_device_out,
    pivot_new_subdir_predictions,
)


def parse_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_floats(value):
    return tuple(float(item) for item in parse_list(value))


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)}")


def write_json(data, path):
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def build_column_config(args):
    return ColumnConfig(
        device=args.device_col,
        subdir=args.subdir_col,
        test_file=args.test_file_col,
        original_file=args.original_file_col,
        audio_id=args.audio_id_col,
        time_slice=args.time_slice_col,
        kl_mean=args.kl_mean_col,
        kl_var=args.kl_var_col,
        js_mean=args.js_mean_col,
        js_var=args.js_var_col,
        l2=args.l2_col,
        cos_sim=args.cos_sim_col,
    )


def normalize_keys(frame, columns):
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = result[column].astype(str)
    return result


def validate_features(frame, feature_cols):
    missing = sorted(set(feature_cols) - set(frame.columns))
    if missing:
        raise ValueError(f"Requested feature columns are missing: {missing}")


def rank_against_reference(predictions, references, device_col, score_col, reference_score_col, group_col=None, tie_margin=1.0):
    rows = []
    grouped_predictions = [(None, predictions)] if group_col is None else predictions.groupby(group_col, dropna=False)
    for group_value, pred_group in grouped_predictions:
        if group_col is None:
            ref_group = references
        else:
            ref_group = references[references[group_col].astype(str) == str(group_value)]
        ref_group = ref_group.dropna(subset=[reference_score_col]).copy()
        ref_group[reference_score_col] = pd.to_numeric(ref_group[reference_score_col], errors="coerce")
        ref_group = ref_group.dropna(subset=[reference_score_col]).sort_values(reference_score_col, ascending=False)

        for _, pred_row in pred_group.iterrows():
            score = float(pred_row[score_col])
            clearly_higher = ref_group[ref_group[reference_score_col] > score + tie_margin]
            clearly_lower = ref_group[ref_group[reference_score_col] < score - tie_margin]
            similar = ref_group[(ref_group[reference_score_col] - score).abs() <= tie_margin]

            higher_name = clearly_higher.iloc[-1][device_col] if not clearly_higher.empty else ""
            lower_name = clearly_lower.iloc[0][device_col] if not clearly_lower.empty else ""
            row = {
                device_col: pred_row[device_col],
                "predicted_score": score,
                "estimated_rank": int(1 + (ref_group[reference_score_col] > score).sum()),
                "reference_device_count": int(len(ref_group)),
                "nearest_clearly_better_device": higher_name,
                "nearest_clearly_worse_device": lower_name,
                "similar_devices_within_1_point": "|".join(similar[device_col].astype(str).tolist()),
            }
            if group_col is not None:
                row[group_col] = group_value
            rows.append(row)
    return pd.DataFrame(rows)


def train(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = build_column_config(args)

    audio_df, device_subdir_df = aggregate_metrics(args.metrics_csv, columns)
    audio_df.to_csv(out_dir / "audio_level_features.csv", index=False)
    device_subdir_df.to_csv(out_dir / "device_subdir_features.csv", index=False)

    subdir_labels, overall_labels = load_human_scores(
        args.scores_json,
        layout=args.json_layout,
        overall_key=args.overall_key,
        device_col=columns.device,
        subdir_col=columns.subdir,
    )
    device_subdir_df = normalize_keys(device_subdir_df, [columns.device, columns.subdir])
    subdir_labels = normalize_keys(subdir_labels, [columns.device, columns.subdir])
    overall_labels = normalize_keys(overall_labels, [columns.device])
    labeled = device_subdir_df.merge(subdir_labels, on=[columns.device, columns.subdir], how="inner")
    if labeled.empty:
        raise RuntimeError("No device-subdir rows matched the JSON human scores.")
    labeled.to_csv(out_dir / "labeled_device_subdir_features.csv", index=False)

    feature_cols = parse_list(args.feature_cols)
    if not feature_cols:
        feature_cols = default_feature_columns(
            labeled,
            columns.device,
            columns.subdir,
            target_cols=["human_score", "overall_human_score"],
        )
    validate_features(labeled, feature_cols)
    interaction_cols = parse_list(args.interaction_features)
    validate_features(labeled, interaction_cols)
    alphas = parse_floats(args.alphas) or DEFAULT_ALPHAS

    subdir_oof, subdir_model, subdir_fold_alphas = nested_leave_one_device_out(
        labeled,
        target_col="human_score",
        device_col=columns.device,
        feature_cols=feature_cols,
        subdir_col=columns.subdir,
        interaction_cols=interaction_cols,
        alphas=alphas,
        weight_cap=args.weight_cap,
    )
    subdir_oof.to_csv(out_dir / "subdir_oof_predictions.csv", index=False)
    subdir_fold_alphas.to_csv(out_dir / "subdir_fold_alphas.csv", index=False)
    subdir_model.weight_table().to_csv(out_dir / "subdir_model_weights.csv", index=False)
    subdir_metrics = regression_ranking_metrics(
        subdir_oof,
        "human_score",
        "predicted_score_oof",
        group_col=columns.subdir,
        tie_margin=args.tie_margin,
        preference_at_margin=args.preference_at_margin,
    )
    write_json(subdir_metrics, out_dir / "subdir_metrics.json")

    overall_model = None
    overall_feature_cols = []
    overall_metrics = None
    overall_oof = pd.DataFrame()
    if not overall_labels.empty:
        overall_table = build_overall_table(subdir_oof, overall_labels, columns.device, columns.subdir)
        overall_feature_cols = [col for col in overall_table.columns if col.startswith("predicted_subdir_")]
        if len(overall_table) >= 4 and overall_feature_cols:
            overall_oof, overall_model, overall_fold_alphas = nested_leave_one_device_out(
                overall_table,
                target_col="overall_human_score",
                device_col=columns.device,
                feature_cols=overall_feature_cols,
                subdir_col=None,
                interaction_cols=None,
                alphas=alphas,
                weight_cap=args.weight_cap,
            )
            overall_oof.to_csv(out_dir / "overall_oof_predictions.csv", index=False)
            overall_fold_alphas.to_csv(out_dir / "overall_fold_alphas.csv", index=False)
            overall_model.weight_table().to_csv(out_dir / "overall_model_weights.csv", index=False)
            overall_metrics = regression_ranking_metrics(
                overall_oof,
                "overall_human_score",
                "predicted_score_oof",
                group_col=None,
                tie_margin=args.tie_margin,
                preference_at_margin=args.preference_at_margin,
            )
            write_json(overall_metrics, out_dir / "overall_metrics.json")

    model_bundle = {
        "columns": columns,
        "subdir_model": subdir_model,
        "overall_model": overall_model,
        "overall_feature_columns": overall_feature_cols,
        "known_subdir_scores": subdir_labels,
        "known_overall_scores": overall_labels,
        "tie_margin": args.tie_margin,
        "preference_at_margin": args.preference_at_margin,
    }
    joblib.dump(model_bundle, out_dir / "device_quality_model.joblib")
    manifest = {
        "metrics_csv": args.metrics_csv,
        "scores_json": args.scores_json,
        "columns": columns.to_dict(),
        "feature_columns": feature_cols,
        "interaction_features": interaction_cols,
        "subdir_final_alpha": subdir_model.alpha,
        "overall_final_alpha": overall_model.alpha if overall_model else None,
        "num_devices": int(labeled[columns.device].nunique()),
        "num_subdirs": int(labeled[columns.subdir].nunique()),
        "subdir_metrics": subdir_metrics,
        "overall_metrics": overall_metrics,
    }
    write_json(manifest, out_dir / "training_manifest.json")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=json_default))


def predict(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.model_bundle)
    columns = bundle["columns"]
    _, device_subdir_df = aggregate_metrics(args.metrics_csv, columns)
    device_subdir_df = normalize_keys(device_subdir_df, [columns.device, columns.subdir])

    subdir_model = bundle["subdir_model"]
    missing = sorted(set(subdir_model.feature_columns) - set(device_subdir_df.columns))
    if missing:
        raise ValueError(f"New-device aggregate is missing trained features: {missing}")
    subdir_predictions = device_subdir_df[[columns.device, columns.subdir, "n_audio", "n_windows_total"]].copy()
    subdir_predictions["predicted_subdir_score"] = subdir_model.predict(device_subdir_df)
    subdir_predictions.to_csv(out_dir / "new_subdir_predictions.csv", index=False)

    subdir_ranking = rank_against_reference(
        subdir_predictions,
        bundle["known_subdir_scores"],
        columns.device,
        "predicted_subdir_score",
        "human_score",
        group_col=columns.subdir,
        tie_margin=bundle["tie_margin"],
    )
    subdir_ranking.to_csv(out_dir / "new_subdir_ranking.csv", index=False)

    overall_model = bundle.get("overall_model")
    if overall_model is not None:
        overall_input = pivot_new_subdir_predictions(subdir_predictions, columns.device, columns.subdir)
        for feature in bundle["overall_feature_columns"]:
            if feature not in overall_input.columns:
                overall_input[feature] = np.nan
        overall_predictions = overall_input[[columns.device]].copy()
        overall_predictions["predicted_overall_score"] = overall_model.predict(overall_input)
        overall_predictions.to_csv(out_dir / "new_overall_predictions.csv", index=False)
        overall_ranking = rank_against_reference(
            overall_predictions,
            bundle["known_overall_scores"],
            columns.device,
            "predicted_overall_score",
            "overall_human_score",
            group_col=None,
            tie_margin=bundle["tie_margin"],
        )
        overall_ranking.to_csv(out_dir / "new_overall_ranking.csv", index=False)
        print(overall_ranking.to_string(index=False))
    else:
        print(subdir_ranking.to_string(index=False))


def add_column_arguments(parser):
    parser.add_argument("--device-col", default="name")
    parser.add_argument("--subdir-col", default="subdir")
    parser.add_argument("--test-file-col", default="test_file")
    parser.add_argument("--original-file-col", default="original_file")
    parser.add_argument("--audio-id-col", default="audio_index")
    parser.add_argument("--time-slice-col", default="time_segment")
    parser.add_argument("--kl-mean-col", default="kl_mean")
    parser.add_argument("--kl-var-col", default="kl_var")
    parser.add_argument("--js-mean-col", default="js_mean")
    parser.add_argument("--js-var-col", default="js_var")
    parser.add_argument("--l2-col", default="l2")
    parser.add_argument("--cos-sim-col", default="cos_sim")


def build_parser():
    parser = argparse.ArgumentParser(description="Train and use the hierarchical device audio quality ranker.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--metrics-csv", required=True)
    train_parser.add_argument("--scores-json", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--json-layout", choices=["device_first", "subdir_first"], default="device_first")
    train_parser.add_argument("--overall-key", default="overall")
    train_parser.add_argument("--feature-cols", default="")
    train_parser.add_argument("--interaction-features", default="")
    train_parser.add_argument("--alphas", default=",".join(str(value) for value in DEFAULT_ALPHAS))
    train_parser.add_argument("--weight-cap", type=float, default=5.0)
    train_parser.add_argument("--tie-margin", type=float, default=1.0)
    train_parser.add_argument("--preference-at-margin", type=float, default=0.8)
    add_column_arguments(train_parser)
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--metrics-csv", required=True)
    predict_parser.add_argument("--model-bundle", required=True)
    predict_parser.add_argument("--output-dir", required=True)
    predict_parser.set_defaults(func=predict)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
