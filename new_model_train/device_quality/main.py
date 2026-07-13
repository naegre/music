import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bootstrap import bootstrap_device_quality
from .column_mapping import parse_column_map, read_mapped_csv
from .config import ColumnConfig
from .labels import load_human_scores, validate_human_scores
from .persistence import load_bundle, save_bundle
from .pipeline import predict_device_quality, train_device_quality
from .validation import validate_and_clean_x_metrics


def _float_list(value):
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Expected comma-separated numbers: {value}") from error


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path, data):
    Path(path).write_text(
        json.dumps(_json_safe(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _concat_result_frames(results, key):
    frames = []
    for subdir, result in results.items():
        frame = result[key].copy()
        if "subdir" not in frame.columns:
            frame.insert(0, "subdir", subdir)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_training_outputs(result, validation_report, label_report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = result["bundle"]
    columns = bundle.columns
    result["normalized_x"].to_csv(output_dir / "normalized_x.csv", index=False)
    result["aggregated"].to_csv(output_dir / "aggregated_name_subdir.csv", index=False)

    subdir_results = result["subdir_results"]
    subdir_oof = _concat_result_frames(subdir_results, "oof")
    subdir_oof.to_csv(output_dir / "subdir_oof_predictions.csv", index=False)
    _concat_result_frames(subdir_results, "fold_selection").to_csv(
        output_dir / "subdir_fold_selection.csv", index=False
    )
    _concat_result_frames(subdir_results, "candidate_metrics").to_csv(
        output_dir / "subdir_candidate_metrics.csv", index=False
    )
    _concat_result_frames(subdir_results, "ablation").to_csv(
        output_dir / "subdir_ablation.csv", index=False
    )

    normalization_frames = []
    weight_frames = []
    metric_rows = []
    for subdir, subdir_result in subdir_results.items():
        normalization = subdir_result["model"].normalizer.summary_frame()
        normalization.insert(0, "model_subdir", subdir)
        normalization_frames.append(normalization)
        weight_frames.append(subdir_result["model"].coefficient_frame())
        metric_rows.append(
            {"scope": "subdir", "subdir": subdir, **subdir_result["oof_metrics"]}
        )
    pd.concat(normalization_frames, ignore_index=True).to_csv(
        output_dir / "normalization_stats.csv", index=False
    )

    overall_weights = bundle.overall_model.coefficient_frame()
    overall_weights.insert(0, "model", "overall")
    overall_weights.insert(0, "subdir", "")
    weight_frames.append(overall_weights)
    pd.concat(weight_frames, ignore_index=True).to_csv(
        output_dir / "model_weights.csv", index=False
    )

    result["stage2_oof"].to_csv(output_dir / "overall_stage2_oof.csv", index=False)
    result["overall_oof"].to_csv(output_dir / "overall_oof_predictions.csv", index=False)
    result["overall_selection"].to_csv(
        output_dir / "overall_fold_selection.csv", index=False
    )
    result["overall_candidate_metrics"].to_csv(
        output_dir / "overall_candidate_metrics.csv", index=False
    )
    result["overall_stage2_selection"].to_csv(
        output_dir / "overall_stage2_subdir_selection.csv", index=False
    )
    metric_rows.append({"scope": "overall", "subdir": "overall", **result["overall_metrics"]})
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)
    _write_json(
        output_dir / "metrics.json",
        {
            "validation": validation_report,
            "labels": label_report,
            "subdir": {
                subdir: subdir_result["oof_metrics"]
                for subdir, subdir_result in subdir_results.items()
            },
            "overall": result["overall_metrics"],
        },
    )
    _write_json(output_dir / "training_metadata.json", bundle.training_metadata)
    save_bundle(bundle, output_dir / "device_quality_model.joblib")
    return metrics_frame


def train_command(args):
    columns = ColumnConfig()
    column_map = parse_column_map(args.column_map)
    raw_input = read_mapped_csv(args.metrics_csv, column_map, columns)
    raw_frame, validation_report = validate_and_clean_x_metrics(raw_input, columns)
    subdir_labels, overall_labels = load_human_scores(
        args.scores_json,
        layout=args.score_layout,
        overall_key=args.overall_key,
        device_col=columns.device,
        subdir_col=columns.subdir,
    )
    subdir_labels, overall_labels, label_report = validate_human_scores(
        subdir_labels, overall_labels, columns
    )
    result = train_device_quality(
        raw_frame,
        subdir_labels,
        overall_labels,
        columns,
        subdir_alphas=args.subdir_alphas,
        gammas=args.gammas,
        auxiliary_alpha=args.auxiliary_alpha,
        overall_alphas=args.overall_alphas,
        rank_lambdas=args.rank_lambdas,
    )
    output_dir = Path(args.output_dir)
    metrics = _write_training_outputs(result, validation_report, label_report, output_dir)
    overall = metrics.loc[metrics["scope"] == "overall"].iloc[0]
    print(f"Model saved to: {output_dir / 'device_quality_model.joblib'}")
    print(
        "Nested LOODO overall: "
        f"MAE={overall['mae']:.3f}, Spearman={overall['spearman']:.3f}, "
        f"pairwise accuracy={overall['pairwise_ranking_accuracy']:.3f}"
    )


def _append_bootstrap_warning(ranking):
    ranking = ranking.copy()
    for index, row in ranking.iterrows():
        if row.get("bootstrap_mode") == "contains_within_only_limited_subdirs":
            warning = "bootstrap_ci_limited_for_single_x"
            current = str(row.get("reliability_warning", "") or "")
            ranking.at[index, "reliability_warning"] = "|".join(
                item for item in (current, warning) if item
            )
    return ranking


def predict_command(args):
    bundle = load_bundle(args.model)
    column_map = parse_column_map(args.column_map)
    raw_input = read_mapped_csv(args.metrics_csv, column_map, bundle.columns)
    raw_frame, validation_report = validate_and_clean_x_metrics(raw_input, bundle.columns)
    point = predict_device_quality(bundle, raw_frame)
    bootstrap = bootstrap_device_quality(
        bundle, raw_frame, n_bootstrap=args.bootstrap, seed=args.seed
    )

    ranking = point["ranking"].merge(
        bootstrap["overall_summary"], on=bundle.columns.device, how="left", validate="one_to_one"
    )
    ranking = _append_bootstrap_warning(ranking)
    subdir_predictions = point["subdir_predictions"].merge(
        bootstrap["subdir_summary"],
        on=[bundle.columns.device, bundle.columns.subdir],
        how="left",
        validate="one_to_one",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subdir_predictions.to_csv(output_dir / "new_subdir_predictions.csv", index=False)
    point["stage2_features"].to_csv(output_dir / "new_stage2_features.csv", index=False)
    ranking.to_csv(output_dir / "new_device_ranking.csv", index=False)
    point["pairwise_probabilities"].to_csv(
        output_dir / "new_pairwise_probabilities.csv", index=False
    )
    bootstrap["subdir_summary"].to_csv(
        output_dir / "new_subdir_confidence_intervals.csv", index=False
    )
    if args.save_bootstrap_replicates:
        bootstrap["overall_replicates"].to_csv(
            output_dir / "bootstrap_overall_replicates.csv", index=False
        )
        bootstrap["subdir_replicates"].to_csv(
            output_dir / "bootstrap_subdir_replicates.csv", index=False
        )
    _write_json(
        output_dir / "prediction_summary.json",
        {
            "input_validation": validation_report,
            "bootstrap_seed": args.seed,
            "bootstrap_replicates": args.bootstrap,
            "devices": ranking.to_dict("records"),
        },
    )
    print(f"Predictions saved to: {output_dir}")
    for _, row in ranking.iterrows():
        print(
            f"{row[bundle.columns.device]}: score={row['overall_prediction']:.2f}, "
            f"rank={int(row['estimated_rank'])}/{int(row['ranking_population'])}, "
            f"95% CI=[{row['overall_ci_low']:.2f}, {row['overall_ci_high']:.2f}]"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Leakage-safe device audio quality evaluation and ranking"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train and validate the ranking pipeline")
    train.add_argument("--metrics-csv", required=True, help="Already aggregated name+subdir+x CSV")
    train.add_argument("--scores-json", required=True, help="Human-score JSON")
    train.add_argument("--output-dir", required=True)
    train.add_argument("--column-map", default="", help="Inline JSON or JSON file: canonical -> input")
    train.add_argument(
        "--score-layout", choices=("subdir_first", "device_first"), default="subdir_first"
    )
    train.add_argument("--overall-key", default="overall")
    train.add_argument("--subdir-alphas", type=_float_list, default=(1.0, 10.0, 100.0))
    train.add_argument("--gammas", type=_float_list, default=(0.0, 0.05, 0.10, 0.20))
    train.add_argument("--auxiliary-alpha", type=float, default=100.0)
    train.add_argument("--overall-alphas", type=_float_list, default=(0.1, 1.0, 10.0))
    train.add_argument("--rank-lambdas", type=_float_list, default=(0.25, 1.0, 4.0))
    train.set_defaults(func=train_command)

    predict = subparsers.add_parser("predict", help="Predict and rank one or more new devices")
    predict.add_argument("--metrics-csv", required=True)
    predict.add_argument("--model", required=True)
    predict.add_argument("--output-dir", required=True)
    predict.add_argument("--column-map", default="", help="Inline JSON or JSON file: canonical -> input")
    predict.add_argument("--bootstrap", type=int, default=500)
    predict.add_argument("--seed", type=int, default=2026)
    predict.add_argument("--save-bootstrap-replicates", action="store_true")
    predict.set_defaults(func=predict_command)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
