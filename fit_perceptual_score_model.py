import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_LOWER_IS_BETTER_HINTS = (
    "distance",
    "dist",
    "l2",
    "fd",
    "fad",
    "kl",
    "js",
    "nll",
    "rmse",
    "mae",
    "error",
    "loss",
)

DEFAULT_HIGHER_IS_BETTER_HINTS = (
    "similarity",
    "sim",
    "cosine_similarity",
    "sisdr",
    "si_sdr",
    "snr",
    "stoi",
    "pesq",
    "mos",
    "score",
)


def infer_metric_direction(column):
    name = column.lower()
    for hint in DEFAULT_LOWER_IS_BETTER_HINTS:
        if hint in name:
            return -1
    for hint in DEFAULT_HIGHER_IS_BETTER_HINTS:
        if hint in name:
            return 1
    return 1


def parse_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_direction_overrides(path):
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    overrides = {}
    for key, value in data.items():
        if isinstance(value, str):
            value = value.lower()
            if value in ("higher", "higher_is_better", "+", "1"):
                overrides[key] = 1
            elif value in ("lower", "lower_is_better", "-", "-1"):
                overrides[key] = -1
            else:
                raise ValueError(f"Unknown direction for {key}: {value}")
        else:
            overrides[key] = 1 if float(value) >= 0 else -1
    return overrides


def select_feature_columns(df, target_col, id_cols, feature_cols, exclude_cols):
    if feature_cols:
        return feature_cols
    excluded = set([target_col]) | set(id_cols) | set(exclude_cols)
    numeric_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


def apply_directions(df, feature_cols, overrides):
    directions = {}
    transformed = df[feature_cols].copy()
    for col in feature_cols:
        direction = overrides.get(col, infer_metric_direction(col))
        directions[col] = direction
        transformed[col] = transformed[col].astype(float) * direction
    return transformed, directions


def corr_safe(y_true, y_pred):
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return {"pearson": np.nan, "spearman": np.nan}
    return {
        "pearson": float(pearsonr(y_true, y_pred).statistic),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
    }


def metric_summary(y_true, y_pred):
    corr = corr_safe(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {**corr, "rmse": rmse, "mae": mae}


class NnlsRegressor:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        y_mean = y.mean()
        y_centered = y - y_mean
        coef, _ = nnls(x, y_centered)
        self.coef_ = coef
        self.intercept_ = y_mean
        return self

    def predict(self, x):
        return np.asarray(x, dtype=np.float64) @ self.coef_ + self.intercept_


def build_model(model_name, random_state):
    if model_name == "ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-4, 4, 25))),
            ]
        )
    if model_name == "elasticnet":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                        alphas=np.logspace(-4, 2, 20),
                        cv=5,
                        random_state=random_state,
                        max_iter=20000,
                    ),
                ),
            ]
        )
    if model_name == "nnls":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", NnlsRegressor()),
            ]
        )
    raise ValueError(f"Unknown model: {model_name}")


def get_linear_weights(pipeline, feature_cols):
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]
    coef_scaled = np.asarray(model.coef_, dtype=np.float64)

    weights_original = coef_scaled / scaler.scale_
    intercept_original = float(model.intercept_ - np.sum(coef_scaled * scaler.mean_ / scaler.scale_))

    rows = []
    for feature, coef_s, coef_o in zip(feature_cols, coef_scaled, weights_original):
        rows.append(
            {
                "feature": feature,
                "weight_on_standardized_feature": float(coef_s),
                "weight_on_direction_adjusted_original_feature": float(coef_o),
            }
        )
    return rows, intercept_original


def cross_validate_predictions(x, y, model_name, n_splits, random_state):
    n = len(y)
    if n_splits <= 1 or n < 4:
        return None

    n_splits = min(n_splits, n)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    preds = np.full(n, np.nan, dtype=np.float64)

    for train_idx, test_idx in kfold.split(x):
        model = build_model(model_name, random_state)
        model.fit(x[train_idx], y[train_idx])
        preds[test_idx] = model.predict(x[test_idx])

    return preds


def write_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Fit an extensible weighted perceptual score model from objective metrics and human scores."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--target_col", required=True, help="Human score column, e.g. MOS or CMOS")
    parser.add_argument("--out_dir", default="perceptual_score_model")
    parser.add_argument("--model", choices=["ridge", "elasticnet", "nnls"], default="ridge")
    parser.add_argument("--feature_cols", default="", help="Comma-separated metric columns. Default: all numeric non-target columns.")
    parser.add_argument("--id_cols", default="key,ref_file,test_file", help="Comma-separated metadata columns copied to predictions.")
    parser.add_argument("--exclude_cols", default="", help="Comma-separated columns to exclude from auto feature selection.")
    parser.add_argument("--direction_json", default="", help="Optional JSON mapping feature -> higher/lower or 1/-1.")
    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=1024)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    if args.target_col not in df.columns:
        raise ValueError(f"target_col not found: {args.target_col}")

    id_cols = [col for col in parse_list(args.id_cols) if col in df.columns]
    feature_cols = parse_list(args.feature_cols)
    exclude_cols = parse_list(args.exclude_cols)
    feature_cols = select_feature_columns(df, args.target_col, id_cols, feature_cols, exclude_cols)
    if not feature_cols:
        raise ValueError("No feature columns selected.")

    working = df.dropna(subset=[args.target_col]).reset_index(drop=True)
    x_raw, directions = apply_directions(working, feature_cols, load_direction_overrides(args.direction_json))
    x = x_raw.to_numpy(dtype=np.float64)
    y = working[args.target_col].to_numpy(dtype=np.float64)

    cv_pred = cross_validate_predictions(x, y, args.model, args.cv_splits, args.random_state)

    model = build_model(args.model, args.random_state)
    model.fit(x, y)
    train_pred = model.predict(x)

    prediction_rows = []
    for i, row in working.iterrows():
        out = {col: row[col] for col in id_cols}
        out["human_score"] = float(y[i])
        out["predicted_score_train_fit"] = float(train_pred[i])
        if cv_pred is not None:
            out["predicted_score_cv"] = float(cv_pred[i])
        for col in feature_cols:
            out[col] = row[col]
            out[f"{col}__direction"] = "higher_is_better" if directions[col] == 1 else "lower_is_better"
        prediction_rows.append(out)

    weights, intercept = get_linear_weights(model, feature_cols)
    for row in weights:
        row["direction"] = "higher_is_better" if directions[row["feature"]] == 1 else "lower_is_better"

    train_metrics = metric_summary(y, train_pred)
    summary = {
        "input_csv": args.input_csv,
        "target_col": args.target_col,
        "model": args.model,
        "num_samples": int(len(y)),
        "features": feature_cols,
        "directions": {k: ("higher_is_better" if v == 1 else "lower_is_better") for k, v in directions.items()},
        "intercept_on_direction_adjusted_original_features": intercept,
        "train_fit": train_metrics,
    }
    if cv_pred is not None:
        summary["cross_validation"] = metric_summary(y, cv_pred)

    predictions_csv = out_dir / "predictions.csv"
    weights_csv = out_dir / "weights.csv"
    summary_json = out_dir / "summary.json"

    write_csv(prediction_rows, predictions_csv)
    write_csv(weights, weights_csv)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {predictions_csv}")
    print(f"wrote {weights_csv}")
    print(f"wrote {summary_json}")


if __name__ == "__main__":
    main()
