import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import kendalltau, pearsonr, spearmanr

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "kl_mean",
    "kl_var",
    "js_mean",
    "js_var",
    "l2",
    "l2_var",
    "cos_sim",
    "cos_sim_var",
]


def extract_score(value):
    """
    支持：

    "name": 85

    或：

    "name": {
        "score": 85
    }
    """
    if isinstance(value, bool):
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, dict):
        for key in ["score", "value", "mos"]:
            if key not in value:
                continue

            try:
                return float(value[key])
            except (TypeError, ValueError):
                return np.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_subdir_scores(
    json_path,
    overall_key="result",
):
    """
    JSON 格式：

    {
        "subdir1": {
            "name1": 85,
            "name2": 80
        },
        "subdir2": {
            ...
        },
        "result": {
            "name1": 84
        }
    }

    本脚本只读取每个 subdir 的评分，
    不使用 result。
    """
    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as file:
        raw = json.load(file)

    rows = []

    for subdir, name_scores in raw.items():
        if subdir == overall_key:
            continue

        if not isinstance(name_scores, dict):
            continue

        for name, value in name_scores.items():
            rows.append(
                {
                    "subdir": str(subdir).strip(),
                    "name": str(name).strip(),
                    "human_score": extract_score(value),
                }
            )

    labels = pd.DataFrame(
        rows,
        columns=[
            "subdir",
            "name",
            "human_score",
        ],
    )

    labels["human_score"] = pd.to_numeric(
        labels["human_score"],
        errors="coerce",
    )

    labels = labels.dropna(
        subset=["human_score"]
    )

    duplicate_mask = labels.duplicated(
        subset=["subdir", "name"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = labels.loc[
            duplicate_mask
        ].sort_values(
            ["subdir", "name"]
        )

        raise ValueError(
            "同一个 subdir + name 存在重复人工评分：\n"
            + duplicates.to_string(index=False)
        )

    return labels


def load_feature_data(
    csv_path,
    id_col,
    requested_features,
):
    """
    当前 CSV 每一行直接视为：

        一个 subdir 下
        一个 name
        一个 id 对应的一组完整音频统计

    不再做时间切片聚合。
    """
    frame = pd.read_csv(csv_path)

    required = [
        "name",
        "subdir",
        id_col,
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"CSV 缺少字段：{missing}"
        )

    frame["name"] = (
        frame["name"]
        .astype(str)
        .str.strip()
    )

    frame["subdir"] = (
        frame["subdir"]
        .astype(str)
        .str.strip()
    )

    available_features = [
        column
        for column in requested_features
        if column in frame.columns
    ]

    if not available_features:
        raise ValueError(
            "CSV 中没有找到可用指标。"
        )

    for column in available_features:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        frame[column] = frame[
            column
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    duplicate_mask = frame.duplicated(
        subset=["name", "subdir", id_col],
        keep=False,
    )

    if duplicate_mask.any():
        examples = frame.loc[
            duplicate_mask,
            ["name", "subdir", id_col],
        ].head(20)

        raise ValueError(
            "发现重复的 name + subdir + id。"
            "本脚本要求每行对应一组完整音频数据。\n"
            + examples.to_string(index=False)
        )

    return frame, available_features


def choose_usable_features(
    train_frame,
    feature_columns,
):
    """
    删除训练集中：

    - 全为空；
    - 只有一个有效值；
    - 所有值完全相同；

    的特征。
    """
    usable = []

    for column in feature_columns:
        values = pd.to_numeric(
            train_frame[column],
            errors="coerce",
        )

        values = values.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if len(values) < 2:
            continue

        if values.nunique() < 2:
            continue

        usable.append(column)

    return usable


def build_model(alpha):
    """
    alpha <= 0：
        普通线性回归

    alpha > 0：
        Ridge 线性回归
    """
    if alpha <= 0:
        regressor = LinearRegression()
    else:
        regressor = Ridge(
            alpha=alpha,
        )

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "regressor",
                regressor,
            ),
        ]
    )


def safe_correlation(
    true_values,
    predicted_values,
):
    true_values = np.asarray(
        true_values,
        dtype=float,
    )

    predicted_values = np.asarray(
        predicted_values,
        dtype=float,
    )

    valid = (
        np.isfinite(true_values)
        & np.isfinite(predicted_values)
    )

    true_values = true_values[valid]
    predicted_values = predicted_values[valid]

    result = {
        "pearson": np.nan,
        "spearman": np.nan,
        "kendall": np.nan,
    }

    if len(true_values) < 2:
        return result

    if (
        len(np.unique(true_values)) < 2
        or len(np.unique(predicted_values)) < 2
    ):
        return result

    result["pearson"] = float(
        pearsonr(
            true_values,
            predicted_values,
        ).statistic
    )

    result["spearman"] = float(
        spearmanr(
            true_values,
            predicted_values,
        ).statistic
    )

    result["kendall"] = float(
        kendalltau(
            true_values,
            predicted_values,
        ).statistic
    )

    return result


def calculate_metrics(
    human_scores,
    predicted_scores,
):
    human_scores = np.asarray(
        human_scores,
        dtype=float,
    )

    predicted_scores = np.asarray(
        predicted_scores,
        dtype=float,
    )

    valid = (
        np.isfinite(human_scores)
        & np.isfinite(predicted_scores)
    )

    human_scores = human_scores[valid]
    predicted_scores = predicted_scores[valid]

    if len(human_scores) == 0:
        return None

    rmse = np.sqrt(
        mean_squared_error(
            human_scores,
            predicted_scores,
        )
    )

    metrics = {
        "n_samples": int(
            len(human_scores)
        ),
        "mae": float(
            mean_absolute_error(
                human_scores,
                predicted_scores,
            )
        ),
        "rmse": float(rmse),
    }

    metrics.update(
        safe_correlation(
            human_scores,
            predicted_scores,
        )
    )

    return metrics


def sanitize_filename(value):
    return re.sub(
        r"[^0-9a-zA-Z._-]+",
        "_",
        str(value),
    )


def split_one_id_per_name(
    labeled_frame,
    seed,
):
    """
    每个 name 随机选择一个 id 所在行作为训练数据。

    该 name 剩余的其他 id 作为验证数据。
    """
    rng = np.random.default_rng(seed)

    train_indices = []

    for _, group in labeled_frame.groupby(
        "name",
        sort=False,
    ):
        candidate_indices = group.index.to_numpy()

        selected_index = rng.choice(
            candidate_indices
        )

        train_indices.append(
            selected_index
        )

    train_mask = labeled_frame.index.isin(
        train_indices
    )

    train_frame = labeled_frame.loc[
        train_mask
    ].copy()

    validation_frame = labeled_frame.loc[
        ~train_mask
    ].copy()

    return train_frame, validation_frame


def run_one_subdir(
    subdir,
    subdir_frame,
    feature_columns,
    alpha,
    seed,
    output_dir,
):
    labeled = subdir_frame[
        subdir_frame["human_score"].notna()
    ].copy()

    unlabeled = subdir_frame[
        subdir_frame["human_score"].isna()
    ].copy()

    n_names = labeled["name"].nunique()

    if n_names < 3:
        print(
            f"[跳过] {subdir}："
            f"只有 {n_names} 个有评分 name。"
        )
        return None

    train_frame, validation_frame = (
        split_one_id_per_name(
            labeled,
            seed=seed,
        )
    )

    usable_features = choose_usable_features(
        train_frame,
        feature_columns,
    )

    if not usable_features:
        print(
            f"[跳过] {subdir}："
            "训练集没有可用特征。"
        )
        return None

    model = build_model(alpha)

    model.fit(
        train_frame[usable_features],
        train_frame["human_score"],
    )

    train_frame[
        "predicted_score"
    ] = model.predict(
        train_frame[usable_features]
    )

    if not validation_frame.empty:
        validation_frame[
            "predicted_score"
        ] = model.predict(
            validation_frame[
                usable_features
            ]
        )

        validation_frame[
            "signed_error"
        ] = (
            validation_frame[
                "predicted_score"
            ]
            - validation_frame[
                "human_score"
            ]
        )

        validation_frame[
            "absolute_error"
        ] = validation_frame[
            "signed_error"
        ].abs()
    else:
        validation_frame[
            "predicted_score"
        ] = np.nan

    if not unlabeled.empty:
        unlabeled[
            "predicted_score"
        ] = model.predict(
            unlabeled[usable_features]
        )

    # 同一个 name 可能有多个验证 id。
    # 排名比较前，先对同一 name 的验证预测取均值。
    if not validation_frame.empty:
        name_validation = (
            validation_frame.groupby(
                "name",
                as_index=False,
            )
            .agg(
                human_score=(
                    "human_score",
                    "first",
                ),
                predicted_score_mean=(
                    "predicted_score",
                    "mean",
                ),
                predicted_score_std=(
                    "predicted_score",
                    "std",
                ),
                n_validation_ids=(
                    "predicted_score",
                    "size",
                ),
            )
        )

        name_validation[
            "human_rank"
        ] = name_validation[
            "human_score"
        ].rank(
            ascending=False,
            method="average",
        )

        name_validation[
            "predicted_rank"
        ] = name_validation[
            "predicted_score_mean"
        ].rank(
            ascending=False,
            method="average",
        )

        name_validation[
            "rank_difference"
        ] = (
            name_validation[
                "predicted_rank"
            ]
            - name_validation[
                "human_rank"
            ]
        )

        name_validation[
            "absolute_rank_difference"
        ] = name_validation[
            "rank_difference"
        ].abs()

        name_validation = (
            name_validation.sort_values(
                "human_rank"
            )
        )

        row_metrics = calculate_metrics(
            validation_frame["human_score"],
            validation_frame[
                "predicted_score"
            ],
        )

        ranking_metrics = calculate_metrics(
            name_validation[
                "human_score"
            ],
            name_validation[
                "predicted_score_mean"
            ],
        )

        mean_rank_error = float(
            name_validation[
                "absolute_rank_difference"
            ].mean()
        )
    else:
        name_validation = pd.DataFrame()

        row_metrics = None
        ranking_metrics = None
        mean_rank_error = np.nan

    subdir_output = (
        output_dir
        / sanitize_filename(subdir)
    )

    subdir_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_frame.to_csv(
        subdir_output / "train_rows.csv",
        index=False,
    )

    validation_frame.to_csv(
        subdir_output / "validation_rows.csv",
        index=False,
    )

    name_validation.to_csv(
        subdir_output
        / "validation_name_ranking.csv",
        index=False,
    )

    unlabeled.to_csv(
        subdir_output
        / "unlabeled_predictions.csv",
        index=False,
    )

    # 保存标准化后的线性模型系数
    regressor = model.named_steps[
        "regressor"
    ]

    coefficients = pd.DataFrame(
        {
            "feature": usable_features,
            "standardized_coefficient": (
                regressor.coef_
            ),
        }
    )

    coefficients[
        "absolute_coefficient"
    ] = coefficients[
        "standardized_coefficient"
    ].abs()

    coefficients = coefficients.sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    coefficients.to_csv(
        subdir_output
        / "model_coefficients.csv",
        index=False,
    )

    result = {
        "subdir": subdir,
        "n_labeled_names": int(n_names),
        "n_train_rows": int(
            len(train_frame)
        ),
        "n_validation_rows": int(
            len(validation_frame)
        ),
        "n_validation_names": int(
            validation_frame[
                "name"
            ].nunique()
        ),
        "n_unlabeled_rows": int(
            len(unlabeled)
        ),
        "n_features": int(
            len(usable_features)
        ),
        "usable_features": (
            "|".join(usable_features)
        ),
        "row_mae": (
            row_metrics["mae"]
            if row_metrics
            else np.nan
        ),
        "row_rmse": (
            row_metrics["rmse"]
            if row_metrics
            else np.nan
        ),
        "name_pearson": (
            ranking_metrics["pearson"]
            if ranking_metrics
            else np.nan
        ),
        "name_spearman": (
            ranking_metrics["spearman"]
            if ranking_metrics
            else np.nan
        ),
        "name_kendall": (
            ranking_metrics["kendall"]
            if ranking_metrics
            else np.nan
        ),
        "mean_absolute_rank_difference": (
            mean_rank_error
        ),
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metrics-csv",
        required=True,
    )

    parser.add_argument(
        "--scores-json",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--id-col",
        default="audio_index",
    )

    parser.add_argument(
        "--overall-key",
        default="result",
    )

    parser.add_argument(
        "--subdir",
        default=None,
        help=(
            "只测试指定 subdir。"
            "不设置则依次测试全部 subdir。"
        ),
    )

    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help=(
            "Ridge 正则化强度。"
            "设置为 0 时使用普通线性回归。"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_frame, feature_columns = (
        load_feature_data(
            args.metrics_csv,
            id_col=args.id_col,
            requested_features=args.features,
        )
    )

    labels = load_subdir_scores(
        args.scores_json,
        overall_key=args.overall_key,
    )

    data = metrics_frame.merge(
        labels,
        on=["subdir", "name"],
        how="left",
        validate="many_to_one",
    )

    if args.subdir is not None:
        requested_subdirs = [
            str(args.subdir).strip()
        ]
    else:
        requested_subdirs = (
            data["subdir"]
            .dropna()
            .unique()
            .tolist()
        )

    summaries = []

    for subdir in requested_subdirs:
        subdir_frame = data[
            data["subdir"] == subdir
        ].copy()

        if subdir_frame.empty:
            print(
                f"[跳过] 找不到 subdir：{subdir}"
            )
            continue

        result = run_one_subdir(
            subdir=subdir,
            subdir_frame=subdir_frame,
            feature_columns=feature_columns,
            alpha=args.alpha,
            seed=args.seed,
            output_dir=output_dir,
        )

        if result is not None:
            summaries.append(result)

            print(
                f"[完成] {subdir}："
                f"Spearman="
                f"{result['name_spearman']:.4f}, "
                f"Kendall="
                f"{result['name_kendall']:.4f}, "
                f"MAE={result['row_mae']:.4f}"
            )

    summary = pd.DataFrame(summaries)

    if not summary.empty:
        summary = summary.sort_values(
            "name_spearman",
            ascending=False,
        )

    summary.to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    print("\n汇总结果：")

    if summary.empty:
        print("没有成功完成的 subdir。")
    else:
        display_columns = [
            "subdir",
            "n_labeled_names",
            "n_validation_names",
            "row_mae",
            "name_spearman",
            "name_kendall",
            "mean_absolute_rank_difference",
        ]

        print(
            summary[
                display_columns
            ].to_string(index=False)
        )

    print(
        f"\n结果已保存到：{output_dir}"
    )


if __name__ == "__main__":
    main()
