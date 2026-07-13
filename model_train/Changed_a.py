from pathlib import Path

import numpy as np
import pandas as pd

from .config import ColumnConfig


def _safe_std(values):
    values = np.asarray(values, dtype=np.float64)

    if values.size < 2:
        return 0.0

    return float(np.std(values, ddof=1))


def _numeric_values(group, column):
    return (
        pd.to_numeric(group[column], errors="coerce")
        .dropna()
        .to_numpy(dtype=np.float64)
    )


def _pooled_mean_variance(group, mean_col, var_col):
    """
    根据每个时间窗口的均值和方差，计算整条音频的总体均值和总体方差。

    pooled_var =
        mean(window_var + window_mean^2)
        - pooled_mean^2
    """
    means = pd.to_numeric(group[mean_col], errors="coerce")
    variances = pd.to_numeric(group[var_col], errors="coerce")

    valid = means.notna() & variances.notna()

    means = means[valid].to_numpy(dtype=np.float64)

    variances = np.maximum(
        variances[valid].to_numpy(dtype=np.float64),
        0.0,
    )

    if means.size == 0:
        return np.nan, np.nan

    pooled_mean = float(means.mean())

    pooled_var = float(
        np.mean(variances + means ** 2)
        - pooled_mean ** 2
    )

    return pooled_mean, max(pooled_var, 0.0)


def _available_audio_group_columns(df, columns):
    """
    动态决定音频级分组字段。

    旧数据有 original_file：
        name + subdir + audio_index + original_file

    新数据没有 original_file：
        name + subdir + audio_index
    """
    group_columns = list(columns.required_audio_group_columns)

    if columns.original_file in df.columns:
        group_columns.append(columns.original_file)

    return group_columns


def _available_metric_columns(df, columns):
    """
    只使用 CSV 中实际存在的指标列。

    因此 l2_var 和 cos_sim_var 可以是可选字段。
    """
    return [
        column
        for column in columns.metric_columns
        if column in df.columns
    ]


def validate_input_columns(df, columns):
    """
    检查真正必需的字段。

    以下字段不再强制要求：
        test_file
        original_file
        time_segment
        l2_var
        cos_sim_var
    """
    required = set(
        columns.required_audio_group_columns
        + columns.required_metric_columns
    )

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"Missing required CSV columns: {missing}"
        )


def aggregate_windows_to_audio(df, columns=None):
    """
    将时间窗口级数据聚合成音频级数据。
    """
    columns = columns or ColumnConfig()

    validate_input_columns(df, columns)

    group_columns = _available_audio_group_columns(
        df,
        columns,
    )

    metric_columns = _available_metric_columns(
        df,
        columns,
    )

    rows = []

    for keys, group in df.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        # group_columns 至少有三个字段，因此 keys 为 tuple
        row = dict(zip(group_columns, keys))

        row["n_windows"] = int(len(group))

        # test_file 存在时保留；不存在时直接跳过
        if columns.test_file in group.columns:
            paths = (
                group[columns.test_file]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            row["test_files"] = "|".join(paths)

        # 对所有实际存在的指标做普通统计
        for metric in metric_columns:
            values = _numeric_values(group, metric)

            if values.size == 0:
                for suffix in (
                    "mean",
                    "median",
                    "std",
                    "min",
                    "max",
                    "range",
                ):
                    row[f"{metric}_audio_{suffix}"] = np.nan

                continue

            row[f"{metric}_audio_mean"] = float(
                values.mean()
            )

            row[f"{metric}_audio_median"] = float(
                np.median(values)
            )

            row[f"{metric}_audio_std"] = _safe_std(
                values
            )

            row[f"{metric}_audio_min"] = float(
                values.min()
            )

            row[f"{metric}_audio_max"] = float(
                values.max()
            )

            row[f"{metric}_audio_range"] = float(
                values.max() - values.min()
            )

        # 需要按照“均值列 + 方差列”计算总体方差的指标
        pooled_pairs = [
            (
                columns.kl_mean,
                columns.kl_var,
                "kl",
            ),
            (
                columns.js_mean,
                columns.js_var,
                "js",
            ),
            (
                columns.l2,
                columns.l2_var,
                "l2",
            ),
            (
                columns.cos_sim,
                columns.cos_sim_var,
                "cos_sim",
            ),
        ]

        for mean_col, var_col, prefix in pooled_pairs:
            # 新方差字段不存在时直接跳过
            if (
                mean_col not in group.columns
                or var_col not in group.columns
            ):
                continue

            pooled_mean, pooled_var = (
                _pooled_mean_variance(
                    group,
                    mean_col,
                    var_col,
                )
            )

            row[
                f"{prefix}_pooled_audio_mean"
            ] = pooled_mean

            row[
                f"{prefix}_pooled_audio_var"
            ] = pooled_var

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_audio_to_device_subdir(
    audio_df,
    columns=None,
):
    """
    将音频级数据继续聚合成设备-子目录级数据。
    """
    columns = columns or ColumnConfig()

    required = set(
        columns.device_subdir_group_columns
        + [columns.audio_id]
    )

    missing = sorted(required - set(audio_df.columns))

    if missing:
        raise ValueError(
            f"Missing audio-level columns: {missing}"
        )

    # 根据当前 DataFrame 中实际存在的字段构造排除列表
    excluded = set(
        _available_audio_group_columns(
            audio_df,
            columns,
        )
        + [
            "test_files",
            "n_windows",
        ]
    )

    feature_cols = [
        col
        for col in audio_df.columns
        if (
            col not in excluded
            and pd.api.types.is_numeric_dtype(
                audio_df[col]
            )
        )
    ]

    rows = []

    for keys, group in audio_df.groupby(
        columns.device_subdir_group_columns,
        dropna=False,
        sort=False,
    ):
        row = dict(
            zip(
                columns.device_subdir_group_columns,
                keys,
            )
        )

        row["n_audio"] = int(len(group))

        row["n_windows_total"] = int(
            group["n_windows"].sum()
        )

        for feature in feature_cols:
            values = (
                pd.to_numeric(
                    group[feature],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(dtype=np.float64)
            )

            if values.size == 0:
                for suffix in (
                    "device_mean",
                    "device_median",
                    "between_audio_std",
                    "standard_error",
                ):
                    row[f"{feature}_{suffix}"] = np.nan

                continue

            std = _safe_std(values)

            row[
                f"{feature}_device_mean"
            ] = float(values.mean())

            row[
                f"{feature}_device_median"
            ] = float(np.median(values))

            row[
                f"{feature}_between_audio_std"
            ] = std

            row[
                f"{feature}_standard_error"
            ] = float(
                std / np.sqrt(values.size)
            )

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_metrics(csv_or_df, columns=None):
    columns = columns or ColumnConfig()

    if isinstance(csv_or_df, (str, Path)):
        window_df = pd.read_csv(csv_or_df)
    else:
        window_df = csv_or_df.copy()

    audio_df = aggregate_windows_to_audio(
        window_df,
        columns,
    )

    device_subdir_df = aggregate_audio_to_device_subdir(
        audio_df,
        columns,
    )

    return audio_df, device_subdir_df
