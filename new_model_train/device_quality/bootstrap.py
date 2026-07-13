import numpy as np
import pandas as pd

from .pipeline import predict_device_quality


def _resample_group(group, columns, rng):
    group = group.reset_index(drop=True)
    n_x = len(group)
    if n_x > 1:
        sampled = group.iloc[rng.integers(0, n_x, size=n_x)].copy().reset_index(drop=True)
        mode = "x_resample_plus_within_parametric"
    else:
        sampled = group.copy()
        mode = "within_parametric_only_limited"
    n_windows = sampled[columns.n_windows].to_numpy(dtype=np.float64)
    n_effective = np.maximum(1.0, (n_windows + 4.0) / 5.0)
    for spec in columns.metric_specs:
        means = sampled[spec.mean_column].to_numpy(dtype=np.float64)
        variances = np.maximum(
            sampled[spec.variance_column].to_numpy(dtype=np.float64), 0.0
        )
        standard_error = np.sqrt(variances / n_effective)
        draws = rng.normal(means, standard_error)
        if spec.key == "cosim":
            draws = np.clip(draws, -0.999999, 0.999999)
        else:
            draws = np.maximum(draws, 0.0)
        sampled[spec.mean_column] = draws
    return sampled, mode


def bootstrap_device_quality(bundle, raw_frame, n_bootstrap=500, seed=2026):
    if n_bootstrap < 20:
        raise ValueError("Use at least 20 bootstrap replicates")
    columns = bundle.columns
    rng = np.random.default_rng(seed)
    groups = list(raw_frame.groupby([columns.device, columns.subdir], sort=False))
    mode_by_device = {}
    for (device, subdir), group in groups:
        mode_by_device.setdefault(str(device), {})[str(subdir)] = (
            "within_parametric_only_limited" if len(group) == 1 else "x_resample_plus_within_parametric"
        )

    overall_rows = []
    subdir_rows = []
    for replicate in range(int(n_bootstrap)):
        sampled_groups = []
        for _, group in groups:
            sampled, _ = _resample_group(group, columns, rng)
            sampled_groups.append(sampled)
        sampled_frame = pd.concat(sampled_groups, ignore_index=True)
        prediction = predict_device_quality(bundle, sampled_frame)
        ranking = prediction["ranking"]
        for _, row in ranking.iterrows():
            overall_rows.append(
                {
                    "replicate": replicate,
                    columns.device: str(row[columns.device]),
                    "overall_prediction": float(row["overall_prediction"]),
                    "estimated_rank": int(row["estimated_rank"]),
                }
            )
        subdir_prediction = prediction["subdir_predictions"]
        for _, row in subdir_prediction.iterrows():
            subdir_rows.append(
                {
                    "replicate": replicate,
                    columns.device: str(row[columns.device]),
                    columns.subdir: str(row[columns.subdir]),
                    "predicted_percentile": float(row["predicted_percentile"]),
                }
            )

    overall_replicates = pd.DataFrame(overall_rows)
    subdir_replicates = pd.DataFrame(subdir_rows)
    overall_summary_rows = []
    for device, group in overall_replicates.groupby(columns.device, sort=False):
        modes = mode_by_device.get(str(device), {})
        limited = sorted(
            subdir for subdir, mode in modes.items() if mode == "within_parametric_only_limited"
        )
        overall_summary_rows.append(
            {
                columns.device: str(device),
                "overall_ci_low": float(group["overall_prediction"].quantile(0.025)),
                "overall_ci_high": float(group["overall_prediction"].quantile(0.975)),
                "possible_rank_low": int(np.floor(group["estimated_rank"].quantile(0.025))),
                "possible_rank_high": int(np.ceil(group["estimated_rank"].quantile(0.975))),
                "bootstrap_replicates": int(len(group)),
                "bootstrap_mode": (
                    "contains_within_only_limited_subdirs"
                    if limited
                    else "x_resample_plus_within_parametric"
                ),
                "limited_ci_subdirs": "|".join(limited),
            }
        )
    subdir_summary = (
        subdir_replicates.groupby([columns.device, columns.subdir], as_index=False)
        .agg(
            subdir_ci_low=("predicted_percentile", lambda values: values.quantile(0.025)),
            subdir_ci_high=("predicted_percentile", lambda values: values.quantile(0.975)),
            bootstrap_replicates=("replicate", "count"),
        )
    )
    mode_rows = [
        {
            columns.device: device,
            columns.subdir: subdir,
            "bootstrap_mode": mode,
        }
        for device, subdirs in mode_by_device.items()
        for subdir, mode in subdirs.items()
    ]
    subdir_summary = subdir_summary.merge(
        pd.DataFrame(mode_rows),
        on=[columns.device, columns.subdir],
        how="left",
        validate="one_to_one",
    )
    return {
        "overall_summary": pd.DataFrame(overall_summary_rows),
        "subdir_summary": subdir_summary,
        "overall_replicates": overall_replicates,
        "subdir_replicates": subdir_replicates,
    }
