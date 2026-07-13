import numpy as np
import pandas as pd

from .config import ColumnConfig


def validate_and_clean_x_metrics(frame, columns=None):
    columns = columns or ColumnConfig()
    missing = sorted(set(columns.required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    data = frame.copy()
    if data.empty:
        raise ValueError("Metrics CSV has no rows")
    key_columns = [columns.device, columns.subdir, columns.audio_id]
    if data[key_columns].isna().any().any():
        bad = data[data[key_columns].isna().any(axis=1)].index.tolist()[:10]
        raise ValueError(f"Null device/subdir/x keys at rows: {bad}")

    data[columns.device] = data[columns.device].astype(str)
    data[columns.subdir] = data[columns.subdir].astype(str)
    data[columns.audio_id] = data[columns.audio_id].astype(str)

    duplicates = data.duplicated(key_columns, keep=False)
    if duplicates.any():
        examples = data.loc[duplicates, key_columns].head(10).to_dict("records")
        raise ValueError(
            "CSV must contain exactly one already-aggregated row per name+subdir+x. "
            f"Duplicate examples: {examples}"
        )

    numeric_columns = [columns.n_windows]
    for spec in columns.metric_specs:
        numeric_columns.extend([spec.mean_column, spec.variance_column])
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    invalid_numeric = data[numeric_columns].isna().any(axis=1)
    if invalid_numeric.any():
        bad = data.loc[invalid_numeric, key_columns].head(10).to_dict("records")
        raise ValueError(f"Non-numeric or missing metric values for: {bad}")
    finite_numeric = np.isfinite(data[numeric_columns].to_numpy(dtype=np.float64)).all(axis=1)
    if not finite_numeric.all():
        bad = data.loc[~finite_numeric, key_columns].head(10).to_dict("records")
        raise ValueError(f"Infinite metric values for: {bad}")

    if (data[columns.n_windows] < 1).any():
        raise ValueError("n_windows must be at least 1")
    if not np.allclose(data[columns.n_windows], np.round(data[columns.n_windows])):
        raise ValueError("n_windows must contain integer counts")
    data[columns.n_windows] = data[columns.n_windows].round().astype(int)

    for spec in columns.metric_specs:
        if (data[spec.variance_column] < 0).any():
            raise ValueError(f"Variance column must be non-negative: {spec.variance_column}")
        if spec.transform == "log1p" and (data[spec.mean_column] < 0).any():
            raise ValueError(f"Positive-distance mean must be non-negative: {spec.mean_column}")

    report = {
        "num_rows": int(len(data)),
        "num_devices": int(data[columns.device].nunique()),
        "num_subdirs": int(data[columns.subdir].nunique()),
        "rows_per_subdir": data.groupby(columns.subdir).size().astype(int).to_dict(),
        "devices_per_subdir": data.groupby(columns.subdir)[columns.device].nunique().astype(int).to_dict(),
        "x_per_subdir": data.groupby(columns.subdir)[columns.audio_id].nunique().astype(int).to_dict(),
    }
    return data, report
