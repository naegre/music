import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ColumnConfig


def _as_score(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("score", "value", "mos"):
            if key in value and isinstance(value[key], (int, float)):
                return float(value[key])
    return None


def _device_first(data, overall_key):
    subdir_rows = []
    overall_rows = []
    for device, scores in data.items():
        if not isinstance(scores, dict):
            continue
        for subdir, raw_score in scores.items():
            score = _as_score(raw_score)
            if score is None:
                continue
            if subdir == overall_key:
                overall_rows.append({"name": str(device), "overall_human_score": score})
            else:
                subdir_rows.append({"name": str(device), "subdir": str(subdir), "human_score": score})
    return pd.DataFrame(subdir_rows), pd.DataFrame(overall_rows)


def _subdir_first(data, overall_key):
    subdir_rows = []
    overall_rows = []
    for subdir, devices in data.items():
        if not isinstance(devices, dict):
            continue
        for device, raw_score in devices.items():
            score = _as_score(raw_score)
            if score is None:
                continue
            if subdir == overall_key:
                overall_rows.append({"name": str(device), "overall_human_score": score})
            else:
                subdir_rows.append({"name": str(device), "subdir": str(subdir), "human_score": score})
    return pd.DataFrame(subdir_rows), pd.DataFrame(overall_rows)


def load_human_scores(path, layout="subdir_first", overall_key="overall", device_col="name", subdir_col="subdir"):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if layout == "device_first":
        subdir_df, overall_df = _device_first(data, overall_key)
    elif layout == "subdir_first":
        subdir_df, overall_df = _subdir_first(data, overall_key)
    else:
        raise ValueError("layout must be device_first or subdir_first")

    if not subdir_df.empty:
        subdir_df = subdir_df.rename(columns={"name": device_col, "subdir": subdir_col})
    if not overall_df.empty:
        overall_df = overall_df.rename(columns={"name": device_col})
    return subdir_df, overall_df


def validate_human_scores(subdir_frame, overall_frame, columns=None):
    columns = columns or ColumnConfig()
    required_subdir = {columns.device, columns.subdir, "human_score"}
    required_overall = {columns.device, "overall_human_score"}
    if not required_subdir.issubset(subdir_frame.columns):
        raise ValueError(f"Subdir labels are missing columns: {sorted(required_subdir - set(subdir_frame.columns))}")
    if not required_overall.issubset(overall_frame.columns):
        raise ValueError(f"Overall labels are missing columns: {sorted(required_overall - set(overall_frame.columns))}")

    subdir = subdir_frame.copy()
    overall = overall_frame.copy()
    subdir[columns.device] = subdir[columns.device].astype(str)
    subdir[columns.subdir] = subdir[columns.subdir].astype(str)
    overall[columns.device] = overall[columns.device].astype(str)
    subdir["human_score"] = pd.to_numeric(subdir["human_score"], errors="coerce")
    overall["overall_human_score"] = pd.to_numeric(
        overall["overall_human_score"], errors="coerce"
    )
    if subdir["human_score"].isna().any() or overall["overall_human_score"].isna().any():
        raise ValueError("Human-score JSON contains non-numeric scores")
    if not np.isfinite(subdir["human_score"]).all() or not np.isfinite(
        overall["overall_human_score"]
    ).all():
        raise ValueError("Human-score JSON contains infinite scores")
    if subdir.duplicated([columns.device, columns.subdir]).any():
        raise ValueError("Human-score JSON has duplicate name+subdir labels")
    if overall.duplicated(columns.device).any():
        raise ValueError("Human-score JSON has duplicate overall device labels")
    report = {
        "num_subdir_labels": int(len(subdir)),
        "num_overall_labels": int(len(overall)),
        "subdir_label_devices": int(subdir[columns.device].nunique()),
        "overall_label_devices": int(overall[columns.device].nunique()),
        "labels_per_subdir": subdir.groupby(columns.subdir).size().astype(int).to_dict(),
    }
    return subdir, overall, report
