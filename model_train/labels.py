import json
from pathlib import Path

import pandas as pd


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


def load_human_scores(path, layout="device_first", overall_key="overall", device_col="name", subdir_col="subdir"):
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
