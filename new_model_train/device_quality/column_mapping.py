import json
from pathlib import Path

import pandas as pd

from .config import ColumnConfig


def parse_column_map(value):
    """Parse canonical-to-input column mapping from inline JSON or a JSON file."""
    if not value:
        return {}
    text = str(value).strip()
    if not text.startswith("{"):
        path = Path(text)
        if not path.exists():
            raise FileNotFoundError(f"Column-map JSON file does not exist: {path}")
        text = path.read_text(encoding="utf-8")
    mapping = json.loads(text)
    if not isinstance(mapping, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in mapping.items()
    ):
        raise ValueError("Column map must be a JSON object of canonical_name -> input_name")
    return mapping


def read_mapped_csv(path, column_map=None, columns=None):
    columns = columns or ColumnConfig()
    frame = pd.read_csv(path)
    mapping = column_map or {}
    canonical = set(columns.required_columns)
    unknown = sorted(set(mapping) - canonical)
    if unknown:
        raise ValueError(f"Unknown canonical names in column map: {unknown}")
    duplicated_inputs = [name for name in mapping.values() if list(mapping.values()).count(name) > 1]
    if duplicated_inputs:
        raise ValueError(
            f"One input column cannot map to multiple canonical columns: {sorted(set(duplicated_inputs))}"
        )
    rename = {input_name: canonical_name for canonical_name, input_name in mapping.items()}
    frame = frame.rename(columns=rename)
    if frame.columns.duplicated().any():
        duplicated = sorted(set(frame.columns[frame.columns.duplicated()].tolist()))
        raise ValueError(f"Column mapping creates duplicate CSV columns: {duplicated}")
    return frame
