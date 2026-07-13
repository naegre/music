from pathlib import Path

import joblib

from .pipeline import DeviceQualityBundle, MODEL_FORMAT_VERSION


def save_bundle(bundle, path):
    if not isinstance(bundle, DeviceQualityBundle):
        raise TypeError("Expected a DeviceQualityBundle")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path):
    bundle = joblib.load(path)
    if not isinstance(bundle, DeviceQualityBundle):
        raise TypeError("The joblib file is not a device-quality model bundle")
    if bundle.format_version != MODEL_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported model format {bundle.format_version}; expected {MODEL_FORMAT_VERSION}"
        )
    return bundle
