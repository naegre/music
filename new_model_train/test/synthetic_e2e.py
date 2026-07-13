import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SUBDIRS = ("clarity", "treble", "lowfreq", "clarity_voice", "dynamics", "spatial")
X_COUNTS = (1, 2, 3, 4, 5, 6)


def metric_row(rng, device, subdir, audio_id, latent, device_index):
    content_shift = 0.10 * int(audio_id)
    noise = rng.normal(0.0, 0.05)
    l2 = max(0.01, 6.0 - 3.8 * latent + content_shift + noise)
    cosim = np.clip(0.20 + 0.72 * latent - 0.015 * int(audio_id) + noise, -0.95, 0.98)
    # Deliberately weak and partly inverted auxiliary metrics.
    inverted = -1.0 if device_index % 5 == 0 else 1.0
    kl = max(0.01, 2.5 - inverted * 0.25 * latent + rng.normal(0.0, 0.45))
    js = max(0.005, 0.55 - inverted * 0.04 * latent + rng.normal(0.0, 0.08))
    n_windows = int(rng.integers(2, 18))
    return {
        "subdir": subdir,
        "name": device,
        "x": str(audio_id),
        "n_windows": n_windows,
        "kl_mean": kl,
        "kl_var": float(rng.uniform(0.01, 0.20)),
        "js_mean": js,
        "js_var": float(rng.uniform(0.001, 0.02)),
        "l2_mean": l2,
        "l2_var": float(rng.uniform(0.01, 0.15)),
        "cosim_mean": cosim,
        "cosim_var": float(rng.uniform(0.0002, 0.005)),
    }


def build_data(directory):
    rng = np.random.default_rng(321)
    devices = [f"device_{index:02d}" for index in range(20)]
    latent = {device: 0.10 + 0.80 * index / 19 for index, device in enumerate(devices)}
    rows = []
    for device_index, device in enumerate(devices):
        for subdir_index, (subdir, x_count) in enumerate(zip(SUBDIRS, X_COUNTS)):
            if (device == "device_03" and subdir == "clarity_voice") or (
                device == "device_07" and subdir == "spatial"
            ):
                continue
            for audio_id in range(x_count):
                if device_index > 0 and x_count > 2 and rng.random() < 0.04:
                    continue
                task_latent = np.clip(
                    latent[device] + rng.normal(0.0, 0.035) + 0.01 * subdir_index,
                    0.0,
                    1.0,
                )
                rows.append(metric_row(rng, device, subdir, audio_id, task_latent, device_index))
    train_frame = pd.DataFrame(rows)
    train_path = directory / "train.csv"
    train_frame.to_csv(train_path, index=False)

    scores = {}
    for subdir_index, subdir in enumerate(SUBDIRS):
        scores[subdir] = {}
        for device_index, device in enumerate(devices[:19]):
            if device == "device_12" and subdir == "dynamics":
                continue
            score = 61.0 + 29.0 * latent[device] + rng.normal(0.0, 1.5)
            if subdir == "clarity" and device == "device_00":
                score = 6.8
            scores[subdir][device] = round(float(score), 3)
    scores["overall"] = {
        device: round(float(62.0 + 28.0 * latent[device] + rng.normal(0.0, 0.6)), 3)
        for device in devices[:18]
    }
    score_path = directory / "scores.json"
    score_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")

    new_rows = []
    for subdir_index, (subdir, x_count) in enumerate(zip(SUBDIRS, X_COUNTS)):
        for audio_id in range(x_count):
            new_rows.append(
                metric_row(rng, "new_device", subdir, audio_id, 0.68, 21)
            )
    new_path = directory / "new.csv"
    pd.DataFrame(new_rows).to_csv(new_path, index=False)
    return train_path, score_path, new_path


def run():
    with tempfile.TemporaryDirectory(prefix="device_quality_e2e_", dir=ROOT) as temp:
        temp_path = Path(temp)
        train_csv, scores_json, new_csv = build_data(temp_path)
        train_output = temp_path / "train_output"
        predict_output = temp_path / "predict_output"
        train_command = [
            sys.executable,
            str(ROOT / "device_quality_main.py"),
            "train",
            "--metrics-csv",
            str(train_csv),
            "--scores-json",
            str(scores_json),
            "--output-dir",
            str(train_output),
            "--subdir-alphas",
            "10",
            "--gammas",
            "0,0.10",
            "--overall-alphas",
            "0.1,1",
            "--rank-lambdas",
            "0.25,1",
        ]
        train_result = subprocess.run(
            train_command, cwd=ROOT, check=True, text=True, capture_output=True, timeout=900
        )
        predict_command = [
            sys.executable,
            str(ROOT / "device_quality_main.py"),
            "predict",
            "--metrics-csv",
            str(new_csv),
            "--model",
            str(train_output / "device_quality_model.joblib"),
            "--output-dir",
            str(predict_output),
            "--bootstrap",
            "20",
            "--seed",
            "7",
        ]
        predict_result = subprocess.run(
            predict_command, cwd=ROOT, check=True, text=True, capture_output=True, timeout=300
        )

        required_train = (
            "aggregated_name_subdir.csv",
            "subdir_oof_predictions.csv",
            "subdir_ablation.csv",
            "overall_oof_predictions.csv",
            "metrics.json",
            "model_weights.csv",
            "device_quality_model.joblib",
        )
        required_predict = (
            "new_subdir_predictions.csv",
            "new_device_ranking.csv",
            "new_pairwise_probabilities.csv",
            "prediction_summary.json",
        )
        assert all((train_output / name).exists() for name in required_train)
        assert all((predict_output / name).exists() for name in required_predict)

        aggregated = pd.read_csv(train_output / "aggregated_name_subdir.csv")
        single = aggregated.loc[aggregated["n_x"] == 1]
        assert len(single) > 0
        assert single["l2_between_x_std"].isna().all()
        assert single["l2_between_x_se"].isna().all()
        assert single["uncertainty_mode"].eq("within_only").all()
        ablation = pd.read_csv(train_output / "subdir_ablation.csv")
        assert set(ablation["ablation"]) == {
            "primary_only",
            "primary_plus_stability",
            "primary_plus_stability_plus_kl_js",
        }
        fold_selection = pd.read_csv(train_output / "subdir_fold_selection.csv")
        assert set(fold_selection["selected_gamma"].round(2)).issubset({0.0, 0.1})
        stage2_selection = pd.read_csv(train_output / "overall_stage2_subdir_selection.csv")
        for _, row in stage2_selection.iterrows():
            safe_devices = set(str(row["safe_subdir_devices"]).split("|"))
            assert str(row["outer_heldout_device"]) not in safe_devices
        weights = pd.read_csv(train_output / "model_weights.csv")
        constrained = weights.loc[weights["constraint"].eq("nonnegative"), "coefficient"]
        assert (constrained >= -1e-12).all()
        bundle = joblib.load(train_output / "device_quality_model.joblib")
        for model in bundle.subdir_models.values():
            assert "device_19" in model.normalizer.fit_devices_
        ranking = pd.read_csv(predict_output / "new_device_ranking.csv")
        assert len(ranking) == 1
        assert np.isfinite(ranking.loc[0, "overall_prediction"])
        assert ranking.loc[0, "overall_ci_low"] <= ranking.loc[0, "overall_ci_high"]
        print(train_result.stdout.strip())
        print(predict_result.stdout.strip())
        print("Synthetic end-to-end test passed")


if __name__ == "__main__":
    run()
