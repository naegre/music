import argparse
import csv
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def read_audio(path):
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    audio = audio.T
    audio = np.mean(audio, axis=0)
    return audio.astype(np.float32), sr


def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return signal.resample_poly(audio, up, down).astype(np.float32)


def peak_limit(audio, peak=0.99):
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs > peak:
        audio = audio * (peak / max_abs)
    return audio.astype(np.float32)


def rms_dbfs(audio, eps=1e-12):
    return 20.0 * np.log10(np.sqrt(np.mean(audio ** 2) + eps))


def normalize_rms(audio, target_dbfs=-23.0):
    gain_db = target_dbfs - rms_dbfs(audio)
    audio = audio * (10.0 ** (gain_db / 20.0))
    return peak_limit(audio)


def normalize_loudness(audio, sr, target_lufs=-23.0):
    try:
        import pyloudnorm as pyln
    except ImportError:
        return normalize_rms(audio, target_lufs), "rms_fallback"

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    audio = pyln.normalize.loudness(audio, loudness, target_lufs)
    return peak_limit(audio), "lufs"


def normalize_audio(audio, sr, method, target):
    if method == "none":
        return peak_limit(audio), "none"
    if method == "rms":
        return normalize_rms(audio, target), "rms"
    if method == "lufs":
        return normalize_loudness(audio, sr, target)
    raise ValueError(f"unknown normalize method: {method}")


def estimate_lag_samples(ref, test, sr, align_sr=8000, max_lag_seconds=10.0):
    ref_ds = resample_audio(ref, sr, align_sr)
    test_ds = resample_audio(test, sr, align_sr)

    ref_ds = ref_ds - np.mean(ref_ds)
    test_ds = test_ds - np.mean(test_ds)
    ref_std = np.std(ref_ds) + 1e-8
    test_std = np.std(test_ds) + 1e-8
    ref_ds = ref_ds / ref_std
    test_ds = test_ds / test_std

    corr = signal.correlate(test_ds, ref_ds, mode="full", method="fft")
    lags = signal.correlation_lags(len(test_ds), len(ref_ds), mode="full")

    max_lag = int(max_lag_seconds * align_sr)
    valid = np.abs(lags) <= max_lag
    corr = corr[valid]
    lags = lags[valid]

    lag_ds = int(lags[int(np.argmax(corr))])
    return int(round(lag_ds * sr / align_sr))


def align_by_lag(ref, test, lag):
    if lag > 0:
        test_start = lag
        ref_start = 0
    else:
        test_start = 0
        ref_start = -lag

    length = min(len(ref) - ref_start, len(test) - test_start)
    if length <= 0:
        raise ValueError("estimated lag leaves no overlap")

    return ref[ref_start:ref_start + length], test[test_start:test_start + length]


def find_pairs(ref_dir, test_dir, suffix=".wav"):
    ref_paths = {p.stem: p for p in Path(ref_dir).glob(f"*{suffix}")}
    test_paths = {p.stem: p for p in Path(test_dir).glob(f"*{suffix}")}
    keys = sorted(set(ref_paths) & set(test_paths))
    return [(key, ref_paths[key], test_paths[key]) for key in keys]


def main():
    parser = argparse.ArgumentParser(
        description="Resample, loudness-normalize, time-align, and crop paired audio files."
    )
    parser.add_argument("--ref_dir", required=True, help="Directory with original/reference wav files")
    parser.add_argument("--test_dir", required=True, help="Directory with device-recorded/generated wav files")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--normalize", choices=["none", "rms", "lufs"], default="lufs")
    parser.add_argument("--target_level", type=float, default=-23.0, help="LUFS or dBFS target")
    parser.add_argument("--align_sr", type=int, default=8000)
    parser.add_argument("--max_lag_seconds", type=float, default=10.0)
    parser.add_argument("--crop_seconds", type=float, default=None, help="Optional fixed duration after alignment")
    parser.add_argument("--suffix", default=".wav")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ref_out = out_dir / "ref"
    test_out = out_dir / "test"
    ref_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(args.ref_dir, args.test_dir, args.suffix)
    if not pairs:
        raise RuntimeError("No paired files found. Make sure stems match, e.g. song001.wav in both dirs.")

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "ref_path",
                "test_path",
                "lag_samples",
                "lag_seconds",
                "duration_seconds",
                "normalization",
                "ref_out",
                "test_out",
            ],
        )
        writer.writeheader()

        for key, ref_path, test_path in pairs:
            print(f"processing {key}")
            ref, ref_sr = read_audio(ref_path)
            test, test_sr = read_audio(test_path)

            ref = resample_audio(ref, ref_sr, args.sample_rate)
            test = resample_audio(test, test_sr, args.sample_rate)

            ref, norm_kind_ref = normalize_audio(ref, args.sample_rate, args.normalize, args.target_level)
            test, norm_kind_test = normalize_audio(test, args.sample_rate, args.normalize, args.target_level)

            lag = estimate_lag_samples(
                ref,
                test,
                args.sample_rate,
                align_sr=args.align_sr,
                max_lag_seconds=args.max_lag_seconds,
            )
            ref_aligned, test_aligned = align_by_lag(ref, test, lag)

            if args.crop_seconds is not None:
                crop_len = int(args.crop_seconds * args.sample_rate)
                ref_aligned = ref_aligned[:crop_len]
                test_aligned = test_aligned[:crop_len]

            length = min(len(ref_aligned), len(test_aligned))
            ref_aligned = peak_limit(ref_aligned[:length])
            test_aligned = peak_limit(test_aligned[:length])

            ref_file = ref_out / f"{key}.wav"
            test_file = test_out / f"{key}.wav"
            sf.write(ref_file, ref_aligned, args.sample_rate, subtype="PCM_24")
            sf.write(test_file, test_aligned, args.sample_rate, subtype="PCM_24")

            writer.writerow(
                {
                    "key": key,
                    "ref_path": str(ref_path),
                    "test_path": str(test_path),
                    "lag_samples": lag,
                    "lag_seconds": lag / args.sample_rate,
                    "duration_seconds": length / args.sample_rate,
                    "normalization": f"{norm_kind_ref}/{norm_kind_test}",
                    "ref_out": str(ref_file),
                    "test_out": str(test_file),
                }
            )

    print(f"wrote {manifest_path}")
    print(f"aligned reference wavs: {ref_out}")
    print(f"aligned test wavs: {test_out}")


if __name__ == "__main__":
    main()
