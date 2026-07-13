import argparse
import csv
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def read_mono(path):
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    audio = audio.T
    audio = np.mean(audio, axis=0)
    return audio.astype(np.float32), sr


def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    gcd = math.gcd(orig_sr, target_sr)
    return signal.resample_poly(audio, target_sr // gcd, orig_sr // gcd).astype(np.float32)


def frame_rms(audio, frame_len, hop_len):
    if len(audio) < frame_len:
        audio = np.pad(audio, (0, frame_len - len(audio)))
    n_frames = 1 + int(np.ceil((len(audio) - frame_len) / hop_len))
    pad_len = (n_frames - 1) * hop_len + frame_len
    if len(audio) < pad_len:
        audio = np.pad(audio, (0, pad_len - len(audio)))

    frames = np.lib.stride_tricks.sliding_window_view(audio, frame_len)[::hop_len]
    return np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)


def smooth_boolean(mask, min_len):
    if min_len <= 1:
        return mask
    kernel = np.ones(min_len, dtype=np.float32)
    hits = np.convolve(mask.astype(np.float32), kernel, mode="same")
    return hits > 0


def mask_to_segments(mask, hop_len, sr):
    segments = []
    in_segment = False
    start = 0
    for i, active in enumerate(mask):
        if active and not in_segment:
            start = i
            in_segment = True
        elif not active and in_segment:
            end = i
            segments.append((start * hop_len, end * hop_len))
            in_segment = False
    if in_segment:
        segments.append((start * hop_len, len(mask) * hop_len))
    return segments


def merge_close_segments(segments, max_gap_samples):
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap_samples:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def detect_segments(
    audio,
    sr,
    silence_db,
    frame_ms,
    hop_ms,
    min_silence_seconds,
    min_segment_seconds,
    pad_seconds,
):
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))
    rms = frame_rms(audio, frame_len, hop_len)
    db = 20.0 * np.log10(rms + 1e-12)

    active = db > silence_db

    # Fill very short inactive gaps inside an active region.
    max_gap_frames = max(1, int(min_silence_seconds * sr / hop_len))
    segments = mask_to_segments(active, hop_len, sr)
    segments = merge_close_segments(segments, max_gap_frames * hop_len)

    min_len = int(min_segment_seconds * sr)
    pad = int(pad_seconds * sr)
    clipped = []
    for start, end in segments:
        start = max(0, start - pad)
        end = min(len(audio), end + pad)
        if end - start >= min_len:
            clipped.append((start, end))
    return clipped


def main():
    parser = argparse.ArgumentParser(
        description="Split a long recording into non-silent audio segments and optionally drop the first pink-noise segment."
    )
    parser.add_argument("--input", required=True, help="Long recording wav")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--silence_db", type=float, default=-40.0)
    parser.add_argument("--frame_ms", type=float, default=50.0)
    parser.add_argument("--hop_ms", type=float, default=10.0)
    parser.add_argument("--min_silence_seconds", type=float, default=0.5)
    parser.add_argument("--min_segment_seconds", type=float, default=1.0)
    parser.add_argument("--pad_seconds", type=float, default=0.05)
    parser.add_argument("--skip_start_seconds", type=float, default=0.0)
    parser.add_argument("--drop_first_segment", action="store_true", help="Drop the first detected active segment, useful for initial pink noise.")
    parser.add_argument("--prefix", default="segment")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = read_mono(args.input)
    audio = resample_audio(audio, sr, args.sample_rate)

    if args.skip_start_seconds > 0:
        audio = audio[int(args.skip_start_seconds * args.sample_rate):]

    segments = detect_segments(
        audio,
        args.sample_rate,
        args.silence_db,
        args.frame_ms,
        args.hop_ms,
        args.min_silence_seconds,
        args.min_segment_seconds,
        args.pad_seconds,
    )

    if args.drop_first_segment and segments:
        dropped = segments[0]
        segments = segments[1:]
        print(
            f"dropped first segment as pink noise: "
            f"{dropped[0] / args.sample_rate:.3f}s-{dropped[1] / args.sample_rate:.3f}s"
        )

    manifest_path = out_dir / "segments.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "start_seconds", "end_seconds", "duration_seconds", "path"],
        )
        writer.writeheader()

        for i, (start, end) in enumerate(segments):
            segment = audio[start:end]
            max_abs = float(np.max(np.abs(segment))) if segment.size else 0.0
            if max_abs > 0.99:
                segment = segment * (0.99 / max_abs)

            path = out_dir / f"{args.prefix}_{i:04d}.wav"
            sf.write(path, segment, args.sample_rate, subtype="PCM_24")
            writer.writerow(
                {
                    "index": i,
                    "start_seconds": start / args.sample_rate,
                    "end_seconds": end / args.sample_rate,
                    "duration_seconds": (end - start) / args.sample_rate,
                    "path": str(path),
                }
            )

    print(f"detected/wrote {len(segments)} segments")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
