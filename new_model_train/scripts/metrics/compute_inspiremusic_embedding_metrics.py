import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm

from inspiremusic.cli.inference import InspireMusicModel, env_variables
from inspiremusic.utils.audio_utils import process_audio


def frechet_distance(x, y, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("FD expects two 2-D embedding matrices")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Embedding dimension mismatch: {x.shape[1]} vs {y.shape[1]}")
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError("FD needs at least two samples per set for covariance estimation")

    mu_x = x.mean(axis=0)
    mu_y = y.mean(axis=0)
    cov_x = np.cov(x, rowvar=False) + np.eye(x.shape[1]) * eps
    cov_y = np.cov(y, rowvar=False) + np.eye(y.shape[1]) * eps

    covmean = sqrtm(cov_x @ cov_y)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu_x - mu_y
    return float(diff @ diff + np.trace(cov_x + cov_y - 2.0 * covmean))


def find_wav_pairs(ref_dir, test_dir):
    ref = {p.stem: p for p in Path(ref_dir).glob("*.wav")}
    test = {p.stem: p for p in Path(test_dir).glob("*.wav")}
    keys = sorted(set(ref) & set(test))
    return [(key, ref[key], test[key]) for key in keys]


@torch.inference_mode()
def extract_embedding(model_wrapper, wav_path, sample_rate, max_audio_seconds, pool):
    device = model_wrapper.device
    audio, _ = process_audio(str(wav_path), sample_rate)

    if max_audio_seconds is not None and max_audio_seconds > 0:
        audio = audio[:, : int(max_audio_seconds * sample_rate)]

    time_start = torch.tensor([0.0], dtype=torch.float64, device=device)
    time_end = torch.tensor([30.0], dtype=torch.float64, device=device)
    chorus = torch.tensor([1], dtype=torch.int, device=device)

    model_input = model_wrapper.model.frontend.frontend_continuation(
        text=None,
        audio_prompt=audio,
        time_start=time_start,
        time_end=time_end,
        chorus=chorus,
        sr=sample_rate,
    )

    token = model_input["audio_token"].to(device).long()
    if token.dim() == 1:
        token = token.unsqueeze(0)

    # Continuation LLM uses semantic tokens. If frontend returns flattened
    # acoustic tokens with four codebooks, use the first codebook.
    if token.shape[1] > 0 and token.shape[1] % 4 == 0:
        token = token.view(token.size(0), -1, 4)[:, :, 0]

    emb = model_wrapper.model.model.llm.speech_embedding(token)
    emb = emb.squeeze(0).float().cpu()

    if pool == "mean":
        pooled = emb.mean(dim=0)
    elif pool == "mean_std":
        pooled = torch.cat([emb.mean(dim=0), emb.std(dim=0)], dim=0)
    else:
        raise ValueError(f"Unknown pool: {pool}")

    return pooled.numpy()


def build_model(args):
    env_variables()
    return InspireMusicModel(
        model_name=args.model_name,
        model_dir=args.model_dir,
        sample_rate=args.sample_rate,
        output_sample_rate=args.output_sample_rate,
        gpu=args.gpu,
        result_dir=args.result_dir,
        fast=args.fast,
        fp16=args.fp16,
    )


def write_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute InspireMusic embedding FD plus paired cosine/L2 metrics."
    )
    parser.add_argument("--ref_dir", required=True, help="Aligned original/reference wav directory")
    parser.add_argument("--test_dir", required=True, help="Aligned device/generated wav directory")
    parser.add_argument("--out_csv", default="inspiremusic_embedding_metrics.csv")
    parser.add_argument("--save_npz", default="inspiremusic_embeddings.npz")
    parser.add_argument("--model_name", default="InspireMusic-1.5B-Long")
    parser.add_argument("--model_dir", default="../../pretrained_models/InspireMusic-1.5B-Long")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--output_sample_rate", type=int, default=48000)
    parser.add_argument("--max_audio_seconds", type=float, default=5.0)
    parser.add_argument("--pool", choices=["mean", "mean_std"], default="mean")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--result_dir", default="exp/eval_dummy")
    args = parser.parse_args()

    pairs = find_wav_pairs(args.ref_dir, args.test_dir)
    if not pairs:
        raise RuntimeError("No paired wav files found. Pairing is done by filename stem.")

    model = build_model(args)
    rows = []
    ref_embs = []
    test_embs = []
    ref_keys = []
    test_keys = []

    for key, ref_path, test_path in pairs:
        print(f"extracting {key}")
        ref_emb = extract_embedding(model, ref_path, args.sample_rate, args.max_audio_seconds, args.pool)
        test_emb = extract_embedding(model, test_path, args.sample_rate, args.max_audio_seconds, args.pool)

        ref_vec = torch.tensor(ref_emb).float()
        test_vec = torch.tensor(test_emb).float()
        cosine_similarity = F.cosine_similarity(ref_vec[None], test_vec[None]).item()
        l2_distance = torch.linalg.norm(ref_vec - test_vec).item()

        rows.append(
            {
                "metric_scope": "pair",
                "metric": "embedding_pairwise",
                "key": key,
                "ref_file": str(ref_path),
                "test_file": str(test_path),
                "cosine_similarity": cosine_similarity,
                "cosine_distance": 1.0 - cosine_similarity,
                "l2_distance": l2_distance,
                "max_audio_seconds": args.max_audio_seconds,
                "pool": args.pool,
            }
        )

        ref_embs.append(ref_emb)
        test_embs.append(test_emb)
        ref_keys.append(key)
        test_keys.append(key)

    ref_mat = np.stack(ref_embs, axis=0)
    test_mat = np.stack(test_embs, axis=0)
    fd = frechet_distance(ref_mat, test_mat)

    rows.insert(
        0,
        {
            "metric_scope": "set",
            "metric": "frechet_distance",
            "key": "ALL",
            "ref_file": str(Path(args.ref_dir)),
            "test_file": str(Path(args.test_dir)),
            "frechet_distance": fd,
            "num_pairs": len(pairs),
            "max_audio_seconds": args.max_audio_seconds,
            "pool": args.pool,
        },
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_csv)

    if args.save_npz:
        np.savez(
            args.save_npz,
            ref_keys=np.array(ref_keys),
            test_keys=np.array(test_keys),
            ref_embeddings=ref_mat,
            test_embeddings=test_mat,
        )

    print(json.dumps({"out_csv": str(out_csv), "frechet_distance": fd, "num_pairs": len(pairs)}, indent=2))


if __name__ == "__main__":
    main()
