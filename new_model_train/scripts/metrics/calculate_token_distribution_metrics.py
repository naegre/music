import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch


def load_distribution_file(path):
    data = torch.load(path, map_location="cpu")

    if isinstance(data, torch.Tensor):
        return {"type": "full", "logprobs": data.float()}

    if "logprobs" in data:
        return {"type": "full", "logprobs": data["logprobs"].float()}

    if "full_logprobs" in data:
        return {"type": "full", "logprobs": data["full_logprobs"].float()}

    if "records" not in data:
        raise ValueError(f"{path} does not contain logprobs/full_logprobs/records")

    records = data["records"]
    if len(records) == 0:
        raise ValueError(f"{path} contains empty records")

    first = records[0]
    for key in ("logprobs", "logp", "full_logprobs"):
        if key in first:
            logprobs = torch.stack([r[key].float() for r in records], dim=0)
            return {"type": "full", "logprobs": logprobs}

    if "top_token_ids" in first and "top_logprobs" in first:
        return {"type": "topk", "records": records}

    raise ValueError(f"Unsupported record format in {path}: {first.keys()}")


def align_full(a, b):
    t = min(a.shape[0], b.shape[0])
    v = min(a.shape[1], b.shape[1])
    return a[:t, :v].float(), b[:t, :v].float()


def kl_from_logprobs(logp, logq):
    p = logp.exp()
    values = p * (logp - logq)
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values.sum(dim=-1)


def js_from_logprobs(logp, logq):
    logm = torch.logaddexp(logp, logq) - math.log(2.0)
    return 0.5 * kl_from_logprobs(logp, logm) + 0.5 * kl_from_logprobs(logq, logm)


def entropy_from_logprobs(logp):
    p = logp.exp()
    values = -(p * logp)
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values.sum(dim=-1)


def topk_record_to_dict(record):
    ids = record["top_token_ids"].detach().cpu().long().view(-1)
    logps = record["top_logprobs"].detach().cpu().float().view(-1)
    return {int(i): float(lp) for i, lp in zip(ids, logps)}


def sparse_union_distribution(record_p, record_q, eps=1e-12):
    p_map = topk_record_to_dict(record_p)
    q_map = topk_record_to_dict(record_q)
    ids = sorted(set(p_map) | set(q_map))

    p_vals = np.array([math.exp(p_map[i]) if i in p_map else 0.0 for i in ids], dtype=np.float64)
    q_vals = np.array([math.exp(q_map[i]) if i in q_map else 0.0 for i in ids], dtype=np.float64)

    p_other = max(1.0 - float(p_vals.sum()), eps)
    q_other = max(1.0 - float(q_vals.sum()), eps)

    p = np.concatenate([p_vals, [p_other]])
    q = np.concatenate([q_vals, [q_other]])

    p = np.maximum(p, eps)
    q = np.maximum(q, eps)
    p /= p.sum()
    q /= q.sum()
    return p, q


def kl_np(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))


def js_np(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_np(p, m) + 0.5 * kl_np(q, m)


def compare_topk(records_p, records_q):
    t = min(len(records_p), len(records_q))
    kl_pq = []
    kl_qp = []
    js = []

    for i in range(t):
        p, q = sparse_union_distribution(records_p[i], records_q[i])
        kl_pq.append(kl_np(p, q))
        kl_qp.append(kl_np(q, p))
        js.append(js_np(p, q))

    return {
        "warning": "Computed from top-k records only. This is an approximation, not exact full-vocabulary KL/JS.",
        "kl_pq": np.asarray(kl_pq, dtype=np.float64),
        "kl_qp": np.asarray(kl_qp, dtype=np.float64),
        "js": np.asarray(js, dtype=np.float64),
    }


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def compare_pair(p_path, q_path, out_dir, prefix=None, save_npz=True, save_json=True):
    p_path = Path(p_path)
    q_path = Path(q_path)
    prefix = prefix or f"{p_path.stem}__vs__{q_path.stem}"

    p_data = load_distribution_file(p_path)
    q_data = load_distribution_file(q_path)

    if p_data["type"] == "full" and q_data["type"] == "full":
        logp, logq = align_full(p_data["logprobs"], q_data["logprobs"])
        kl_pq = kl_from_logprobs(logp, logq).numpy()
        kl_qp = kl_from_logprobs(logq, logp).numpy()
        js = js_from_logprobs(logp, logq).numpy()
        entropy_p = entropy_from_logprobs(logp).numpy()
        entropy_q = entropy_from_logprobs(logq).numpy()
        result_kind = "full"
        extra_arrays = {"entropy_p": entropy_p, "entropy_q": entropy_q}
        warning = ""
    elif p_data["type"] == "topk" and q_data["type"] == "topk":
        compared = compare_topk(p_data["records"], q_data["records"])
        kl_pq = compared["kl_pq"]
        kl_qp = compared["kl_qp"]
        js = compared["js"]
        result_kind = "topk_approx"
        extra_arrays = {}
        warning = compared["warning"]
    else:
        raise ValueError(
            f"Cannot compare mixed dump types: {p_data['type']} vs {q_data['type']}. "
            "Re-dump both files as full logprobs for exact KL/JS."
        )

    npz_path = out_dir / f"{prefix}.npz"
    if save_npz:
        np.savez(npz_path, kl_pq=kl_pq, kl_qp=kl_qp, js=js, **extra_arrays)

    step_csv_path = out_dir / f"{prefix}.per_step.csv"
    with step_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["step", "kl_pq", "kl_qp", "js"]
        if "entropy_p" in extra_arrays:
            fieldnames += ["entropy_p", "entropy_q"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(js)):
            row = {"step": i, "kl_pq": float(kl_pq[i]), "kl_qp": float(kl_qp[i]), "js": float(js[i])}
            if "entropy_p" in extra_arrays:
                row["entropy_p"] = float(extra_arrays["entropy_p"][i])
                row["entropy_q"] = float(extra_arrays["entropy_q"][i])
            writer.writerow(row)

    summary = {
        "p": str(p_path),
        "q": str(q_path),
        "type": result_kind,
        "num_steps": int(len(js)),
        "kl_pq": summarize(kl_pq),
        "kl_qp": summarize(kl_qp),
        "js": summarize(js),
        "output_npz": str(npz_path) if save_npz else "",
        "per_step_csv": str(step_csv_path),
    }
    if warning:
        summary["warning"] = warning
    if "entropy_p" in extra_arrays:
        summary["entropy_p"] = summarize(extra_arrays["entropy_p"])
        summary["entropy_q"] = summarize(extra_arrays["entropy_q"])

    if save_json:
        json_path = out_dir / f"{prefix}.summary.json"
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["summary_json"] = str(json_path)

    return summary


def flatten_summary(summary):
    row = {
        "p_file": summary["p"],
        "q_file": summary["q"],
        "type": summary["type"],
        "num_steps": summary["num_steps"],
        "kl_pq_mean": summary["kl_pq"]["mean"],
        "kl_pq_std": summary["kl_pq"]["std"],
        "kl_pq_p50": summary["kl_pq"]["p50"],
        "kl_pq_p90": summary["kl_pq"]["p90"],
        "kl_pq_p95": summary["kl_pq"]["p95"],
        "kl_qp_mean": summary["kl_qp"]["mean"],
        "kl_qp_std": summary["kl_qp"]["std"],
        "js_mean": summary["js"]["mean"],
        "js_std": summary["js"]["std"],
        "js_p50": summary["js"]["p50"],
        "js_p90": summary["js"]["p90"],
        "js_p95": summary["js"]["p95"],
        "js_max": summary["js"]["max"],
        "per_step_csv": summary.get("per_step_csv", ""),
        "summary_json": summary.get("summary_json", ""),
        "output_npz": summary.get("output_npz", ""),
        "warning": summary.get("warning", ""),
    }
    if "entropy_p" in summary:
        row["entropy_p_mean"] = summary["entropy_p"]["mean"]
        row["entropy_q_mean"] = summary["entropy_q"]["mean"]
    return row


def write_summary_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_pt_pairs(p_dir, q_dir):
    p_files = {p.stem: p for p in Path(p_dir).glob("*.pt")}
    q_files = {q.stem: q for q in Path(q_dir).glob("*.pt")}
    keys = sorted(set(p_files) & set(q_files))
    return [(key, p_files[key], q_files[key]) for key in keys]


def main():
    parser = argparse.ArgumentParser(
        description="Compare InspireMusic next-token probability dumps with KL and JS divergence."
    )
    parser.add_argument("--p", default=None, help="Reference/original probability dump .pt")
    parser.add_argument("--q", default=None, help="Test/degraded/generated probability dump .pt")
    parser.add_argument("--p_dir", default=None, help="Directory of reference/original probability dumps")
    parser.add_argument("--q_dir", default=None, help="Directory of test/degraded/generated probability dumps")
    parser.add_argument("--out_dir", default="token_distribution_metrics")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--summary_csv", default="summary.csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.p_dir and args.q_dir:
        pairs = find_pt_pairs(args.p_dir, args.q_dir)
        if not pairs:
            raise RuntimeError("No .pt pairs found. Pairing is done by filename stem.")
        for key, p_path, q_path in pairs:
            print(f"comparing {p_path} vs {q_path}")
            pair_prefix = f"{args.prefix}_{key}" if args.prefix else key
            summary = compare_pair(p_path, q_path, out_dir, prefix=pair_prefix)
            rows.append(flatten_summary(summary))
    elif args.p and args.q:
        summary = compare_pair(args.p, args.q, out_dir, prefix=args.prefix)
        rows.append(flatten_summary(summary))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        raise ValueError("Use either --p/--q for one pair or --p_dir/--q_dir for paired directories.")

    summary_csv_path = out_dir / args.summary_csv
    write_summary_csv(rows, summary_csv_path)
    print(f"wrote summary csv: {summary_csv_path}")


if __name__ == "__main__":
    main()
