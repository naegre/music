import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--out_dir", default="exp/next_token_probs")
    parser.add_argument("--model_dir", default="../../pretrained_models/InspireMusic-1.5B-Long")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--text", default="")
    parser.add_argument("--chorus", type=int, default=1)  # 1 roughly maps to verse in CLI path
    parser.add_argument("--time_start", type=float, default=0.0)
    parser.add_argument("--time_end", type=float, default=30.0)
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--max_prompt_seconds", type=float, default=5.0)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--save_full_prob", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    os.environ["PYTHONIOENCODING"] = "UTF-8"

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "third_party" / "Matcha-TTS"))

    import torch
    from inspiremusic.cli.inspiremusic import InspireMusic
    from inspiremusic.utils.audio_utils import process_audio

    device = torch.device("cuda" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    wav_dir = Path(args.wav_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    music = InspireMusic(
        model_dir=args.model_dir,
        load_jit=True,
        load_onnx=False,
        dtype="fp16",
        fast=False,
        fp16=True,
    )

    time_start = torch.tensor([args.time_start], dtype=torch.float64, device=device)
    time_end = torch.tensor([args.time_end], dtype=torch.float64, device=device)
    chorus = torch.tensor([args.chorus], dtype=torch.int, device=device)

    wavs = sorted(wav_dir.rglob("*.wav"))
    print(f"Found {len(wavs)} wav files")

    for wav_path in wavs:
        rel = wav_path.relative_to(wav_dir)
        dump_path = out_dir / rel.with_suffix(".pt")
        dump_path.parent.mkdir(parents=True, exist_ok=True)

        if dump_path.exists():
            print(f"skip existing: {dump_path}")
            continue

        print(f"processing: {wav_path}")
        audio, _ = process_audio(str(wav_path), args.sample_rate)

        if audio.size(1) < args.sample_rate:
            print(f"skip short audio: {wav_path}")
            continue

        max_prompt_samples = int(args.max_prompt_seconds * args.sample_rate)
        audio = audio[:, :max_prompt_samples]

        model_input = music.frontend.frontend_continuation(
            args.text,
            audio,
            time_start,
            time_end,
            chorus,
            args.sample_rate,
        )

        gen = music.model.llm.inference(
            text=model_input["text_token"].to(device),
            text_len=model_input["text_token_len"].to(device),
            audio_token=model_input["audio_token"].to(device),
            audio_token_len=model_input["audio_token_len"].to(device),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32, device=device),
            prompt_text_len=torch.tensor([0], dtype=torch.int32, device=device),
            prompt_audio_token=torch.zeros(1, 0, dtype=torch.int32, device=device),
            prompt_audio_token_len=torch.tensor([0], dtype=torch.int32, device=device),
            embeddings=model_input["embeddings"],
            duration_to_gen=args.time_end - args.time_start,
            task="continuation",
            one_step_only=True,
            prob_dump_path=str(dump_path),
            prob_top_k=args.top_k,
            save_full_prob=args.save_full_prob,
        )

        for _ in gen:
            pass


if __name__ == "__main__":
    main()