# music
inspiremusic/llm/llm.py
examples/music_generation/batch_next_token_prob.py

cd /mnt/data/pyf_sd/inspiremusic/InspireMusic/examples/music_generation

python batch_next_token_prob.py \
  --wav_dir /path/to/your/wav_folder \
  --out_dir exp/next_token_probs \
  --model_dir ../../pretrained_models/InspireMusic-1.5B-Long \
  --gpu 0 \
  --top_k 100


all D:
python batch_next_token_prob.py \
  --wav_dir /path/to/your/wav_folder \
  --out_dir exp/next_token_probs_full \
  --gpu 0 \
  --save_full_prob

check result:
import torch

x = torch.load("exp/next_token_probs/some_audio.pt")
print(x["top_token_ids"])
print(x["top_logprobs"])
