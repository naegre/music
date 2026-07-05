# InspireMusic Audio Evaluation Tools

这组脚本用于评估原始音频和设备播放/录音音频之间的差异，包含：

- 长录音静音切段，并可丢弃开头粉噪段
- 成对音频预处理：重采样、响度归一化、时间对齐、裁剪
- 使用 InspireMusic 自身 audio/token embedding 计算 FD、cosine、L2
- 使用已保存的 next-token 概率分布计算 KL 散度和 JS 距离

## Environment

建议直接在已经跑通 InspireMusic 的 conda 环境中使用：

```bash
conda activate inspiremusic_py310
pip install -r requirements-audio-eval.txt
```

如果要计算 InspireMusic embedding FD，脚本需要能 import InspireMusic 项目代码。推荐在项目的 `examples/music_generation` 目录运行，或者保证 `PYTHONPATH` 包含 InspireMusic 仓库根目录。

```bash
cd /mnt/data/pyf_sd/inspiremusic/InspireMusic/examples/music_generation
```

`pyloudnorm` 是可选依赖。如果没有安装，`preprocess_audio_pairs.py` 在使用 `--normalize lufs` 时会自动退回 RMS 归一化。

## File Naming

成对评估默认按文件名 stem 配对。例如：

```text
original_wavs/song001.wav
device_wavs/song001.wav
```

这两个文件会被认为是一对。

## 1. Split Long Device Recording

如果设备录音是一整条长音频，并且结构类似：

```text
开头粉噪 + 空白 + 音频段1 + 空白 + 音频段2 + ...
```

先用静音检测切段，并丢弃第一个粉噪段：

```bash
python split_recording_by_silence.py \
  --input /path/to/long_recording.wav \
  --out_dir /path/to/device_segments \
  --sample_rate 24000 \
  --silence_db -40 \
  --min_silence_seconds 0.5 \
  --min_segment_seconds 1.0 \
  --drop_first_segment
```

输出：

```text
device_segments/segment_0000.wav
device_segments/segment_0001.wav
device_segments/segments.csv
```

如果切得太碎，可以尝试：

```bash
--silence_db -35 --min_silence_seconds 0.8
```

如果漏掉轻声音频，可以尝试：

```bash
--silence_db -45
```

## 2. Preprocess Paired Audio

预处理会做：

```text
转单声道 -> 重采样 -> 响度归一化 -> 互相关估计延迟 -> 时间对齐 -> 裁成同样长度
```

严格模拟 InspireMusic continuation 默认只看前 5 秒时：

```bash
python preprocess_audio_pairs.py \
  --ref_dir /path/to/original_wavs \
  --test_dir /path/to/device_segments \
  --out_dir /path/to/aligned_wavs \
  --sample_rate 24000 \
  --normalize lufs \
  --target_level -23 \
  --crop_seconds 5
```

输出：

```text
aligned_wavs/ref/*.wav
aligned_wavs/test/*.wav
aligned_wavs/manifest.csv
```

`manifest.csv` 会记录每对音频的延迟估计和输出路径。

如果要评估整段音频质量，可以不传 `--crop_seconds`，或者设成更长时间，例如 `30`。

## 3. Compute InspireMusic Embedding FD / Cosine / L2

这个脚本使用 InspireMusic 自己的 audio token embedding：

```text
wav -> InspireMusic tokenizer -> audio token ids -> llm.speech_embedding -> pooled embedding
```

运行：

```bash
python compute_inspiremusic_embedding_metrics.py \
  --ref_dir /path/to/aligned_wavs/ref \
  --test_dir /path/to/aligned_wavs/test \
  --model_dir ../../pretrained_models/InspireMusic-1.5B-Long \
  --model_name InspireMusic-1.5B-Long \
  --gpu 0 \
  --max_audio_seconds 5 \
  --out_csv /path/to/results/inspiremusic_embedding_metrics.csv \
  --save_npz /path/to/results/inspiremusic_embeddings.npz
```

CSV 中包含：

- `frechet_distance`：集合级 FD，比较两个目录的整体 embedding 分布
- `cosine_similarity` / `cosine_distance`：逐文件配对相似度
- `l2_distance`：逐文件配对 L2 距离
- `ref_file` / `test_file`：明确标注计算的是哪两个文件

注意：FD 是集合级指标，样本数太少时协方差估计会不稳定。逐文件比较请主要看 cosine/L2。

## 4. Compute KL / JS From Next-token Distributions

脚本读取之前保存的 InspireMusic next-token 概率分布 `.pt` 文件。

推荐 `.pt` 格式：

```python
{
    "logprobs": torch.Tensor,  # shape: [T, vocab_size]
}
```

如果 `.pt` 只包含 `records` 里的 `top_token_ids/top_logprobs`，脚本也能计算 top-k 近似 KL/JS，但正式实验建议保存完整 `logprobs`。

### Single Pair

```bash
python calculate_token_distribution_metrics.py \
  --p /path/to/original_probs/song001.pt \
  --q /path/to/device_probs/song001.pt \
  --out_dir /path/to/results/token_distribution_metrics \
  --prefix song001 \
  --summary_csv token_distribution_summary.csv
```

### Directory Pair

如果两个目录下 `.pt` 文件名一致：

```text
original_probs/song001.pt
device_probs/song001.pt
```

可以批量计算：

```bash
python calculate_token_distribution_metrics.py \
  --p_dir /path/to/original_probs \
  --q_dir /path/to/device_probs \
  --out_dir /path/to/results/token_distribution_metrics \
  --summary_csv token_distribution_summary.csv
```

输出：

```text
token_distribution_metrics/token_distribution_summary.csv
token_distribution_metrics/<key>.per_step.csv
token_distribution_metrics/<key>.summary.json
token_distribution_metrics/<key>.npz
```

`token_distribution_summary.csv` 会按文件对汇总：

- `p_file` / `q_file`
- `kl_pq_mean`：KL(P || Q)
- `kl_qp_mean`：KL(Q || P)
- `js_mean`
- `js_p50` / `js_p90` / `js_p95`
- `num_steps`

其中 `P` 通常放原始音频分布，`Q` 放设备播放/录音音频分布。

## Recommended Pipeline

典型流程：

```bash
# 1. 长录音切段，丢弃开头粉噪
python split_recording_by_silence.py \
  --input /path/to/long_recording.wav \
  --out_dir /path/to/device_segments \
  --sample_rate 24000 \
  --silence_db -40 \
  --min_silence_seconds 0.5 \
  --drop_first_segment

# 2. 与原始音频配对并预处理
python preprocess_audio_pairs.py \
  --ref_dir /path/to/original_wavs \
  --test_dir /path/to/device_segments \
  --out_dir /path/to/aligned_wavs \
  --sample_rate 24000 \
  --normalize lufs \
  --target_level -23 \
  --crop_seconds 5

# 3. 计算 InspireMusic embedding FD/cosine/L2
python compute_inspiremusic_embedding_metrics.py \
  --ref_dir /path/to/aligned_wavs/ref \
  --test_dir /path/to/aligned_wavs/test \
  --model_dir ../../pretrained_models/InspireMusic-1.5B-Long \
  --gpu 0 \
  --max_audio_seconds 5 \
  --out_csv /path/to/results/inspiremusic_embedding_metrics.csv

# 4. 如果已经保存了 next-token logprobs，则计算 KL/JS
python calculate_token_distribution_metrics.py \
  --p_dir /path/to/original_probs \
  --q_dir /path/to/device_probs \
  --out_dir /path/to/results/token_distribution_metrics \
  --summary_csv token_distribution_summary.csv
```

## Notes

- InspireMusic continuation 默认最多使用约 5 秒音频 prompt。因此如果目标是模拟模型真实输入，建议 `--crop_seconds 5` 和 `--max_audio_seconds 5`。
- 如果目标是评价整段设备播放质量，可以改用更长的裁剪长度，或不裁剪。
- KL/JS 比较的是模型 next-token 分布差异，不是 waveform 级失真。
- FD/cosine/L2 使用的是 InspireMusic embedding 空间，结果更接近模型感知到的音频 token 差异。
- 做任何配对指标前，时间对齐都很重要，否则延迟会严重污染结果。

## 5. Fit A Perceptual Score Model

如果你有人类感知实验分数，例如 MOS/CMOS，可以用 `fit_perceptual_score_model.py` 学习一个可扩展的指标融合模型。输入 CSV 至少需要包含一列人类分数，以及若干客观指标列。

示例输入：

```text
key,human_mos,js_mean,kl_pq_mean,cosine_distance,l2_distance,mel_l1,sisdr
song001,4.2,0.031,0.210,0.08,1.52,0.14,18.3
song002,3.7,0.055,0.340,0.12,2.10,0.21,15.6
```

默认会自动选择所有数值型非目标列作为特征，并根据列名自动判断方向：`kl/js/l2/fd/distance/loss/error` 等会按越小越好处理，`similarity/sisdr/snr/stoi/pesq` 等会按越大越好处理。

推荐先使用 Ridge：

```bash
python fit_perceptual_score_model.py \
  --input_csv /path/to/combined_metrics_with_human_scores.csv \
  --target_col human_mos \
  --model ridge \
  --out_dir /path/to/results/perceptual_score_model
```

如果想指定特征列：

```bash
python fit_perceptual_score_model.py \
  --input_csv /path/to/combined_metrics_with_human_scores.csv \
  --target_col human_mos \
  --feature_cols js_mean,kl_pq_mean,cosine_distance,l2_distance,mel_l1,sisdr \
  --model ridge \
  --out_dir /path/to/results/perceptual_score_model
```

如果希望所有权重都非负、解释成“每个正向指标都贡献更高感知分数”，可以用 NNLS：

```bash
python fit_perceptual_score_model.py \
  --input_csv /path/to/combined_metrics_with_human_scores.csv \
  --target_col human_mos \
  --model nnls \
  --out_dir /path/to/results/perceptual_score_model_nnls
```

输出：

```text
perceptual_score_model/predictions.csv
perceptual_score_model/weights.csv
perceptual_score_model/summary.json
```

其中：

- `predictions.csv`：每条样本的人类分数、训练集拟合预测分数、交叉验证预测分数
- `weights.csv`：每个指标的学习权重，便于解释综合评分公式
- `summary.json`：Pearson、Spearman、RMSE、MAE 等拟合效果

可以用 `--direction_json` 手动覆盖指标方向。JSON 示例：

```json
{
  "js_mean": "lower",
  "kl_pq_mean": "lower",
  "cosine_similarity": "higher",
  "sisdr": "higher"
}
```

然后运行：

```bash
python fit_perceptual_score_model.py \
  --input_csv /path/to/combined_metrics_with_human_scores.csv \
  --target_col human_mos \
  --direction_json metric_directions.json \
  --model ridge \
  --out_dir /path/to/results/perceptual_score_model
```
