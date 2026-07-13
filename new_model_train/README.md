# 设备音频质量评估与排名

本项目把设备录音与原始音频之间的 InspireMusic 指标，校准为各任务质量百分位和设备 overall 分数，并估计新设备在已知设备中的排名。

核心原则：L2 和 cosine 是主特征；KL/JS 只能拟合主模型的 OOF 残差，并由嵌套 Leave-One-Device-Out（LOODO）决定是否以很小的 `gamma` 启用。

## 1. 输入不是窗口级数据

训练和预测 CSV **必须已经是音频 `x` 级别的聚合结果**。每行唯一对应：

```text
一个设备 name + 一个任务 subdir + 一段测试音频 x
```

程序不会再做“窗口 -> 音频”聚合。若同一个 `name+subdir+x` 出现多行，数据验证会直接报错。

默认列：

```text
subdir,name,x,n_windows,
kl_mean,kl_var,js_mean,js_var,
l2_mean,l2_var,cosim_mean,cosim_var
```

示例：

```csv
subdir,name,x,n_windows,kl_mean,kl_var,js_mean,js_var,l2_mean,l2_var,cosim_mean,cosim_var
clarity,xiaomi,2,6,5.13123,1.2654,0.4261,0.0123,3.23135,0.1235,0.83215,0.0012
clarity,ipad,2,6,4.91200,1.1020,0.4010,0.0108,2.74500,0.1010,0.87120,0.0010
treble,xiaomi,0,11,4.60000,0.9300,0.3900,0.0090,3.01000,0.0900,0.85000,0.0008
```

约定：

- `n_windows` 是构成该 `x` 统计量的窗口数。
- `*_mean` 和 `*_var` 是窗口指标的均值和方差；方差必须非负。
- 原始窗口长度为 5 秒、步长为 1 秒。程序按 `n_effective=max(1,(n_windows+4)/5)` 保守估计有效样本数。
- 相同 `subdir+x` 在不同设备中必须始终对应同一段原始音频。
- `x` 可以是数字或字符串；程序内部统一按字符串处理。

人工评分 JSON 默认是 subdir-first：

```json
{
  "clarity": {
    "xiaomi": 75.0,
    "ipad": 80.0
  },
  "lowfreq": {
    "xiaomi": 72.0,
    "ipad": 78.0
  },
  "overall": {
    "xiaomi": 79.0,
    "ipad": 83.0
  }
}
```

CSV 或 JSON 可以缺少设备、整个 subdir 或少量 `x`。没有人工分数的 CSV 设备会参与最终部署模型的 robust 参考基准，但不会进入监督训练。

仓库中的 `examples/sample_x_metrics.csv`、`examples/sample_human_scores.json` 和 `examples/column_map_zh.json` 可用于核对文件格式。示例只有两台设备，不足以训练嵌套 LOODO 模型；正式训练至少需要 5 台 overall 标注设备，建议接近当前实验的约 20 台规模。

## 2. 环境安装

设备质量模型本身可在 CPU 上训练，建议 Python 3.10：

```bash
conda create -n audio_quality python=3.10 -y
conda activate audio_quality
pip install -r requirements-audio-eval.txt
```

如果还要运行 InspireMusic embedding 或 next-token 提取脚本，请使用已经能运行 InspireMusic 的环境，并先按服务器 CUDA/PyTorch 兼容关系安装 PyTorch。`requirements-audio-eval.txt` 不包含 `torch`，避免覆盖现有 CUDA 版本。

## 3. 训练

标准命令：

```bash
python device_quality_main.py train \
  --metrics-csv /path/to/device_x_metrics.csv \
  --scores-json /path/to/human_scores.json \
  --output-dir /path/to/device_quality_model
```

默认搜索：

```text
subdir primary alpha: 1,10,100
auxiliary gamma:       0,0.05,0.10,0.20
overall alpha:         0.1,1,10
overall rank lambda:   0.25,1,4
```

约 20 台设备、6 个 subdir 的严格嵌套 LOODO 会执行较多小模型拟合，训练可能需要数分钟到更久；这是一次性成本。调试数据管道时可缩小候选集合，例如：

```bash
python device_quality_main.py train \
  --metrics-csv /path/to/device_x_metrics.csv \
  --scores-json /path/to/human_scores.json \
  --output-dir /path/to/debug_model \
  --subdir-alphas 10 \
  --gammas 0,0.10 \
  --overall-alphas 0.1,1 \
  --rank-lambdas 0.25,1
```

正式结果建议使用默认完整 `gamma` 集合。程序只接受 `0,0.05,0.10,0.20` 中的值。

若 JSON 是 device-first，可加：

```bash
--score-layout device_first
```

### 中文列名映射

`--column-map` 接受“标准英文列名 -> 输入 CSV 列名”的 JSON 字符串或 JSON 文件：

```bash
python device_quality_main.py train \
  --metrics-csv metrics_zh.csv \
  --scores-json scores.json \
  --output-dir model_output \
  --column-map '{"subdir":"类型","name":"设备名称","x":"音频文件序号","n_windows":"窗口数量","kl_mean":"KL均值","kl_var":"KL方差","js_mean":"JS均值","js_var":"JS方差","l2_mean":"L2均值","l2_var":"L2方差","cosim_mean":"余弦均值","cosim_var":"余弦方差"}'
```

也可以把该 JSON 保存为 `column_map.json`，然后使用 `--column-map column_map.json`。预测 CSV 可以单独指定自己的映射。

## 4. 模型做了什么

### 4.1 配对 robust 标准化

所有变换都在对应训练折内拟合。程序先变换指标：

```text
L2、KL、JS: log1p
cosine:     clip 到 (-1,1) 后做 atanh/Fisher 变换
```

随后在每个 `subdir+x` 内跨训练设备计算 `median/MAD`。若 MAD 为 0，依次回退到 IQR、标准差和常数尺度。质量方向统一为：

```text
L2 quality   = -robust_z(log1p(l2_mean))
Cos quality  =  robust_z(atanh(clipped cosim_mean))
KL quality   = -robust_z(log1p(kl_mean))
JS quality   = -robust_z(log1p(js_mean))
```

因此所有 quality 都是越高越好。

### 4.2 聚合不同 x

同一 `name+subdir` 下的每个 `x` 等权。`n_windows` 较大不会直接赋予更大的质量权重。输出包含每个 quality 的：

- mean、median
- `between_x_std`、`between_x_se`
- x 内不确定性和合并后的 `total_se`
- `n_x`、`coverage_ratio`、`n_windows_total`、`has_multiple_x`

若 `n_x=1`：

```text
mean = median = 该 x 的 quality
between_x_std = NaN
between_x_se  = NaN
uncertainty_mode = within_only
```

这时置信区间只能使用窗口方差和 `n_windows` 做参数化近似，不能伪造跨音频波动为 0。

### 4.3 独立 subdir 模型

每个 subdir 单独训练。人工分数先转成折内排名百分位，避免 `6.8` 对比约 `70` 的极端值控制损失。

第一阶段只使用 L2/cosine quality 的 mean、median 和少量稳定性/不确定性特征。所有特征已统一为“越高越好”，模型系数约束为非负，因此更低 L2 或更高 cosine 不会使预测质量下降。

第二阶段使用 KL/JS 拟合第一阶段的 **设备留一 OOF 残差**：

```text
prediction = primary_prediction + gamma * auxiliary_correction
```

`gamma` 仅从 `0,0.05,0.10,0.20` 中选择。只有当内层 OOF 排序改善且 MAE/排序没有明显退化时才接受非零值，否则自动回到 `gamma=0`。

### 4.4 Overall 软排序模型

overall 主要输入各 subdir 的交叉拟合预测百分位。缺失整个 subdir 时，仅使用当前训练折中位数填充，并增加 missing indicator。

各 subdir 质量权重约束非负。目标同时包含 overall 分数回归和设备对软排序：

```text
P(A > B) = sigmoid(0.693 * (score_A - score_B))
```

因此 2 分差约对应 80% 偏好概率，3 分差约对应 88.9%。该映射只用于 overall，不用于 subdir 人工分数。

### 4.5 防泄漏边界

- 外层 held-out 设备不会参与该折任何 robust 基准、subdir 模型或 `gamma` 选择。
- overall 训练设备的 subdir 输入由设备级交叉拟合生成，不使用该设备自身的 subdir 标签拟合其特征。
- 缺失值填充、特征缩放和 overall 超参数选择均在对应训练折内完成。
- 不随机拆分窗口或 `x`；验证分组始终是完整设备 `name`。
- 训练会话缓存只复用训练设备集合与候选配置完全相同的折，不跨折共享统计量。

## 5. 训练输出

`--output-dir` 主要文件：

| 文件 | 内容 |
| --- | --- |
| `normalized_x.csv` | 每个原始 `x` 的变换、robust quality 和 x 内 SE |
| `aggregated_name_subdir.csv` | 等权聚合后的设备-subdir 特征、coverage 与可靠性 |
| `normalization_stats.csv` | 最终部署模型的 median/MAD 及 fallback 记录 |
| `subdir_oof_predictions.csv` | subdir 嵌套 LOODO 预测 |
| `subdir_ablation.csv` | primary、primary+stability、再加 KL/JS 的消融 |
| `subdir_candidate_metrics.csv` | 每个 alpha/gamma 候选的 OOF 指标 |
| `subdir_fold_selection.csv` | 每个外层设备折选择的模型与 gamma |
| `overall_stage2_oof.csv` | overall 使用的交叉拟合 subdir 百分位 |
| `overall_oof_predictions.csv` | overall 嵌套 LOODO 预测 |
| `overall_candidate_metrics.csv` | overall 候选配置的外层 OOF 指标 |
| `metrics.csv` / `metrics.json` | MAE、RMSE、Pearson、Spearman、Kendall 和设备对指标 |
| `model_weights.csv` | subdir 与 overall 权重；辅助权重同时给出乘 gamma 后的有效值 |
| `training_metadata.json` | 最终配置、偏好映射和防泄漏说明 |
| `device_quality_model.joblib` | 预测所需完整模型 |

设备对指标包括：

- `pairwise_ranking_accuracy`：subdir 按非并列百分位比较方向；overall 只统计人工差异至少 2 分的设备对。
- `pairwise_brier_score`：预测偏好概率与软偏好目标的均方差，越低越好。
- `pairwise_log_loss`：软标签交叉熵，越低越好。

subdir 没有 2 分偏好概率定义，因此 Brier/log loss 只对 overall 报告。

## 6. 新设备预测

新设备 CSV 与训练 CSV 格式相同，仍是一行一个 `name+subdir+x`：

```bash
python device_quality_main.py predict \
  --metrics-csv /path/to/new_device_x_metrics.csv \
  --model /path/to/device_quality_model/device_quality_model.joblib \
  --output-dir /path/to/new_device_result \
  --bootstrap 500 \
  --seed 2026
```

一个 CSV 可以包含多台新设备。默认 bootstrap 在每个 `name+subdir` 内按 `x` 重采样，并根据窗口方差增加 x 内参数化扰动。

主要输出：

| 文件 | 内容 |
| --- | --- |
| `new_subdir_predictions.csv` | 各 subdir 百分位、coverage、可靠性和 95% CI |
| `new_stage2_features.csv` | overall 实际输入及 missing 信息 |
| `new_device_ranking.csv` | overall 分数、已知设备中排名、分数/排名区间和邻近设备 |
| `new_pairwise_probabilities.csv` | `P(新设备 > 每台已知设备)` |
| `new_subdir_confidence_intervals.csv` | subdir bootstrap CI 汇总 |
| `prediction_summary.json` | 机器可读预测摘要和输入验证信息 |

加 `--save-bootstrap-replicates` 可保存所有重复样本结果。

### Bootstrap 限制

- `n_x>1`：按 x 非参数重采样，并加入 x 内参数化扰动。
- `n_x=1`：只能根据 `mean/var/n_windows` 参数化采样，输出会标记 `within_parametric_only_limited`。
- 当前区间条件于已拟合模型，反映音频内容和窗口统计不确定性，不包含“重新招募一批设备和人工评分者”带来的模型训练不确定性。

## 7. 现有音频处理脚本

这些脚本保留用于从长录音生成模型指标。它们不属于 `device_quality` 的 x 级再次聚合。

### 开头粉噪和静音切段

粉噪位于开头且与实际内容由静音分隔时：

```bash
python scripts/audio/split_recording_by_silence.py \
  --input /path/to/long_recording.wav \
  --out_dir /path/to/device_segments \
  --sample_rate 24000 \
  --silence_db -40 \
  --min_silence_seconds 0.5 \
  --drop_first_segment
```

先人工抽查切分 CSV 和波形，确认第一段确实是粉噪后再使用 `--drop_first_segment`。

### 重采样、响度归一化和时间对齐

```bash
python scripts/audio/preprocess_audio_pairs.py \
  --ref_dir /path/to/original_wavs \
  --test_dir /path/to/device_segments \
  --out_dir /path/to/aligned_wavs \
  --sample_rate 24000 \
  --normalize lufs \
  --target_level -23 \
  --crop_seconds 5
```

### InspireMusic embedding FD/cosine/L2

从 InspireMusic 项目可导入的位置运行：

```bash
python scripts/metrics/compute_inspiremusic_embedding_metrics.py \
  --ref_dir /path/to/aligned_wavs/ref \
  --test_dir /path/to/aligned_wavs/test \
  --model_dir ../../pretrained_models/InspireMusic-1.5B-Long \
  --gpu 0 \
  --max_audio_seconds 5 \
  --out_csv /path/to/results/inspiremusic_embedding_metrics.csv
```

### Next-token KL/JS

```bash
python scripts/metrics/calculate_token_distribution_metrics.py \
  --p_dir /path/to/original_probs \
  --q_dir /path/to/device_probs \
  --out_dir /path/to/results/token_distribution_metrics \
  --summary_csv token_distribution_summary.csv
```

## 8. 模块结构

```text
.
├── device_quality/               核心训练、验证、预测和 bootstrap 包
├── scripts/
│   ├── audio/                    切段、重采样、响度归一化和时间对齐
│   ├── metrics/                  InspireMusic embedding 与 KL/JS 指标
│   └── legacy/                   旧版感知分数基线，仅供对照
├── tests/
│   └── synthetic_e2e.py          20 设备、6 subdir 端到端测试
├── examples/                     CSV、JSON 和中文列映射示例
├── docs/research/                研究方案文档
├── device_quality_main.py        train/predict 根目录入口
├── requirements-audio-eval.txt  Python 依赖
├── .gitignore                    大模型、原始音频和运行产物排除规则
└── README.md
```

## 9. 合成端到端测试

测试会在临时目录生成约 20 台设备、6 个 subdir，覆盖 `n_x=1`、x 缺失、整个 subdir 缺失、JSON 标签缺失、极端 subdir 分数和反常 KL/JS，然后真正调用 train/predict CLI。临时数据会自动删除：

```bash
python tests/synthetic_e2e.py
```

测试为了控制耗时使用缩小后的候选集合；正式数据训练仍建议使用第 3 节默认命令。
