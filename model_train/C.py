device_subdir_df = normalize_keys(
    device_subdir_df,
    [columns.device, columns.subdir],
)

subdir_labels = normalize_keys(
    subdir_labels,
    [columns.device, columns.subdir],
)

overall_labels = normalize_keys(
    overall_labels,
    [columns.device],
)

match_columns = [
    columns.device,
    columns.subdir,
]

# ---------------------------------------------------------
# 1. 检查人类评分表是否包含 human_score
# ---------------------------------------------------------
if "human_score" not in subdir_labels.columns:
    raise ValueError(
        "Human-score JSON did not produce a human_score column. "
        "Please check --json-layout and the JSON structure."
    )

# 将评分统一转为数值。
# None、空字符串、非数值字符串都会变成 NaN。
subdir_labels["human_score"] = pd.to_numeric(
    subdir_labels["human_score"],
    errors="coerce",
)

# ---------------------------------------------------------
# 2. 删除 JSON 中存在记录、但评分值无效的条目
# ---------------------------------------------------------
invalid_score_labels = subdir_labels[
    subdir_labels["human_score"].isna()
].copy()

if not invalid_score_labels.empty:
    invalid_score_labels.to_csv(
        out_dir / "invalid_human_scores.csv",
        index=False,
    )

    print(
        "[train] Skipping "
        f"{len(invalid_score_labels)} human-score entries "
        "because their scores are empty or non-numeric."
    )

subdir_labels = subdir_labels.dropna(
    subset=["human_score"]
).copy()

# ---------------------------------------------------------
# 3. 检查是否存在重复的 name + subdir 评分
# ---------------------------------------------------------
duplicate_mask = subdir_labels.duplicated(
    subset=match_columns,
    keep=False,
)

if duplicate_mask.any():
    duplicate_labels = (
        subdir_labels.loc[duplicate_mask]
        .sort_values(match_columns)
        .copy()
    )

    duplicate_labels.to_csv(
        out_dir / "duplicate_human_scores.csv",
        index=False,
    )

    raise ValueError(
        "Duplicate human scores were found for the same "
        f"{columns.device} + {columns.subdir}. "
        "See duplicate_human_scores.csv."
    )

# ---------------------------------------------------------
# 4. 使用 left merge 显式检查每条特征是否有评分
# ---------------------------------------------------------
matched = device_subdir_df.merge(
    subdir_labels,
    on=match_columns,
    how="left",
    indicator="_human_score_match",
    validate="many_to_one",
)

# ---------------------------------------------------------
# 5. 没有评分的 name + subdir：保存后跳过
# ---------------------------------------------------------
unlabeled = matched[
    matched["human_score"].isna()
].copy()

if not unlabeled.empty:
    unlabeled["_skip_reason"] = (
        "missing_human_score_for_name_subdir"
    )

    unlabeled.to_csv(
        out_dir / "skipped_unlabeled_device_subdir_features.csv",
        index=False,
    )

# ---------------------------------------------------------
# 6. 只保留有有效人类评分的数据用于训练
# ---------------------------------------------------------
labeled = matched[
    matched["human_score"].notna()
].copy()

labeled = labeled.drop(
    columns=["_human_score_match"],
    errors="ignore",
)

# human_score 再次确保为 float
labeled["human_score"] = labeled[
    "human_score"
].astype(float)

# ---------------------------------------------------------
# 7. 输出匹配统计
# ---------------------------------------------------------
total_rows = len(device_subdir_df)
matched_rows = len(labeled)
skipped_rows = len(unlabeled)

print(
    "[train] Human-score matching summary:"
)

print(
    f"  total device-subdir rows : {total_rows}"
)

print(
    f"  matched training rows    : {matched_rows}"
)

print(
    f"  skipped unlabeled rows   : {skipped_rows}"
)

if total_rows > 0:
    print(
        "  matched ratio           : "
        f"{matched_rows / total_rows:.2%}"
    )

if skipped_rows > 0:
    print(
        "[train] Unlabeled rows were skipped. "
        "Details were saved to "
        "skipped_unlabeled_device_subdir_features.csv."
    )

# ---------------------------------------------------------
# 8. 检查过滤后是否仍有数据可以训练
# ---------------------------------------------------------
if labeled.empty:
    raise RuntimeError(
        "No device-subdir rows have valid human scores. "
        "Please check the name/subdir values in the metrics CSV "
        "and human-score JSON."
    )

num_labeled_devices = labeled[
    columns.device
].nunique()

if num_labeled_devices < 2:
    raise RuntimeError(
        "At least two labeled devices are required for "
        "leave-one-device-out training, but only "
        f"{num_labeled_devices} labeled device was found after "
        "skipping rows without human scores."
    )

# 保存真正进入训练的数据
labeled.to_csv(
    out_dir / "labeled_device_subdir_features.csv",
    index=False,
)
