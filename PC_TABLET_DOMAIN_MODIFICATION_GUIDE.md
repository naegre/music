# PC / Tablet 双域支持修改指南

> 基准：PC Quality Demo 0.3.0  
> 文档日期：2026-07-30  
> 目标：在不更换、不重训当前类别模型和总体模型的前提下，为现有 PC-only 程序增加 Tablet 预测。  
> 使用方式：本指南面向另一台已经做过本地修改的电脑，按函数逐项合并；不要用本机整文件直接覆盖另一台电脑的源码。

---

## 0. 验证状态

PC/Tablet 共用模型、按域排名、Tablet 雷达、导出和图表等共同核心，已在 `PC Quality Demo 0.3.0` 的隔离副本中完成参考实现验证。没有覆盖当前正式源码，也没有修改或重新保存两个 `.joblib`。

验证结果：

- 全量自动化测试：`150 passed`；
- 内置模型自检：通过；
- 外部模型加载自检：通过；
- GUI 自检：通过；
- 原 PC 示例总体分数仍为 `76.5247869455812`，排名集合仍为“1 台新 PC + 5 台固定参考”；
- 真实内置类别模型 + 内置总体模型的 Tablet 端到端样例通过，单台 Tablet 的 rank/percentile/win 为 `1/50/50`，四张图和 8 个 CSV 均可生成；
- 单个第二份 CSV 内混合 PC/Tablet 时，会在任何成功、失败或类别输出写入前整体拒绝；
- Tablet 的核心预测结果、排名、图表和雷达候选中没有固定 5 台 PC。

2026-07-30 后续确认的新规则是：不同批次之间不使用手动 domain 锁；新批次若是另一 domain，自动清空旧域结果并切换。上面的 `150 passed` 验证了共同核心和旧的“跨域拒绝”版本，不应冒充新自动切换分支已经完成安装包验证。新分支必须按第 20、21 节补测。

另一台电脑已有本地改动，因此合并后仍必须重新执行测试；不能把本次结果直接当成另一台修改版 EXE 已经验证完成。

---

## 1. 已确认的产品规则

### 1.1 三份输入

第一份“其他 15 项评分 CSV”是长期维护的全设备特征库：

```csv
name,domain,loudness,dynamics_1_S,...,artefacts_3_S
PC_A,pc,80.0,79.0,...,84.0
TABLET_A,tablet,81.0,80.0,...,85.0
```

它可以同时包含 PC 和 Tablet。`domain` 是权威域信息，只允许：

```text
pc
tablet
```

第二份音频指标 CSV 不需要增加 `domain`。程序读取该文件中实际出现的 `name`，去第一份 CSV 查询 domain，再判断本批是否混域。

第三份展示与价值 CSV 保持：

```csv
name,device_name,values
```

第三份表不负责定义 domain。

### 1.2 每次第二份 CSV 独立判域并自动切换

程序不设置需要用户手动解除的 domain 锁。每次导入或更新第二份音频指标 CSV，都重新执行以下流程：

1. 读取第二份 CSV 中本批出现的全部非空 `name`；
2. 与第一份 CSV 按 `name` 取交集；
3. 只有交集中的设备才进入类别模型和总体模型计算；
4. 从第一份 CSV 查询这些交集设备的 `domain`；
5. 交集设备必须全部为 `pc` 或全部为 `tablet`。

随后按本批判定结果处理：

- 本批与当前已保存结果同域：继续累积新设备并重新排名；
- 本批与当前已保存结果异域：自动清空旧域的成功、失败和类别预测记录，然后切换到本批 domain；
- 本批自身同时含 PC 和 Tablet：整批拒绝，且保留当前已有结果，不先清空；
- 本批没有任何 name 能在第一份 CSV 找到：不切换 domain、不清空已有结果，只按现有失败规则登记缺失 name；
- 用户主动清空预测记录：当前结果 domain 回到未确定状态；
- 成功替换第一份评分 CSV：清空全部旧预测记录，等待下一份第二 CSV 重新判域。

| 当前结果 domain | 本批判定 | 处理 |
|---|---|---|
| 未确定 | PC | 保存 PC 结果并设为 PC |
| 未确定 | Tablet | 保存 Tablet 结果并设为 Tablet |
| PC | PC | 累积新 PC 并重新排名 |
| Tablet | Tablet | 累积新 Tablet 并重新排名 |
| PC | Tablet | 自动清空 PC，改为 Tablet |
| Tablet | PC | 自动清空 Tablet，改为 PC |
| 任意 | 批内混域 | 拒绝本批，旧结果不变 |
| 任意 | 无 name 交集 | 不切换，只登记缺名失败 |

因此仍然不会把 PC 和 Tablet 放进同一个排名集合，但这是“每批重新判断 + 必要时自动替换旧域结果”，不是“第一批锁定、跨域必须手动清空”。

### 1.3 PC 模式

PC 模式保持原逻辑：

- Score 使用 PC offset；
- Rank 使用新 PC + 固定 5 台参考 PC；
- `expected_pairwise_win_percent` 包含固定 5 台参考 PC；
- 固定 PC 继续进入图表、雷达图和导出；
- 现有文件名和主要字段尽量保持不变。

### 1.4 Tablet 模式

Tablet 模式：

- 使用当前相同类别模型；
- 使用当前相同总体 Score/Rank bundle；
- Score 不使用 PC offset；
- Rank 只在当前会话 Tablet 内计算；
- `expected_pairwise_win_percent` 只比较当前 Tablet；
- 固定 5 台 PC 不进入计算、GUI、四张图或 Tablet 结果导出；
- 不检查 Tablet 是否曾出现在模型训练数据中；
- 因为未检查训练重合，Tablet 结果只能作为 demo 展示，不能声称为严格 OOF 或严格 inductive 泛化结果。

### 1.5 Tablet 雷达归一化

Tablet 模式每一维使用当前会话全部成功 Tablet 作为锚点：

\[
\text{当前最低}=60,\qquad
\text{当前最高}=100
\]

映射公式：

\[
r
=
60+40\frac{x-x_{\min}}{x_{\max}-x_{\min}}
\]

最后裁剪到：

\[
[0,120]
\]

如果只有一台 Tablet，或某一维所有 Tablet 完全相同，则该维统一设为：

\[
80
\]

显示名称 CSV 的隐藏行不应改变数学口径。因此建议 Tablet 雷达锚点使用当前会话全部成功 Tablet，而绘图仍只显示有有效 `device_name` 的设备。

### 1.6 单台 Tablet 排名

只有一台 Tablet 时：

```text
estimated_rank_1_is_best = 1
rank_percentile = 50
expected_pairwise_win_percent = 50
```

50 表示没有同域比较对象，不表示完成了一次真实胜负比较。

---

## 2. 为什么不需要重训 joblib

这次修改没有改变 19 个模型特征的名称、数量或顺序。

当前 Score 模型本来就包含：

\[
\hat y
=
b+dI_{\mathrm{PC}}+\mathbf z^\top\boldsymbol\beta
\]

只要推理时把 Tablet 的 domain 正确设为 `tablet`：

\[
I_{\mathrm{PC}}=0
\]

就会自动去掉 PC offset。

Rank 模型本来就是一个共享 utility 函数。PC 和 Tablet 的差别在于“和哪些设备一起比较”，不需要改 Rank 系数：

- PC：新 PC + 5 台固定 PC；
- Tablet：当前会话 Tablet。

因此以下文件不应修改或重新序列化：

```text
category_model_bundle_inductive.joblib
overall_compact_positive_bundle.joblib
```

内置模型 SHA-256 也不应改变。

---

## 3. 推荐新增一个小型域工具文件

新增：

```text
src/pc_quality_demo/domains.py
```

建议内容：

```python
"""Shared PC/Tablet domain validation."""

from __future__ import annotations


PC_DOMAIN = "pc"
TABLET_DOMAIN = "tablet"
VALID_DOMAINS = frozenset({PC_DOMAIN, TABLET_DOMAIN})


def normalize_domain(value: object) -> str:
    domain = str(value).strip().lower()
    if domain not in VALID_DOMAINS:
        raise ValueError(
            "domain 只允许 pc 或 tablet，"
            f"实际收到：{value!r}"
        )
    return domain


def domain_display_name(domain: str | None) -> str:
    if domain == TABLET_DOMAIN:
        return "Tablet"
    if domain == PC_DOMAIN:
        return "PC"
    return "设备"
```

这样 `input_data.py`、`overall_runtime.py`、`session.py` 和图表代码共用同一规则，避免出现一个文件接受 `Tablet`、另一个文件却拒绝的情况。

不要从 `input_data.py` 导入域函数到 `overall_runtime.py`，因为 `input_data.py` 已经从 `overall_runtime.py` 导入特征常量，会产生循环依赖。

---

## 4. 修改第一份 CSV 的读取

文件：

```text
src/pc_quality_demo/input_data.py
```

先导入：

```python
from .domains import normalize_domain
```

在 `read_auxiliary_scores()` 中，将：

```python
required = {"name", *AUXILIARY_FEATURES}
```

改成：

```python
required = {"name", "domain", *AUXILIARY_FEATURES}
```

将：

```python
result = frame[["name", *AUXILIARY_FEATURES]].copy()
```

改成：

```python
result = frame[["name", "domain", *AUXILIARY_FEATURES]].copy()
```

完成 name 校验后，增加：

```python
if result["domain"].isna().any():
    invalid_names = result.loc[
        result["domain"].isna(), "name"
    ].astype(str).tolist()
    raise ValueError(
        "其他评分 CSV 的 domain 不能缺失："
        f"{invalid_names[:10]}"
    )

normalized_domains: list[str] = []
domain_errors: list[str] = []
for _, row in result.iterrows():
    try:
        normalized_domains.append(
            normalize_domain(row["domain"])
        )
    except ValueError:
        normalized_domains.append("")
        domain_errors.append(str(row["name"]))

if domain_errors:
    raise ValueError(
        "其他评分 CSV 的 domain 只允许 pc 或 tablet："
        f"{domain_errors[:10]}"
    )

result["domain"] = normalized_domains
```

继续保留原有 15 个特征的数值校验。

返回值仍然按 name 建索引：

```python
return result.set_index("name", drop=False)
```

### 4.1 这个函数应该允许混域

不要在 `read_auxiliary_scores()` 中写：

```python
if result["domain"].nunique() != 1:
    raise ...
```

第一份表本来就允许保存大量 PC 和 Tablet。混域判断只能针对第二份音频 CSV 本批实际出现的 name。

---

## 5. 为点预测保存 domain

文件：

```text
src/pc_quality_demo/overall_runtime.py
```

导入：

```python
from .domains import PC_DOMAIN, normalize_domain
```

给 `OverallPointPrediction` 增加字段：

```python
@dataclass(frozen=True)
class OverallPointPrediction:
    name: str
    domain: str
    predicted_overall: float
    rank_utility: float
    features: dict[str, float]
    ...
```

如果另一台电脑有很多旧调用，可以暂时把 `domain` 放在所有无默认字段的最后；不要把无默认字段放到已有默认字段之后，否则 dataclass 会报错。

所有固定参考点创建时明确写：

```python
domain=PC_DOMAIN,
```

---

## 6. 修改总体单设备预测

文件：

```text
src/pc_quality_demo/overall_runtime.py
```

把 `predict_device()` 改成兼容旧调用的形式：

```python
def predict_device(
    self,
    name: str,
    values: Mapping[str, object],
    *,
    domain: str = PC_DOMAIN,
    category_standard_deviations: Mapping[str, object] | None = None,
    n_samples: int = DEFAULT_MONTE_CARLO_DRAWS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> OverallPointPrediction:
```

函数开头增加：

```python
clean_domain = normalize_domain(domain)
```

旧代码：

```python
if clean_name in self.reference_names:
    raise ValueError("设备 name 与固定参考 PC 重复")
```

建议改成：

```python
if (
    clean_domain == PC_DOMAIN
    and clean_name in self.reference_names
):
    raise ValueError("PC 设备 name 与固定参考 PC 重复")
```

构造总体模型输入时，将硬编码：

```python
{"name": clean_name, "domain": "pc", **numeric_values}
```

改成：

```python
{
    "name": clean_name,
    "domain": clean_domain,
    **numeric_values,
}
```

返回结果时增加：

```python
return OverallPointPrediction(
    name=clean_name,
    domain=clean_domain,
    predicted_overall=score,
    rank_utility=utility,
    features=numeric_values,
    **uncertainty_values,
)
```

这里不需要自行减去 PC offset。现有 `_predict_score()` 已经通过：

```python
is_pc = frame["domain"].astype(str).eq("pc")
```

自动完成：

- PC：使用 offset；
- Tablet：不使用 offset。

蒙特卡洛总体分数传播也已经读取 frame 的 domain，因此不需要另外写 Tablet 公式。

---

## 7. 修改总体排名汇总

文件：

```text
src/pc_quality_demo/overall_runtime.py
```

当前 `summarize_session()` 总是拼接 5 台参考 PC。建议增加显式 domain 参数：

```python
def summarize_session(
    self,
    predictions: Iterable[OverallPointPrediction],
    *,
    domain: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
```

建议核心逻辑：

```python
prediction_list = list(predictions)
names = [prediction.name for prediction in prediction_list]
if len(names) != len(set(names)):
    raise ValueError("当前会话包含重复设备 name")

point_domains = {
    normalize_domain(prediction.domain)
    for prediction in prediction_list
}
if len(point_domains) > 1:
    raise ValueError("当前会话不能混合 PC 和 Tablet")

if domain is None:
    clean_domain = (
        next(iter(point_domains))
        if point_domains
        else PC_DOMAIN
    )
else:
    clean_domain = normalize_domain(domain)

if point_domains and point_domains != {clean_domain}:
    raise ValueError("会话 domain 与预测记录 domain 不一致")

session_utilities = np.asarray(
    [
        prediction.rank_utility
        for prediction in prediction_list
    ],
    dtype=float,
)

if clean_domain == PC_DOMAIN:
    all_names = [*self.reference_names, *names]
    all_utilities = np.concatenate(
        [self.reference_utilities, session_utilities]
    )
else:
    all_names = names
    all_utilities = session_utilities

combined = rank_summary(all_names, all_utilities)
```

构造 `score_table` 时增加：

```python
"domain": [
    prediction.domain
    for prediction in prediction_list
],
```

后面的 merge 和排序可以继续使用。

原函数若对空 `prediction_list` 有提前返回分支，还要在该空表中显式增加：

```python
empty["domain"] = pd.Series(dtype=str)
```

空 Tablet 会话也必须保留明确列结构，避免后续 merge、GUI 空态或 CSV 导出依赖 pandas 猜测空表字段。

`rank_summary()` 本身不需要修改。它已经规定：

- 单设备：rank=1，percentile=50，win=50；
- 多设备：按 utility 差计算胜率。

Tablet 分支只向它传当前 Tablet，自然不会把固定 PC 计入 `win_percent`。

---

## 8. 会话保存 current_domain，但不把它当作锁

文件：

```text
src/pc_quality_demo/session.py
```

导入：

```python
from .domains import (
    PC_DOMAIN,
    TABLET_DOMAIN,
    normalize_domain,
)
```

在 `PredictionSession.__init__()` 中增加：

```python
self.current_domain: str | None = None
```

`current_domain` 只表示“当前内存中已保存结果属于哪个域”，供排名、固定参考选择、图表和 GUI 使用；它不是拒绝新 domain 的锁。

如果已经按旧版指南写成 `active_domain`，可以暂时保留变量名，但必须删除“不同域就拒绝”的逻辑。为了避免继续误解，推荐统一改名为 `current_domain`。

建议增加两个只读属性：

```python
@property
def effective_domain(self) -> str:
    """Use PC as the pre-prediction legacy display mode."""

    return self.current_domain or PC_DOMAIN


@property
def includes_fixed_references(self) -> bool:
    return self.effective_domain == PC_DOMAIN
```

### 8.1 清空预测记录时重置当前域

在 `clear_predictions()` 最后增加：

```python
self.current_domain = None
```

这样：

- 用户主动清空后回到“尚未判域”；
- 自动跨域切换可以复用同一个清空函数；
- 切换模型后不会保留旧模型产生的域状态。

### 8.2 成功替换第一份 CSV 时清空全部预测记录

替换第一份 CSV 的正确顺序应是：

```python
candidate = read_auxiliary_scores(path)

# 新文件已经完整读取并通过校验后，才清除旧预测状态。
self.clear_predictions()
self.auxiliary = candidate
self.auxiliary_source = str(Path(path).resolve())
```

不要先清空再读取新文件，否则新 CSV 格式错误时也会意外丢失旧结果。

这里的“清空全部记录”指：

- `successes`；
- `category_outputs`；
- `failures`；
- `current_domain`。

第三份 `name/device_name/values` 展示表可以继续保留，因为它不参与 domain 判定；如果你的产品希望连展示表也一起换，再单独调用 `clear_all()`。

旧版 `clear_existing=False` 不能再允许“换了第一份表却保留旧预测”。GUI 可以在执行前询问用户是否确认，但用户确认且新文件校验成功后，session 层必须无条件清空旧预测状态。

---

## 9. 由第二份 CSV 的 name 判断批次 domain

文件：

```text
src/pc_quality_demo/session.py
```

在 `PredictionSession` 中增加辅助方法：

```python
def _resolve_metrics_batch_domain(
    self,
    names: list[str],
) -> str | None:
    if self.auxiliary is None:
        raise ValueError("请先导入包含其他 15 项评分的 CSV")

    known_names = [
        name for name in names
        if name in self.auxiliary.index
    ]
    if not known_names:
        return None

    domains = {
        normalize_domain(self.auxiliary.loc[name, "domain"])
        for name in known_names
    }
    if len(domains) != 1:
        details = {
            name: str(self.auxiliary.loc[name, "domain"])
            for name in known_names
        }
        raise ValueError(
            "本次音频指标 CSV 同时包含 PC 和 Tablet；"
            f"请拆分后分别导入：{details}"
        )

    return next(iter(domains))
```

这个函数只负责判断“本次第二份 CSV 的有效 name 是否为单一 domain”，不再把本批与历史 domain 比较，也不再要求手动清空。

### 9.1 推荐使用“先计算临时结果，最后一次提交”

不要在刚识别出异域时立刻删除旧结果。更稳妥的方式是：先把本批成功、类别输出和失败记录写入局部临时容器；整批处理结束后，再一次性切换会话状态。

核心结构：

```python
batch_domain = self._resolve_metrics_batch_domain(names)
switching_domain = (
    batch_domain is not None
    and self.current_domain is not None
    and batch_domain != self.current_domain
)

pending_successes = OrderedDict()
pending_categories: dict[str, list[CategoryPrediction]] = {}
pending_failures = OrderedDict()

for name in names:
    # 跨域时旧结果稍后会整体清空，因此不能把旧域同名设备
    # 误判成“当前会话已存在”。
    if not switching_domain and name in self.successes:
        pending_failures[name] = FailureRecord(
            name,
            "当前结果中已存在相同 name",
        )
        continue

    try:
        if name not in self.auxiliary.index:
            raise ValueError("其他评分 CSV 中找不到该 name")

        # 沿用原有类别预测、19 特征组装和总体预测代码；
        # 但先写入 pending_*，不要在循环中改 self.successes。
        ...
        pending_successes[name] = point
        pending_categories[name] = category_predictions
    except Exception as exc:
        pending_failures[name] = FailureRecord(name, str(exc))

# 提交点：本批判域和逐设备处理均已完成。
if switching_domain:
    self.clear_predictions()
if batch_domain is not None:
    self.current_domain = batch_domain

for name, point in pending_successes.items():
    self.successes[name] = point
    self.category_outputs[name] = pending_categories[name]
    self.failures.pop(name, None)
self.failures.update(pending_failures)
```

计数也应根据 `pending_successes/pending_failures` 生成，再保持原有返回字段：

```python
return {
    "added": len(pending_successes),
    "failed": len(pending_failures),
    "seen": len(names),
}
```

执行顺序不能颠倒：

1. 完整读取并校验第二份 CSV；
2. 提取非空唯一 name 并与第一份 CSV 取交集；
3. 确认本批不是 PC/Tablet 混合；
4. 在临时容器中完成逐设备预测或失败登记；
5. 到达提交点后，异域才清空旧状态并切换；同域则直接合并。

因此：

- 同域批次继续累积；
- 异域批次自动清空并切换；
- 混域批次、CSV schema 错误或提交前的非预期整体异常不会删除旧结果；
- 异域批次本身全部预测失败时，仍应清除旧域、把 `current_domain` 切换到新域，并只保留本批失败记录。

第一份表找不到的 name 仍可沿用当前逻辑，逐台记录：

```text
其他评分 CSV 中找不到该 name
```

如果整个音频文件的 name 都不在第一份表中，则无法解析批次域，全部作为缺失 name 失败，`current_domain` 和已有成功结果保持不变。

### 9.2 调用总体模型时传 domain

将：

```python
point = self.overall_runtime.predict_device(
    name,
    {**auxiliary_features, **category_features},
    category_standard_deviations=category_standard_deviations,
)
```

改为：

```python
if name not in self.auxiliary.index:
    raise ValueError("其他评分 CSV 中找不到该 name")
if batch_domain is None:
    raise RuntimeError("内部错误：有效设备批次缺少 domain")

point = self.overall_runtime.predict_device(
    name,
    {**auxiliary_features, **category_features},
    domain=batch_domain,
    category_standard_deviations=category_standard_deviations,
)
```

第一段 name 检查保证只有同时存在于两份 CSV 的设备才进入模型。只要当前 name 有效，`batch_domain` 按前面的批次解析就一定不是 `None`；第二段只是防止未来改代码时破坏这一不变量。若整个批次的 name 都不存在，每台仍记录原有“找不到该 name”失败。

类别模型本身不需要 domain。它仍然只处理 L2/cosine。

---

## 10. Tablet 训练重合的诚实标记

用户已确认 Demo 暂时不检查 Tablet 是否出现在类别或总体训练数据中。

这表示 Tablet 输出不应该声称为严格 OOF，也不应该无条件声称为严格 inductive。

最小做法是在 README 和 GUI 状态中写明：

```text
Tablet demo：未检查与模型训练设备的重合，仅演示推理功能。
```

更可审计的做法是在导出表增加：

```text
evaluation_scope
```

值为：

```text
pc                -> inductive_demo
tablet            -> demo_training_overlap_unchecked
fixed_reference   -> active_bundle_reference
```

这是状态字段，不进入模型计算，也不是通用警告文本。

如果暂时不想增加输出列，至少不要在文档中把 Tablet 结果描述成严格外测。

---

## 11. Session 中所有排名调用都要传域

文件：

```text
src/pc_quality_demo/session.py
```

所有：

```python
self.overall_runtime.summarize_session(
    self.successes.values()
)
```

改成：

```python
self.overall_runtime.summarize_session(
    self.successes.values(),
    domain=self.effective_domain,
)
```

至少检查：

- `snapshot()`
- `all_device_predictions()`

不要只改 GUI 调用，否则 CSV 导出仍可能偷偷加入 5 台 PC。

---

## 12. Session 输出真实 domain

### 12.1 `snapshot()` 的最终 19 特征

把硬编码：

```python
"domain": "pc",
```

改成：

```python
"domain": prediction.domain,
```

### 12.2 `_point_record()`

增加：

```python
"domain": prediction.domain,
```

### 12.3 `all_feature_table()`

把：

```python
"domain": "pc",
```

改成：

```python
"domain": prediction.domain,
```

固定参考 `OverallPointPrediction` 已经保存 `domain="pc"`，所以不用根据 source 再硬编码。

### 12.4 类别输出

`category_predictions.csv` 建议增加 `domain`：

```python
category_columns = [
    "name",
    "domain",
    "subdir",
    ...
]
```

会话类别行：

```python
"domain": self.effective_domain,
```

固定参考类别行：

```python
"domain": PC_DOMAIN,
```

---

## 13. Tablet 模式完全排除固定参考

文件：

```text
src/pc_quality_demo/session.py
```

### 13.1 `_all_point_predictions()`

建议改为：

```python
def _all_point_predictions(
    self,
) -> list[tuple[OverallPointPrediction, str]]:
    session_points = [
        (prediction, SESSION_DEVICE_SOURCE)
        for prediction in self.successes.values()
    ]
    if not self.includes_fixed_references:
        return session_points
    return [
        *[
            (prediction, FIXED_REFERENCE_SOURCE)
            for prediction
            in self.overall_runtime.reference_predictions()
        ],
        *session_points,
    ]
```

这会同时影响：

- GUI 表；
- 分数图；
- 排名图；
- 价值图及趋势线；
- 雷达候选；
- `device_predictions.csv`；
- `final_19_features.csv`。

### 13.2 `all_category_table()`

只有：

```python
self.includes_fixed_references
```

为真时才创建固定参考类别行。

Tablet 模式：

```python
references = pd.DataFrame(columns=...)
```

然后仅返回 session categories。

### 13.3 空 Tablet 结果

当前 PC 模式即使没有新设备也有 5 台参考，因此部分函数默认表不会为空。

Tablet 批次全部失败时，以下函数必须能返回带正确列名的空表：

- `all_device_predictions()`
- `all_feature_table()`
- `all_category_table()`
- `chart_data()`
- `export_csvs()`

构造空 DataFrame 时显式传 `columns=[...]`，不要依赖 pandas 从空列表猜列，否则后续 merge 会因缺列失败。

---

## 14. Tablet 雷达归一化

文件：

```text
src/pc_quality_demo/session.py
```

当前 `chart_data()` 固定选择：

```python
fixed = chart["source"].eq(FIXED_REFERENCE_SOURCE)
reference = chart.loc[fixed]
```

建议把“锚点表”分支独立出来。

伪代码：

```python
unfiltered_features = self.all_feature_table()

if self.includes_fixed_references:
    anchors = unfiltered_features.loc[
        unfiltered_features["source"].eq(
            FIXED_REFERENCE_SOURCE
        )
    ].copy()
else:
    anchors = unfiltered_features.loc[
        unfiltered_features["source"].eq(
            SESSION_DEVICE_SOURCE
        )
    ].copy()

anchors["clarity_raw"] = anchors[CATEGORY_FEATURES[0]]
anchors["lowfreq_raw"] = anchors[CATEGORY_FEATURES[1]]
anchors["fullness_raw"] = anchors[CATEGORY_FEATURES[2]]
anchors["edginess_raw"] = anchors[CATEGORY_FEATURES[3]]
anchors["other_raw"] = (
    self.overall_runtime.auxiliary_score_contributions(
        anchors
    )
)
```

每一维：

```python
low = float(anchors[raw_column].min())
high = float(anchors[raw_column].max())
span = high - low

chart[f"{raw_column}_reference_min"] = low
chart[f"{raw_column}_reference_max"] = high

if not np.isfinite([low, high]).all():
    chart[radar_column] = np.nan
    scaling_status.append(
        f"{raw_column}:invalid_anchor"
    )
elif span <= 1e-12:
    if self.effective_domain == TABLET_DOMAIN:
        chart[radar_column] = 80.0
        scaling_status.append(
            f"{raw_column}:tablet_zero_span_midpoint"
        )
    else:
        chart[radar_column] = np.nan
        scaling_status.append(
            f"{raw_column}:zero_reference_span"
        )
else:
    chart[radar_column] = np.clip(
        60.0
        + 40.0
        * (chart[raw_column] - low)
        / span,
        0.0,
        120.0,
    )
```

为了减少对旧导出消费者的影响，可以继续保留历史列名：

```text
*_reference_min
*_reference_max
```

另外新增：

```text
radar_anchor_domain
radar_anchor_count
radar_scaling_basis
```

例如：

```python
chart["radar_anchor_domain"] = self.effective_domain
chart["radar_anchor_count"] = len(anchors)
chart["radar_scaling_basis"] = (
    "pc_fixed_reference"
    if self.includes_fixed_references
    else "tablet_session"
)
```

---

## 15. Tablet 雷达默认选择

文件：

```text
src/pc_quality_demo/app.py
```

PC 保持：

- 左侧：固定 5 台 PC；
- 右侧：最多 7 台新 PC。

Tablet 没有固定设备。建议默认：

- 按当前 Rank 顺序排列；
- 左侧放前 7 台；
- 右侧放第 8～14 台；
- 用户仍可在两侧选择任意 Tablet，每侧最多 7 台。

在 `_sanitize_radar_selections()` 中分支：

```python
if self.session.effective_domain == TABLET_DOMAIN:
    session_names = (
        chart_data.loc[
            chart_data["source"].eq(SESSION_DEVICE_SOURCE),
            "name",
        ]
        .astype(str)
        .tolist()
    )
    if not self.radar_left_customized:
        self.radar_left_names = tuple(
            session_names[:MAX_RADAR_DEVICES_PER_SIDE]
        )
    if not self.radar_right_customized:
        start = MAX_RADAR_DEVICES_PER_SIDE
        stop = 2 * MAX_RADAR_DEVICES_PER_SIDE
        self.radar_right_names = tuple(
            session_names[start:stop]
        )
else:
    # 保留当前 PC 默认逻辑
    ...
```

如果希望 7 台以内全部默认显示在右侧，也只需调整这一个函数，不影响模型和导出。

清空会话后，最好同时将：

```python
self.radar_left_customized = False
self.radar_right_customized = False
```

否则上一个域的自定义选择状态可能影响新会话默认布局。

建议把这几行封装成 `_reset_radar_selections()`，并在以下位置调用：

- `clear_session()`；
- 成功替换第一份 CSV 后；
- `add_metrics_file()` 导入前后 `current_domain` 不同时；
- `_on_models_changed()`；
- 恢复内置模型后；
- 其他任何会清空或改变当前结果 domain 的入口。

跨域自动切换后如果不重置两个 `customized` 标志，旧设备 name 虽会被过滤掉，但界面可能仍认为用户已经完成自定义，从而不为新域生成默认雷达选择。

---

## 16. 图表标题动态化

文件：

```text
src/pc_quality_demo/charts.py
```

建议让图表从 `chart_data["domain"]` 推断域，而不是在四个调用处重复传参数。

增加：

```python
from .domains import (
    PC_DOMAIN,
    TABLET_DOMAIN,
    domain_display_name,
    normalize_domain,
)
```

辅助函数：

```python
def _chart_domain(frame: pd.DataFrame) -> str | None:
    if frame.empty or "domain" not in frame.columns:
        return None
    domains = {
        normalize_domain(value)
        for value in frame["domain"]
    }
    if len(domains) != 1:
        raise ValueError("chart_data 不能混合 PC 和 Tablet")
    return next(iter(domains))
```

### 16.1 Score 图

把固定标题：

```python
"PC 总体分数预测（Score 模型）"
```

改成：

```python
domain = _chart_domain(prepared)
label = domain_display_name(domain)
axis.set_title(
    f"{label} 总体分数预测（Score 模型）",
    fontsize=13,
)
```

### 16.2 Rank 图

PC：

```text
当前 PC 排名（固定参考设备 + 新设备，共 N 台）
```

Tablet：

```text
当前 Tablet 排名（仅当前会话，共 N 台）
```

空图提示也要按域或使用中性文本：

```text
导入有效设备后，这里将显示当前同域设备排名
```

### 16.3 价值—分数图

标题可改为：

```python
f"{label} 设备价值与总体分数"
```

Tablet 模式传入的数据已经排除固定 PC，因此线性回归自动只使用当前 Tablet，无需修改回归公式。

### 16.4 图例

Tablet 图只有 `session_device`，不会出现“固定参考设备”图例。

可以把 `SESSION_DEVICE_SOURCE` 的显示文字从“新设备”改成更中性的“当前设备”；如果要保持 PC 旧界面完全不变，则不要改全局样式，只在 Tablet 模式动态显示“当前 Tablet”。

### 16.5 雷达不可用提示

`_plot_radar_side()` 目前有固定文字：

```text
固定参考某维无差异，无法归一化
```

Tablet 模式不一定有固定参考，应改成中性文字：

```text
以下设备存在不可用雷达维度，无法归一化
```

正常的 Tablet 单设备或零跨度维度已经按规则映射到 80，不应触发该提示；只有非有限锚点等真正异常才保持不可用。

---

## 17. GUI 状态信息

文件：

```text
src/pc_quality_demo/app.py
```

当前 `_build_layout()` 中还有两处 PC-only 固定文字：

```text
PC 音频主观质量预测 Demo
仅预测 PC · 本地会话 · 失败设备不进入排名
```

建议改成不会随模式过期的中性标题：

```text
音频设备主观质量预测 Demo
支持 PC / Tablet · 当前结果集单域 · 失败设备不进入排名
```

`__init__()` 中的雷达默认说明也不要永久写成：

```text
左侧：固定参考设备
右侧：新设备
```

可以初始化为：

```text
左侧：等待设备
右侧：等待设备
```

再由 `_sanitize_radar_selections()` 根据当前 PC/Tablet 模式和实际选择数量更新。

第一份 CSV 的按钮或提示最好明确为：

```text
导入其他 15 项评分与 domain CSV
```

但不要在 GUI 重新读取第二份 CSV 或重复判断 domain。批次解析、混域拒绝、自动清空和域切换应只由 `PredictionSession.add_metrics_file()` 完成，避免 GUI 与命令行/测试走出两套规则。

导入成功后的状态建议显示：

```python
previous_domain = self.session.current_domain
summary = self.session.add_metrics_file(path)
new_domain = self.session.current_domain
switched = (
    previous_domain is not None
    and new_domain is not None
    and previous_domain != new_domain
)
if switched:
    self._reset_radar_selections()
self.refresh()

domain_label = domain_display_name(
    new_domain
)
message = f"当前模式：{domain_label}；"
if switched:
    message += "已自动清空上一域结果；"
message += (
    f"已读取 {summary['seen']} 台："
    f"成功 {summary['added']} 台，"
    f"失败 {summary['failed']} 台。"
)
self.status_text.set(message)
```

为保持旧调用兼容，建议不要改变 `add_metrics_file()` 的返回结构，继续返回 `added/failed/seen`；GUI 比较调用前后的 `self.session.current_domain` 就能知道是否发生了自动切换。

清空状态：

```text
预测记录已清空；下一份音频指标 CSV 将重新决定 PC 或 Tablet。
```

Tablet 模式可以在详情栏固定显示：

```text
Tablet demo 未检查与模型训练设备的重合。
```

---

## 18. 导出规则

`export_csvs()` 的文件名建议保持不变，以免破坏另一台电脑已有的调用脚本。

Tablet 模式下：

| 文件 | 内容 |
|---|---|
| `device_predictions.csv` | 当前成功 Tablet |
| `combined_reference_rankings.csv` | 当前 Tablet 排名；文件名仅为兼容保留 |
| `category_predictions.csv` | 当前 Tablet 四类结果 |
| `final_19_features.csv` | 当前 Tablet 19 维输入 |
| `failures.csv` | 当前失败记录 |
| `model_manifest.csv` | 当前模型身份 |
| `device_plot_metadata.csv` | 第三份展示表 |
| `chart_data.csv` | 当前可绘图 Tablet |

这里的 `device_plot_metadata.csv` 是第三份用户输入的原样审计副本，不是预测结果表；如果第三份表本身同时保存 PC 和 Tablet，它可以包含固定 PC。要求排除固定 PC 的是其余核心预测、排名、特征、类别和绘图结果。若希望连输入副本也按域过滤，必须另行通过第一份 CSV 的 `domain` 做 join，不能只凭第三份表判断。

在 `combined_reference_rankings.csv` 的列选择中增加：

```python
"domain",
```

建议再增加：

```python
"source",
```

Tablet 全部为 `session_device`。

不要因为文件名仍叫 `combined_reference_rankings.csv` 就重新加入固定 PC。文件内容必须以 active domain 为准。

---

## 19. 示例和模板

至少修改：

```text
examples/auxiliary_scores_template.csv
examples/auxiliary_scores_example.csv
```

在 `name` 后增加 `domain`。

例如：

```csv
name,domain,loudness,dynamics_1_S,dynamics_2,treble_1,treble_2,balance,spatial_1,spatial_2,spatial_3_S,spatial_4_S,spatial_5_S,spatial_6_S,artefacts_1_S,artefacts_2,artefacts_3_S
SAMPLE_PC,pc,80,79,78,88,82,90,85,77,83,82,82,82,85,84,85
SAMPLE_TABLET,tablet,81,80,79,89,83,91,86,78,84,83,83,83,86,85,86
```

第二份和第三份模板不需要 domain。

---

## 20. 必须新增或修改的测试

### 20.1 输入测试

1. 第一份 CSV 缺 `domain` 时拒绝。
2. 空 domain 拒绝。
3. `phone` 等未知值拒绝。
4. ` PC `、`Tablet` 正规化为 `pc/tablet`。
5. 第一份 CSV 同时有 PC 和 Tablet 时允许。
6. 读取结果保留 domain。

### 20.2 批次和会话测试

1. 第二份音频 CSV 的所有 name 查到 PC 时，批次为 PC。
2. 全部查到 Tablet 时，批次为 Tablet。
3. 同一音频文件解析出两种域时，在任何预测写入前整体拒绝。
4. PC 会话继续导入 PC 成功。
5. Tablet 会话继续导入 Tablet 成功。
6. 已有 PC 结果时导入纯 Tablet 批次：旧 PC 成功、失败和类别记录全部清空，只保留新 Tablet 结果。
7. 已有 Tablet 结果时导入纯 PC 批次：旧 Tablet 记录全部清空，新 PC 排名重新包含固定 5 台参考。
8. 自动跨域切换不需要用户先调用 `clear_predictions()`。
9. 混域批次被拒绝时，旧结果和 `current_domain` 保持不变，不能先清空。
10. 第二份 CSV 无法读取或 schema 错误时，旧结果保持不变。
11. 第二份 CSV 的 name 全都不在第一份表时，不切换、不清空，并记录现有缺失 name 失败原因。
12. 部分 name 不在第一份表时，只有交集设备进入模型；缺失设备只记失败。
13. `clear_predictions()` 后 `current_domain is None`。
14. 成功替换第一份 CSV 后全部预测记录清空；若新第一份 CSV 校验失败，旧第一表和旧预测记录保持不变。
15. 异域新批次全部模型预测失败时，旧域仍被清除、`current_domain` 切换，并且只留下新域失败记录。
16. 连续执行 PC → Tablet → PC，两次都自动清空上一域，最后 PC 固定 5 台参考正确恢复。
17. 在临时结果提交前模拟整体异常，旧结果不发生部分修改。

### 20.3 Score 测试

使用相同 19 特征分别预测 PC 和 Tablet，在都未触发 0/100 clipping 的情况下：

\[
\hat y_{\mathrm{PC}}
-
\hat y_{\mathrm{Tablet}}
=
\text{pc\_offset}
\]

同时验证输出对象保存正确 domain。

### 20.4 Rank 和胜率测试

1. 一台 Tablet：
   - rank=1；
   - percentile=50；
   - win=50；
   - combined 长度为 1。
2. 两台 Tablet：
   - combined 长度为 2；
   - 不出现 5 个参考 name；
   - 两个方向概率互补；
   - 两台 expected win percent 总和约为 100。
3. PC 回归测试：
   - combined 长度仍为“新 PC 数 + 5”；
   - 原有参考名不变；
   - 原有数值指纹不变。

### 20.5 导出测试

Tablet 模式所有核心结果表：

```python
assert frame["domain"].eq("tablet").all()
```

并检查不存在：

```text
CG-
MG-
pro14-
pro16-
xx14-
```

PC 模式继续包含这 5 台。

再验证每次跨域自动切换后，全部核心导出、排名、趋势线输入和 `chart_data` 都不包含上一域 name。

### 20.6 雷达测试

1. 两台 Tablet 某维低/高映射到 60/100。
2. 单台 Tablet 五维全部映射到 80。
3. 多台 Tablet 某一维完全相同时，该维全部为 80。
4. Tablet 雷达候选不包含固定 PC。
5. PC 雷达继续使用固定 5 台锚点。
6. 隐藏某个 Tablet 的 `device_name` 不应改变其他 Tablet 的雷达缩放。
7. PC/Tablet 自动切换后，左右雷达选择和两个 `customized` 状态被重置，新域默认设备能正常出现。

### 20.7 图表测试

1. PC 标题仍显示 PC 和固定参考。
2. Tablet 标题显示 Tablet/当前会话。
3. Tablet 四张图不出现固定参考名称。
4. Tablet 价值趋势线只使用 Tablet 点。
5. 空 Tablet 结果仍能渲染空态，不抛异常。

---

## 21. 推荐测试顺序

在另一台源码电脑的项目根目录：

```powershell
.\.venv-build\Scripts\python.exe -m pytest -q
.\.venv-build\Scripts\python.exe -m pc_quality_demo --self-test
.\.venv-build\Scripts\python.exe -m pc_quality_demo --external-model-self-test
.\.venv-build\Scripts\python.exe -m pc_quality_demo --ui-self-test
```

旧版“134 passed”只证明旧源码通过。加入 Tablet 后必须以新测试总数和本次运行结果为准，不能继续引用旧证据。

建议再人工验证两组文件：

1. 仅 PC 音频 CSV：
   - 行为和旧版完全一致；
   - 5 个参考仍存在。
2. 仅 Tablet 音频 CSV：
   - 无固定 PC；
   - Score、Rank、win percent、四张图和导出均可用。

然后按顺序验证：

1. 导入纯 PC 第二份 CSV，再导入另一个纯 PC CSV：结果同域累积。
2. 紧接着导入纯 Tablet CSV：无需手动清空，旧 PC 记录自动消失，只剩 Tablet。
3. 再导入纯 PC CSV：旧 Tablet 自动消失，固定 5 台 PC 恢复。
4. 在已有有效结果时导入故意混域的第二份 CSV：它在任何清空和预测前整体失败，旧结果保持不变。
5. 导入 name 全部不在第一份表中的第二份 CSV：不切换当前域、不删除已有成功结果。
6. 替换第一份 CSV：新文件有效时清空全部预测记录；新文件无效时保留旧第一表和旧结果。

---

## 22. 版本和安装包

这是新增用户功能，建议从：

```text
0.3.0
```

升级为：

```text
0.4.0
```

至少同步检查：

```text
src/pc_quality_demo/version.py
pyproject.toml
src/pc_quality_demo/assets/model_manifest.json
packaging/pyinstaller/PCQualityDemo.version_info.txt
packaging/inno/PCQualityDemo.iss
README.md
CHANGELOG.md
发布文件名
```

模型文件内容和模型 SHA 不变。`model_manifest.json` 中的 app version 可以更新，但两个模型的哈希不能因为代码功能修改而随意改写。

Python 源码发生变化后必须重新构建，不能使用 `-SkipBuild` 复用旧 EXE：

```powershell
.\scripts\make_release.ps1 `
  -PythonExe .\.venv-build\Scripts\python.exe
```

最终应重新验证：

- 安装；
- 启动；
- PC 三份 CSV；
- Tablet 三份 CSV；
- 单域累积；
- PC → Tablet 自动清空并切换；
- Tablet → PC 自动清空并切换；
- 单个第二份 CSV 内混域时拒绝且保留旧结果；
- 替换第一份 CSV 后清空全部预测记录；
- 8 个 CSV；
- 4 张 PNG；
- 升级覆盖；
- 无 Python 环境运行。

---

## 23. 最容易漏改的地方

1. `read_auxiliary_scores()` 仍然把 domain 丢掉。
2. `OverallRuntime.predict_device()` 仍硬编码 `"pc"`。
3. Score 点预测正确，但总体 Monte Carlo frame 仍写死 PC。
4. `snapshot()` 或 `all_feature_table()` 导出仍写死 PC。
5. GUI 已排除固定 PC，但 `summarize_session()` 的 win percent 仍包含它们。
6. 图表不显示参考 PC，但趋势线的数据仍包含参考 PC。
7. Tablet `all_device_predictions()` 空表时因没有固定参考而 merge 崩溃。
8. Tablet 雷达仍偷偷用固定 PC 做锚点。
9. `clear_predictions()` 没有把 `current_domain` 重置为 `None`。
10. 仍保留旧的“跨域拒绝、要求手动清空”代码，没有改成自动切换。
11. 在确认新批次为单一 domain 之前就清空旧结果，导致混域或坏 CSV 也误删数据。
12. 混域检查放在逐设备循环内，导致部分结果已写入。
13. 替换第一份 CSV 时先清空再校验新文件，导致坏文件也误删旧记录。
14. 示例 CSV 没有 domain，导致安装包演示失败。
15. 只运行单元测试，没有重建并测试冻结后的 EXE。

---

## 24. 最终验收口径

完成后，同一套模型和同一套程序应满足：

```text
辅助 CSV 可以长期保存 PC + Tablet
                    ↓
音频 CSV 的 name 决定本批设备集合
                    ↓
通过辅助 CSV 查询并验证批次 domain
                    ↓
批内混域？── 是 → 整批拒绝，保留旧结果
       │
       否
       ↓
无 name 交集？── 是 → 只登记缺名失败，不切换
       │
       否
       ↓
与当前结果同域？── 是 → 累积设备并重新排名
       │
       否
       ↓
自动清空旧域结果并切换到本批 domain
       ↓
PC     → 固定 5 PC + 当前 PC 排名/胜率/图表
Tablet → 仅当前 Tablet 排名/胜率/图表
```

PC 的旧预测数值必须保持不变。Tablet 只改变：

- Score 是否启用 PC offset；
- 当前排名比较集合；
- 固定参考是否进入展示和导出；
- 雷达归一化锚点；
- GUI 的域相关文字。

类别模型、19 维特征顺序、Score 系数、Rank 系数和两个 joblib 均不改变。
