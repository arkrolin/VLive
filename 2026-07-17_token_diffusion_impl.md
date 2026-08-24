# 语义 Token 扩散 — 实现文档

- 文档类型:实现规范(implementation spec)
- 日期:2026-07-17
- 状态:数据侧待实现(本文档);模型侧由作者在服务器扩展
- 前置结论:参数语义嵌入聚类已由作者验证有效(这类超简单语义可分)
- 关联:`2026-07-17_hetero_layer_restructure.md`(三层拆分)、
  `2026-07-16_text_modality_design.md`(文本模态)

---

## 0. 要解决的根本矛盾

hetero 层是**变语义**的,不只是变长:

```
机型X 的 hetero: 第3列 = PARAM_HAND_L
机型Y 的 hetero: 第3列 = PARAM_SHOUWAN_L
```

现方案用 FSQ 按机型压成**固定维匿名向量**,跨机型共享的扩散去噪器拿到隐向量第 k 维
时,不知道它在当前机型代表什么 → 每机型死记一套私有编码 → 不泛化、新机型崩。

**解法**:不压成匿名向量,而是编码成一组**自带语义身份的 token**,扩散在语义空间去噪。

```
每个参数 p → token = [ 语义嵌入 e_p (它是什么) | 值 v_p (它现在多少) ]
                       ↑ 参数名文本编码(冻结,跨机型对齐)
```

- 种类数不同 → attention 天然吃变长。
- 每列语义不同 → 语义由 e_p 携带,不靠位置;"左臂"在所有机型对齐到同一嵌入簇。
- 新机型泛化 → 新参数只要能给出语义描述就有 e_p,零样本加入。

**副作用**:FSQ 的"变长压定长"职责消失,可降级为时序 patch 压缩或整体去掉。

---

## 1. 职责边界(本文档只做数据侧)

| 侧 | 内容 | 谁做 |
|---|---|---|
| **数据侧** | 参数语义资产:每个参数 id → 语义描述文本 + 逐参数归一化范围 + 三层标签 | 本次(本地) |
| 模型侧 | 语义嵌入编码、token 构造、DiT token 扩散、去 FSQ | 作者(服务器) |

数据侧产出让模型侧"开箱即用":拿到分片 + 语义资产就能构 token,不必回头补数据。

---

## 2. 数据侧要产出的三样资产

### 2.1 参数语义描述表 `param_semantics.json`

每个出现过的参数 id → 一句英文语义描述(喂给冻结文本编码器得到 e_p)。

```json
{
  "PARAM_ARM_L":     {"desc": "left arm rotation",      "group": "A_active", "bucket": "limb_arm", "src": "curated"},
  "PARAM_SHOUWAN_L": {"desc": "left wrist rotation",     "group": "A_active", "bucket": "limb_arm", "src": "curated"},
  "PARAM_XYZ_PRIV":  {"desc": "xyz priv",                "group": "C_nonmotion", "bucket": "tail", "src": "auto"}
}
```

**覆盖策略**(实测:5197 个 distinct id,top 484 覆盖 69.1% 列出现):
- **curated**:>=10 机型的高频 id(484 个)人工/半自动精选翻译。含拼音词
  (`SHOUWAN`=wrist、`SHIZHI`=index finger、`TUI`=leg…),复用
  `step4_constraints.PINYIN_MAP` 的翻译资产,扩充部件词表。
- **auto**:长尾(3615 个单机型私有)算法兜底 —— 去 `PARAM_` 前缀、按 `_` 切 token、
  拼音转写、side/axis 后缀(`_L`/`_R`/`_X`/`_Y`)展开成 "left/right/…"。质量够 e_p 用。
- `src` 字段标注来源,便于审计与后续增补。

### 2.2 逐参数归一化范围 `hetero_ranges`(写入 semantic_schema.json)

**现状问题**:hetero 存的是**原始值,未归一化**,逐列范围差异极大
(实测 `ARM_L` [0,4.9]、`ARM_L_02` [-0.7,0.56]、`TOUSHI` [0,1])。token 的 value
字段 v_p 必须逐参数归一到 [-1,1],否则扩散尺度不一致。

- 复用 core 的做法:全量扫描每个 hetero id 的 min/max(类比 `scan.scan_ranges` /
  `l2d-pipeline ranges`),写入 schema 的 `hetero_ranges`。
- 归一化:`v_norm = clip((v-min)/(max-min)*2-1, -1, 1)`,与 `align.normalize_core` 同式。
- 反归一化用于回放:`align.denormalize_core` 同款逆变换(渲染注入需原值)。

### 2.3 三层标签 `hetero_labels.json`(已产出)

`tools/classify_hetero.py --label` 已生成:每机型每列打 `A_active`/`B_passive`/
`C_nonmotion` + 细分 bucket。token 扩散用它:A 组参与动作扩散;C 组(非运动,38.5%)
旁路,回放注入原值,不进扩散目标。

---

## 3. 分片新增字段(供模型侧构 token)

现分片已有 `hetero_ids`(N/A,整片共用)、`hetero_targets`、`hetero_latent_raw`。
token 扩散额外需要(可运行时从上述资产 join,或固化进分片):

```python
{
  # 已有
  "hetero_ids":        (M,)  str,   # 第 j 列参数 id
  "hetero_targets":    (M,)  str,   # Parameter | PartOpacity
  "hetero_latent_raw": (N,G,M),     # 原始值
  # 新增(本次数据侧产出)
  "hetero_desc":       (M,)  str,   # 第 j 列语义描述(from param_semantics)
  "hetero_group":      (M,)  str,   # A_active | B_passive | C_nonmotion
  "hetero_norm":       (N,G,M),     # 逐参数归一化到[-1,1]的值(v_p)
}
```

模型侧:`e_p = TextEnc(hetero_desc[j])`(冻结,可缓存),`token_j = [e_p | hetero_norm[:,:,j]]`。
core 21 维同理造 token(desc 用 core_params 的标准语义,group 恒为显式)。

---

## 4. 实现任务(数据侧)

1. **`tools/build_param_semantics.py`**:扫全部分片 hetero_ids → 生成
   `param_semantics.json`。curated 高频 + auto 长尾。内置扩展部件词表(在
   PINYIN_MAP 基础上补 wrist/finger/leg/shoulder 等身体部件拼音)。
2. **`hetero_ranges` 扫描**:全量扫 hetero 原始值域 → 写 semantic_schema.json。
   可加进 `l2d-pipeline ranges`(加 `--hetero` flag)或独立脚本。
3. **分片增强**(可选,或运行时 join):把 `hetero_desc`/`hetero_group`/`hetero_norm`
   写进分片。建议先出 param_semantics + hetero_ranges 两个资产文件,分片增强留到
   模型侧确认 token 格式后再落,避免返工。

**先做 1+2**(纯资产,不改分片结构,零返工风险),3 等模型侧格式定稿。

---

## 5. 与文本模态的统一(备注)

动作 token 化 = 文本模态设计层次 2 的同一范式:
- 条件 = 文本 token 序列;动作 = 参数 token 序列;全在 attention 对齐。
- 参数语义描述(2.1)与文本层次 4 的语义扩写**共用翻译资产**(拼音→英文)。
两个改动是一件事,模型侧统一设计。
