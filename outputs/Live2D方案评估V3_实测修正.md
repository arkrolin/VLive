# Live2D 动作生成 · 方案评估 V3（329 模型实测修正版）

- 日期：2026-08-05
- 状态：基于 `standrad-live-2d/` 全语料实测，修正 V2 中三条被数据推翻的判断
- 前置：`Live2D动作生成_方案设计V2.md`（架构骨架仍然有效）、
  `2026-07-17_hetero_layer_restructure.md`、`2026-07-17_token_diffusion_impl.md`
- 复现脚本：`tools/survey_dataset.py`、`tools/param_fingerprint.py`、
  `tools/survey_parts.py`、`tools/survey_conflicts.py`、`tools/peek_moc3.py`

---

## 0. 一句话结论

**扫掠探针单独不够，但它不需要够。** 你担心的「模型能不能认出装饰物 / 布料」是个被设错的目标：
组件身份不需要模型去*推断*，moc3 的 Part 树里明文写着，而且 Part 命名的跨模型标准化程度
比参数命名高一个数量级（13 个部件在 ≥90% 的模型中同名出现，vs 参数只有 20 个标准 id）。
真正需要计算的只是「参数 → 哪些 Part」这个映射，而这是扫掠探针的**确定性输出**，不是学习问题。

至于那些确实认不出来的（左边第几缕头发、哪片裙摆）——它们占手工参数的 19.5%，
**同时也正是应该从生成目标里删掉的东西**。认不出和不用认，恰好是同一批参数。

---

## 1. 语料实测基线（先把家底摊开）

`uv run --no-project python tools/survey_dataset.py`

| 项 | 实测值 |
|---|---|
| 模型数 | **329**（315 个 moc3 可解析） |
| motion 文件总数 | **8279**（去重后），中位 19 个/模型，max 83 |
| 动作总时长 | **625.7 分钟 ≈ 10.4 小时**，全部 30 fps |
| 单条 motion 时长 | 中位 4.0 s，p95 9.2 s，max 21.9 s |
| 曲线数/motion | 中位 135，p95 212，max 469 |
| 动画参数数/模型 | 中位 **114**，p95 174，max 313 |
| distinct 参数 id | **7976**（motion 中）/ 5981（moc3 中） |
| 单模型私有参数 id | **67.1%**（4016/5981） |
| Part 数/模型 | 中位 **22**，max 209 |
| distinct Part id（归一后） | **373** |
| 单模型私有 Part | **40.5%** |

**监督预算判断**：10.4 小时、8279 段、平均每段 135 条曲线，对「一次性表演动作」这个目标
是**充足的**。参照 text-to-motion 领域，HumanML3D 是 28.6 小时 / 14616 段；你在段数上同量级，
时长约 1/3，但你的每段维度更高且**同一角色内高度一致**（不需要跨人体型泛化）。
数据不是瓶颈，这个判断成立。

**长度分布是个好消息**：p95 只有 9.2 秒。按 30fps 定长 300 帧（10 秒）padding 即可覆盖 95%，
不需要做可变长度或自回归续接。这砍掉了 V2 §5.4 里 RTC 块间续接的全部工作量——
你选「一次性表演动作」这个目标，正好绕开了动画里最麻烦的一块。

---

## 2. 被数据推翻的三条 V2 判断

### 2.1 「读 model3.json 的 Groups 拿眨眼/口型排除表」— **完全失效**

```
model3 Groups non-empty : 1/329 (0.3%)   kinds={'LipSync': 1}
cdi3 / DisplayInfo      : 0/329 (0.0%)
pose3                   : 0/329 (0.0%)
expressions             : 0/329 (0.0%)
physics3                : 276/329 (83.9%)
```

手游解包资源只保留了运行时必需的最小集合。`Groups`、`cdi3`、`Pose`、`Expressions` 全没有。

**影响**：
- 没有 `Groups.EyeBlink` → 无法用官方声明识别眨眼通道，只能靠行为指纹（见 §4.2，这条路可行）。
- 没有 `cdi3` → **没有任何人类可读的参数名**。这是 V2 里我假设可用的语义源，实际不存在。
  你 `build_param_semantics.py` 走 ID 文本分解是对的，因为没有别的选择。
- 好消息：`physics3` 有 83.9%，这是唯一保留下来的高价值元数据。

### 2.2 「physics3 的 Output 参数应从生成目标剔除」— **方向对，理由错**

我原以为手工 motion 不会去写物理输出通道。实测（同模型内，非跨模型聚合）：

```
89 个有 physics 的模型中，88 个存在冲突
冲突参数数/模型: 中位 23,  占手工参数的 19.5%
top 冲突 id: PARAM_HAIR_FRONT 85.4% / PARAM_HAIR_SIDE_R 62.9% / PARAM_HAIR_SIDE_L 62.9%
             PARAM_HAIR_BACK 37.1% / PARAM_RU_Y 29.2% / PARAM_BUST_X 18.0%
```

**作者是系统性地手 K 头发的**，即使 physics 会驱动同一批参数。这大概率是 Cubism 2 时代
（无物理演算）遗留的工作习惯，或者项目里物理被关闭。

**但结论反而更强了**，因为三条独立证据指向同一批参数：

| 证据 | 数字 | 含义 |
|---|---|---|
| physics 冲突 | 占手工参数 **19.5%** | 运行时会被物理覆盖/叠加，手 K 值的实际贡献不确定 |
| 行为指纹检索 | 头发类 **R@1 = 0%** | 跨模型完全不可区分，学不到可迁移的表征 |
| hetero 统计（你的） | 头发+衣物仅 **4.7%** 语义占比 | 建模收益本来就低 |

→ **把 physics3 的 Output 集合整体移出生成目标，播放时交给物理引擎。**
不是因为「作者不写」，而是因为「难学、易被覆盖、且有免费的替代方案」。
这一刀砍掉约 19.5% 的目标维度，且**预期质量不降反升**——物理演算比手 K 更连贯。

反向红利仍然成立：physics 的 **Input** 高度集中在
`PARAM_ANGLE_X/Y/Z` + `PARAM_BODY_ANGLE_X/Y/Z/WAVE`。
你把这 6-7 个 root 通道生成好，物理系统免费产出几十个次级运动通道。

### 2.3 「Stepped 段占 41% → 需要离散生成头」— **过度反应**

段级统计确实是 Linear 44.2% / Stepped 41.2% / Bezier 14.6%。但**参数级**统计完全不同：

```
11499 个 (模型,参数) 对：
  48.3%  continuous
  33.4%  constant (整个模型所有 motion 里只有 1 个取值)  ← 真正的大头
  16.4%  stepped-mixed (50-80% 段为 Stepped)
   0.8%  stepped-dominant (>80%)
   0.7%  binary
   0.5%  few-level (3-5)
```

Stepped 段多，是因为它被用来表达「长时间保持不变」——**一个 Stepped 段可以覆盖好几秒**。
真正的离散开关型参数只有约 **2%**，不值得为它设专门的分类头。

**但暴露出一个更大的问题：33.4% 的参数是常量。**
三分之一的建模预算浪费在永远不动的通道上。这些必须在数据侧直接剔除——按 (模型,参数) 粒度，
不是按 id 全局剔除（同一个 id 在 A 模型是常量、在 B 模型可能是主运动）。

另外 **PartOpacity 占全部曲线的 14.6%**，这是图层显隐开关，是真离散。
建议 v1 直接旁路（回放时注入原值），不进扩散目标。

---

## 3. 你的核心问题：扫掠探针够不够

### 3.1 先把问题拆对

「模型能识别出不同参数对应的组件吗」隐含了一个假设：模型需要从视觉里*认出*
「这是一片布料」。但实际链路是：

```
参数 p  --扫掠--> 哪些 Drawable 的顶点动了  --moc3 父索引--> 哪些 Part  --Part 名--> 语义
        (确定性计算)                        (查表)              (跨模型标准化)
```

三步里没有一步需要「识别」。`csmGetDrawableParentPartIndices` 和 `csmGetPartIds` 是
Cubism Core 的公开 API，Drawable→Part 的归属是 moc3 里存好的结构。

### 3.2 关键实测：Part 层比参数层干净一个数量级

`uv run --no-project python tools/survey_parts.py`

| 阈值 | Part（归一后）| Parameter |
|---|---|---|
| 出现在 ≥90% 模型 | **13** | 20 |
| ≥50% | 18 | 42 |
| ≥10% | 54 | 184 |
| 单模型私有 | **40.5%** | **67.1%** |
| 总种类 | **373** | 5981 |
| 每模型个数（中位）| **22** | 114 |

≥90% 覆盖的 13 个 Part（实测覆盖率）：

```
HAIR_BACK 97.1%   ARM_R 96.8%   ARM_L 94.6%   MOUTH 94.3%   FACE 94.0%
BODY 93.0%        BROW 93.0%    EYE 93.0%     EYE_BALL 93.0%
HAIR_FRONT 92.7%  EAR 92.7%     NOSE 92.7%    NECK 92.1%
再往下：HAIR_SIDE 86.0%  BACKGROUND 72.7%  HAND_R 69.5%  HAND_L 67.6%  YIFU(衣服) 59.7%
```

这是 **Cubism 官方模板的部件命名规范**，绝大多数作者遵守。对比参数层 481 种手臂命名变体、
最高频 `PARAM_HAND_L` 仅 60.7% —— 同一件事，在 Part 层是 `ARM_L` 94.6%。

**这就是答案的核心：把参数投影到 Part 上，跨模型对齐问题基本消失。**

### 3.3 三层证据的实测能力与盲区

`uv run --no-project --with numpy python tools/param_fingerprint.py --limit 80`

跨模型检索实验：给定模型 A 的参数 p，在模型 B 的全部参数里检索最近邻。

```
=== EXACT-ID        recall@1 = 13.5%   recall@5 = 32.8%   (随机基线 0.94%)
=== SEMANTIC-BUCKET   top1   = 44.9%   in-top5 = 71.5%
```

| 证据源 | 能回答 | 实测强度 | 盲区（实测） |
|---|---|---|---|
| ① ID 文本 | 粗语义分类 | 67.1% 私有 id，标准名仅 20 个 | `PARAM_14` / `PARAM_2` 完全无信息 |
| ② 行为指纹 | **这是什么*类型*的自由度** | 类型判对 **44.9%**，精确定位 13.5% | **全部头发参数 R@1 = 0%** |
| ③ 几何探针 | **它动的是画面哪块、怎么动** | Part 名 13 个 ≥90% 共有 | 参数耦合、约 20% 模型 Part 名为纯数字 |

**②的盲区分布极其干净地印证了你的担忧，同时也给出了解法**：

```
能识别（R@1）:  ArmR_bigRoll 88%  Fingers_R_Move 80%  HANDR_Z 64%
                ANGLE_Y 60%  EYE_R_OPEN 60%  BROW_R_Y 45%  SHOUWAN_R 44%
识别不了（R@1=0%）: HAIR_L00 / HAIR_R00 / HAIR_LZ / HAIR_RZ / HAIR_WAVE
                    HAIR_BACK_WAVE / HAIRBACK_01 / QUNZI_SHAKE(裙摆, medRank 92)
                    SHOULDER_L / SHOULDER_R
```

头发/裙摆之所以 R@1 = 0，**不是因为特征不够，而是因为它们在时间行为上真的等价**——
左边发丝和右边发丝都是低频跟随摆动，统计上不可分。区别**只在空间位置**。
而空间位置正是扫掠探针唯一能给、也必然能给的东西。

**②和③是严格正交的**：②给「什么类型」，③给「在哪、朝哪动」。缺任何一个都不完整：
- 只有② → 知道是慢变连续量，不知道是左发还是右发
- 只有③ → 知道在左侧发丝区、绕根部旋转，但不知道它是主运动还是被动跟随

### 3.4 扫掠探针的真实局限（不回避）

**(a) 单参数独立扫掠会漏掉耦合。**
Live2D 的 Deformer 可以由 2 个参数联合控制（二维关键帧网格），且 Deformer 可嵌套。
独立扫掠 `p` 时其它参数固定在默认值，会漏掉交互项。

*补救*：只对**同 Part 内**的参数做成对 2D 扫掠。每模型中位 22 个 Part，
Part 内参数中位数不到 10，成对组合约 45 对 × 5×5 网格 = 1125 次 `csmUpdateModel`，
纯 CPU 亚秒级。全语料 315 模型也就几分钟。成本完全可控。

**(b) 约 20% 的模型 Part 名是纯数字。**
实测有 `00`/`01`/`02`/`12`/`13` 这类偷懒命名（覆盖率 20-41%）。这些模型的 Part 名无语义。

*补救*：Part 名无语义时，用该 Part 的**几何签名 + 该 Part 下参数的行为指纹**做跨模型检索，
匹配到有语义的 Part 簇。因为 Part 只有 22 个且空间分布固定（脸在上、脚在下、手在两侧），
这个检索比参数级检索容易得多。

**(c) PartOpacity 切换导致的显隐变化，顶点位置不变。**
扫掠时必须同时记录 `csmGetDrawableOpacities` 和 `csmGetDrawableDynamicFlags` 的
`csmIsVisible` 位，否则会把「让某图层消失」的参数误判为「无影响」。

**(d) 装饰物/道具确实区分不了——但不需要区分。**
实测私有 Part 里有 `YINGYUANDENG`、`Kuaizi`（筷子）、`Noodle`、`DanGao`（蛋糕）、`Music`
这类道具。它们：命名私有、几何签名互不相同也无从对齐、行为上多为开关。
这些落在你 hetero-C（非运动，38%）里，**本来就该旁路**。
「认不出」和「不用认」在这里是同一批对象。

---

## 4. 生成目标的重新定义（维度账）

按 (模型, 参数) 粒度逐级筛，每一步都是读文件就能算的确定性规则：

| 步骤 | 规则 | 剩余（中位/模型） |
|---|---|---|
| 起点 | motion 中出现的 Parameter 曲线 | **114** |
| −常量通道 | 该模型全部 motion 中 unique 值 ≤1 | ~76（−33.4%） |
| −physics 输出 | 出现在该模型 `physics3.json` 的 Destination | ~53（−19.5%） |
| −PartOpacity | Target == PartOpacity，旁路回放 | 53（本就不在 Parameter 内） |
| −非运动 C 组 | 你的 `classify_hetero` 标 C（道具/特效/宏/长尾） | **~35-40** |
| = 生成目标 | 连续主动运动通道 | **约 35-40 维** |

和 V2 估的 25-40 吻合，但现在每个数字都有出处。

**注意**：C 组的剔除要谨慎。你统计里 C 组占 hetero 的 38%，其中「长尾私有」24% 是
`classify_hetero` 未命中的兜底，里面**混有真实的手臂/身体参数**（因为拼音词表不全）。
建议：C 组剔除**只信 fx/macro/prop 三个明确 bucket，tail 不剔除**，改为交给几何探针二次分诊——
tail 参数若扫掠后影响 `ARM_*`/`BODY`/`HAND_*` Part，就提回 A 组。
这正是几何探针相对 ID 正则的核心增量价值。

---

## 5. 对你现有方案的三条具体修改建议

你 `2026-07-17_token_diffusion_impl.md` 的语义 token 方向是对的，与 AnyTop / URMA 的独立结论一致
（URMA arXiv:2409.06366 做过干净消融：token 集合 > 多头 > 维度 padding）。
以下是基于实测的具体修改。

### 5.1 `param_semantics.json` 的 desc 应该是三路融合，不是单靠 ID 分解

现在 `build_param_semantics.py` 的 `describe()` 只做 ID 文本分解——这是实测**最弱的一路证据**
（67.1% 私有，且 `PARAM_14` 这类完全无解）。建议改为拼接式描述符：

```python
# 现在
"PARAM_14": {"desc": "14"}                      # -> 无用的 e_p

# 建议
"PARAM_14": {
  "desc": "left side hair strand, slow continuous sway, "
          "rotates about upper anchor, affects 3 meshes in part HAIR_SIDE",
  "src": {
    "id_text":  "",                              # ① 无信息
    "behaviour": "slow continuous, low frequency, bidirectional, "
                 "always active",                # ② 行为指纹 -> 模板句
    "geometry":  "part=HAIR_SIDE, region=upper-left, "
                 "field=rotational, extent=3 meshes, area=2.1%",  # ③ 扫掠
  }
}
```

三路都转成**自然语言模板句**再拼接，喂同一个冻结文本编码器。好处：
- 不改模型侧接口，`e_p = TextEnc(desc)` 一行不动
- 三路证据自动加权（文本编码器会处理冗余）
- 缺任何一路都能降级工作（ID 无语义时靠②③，无渲染环境时靠①②）

行为指纹 → 模板句的映射，用 `tools/param_fingerprint.py` 已算出的 32 维特征做规则分箱即可，
不需要学习。例如 `event_rate > 0.3 且 event_dur < 0.3s` → "brief periodic pulses"（眨眼）。

### 5.2 归一化范围：用 motion 实测值域，不是 moc3 声明值域

`hetero_ranges` 你计划全量扫 min/max。补一条：**扫的是 motion 里的实际出现值，
不是 `csmGetParameterMinimumValues` 的声明范围**。原因是大量参数的声明范围远大于实际使用范围
（作者留了余量），用声明范围归一化会把有效信号压到 [-0.1, 0.1] 的小区间。

同时务必加下限截断：`scale = max(q99 - q01, eps_floor)`，`eps_floor` 建议取该参数声明范围的 5%。
33.4% 的常量通道即使被剔除，仍会有低使用率通道让归一化数值爆炸。

### 5.3 core 21 维 / hetero 的二分可以取消

你的 core 是按 ≥90% 跨机型频率选的，本质是「命名标准化程度」的代理指标，不是重要性。
一旦走全 token 化（每参数一个 token + 语义嵌入），**core 和 hetero 的区分就没有必要了**——
attention 天然吃变长，语义由 e_p 携带。

保留 core 的唯一理由是它有稳定的列序可做位置编码，但 token 方案里位置编码本就不该用。
建议：**统一为一个参数 token 集合**，A/B/C 分组降级为 token 上的一个 group embedding
（或直接体现在 desc 文本里），而不是结构上的分层。这会简化模型侧不少。

---

## 6. 关于「参考视频」——一个方向修正

你说「参考视频指同一角色已有的 Live2D 动画，提供视频是帮助模型理解图层」。

**如果视频是由该角色已有 motion 渲染出来的，那么视频的信息量 ⊆ (motion 文件 + moc3)。**
让 VLM 从像素里反推「哪个参数控制哪个图层」，是在用一条有损、昂贵、不可验证的路径，
去获取一份你本来就能精确计算的信息。扫掠探针给的是**精确形变场**，视觉编码器给的是
**有损的像素级近似**——后者严格弱于前者。

所以我的建议是：**「理解图层」这件事不要交给视频**，交给扫掠探针 + Part 树。

视频/视觉模态的真正价值在另外三处，这三处扫掠探针替代不了：

1. **角色外观理解**（这个角色是短发还是长发、穿裙子还是裤子、有没有猫耳）→
   影响动作风格的先验。这需要看**渲染图**，但只需要静态图，不需要视频。
   一张 T-pose 渲染图 + 几张扫掠对比图就够。
2. **few-shot 风格条件**：「模仿这段参考动画的节奏和幅度」——
   但这里更该直接用参考 motion 的**参数曲线**做条件，而不是视频。同域数据，何必绕一圈。
3. **闭环质检**：生成 → 渲染 → VLM 判断「这看起来像不像『开心地挥手』」。
   这是视频模态真正不可替代的地方，用于自动评测和 RL 后训练的奖励信号。

按「效果优先」，我建议 v1 **完全不上视觉编码器**。理由：
- 你的参考视频与 motion 同域，信息冗余
- 图层理解由探针精确解决
- 视觉编码器会显著增加训练成本和不稳定性
- 留到 M3 做闭环质检时再引入

---

## 7. 修正后的分期路线

你的四个回答（同域参考动画 / 一次性表演动作 / 需要身体手臂 / 不需人工二编）
砍掉了三块工作量：无需 RTC 续接、无需贝塞尔曲线拟合（Linear 段密铺 30fps 直出）、
无需真人视频反解。剩下的路线：

**M0 · 数据地基**（纯确定性，无模型）
- 接 Cubism Core：Windows DLL + ctypes 直调即可，需要的 API 约 10 个
  （`csmReviveMocInPlace` / `csmInitializeModelInPlace` / `csmGetParameterIds` /
  `csmSetParameterValues` / `csmUpdateModel` / `csmGetDrawableVertexPositions` /
  `csmGetDrawableParentPartIndices` / `csmGetPartIds` / `csmGetDrawableOpacities` /
  `csmGetDrawableDynamicFlags`）。不需要 GPU、不需要 OpenGL 上下文。
  参考实现可看 `live2d-py`（Gitee 阿东/live2d-py）的 Core 封装。
- 产出 `param_probe.json`：每参数 → 影响的 Drawable 集合、父 Part、位移场统计
  （质心、主方向、旋转 vs 平移 vs 缩放、影响面积占比）
- 产出 `param_behaviour.json`：`tools/param_fingerprint.py` 的 32 维特征（已可用）
- 合成 `param_semantics.json` v2：三路模板句拼接（§5.1）
- 目标筛选：常量 / physics-out / C 组剔除，落 `gen_mask.json`

**M1 · 单模型验证**（不碰跨模型，不碰 VLM）
- 挑 motion 数最多的模型（有 83 个的那个）
- 固定维度、flow matching、文本条件用 CLIP/T5 文本编码器
- 目标：「挥手」「点头」「歪头笑」「双手叉腰」能生成得自然
- **这一步跑不通，后面全是空中楼阁**

**M2 · 跨模型**
- 换成参数 token 集合 + 轴向注意力（时间轴 / 参数轴分解）
- 全 329 模型联合训练
- 验收：留出 20 个模型零样本，只给 param_semantics，看生成质量

**M3 · 闭环**
- 渲染 → VLM 打分 → 自动评测集
- 到这一步再考虑视觉模态和 RL 后训练

---

## 8. 待你决策的三件事

1. **常量通道剔除后，回放时注入什么值？** 该模型 motion 里的那个唯一值？还是 moc3 的
   `csmGetParameterDefaultValues`？两者可能不一致。建议前者（作者的实际选择），
   但需要落一份 `static_pose.json`。

2. **physics 输出通道整体剔除，是否接受「和原始 motion 不完全一致」？**
   剔除后播放效果依赖运行时物理演算是否开启。如果你的播放器不跑 physics，
   头发会完全不动。需要确认播放侧。

3. **Part 名为纯数字的那约 20% 模型，v1 是先排除还是靠几何检索兜底？**
   排除更省事（还剩 250+ 模型，够用）；兜底更完整但要多写一个 Part 级检索模块。
   按效果优先，我倾向 **M1/M2 先排除，M3 再补**。
