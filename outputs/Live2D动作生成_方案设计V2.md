# Live2D 动作生成方案 V2

**从「2D 机体动作参数」收敛到「`.motion3.json` 生成」**

> 版本：V2（2026-08-05）
> 与 V1 的关系：V1《VLA与具身智能扩散动作生成_深度调研报告》给出的**通用骨架**（flow matching 动作专家 + VLM 梯度隔离 + action chunking + RTC 续接）依然成立。V2 只改三处：**动作头从"固定维度"改为"参数即 Token"**、**新增通道分诊机制**、**新增以渲染器为核心的数据引擎**。

---

## 0. 一页纸结论

| 你提出的问题 | 结论 |
|---|---|
| 参数命名/数量跨模型完全不一致 | 不要固定维动作头。用 **AnyTop 式 per-parameter token**，参数身份由「文本名 + 扫掠探针指纹 + 数值范围 + Deformer 拓扑」四路合成。已有三方独立证据表明 **token 集合 > 多头 > padding** |
| 需要视频模态帮模型认知参数含义 | **对，但更好的做法不是渲染图像，是读顶点。** Cubism Core 的 `csmGetDrawableVertexPositions` 在 `csmUpdateModel` 后直接返回形变后顶点，**不需要 GPU**。扫掠一个参数就能拿到精确形变场。渲染图像只用于给 VLM 做自动语义标注 |
| 不同自由度统计差异巨大（衣服不动、眼睛几秒眨一次） | **三路分诊**：连续通道走 flow matching；事件通道走点过程 + 模板核；静态通道**不生成**。并且——衣服/头发**根本不该出现在 motion 文件里**（它们是 `physics3.json` 的输出，运行时会被物理覆盖），眨眼**默认也不该生成**（`Groups.EyeBlink` 是 SDK 自动效果） |
| few-shot 适配新模型 | 零样本靠探针（无需训练）；要精修就只训**参数 token 嵌入的残差**（Live2D 版 textual inversion），几百参数、不动主干 |
| 这条路有没有先例 | **没有。** 认真检索过中英日文，学术界与开源界均无"生成 motion3.json"的工作。最接近的是规则+Agent 提示词的 `live2d-add-motion-sample-web-ui`。**天花板就是你自己，但也意味着没有现成基线，评测体系要自建** |

**最大的风险不在模型，在数据。** 合法可用、带成对手工 motion 的高质量 Live2D 模型只有**几十个量级**。整个方案的成败取决于第 6 节的数据引擎能否跑通。

---

## 1. 问题重述：这是三个问题的叠加

把你的需求拆开，它不是一个 VLA 问题，而是三个：

**(1) 跨本体问题 (cross-embodiment)。** 每个 Live2D 模型就是一个"机器人本体"，参数空间互不兼容。这在机器人领域是老问题，有成熟答案。

**(2) 你拥有一个仿真器。** 这是机器人领域求之不得的东西。Cubism Core 是一个**确定性的、微秒级的、可任意查询的正向模型** `f: 参数向量 → 全部 Drawable 顶点位置`。这意味着：
- 参数语义可以**测量**，不用猜
- 可以合成无限量的 (参数 → 视觉) 配对数据
- 可以做 analysis-by-synthesis 反解视频（第 6.3 节）

**这是你相对所有 VLA 工作的结构性优势，方案设计应该围绕它展开，而不是照抄机器人的做法。**

**(3) 极端异构的通道统计。** 这是最容易被低估、也最容易让 v1 模型"看起来能跑但效果很差"的一点。第 4 节专门处理。

---

## 2. 必须先写死的格式事实

这一节是写代码的依据，全部核对过官方 Schema（`github.com/Live2D/CubismSpecs`）。

### 2.1 `.motion3.json` 的 Segments 编码

```
Segments = [ t0, v0,  seg_id, ...,  seg_id, ...,  ... ]
             └初始点┘  └───段1───┘   └───段2───┘
```

- 曲线以**初始点 (t, v) 两个数字**开头，第 3 个数字才是第一个段标识符
- 段标识符取值与消耗：
  | id | 类型 | 后跟点数 | 后跟数字数 |
  |---|---|---|---|
  | `0` | Linear | 1 | 2 |
  | `1` | Cubic Bézier | 3（P1,P2,P3） | 6 |
  | `2` | Stepped | 1 | 2 |
  | `3` | InverseStepped | 1 | 2 |
- Bézier 的 P0 是上一段末点。官方约定手柄 t 分量为三等分点
- `Target` ∈ `{"Parameter", "PartOpacity", "Model"}`；`Target=="Model"` 时 `Id` 只能是 `"Opacity"` / `"EyeBlink"` / `"LipSync"`

最小合法示例（官方）：
```json
{
  "Version": 3,
  "Meta": { "Duration":1, "Fps":120, "CurveCount":1,
            "TotalSegmentCount":1, "TotalPointCount":2 },
  "Curves": [
    { "Target":"Parameter", "Id":"ParamEyeLOpen", "Segments":[0,0, 0, 1,1] }
  ]
}
```

**⚠️ 序列化器必须自动计算并保证 `CurveCount` / `TotalSegmentCount` / `TotalPointCount` 一致。** SDK 按这几个数字预分配内存，写错会崩溃或静默截断。这是最容易踩的工程坑。

**⚠️ `AreBeziersRestricted`**：Unity 导入器检查此标志，为 `false` 会警告曲线可能不匹配。生成侧建议置 `true` 并严格生成受限贝塞尔；或 v1 阶段干脆全用 Linear 段密铺。

**Fade 优先级**：曲线级 `FadeInTime`/`FadeOutTime` > model3.json 中该 motion 条目级 > motion3.json Meta 级 > 默认 1 秒。

### 2.2 参数的 min/max/default 不在任何 JSON 里

在 `.moc3` 二进制中。三条获取路径：

| 路径 | 说明 |
|---|---|
| Cubism Core C API | `csmGetParameterMinimumValues` / `MaximumValues` / `DefaultValues` / `csmGetParameterRepeats`（5.x 新增）/ `csmGetParameterKeyValues` |
| `py-moc3`（PyPI） | 纯 Python 零依赖，支持 moc3 v3.0~v5.0，`moc["parameter.min_values"]`，CLI `moc3 params model.moc3 --json`。成熟度低但 round-trip 字节一致测试过，批量扫描够用 |
| `vtubing/moc3`（Rust） | 干净室实现，v3.0~v5.0 |

**绝不能硬编码 ±30。** 官方明确鼓励建模者改宽范围（"要转更大角度就设 -45~45"），实际模型范围五花八门。归一化必须逐模型读实际范围。

### 2.3 三张排除表（生成前必须扣掉）

```python
EXCLUDE = set()
# 1) 物理驱动输出 —— 写了也会被物理演算覆盖，纯训练噪声
EXCLUDE |= {o["Destination"]["Id"]
            for s in physics3["PhysicsSettings"] for o in s["Output"]}
# 2) LipSync 组 —— 交给 MotionSync SDK / 响度映射在运行时叠加
# 3) EyeBlink 组 —— 交给 SDK 自动眨眼（见 4.3，但保留显式覆盖通道）
for g in model3.get("Groups", []):
    if g["Name"] in ("LipSync", "EyeBlink"):
        EXCLUDE |= set(g["Ids"])
```

**这一步直接解决了你担心的一半问题。** "衣服动作变化很小" —— 因为衣服摆动本来就是 `physics3.json` 的 Output，不该生成；"眼睛几秒眨一次" —— 因为眨眼是 SDK 的 `EyeBlink` 自动效果，不该生成。**把它们从生成目标里删掉，通道统计的极端不平衡问题立刻缓解一大半。**

反过来，`physics3.json` 的 **Input** 通常是 `ParamAngleX/Y/Z`、`ParamBodyAngleX/Y/Z` —— 这些正是你**必须**生成好的核心通道，因为整个头发/衣物的次级动态都由它们驱动。**生成好 6 个 root 通道，物理系统会免费送你几十个通道的高质量次级运动。** 这是 Live2D 相对纯骨骼动画的巨大便利。

### 2.4 参数拓扑从哪来

三张图，都可以作为注意力偏置：

1. **Deformer 层级**：ArtMesh → WarpDeformer / RotationDeformer 的父子树。这是 Live2D 的"骨架"
2. **物理关联图**：`physics3.json` 的 Input → Output 有向边（`ParamAngleX` → `ParamHairFront` 等）
3. **cdi3 参数组树**：`ParameterGroups` 支持嵌套（`ParamGroupFace` → `ParamGroupEyes` → …）。**注意 cdi3 是可选文件，大量解包模型没有**，不能作为必需输入

### 2.5 标准参数 ID 与遵守程度

官方标准表：`docs.live2d.com/en/cubism-editor-manual/standard-parameter-list/`。核心约定："眼睛和嘴闭合为 0、张开为 1"。

遵守程度的现实：
- Cubism 3+ 的 VTuber 模型**遵守度高**（因为 VTube Studio / Animaze / IRIAM 都按这套 ID 映射）
- Cubism 2.1 老模型与手游解包资源大量使用 `PARAM_ANGLE_X` **大写下划线**风格
- 日式模型常见 `ParamA/I/U/E/O`（五母音口型）、`ParamTere`（照れ=脸红）等非标准 ID
- cdi3 的 `Name` 是建模者手填的显示名，**语言不固定**（日文极常见），是有价值的弱信号但**不能当稳定标签**

→ 所以文本编码器必须是**多语言**的（mT5 / BGE-M3 / multilingual-E5），纯英文 T5 会在日文名上失效。

---

## 3. 核心架构：参数即 Token

### 3.1 为什么不能用固定维动作头

三种跨本体方案的对比，机器人领域已经做过干净的消融（URMA，arXiv:2409.06366，CoRL 2024）：

| 方案 | 机制 | 代表 | 零样本新本体 | 对你的适配度 |
|---|---|---|---|---|
| **维度 padding** | zero-pad 到最大维 + embodiment ID | π0 | 差 | ❌ URMA 实测**显著最差**。你的参数数 20~200，padding 到 200 会让 90% 的槽位是噪声 |
| **多头解码** | 共享主干 + 每本体一个 head | GR00T N1、UniTalker | 不支持（新本体必须有数据训头） | ⚠️ 可作为头部高频模型的精修补丁，不能作主方案 |
| **Token 集合** | 每自由度一个 token，共享输出头 | AnyTop、CrossFormer、URMA、MetaMorph | 好 | ✅ **唯一能零样本上新模型的路** |

四方独立证据都指向 token 集合：AnyTop（图形学，SIGGRAPH 2025，arXiv:2502.17327）、URMA（腿足机器人，arXiv:2409.06366）、CrossFormer（20 种本体，arXiv:2408.11812）、Embedding Morphology into Transformers（MERL 2026，arXiv:2603.00182，明确验证 per-joint token + 拓扑 bias + 关节属性优于 vanilla π0.5 基线）。

### 3.2 参数身份嵌入：四路合成

每个参数 `p` 得到一个身份向量 `e_p ∈ R^F`，加到该参数所有帧的 token 上（AnyTop 的加性偏置做法）：

```
e_p = MLP( concat[
    E_text( "ParamEyeLOpen | 左目 開閉 | ParamGroupEyes" ),   # 多语言文本编码器
    E_probe( 扫掠探针指纹 ),                                  # ★ 见 3.3
    E_num( [min, default, max, repeat_flag, is_physics_input] ),
    E_topo( Deformer 树中的深度 / 类型 one-hot )
])
```

**训练时对 `E_text` 做 10~30% 随机 dropout。** 强迫模型在名字缺失或无语义（`Param14`）时，从探针指纹 + 数值范围 + 拓扑位置推断身份。这一招来自 SAME（SIGGRAPH Asia 2023）的随机关节遮蔽，也直接对应你说的"命名完全不一致"。

同时对参数集合做 **随机子集 drop**（保留 60~100%）作为数据增强，模拟"不同模型参数数量差异巨大"。

### 3.3 扫掠探针：不需要 GPU 的精确形变场

**这是整个方案里我认为最有价值、且没有先例的一步。** 图形学侧我没有找到"通过扰动绑定学习参数语义"的直接论文（机器人侧有：Active Embodiment Identification, arXiv:2605.08020；UPESI, arXiv:2109.13438），你这里是空白区。

管线（**纯 CPU，无需渲染**）：

```
for p in parameters:
    for v in linspace(min_p, max_p, K=9):
        csmSetParameterValues(所有参数=default, p=v)
        csmUpdateModel()
        V[p][v] = csmGetDrawableVertexPositions()   # 形变后顶点，精确
    Δ[p] = V[p] - V[p][default]
```

从 `Δ[p]` 提取指纹（低维、模型无关）：
- **空间**：受影响 Drawable 的集合与占比、位移质心（归一化到画布坐标）、包围盒、主方向（PCA 第一主成分）
- **幅度**：最大位移 / 平均位移，相对角色高度归一化
- **类型**：位移场是接近平移 / 旋转 / 缩放 / 局部非刚性（对 Δ 做仿射拟合，看残差）
- **非线性度**：`Δ` 随 `v` 的曲率（有些参数是分段的）
- **不透明度**：`csmGetDrawableOpacities` 的变化（区分形变参数 vs 显隐切换参数）

再加一路**语义标注**（这一路要渲染）：把 `v=min / default / max` 三张图 + 差异热力图喂给 VLM，让它输出一句话，例如"这个参数控制左眼上眼睑下降，取 0 时完全闭合"。这句话与参数 ID 拼接后一起进 `E_text`。**`Param14` 的语义问题在这里被彻底解决。**

配合 `csmGetDrawableMasks` / `csmGetDrawableParentPartIndices` 可以回溯到 Part 层级，正好对应你说的"逐资源图层渲染"。

> 成本估计：P=100 个参数 × K=9 个采样点 = 900 次 `csmUpdateModel`，微秒级，整个探针**秒级完成**。VLM 标注 100 次调用，几分钟。**一个新模型的完整"身份证"可以在 5 分钟内自动生成。**

### 3.4 轴向 DiT 与拓扑注意力偏置

动作张量：`X ∈ R^{T × P × D}`，`T=120`（4 秒 @30fps），`P` 可变（≤128，带 mask），`D=4`：
```
D = [ 归一化值 v̂ = (v - default) / max(range, ε), Δv̂, Δ²v̂, active_mask ]
```

去噪网络每层四段（AnyTop 的三段 + 一段 cross-attn）：

1. **参数轴注意力**（同一帧内跨参数全连接）。拓扑以**加性 bias** 注入而非 mask：
   ```
   a_ij = (q_i·k_j + a^D_ij + a^R_ij + a^Phys_ij) / √F
   ```
   - `a^D`：Deformer 树上的图距离（截断到 d_max）
   - `a^R`：关系类型（parent / child / sibling / self / leaf / none）
   - `a^Phys`：物理 Input→Output 关联（额外一类）

   用 bias 而非 mask 的理由：拓扑远的参数仍可通信（眼睛和嘴在 Deformer 树上可能很远，但表情上强相关），只是先验被削弱。

2. **时间轴注意力**（每参数独立，窗口 `W`）。⚠️ **窗口大小要谨慎**：眨眼 3~6 帧、头部转动 30~90 帧，同一个 `W` 无法兼顾。建议**多尺度**：偶数层 `W=16`，奇数层 `W=64`。

3. **Cross-attention → VLM condition token**（V1 的梯度隔离在此生效）

4. **FFN**，AdaLN 注入 flow timestep

**静态骨架信息**沿用 AnyTop 的做法：把 `{min, default, max, is_physics_input, probe 指纹}` 投影后**拼成"第 0 帧"**，张量变为 `(T+1) × P × F`。极简且有效。

算力：`T=120, P=128, F=384`。参数轴 `120 × 128² × F ≈ 7.5e8`；时间轴（窗口化）`128 × 120 × 64 × F ≈ 3.8e8`。单卡完全可以跑。

### 3.5 与 VLM 的接口（沿用 V1）

不变的部分：
- VLM 只输出 condition token，**动作专家的梯度在注意力层对主干 K/V 施加 stop-gradient**（Knowledge Insulation, arXiv:2505.23705）
- Flow matching 目标，时间步用 `Beta(1.5, 1)` 偏重高噪声端
- 块间续接用 **RTC**（arXiv:2506.07339），`execution_horizon` 8~12

VLM 在这里的具体职责（比 V1 更明确了）：
1. 理解文本指令
2. **看角色外观**（决定动作风格：机械体 vs 少女 vs 兽耳）
3. **看探针图**（辅助生成参数语义描述，这一路可以离线跑完缓存，不必在线）
4. 看参考视频（few-shot 风格条件）

> **务必先验证 VLM 是不是必需的。** 见第 9 节 M1/M2 的对照实验。如果指令空间不大，一个多语言文本编码器 + flow policy 可能就够了。

---

## 4. 异构通道：三路分诊

这是你提的第三个问题，也是最容易毁掉效果的一个。

### 4.1 扩散模型在极端不平衡通道上的五种失败模式

1. **均值回归塌陷（最致命）**。x₀-prediction + MSE 的最优解是条件期望。眨眼的**时间位置**在给定文本条件下几乎完全随机，`E[blink_t | text] ≈ 边缘均值 ≈ 0.03~0.08`。结果：模型输出一条恒定的"半眯眼"直线，**永不闭合**。这不是训练不足，这是目标函数的最优解。
2. **归一化两难**。全局标准化 → 眨眼被淹没；per-channel 标准化 → 近静态通道 `σ→0`，除法爆炸，纯噪声被放大成剧烈抖动。
3. **扩散的频域偏置**。脉冲是高频窄带信号，在标准 schedule 下很早就 SNR<1，模型在绝大多数时间步上"看不到"眨眼。
4. **时间尺度不匹配**。3~6 帧 vs 30~90 帧，共用一个 attention 窗口必然平均掉短脉冲。
5. **指标失明**。眨眼在总 MSE 里贡献 <0.5%，完全忽略它 loss 几乎不变。

### 4.2 自动通道分类

对每个参数在训练语料上统计：
```
特征 = [ 处于 default 的帧占比, 峰度 kurtosis, 自相关衰减时间 τ,
        功率谱质心, 峰值/中位数比, 活跃帧比例, 探针视觉幅度 ]
```
聚成三类：

| 类别 | 典型成员 | 处理 |
|---|---|---|
| **连续 continuous** | AngleX/Y/Z, BodyAngle*, EyeBall*, Brow*, MouthForm, Breath | flow matching 主通道 |
| **事件 event** | EyeLOpen/ROpen（若不排除）、表情瞬切、PartOpacity 显隐 | 点过程 + 模板核（4.3） |
| **静态 static** | 服装、配饰、罕用变形 | **不生成**，冻结在 default |

分类可人工抽检修正，结果作为模型的一部分缓存。

### 4.3 事件通道：眨眼的正确答案是"别生成"

**优先级最高的答案是 2.3 节的排除表**：`Groups.EyeBlink` 里的参数交给 SDK 自动眨眼（间隔对数正态 / 泊松，均值 3~5 秒，时长 100~400ms）。这条路的性价比极高，**建议直接作为 v1 基线**，绝大多数商业管线就是这么做的。

但你会需要一个**显式覆盖通道**——指令说"眨个眼""眯起眼睛笑"时，自动眨眼给不了。设计：

- 模型输出一路**稀疏事件序列** `{(t_k, type_k, duration_k, amplitude_k)}`
- `type` 从数据里聚类出的少量模板（close-hold-open 单眨、双眨、单侧 wink、长眯）
- 采样后用固定形状核卷积展开成曲线，写入 motion 文件，同时在运行时**临时禁用该区间的自动眨眼**（或用 `Target:"Model", Id:"EyeBlink"` 曲线做权重压制）

**论文支撑**（这条线有明确先例）：
- **Identity-Preserving Realistic Talking Face Generation**（arXiv:2005.12318）有独立小节 *Spontaneous Blink Generation*，明确论证眨眼与语音无依赖，用 GMMN 的 MMD 损失匹配真实眨眼的**频率、时长、间隔分布**。**这是你这条线的第一参考文献，问题设定几乎一模一样。**
- **DFA-NeRF**（arXiv:2201.00785）：音频与唇动强相关、与眨眼弱相关，故把眨眼用 Transformer-VAE + 高斯过程单独建模。
- **MoDiT**（arXiv:2412.09296）三大问题之一直接点名 "unnatural blinking behavior"。
- **反方证据**：VASA-1（arXiv:2404.10667）刻意把唇动/表情/凝视/眨眼放进同一个扩散 latent 联合生成，明确反对分离建模——但它有 VoxCeleb2 级别的 6000+ identity 数据量。
- **判据**：数据量不足时用解耦，数据量充足时联合建模更自然。**你的数据量不足 → 选解耦。**

### 4.4 归一化：方差下限与显式 mask

```python
scale_c = max(q99_c - q01_c, 0.05 * (max_c - min_c))   # ★ 双重下限
v_hat   = (v - default_c) / scale_c
```
- 用 q01/q99 分位数而非 min/max（抗离群）
- **下限用参数物理量程的 5%，而不是一个绝对小数** —— 这比 V1 里说的 `eps_floor` 更稳，因为不同参数量程差 30 倍（±30 vs 0~1）
- 额外一路 `active_mask` 通道显式告诉模型"这个参数在这段里有没有动"，避免模型靠幅度猜

### 4.5 损失重加权（如果暂时不做分诊的兜底）

```
w_c ∝ 1 / max(σ_c, σ_floor)      # 通道级
w_{c,t} ×= 5~10  if |Δv̂_{c,t}| > τ   # 峰值边沿加权
+ STFT 幅度谱 L1（仅事件通道）        # 强制保留高频能量
```
另可对事件通道用**更慢的加噪 schedule**，延长其高频可学习的时间步窗口。

---

## 5. 输出表示：稠密帧还是关键帧

### 5.1 v1：稠密帧 + 闭式曲线拟合（推荐先做这个）

生成 `T×P` 稠密张量 @30fps → 后处理转 Segments。

拟合管线：
1. **Visvalingam–Whyatt 简化**选关键帧（按三角形面积，比 RDP 更适合平滑曲线）
2. 每段做**带 Laplacian 正则的最小二乘贝塞尔拟合**（有闭式解，见 BiMotion 的做法：5 万顶点 ×200 帧在消费级 CPU 上 <1s；你的是 1D 单通道，更简单）
3. **事件通道跳过简化** —— 脉冲一旦被简化算法当噪声删掉就彻底没了
4. 误差超阈值的段回退为 Linear 密铺

理由：张量形状固定，扩散最稳；曲线拟合是成熟闭式问题；先出效果。

### 5.2 v2：关键帧直出

用 **CondMDI**（arXiv:2405.11126，SIGGRAPH 2024）的 mask-concat 机制，直接生成 `(t_k, value_k, in_tangent_k, out_tangent_k)` 四元组 —— **正好就是 motion3.json 的 Bézier segment 格式，生成即可落盘**。

- CondMDI 的做法极简：`x̃_t = m⊙c + (1-m)⊙x_t`，并把 mask **concat 进输入**；训练时随机采样关键帧位置**和通道子集**。它系统对比过 imputation / reconstruction guidance，结论是**训练时喂 mask 明显优于纯推理期引导**
- **sMDM**（arXiv:2503.13859）验证了稀疏关键帧的可行性：用 VW 简化选关键帧、mask 排除非关键帧出 self-attention（`O(N²)→O(K²)`）、线性插值重建。关键工程细节：**把输入输出线性层换成带正弦激活的 Lipschitz MLP**，保证插值平滑不丢高频。在 HumanML3D 上 FID 与 R-Precision 均优于 MDM
- **DanceNet3D**（arXiv:2103.10206）明确以动画工业实践为动机，把任务表述为"关键姿态之间的运动曲线预测"，是最贴合的先例
- **AutoKeyframe**（SIGGRAPH 2025，DOI 10.1145/3721238.3730664）自回归扩散直接生成关键帧，且**关键帧可被动画师直接编辑**再作为控制信号

**什么时候值得升到 v2**：当"生成结果需要美术二次编辑"成为硬需求时。稠密 Linear 密铺的文件人类几乎无法手编。

### 5.3 Loop 约束（idle 动作必需）

如果生成 idle 循环动作，需要额外约束首尾连续：
- 训练时对 loop 样本做**环形 padding**，让时间注意力跨越边界
- 采样后强制 `v(T) = v(0)` 且 `v'(T) = v'(0)`，用最后几帧的软混合实现
- 或直接在**环形隐空间**生成（把时间维当周期信号，用循环卷积）

---

## 6. 数据引擎：最大的资产和最大的瓶颈

### 6.1 现实评估

| 来源 | 量级 | 质量 | 法务 |
|---|---|---|---|
| Live2D 官方 Sample（Hiyori/Haru/Mao/Natori…） | **10~20 个** | 高，带完整 motion3 + physics3 + cdi3 | Free Material License，个人项目 OK，但禁止用于有损角色形象用途 |
| `Eikanya/Live2d-model`（3.3k star，手游解包） | **数百~上千** | 参差，moc2/moc3 混杂 | ⚠️ **全是游戏公司版权资产，无授权**。个人研究自用需自行判断 |
| nizima / BOOTH | 付费为主 | 高 | 逐个 License |

**结论：真正"干净且带成对手工 motion"的只有几十个模型。** 直接监督训练不可能撑起一个泛化模型。数据引擎是必须的。

### 6.2 五条数据路线

```
① 探针库（每模型一次性）    → 参数身份证，不是训练数据但是一切的前提
② 真实 motion3.json          → 少量高质量金标准（几百条）
③ 程序化 + 渲染 + VLM 标注   → ★ 无限量合成配对数据
④ 跨模型重定向增强           → 一份 motion 变 N 份
⑤ 面捕弱监督                 → 量大、风格偏、只覆盖面部
```

**③ 是主力。** 具体做法：
1. 用程序化动作生成器造参数曲线：Perlin 噪声 + 从②学到的关键帧模板库 + 物理合理的速度/加速度约束
2. 渲染成视频
3. 用 VLM 看视频**反向生成文本描述**（"角色缓慢向左歪头，然后微笑着点了两下头"）
4. 得到 `(text, video, params, model_id)` 四元组

这条路的关键是**程序化生成器的先验质量**。纯随机曲线渲染出来不像"动作"，VLM 也标不出有意义的文本。建议：从②的真实 motion 里提取运动基元（segment 级聚类），再随机组合 + 时间伸缩 + 幅度扰动 + 跨参数迁移。

**④ 是免费的 10 倍数据。** 同一条参数轨迹通过探针指纹做语义对齐后，可以重定向到另一个模型上重新渲染。这直接对应 SAME 的随机关节 drop 增强，也是跨模型泛化能力的主要来源。

**⑤ 的局限必须清醒**（调研已确认）：
- 只覆盖头部/面部。`ParamBodyAngle*` 在 Animaze 标准里明确标注 "not tracked by face trackers"；`ParamArm*` / `ParamHandL/R` / `ParamShoulderY` 完全提取不到
- 映射是标定依赖的启发式，不是 ground truth（`facial-landmarks-for-cubism` 作者明说"几乎每个参数都做成可调的"）
- 真人的运动幅度/时序 ≠ 手工 Live2D 动作的夸张化、卡点化风格。直接训会产出"很像面捕录制"而不像"美术做的动作"
- → **定位为预训练弱信号，不能作主数据**

### 6.3 逆向模型 + analysis-by-synthesis（不需要可微渲染器）

这是把"任意参考视频"变成训练数据的关键。

**你不需要可微渲染器。** 因为 `csmGetDrawableVertexPositions` 给出的是参数到顶点的映射，用**有限差分**就能拿到雅可比：

```
J[:, p] = (V(θ + δ·e_p) - V(θ - δ·e_p)) / (2δ)     # P 次前向，微秒级
```

P=100 时，一次雅可比 = 200 次 `csmUpdateModel` ≈ 亚毫秒。于是可以做 **Gauss-Newton 拟合**：

```
目标: min_θ  ‖ Π(V(θ)) - landmarks_ref ‖² + λ‖θ - θ_prev‖² + μ‖θ̈‖²
      (Π 是投影到 2D 关键点/掩码的算子，λ 时序平滑，μ 抗抖动)
```

流程：
1. 训一个 **inverse model**（video → params），用 ③ 的合成数据监督。因为合成数据无限，这个模型可以训得很好
2. 对新视频，inverse model 出初值 → Gauss-Newton 精修 → 高质量伪标签
3. 伪标签进训练集，飞轮转起来

**同域（渲染的 Live2D 视频）几乎可以完美反解**，因为正向模型是精确已知的。**跨域（真人视频、动画片段）**需要先提取语义中间量（2D landmark / DWPose / MediaPipe blendshape），再通过探针指纹做软匹配。

### 6.4 工具链选型

| 用途 | 推荐 | 备注 |
|---|---|---|
| moc3 参数抽取 | `py-moc3` (PyPI) 或 Cubism Core C API | 前者纯 Python 零依赖，批量扫描首选 |
| Python 驱动 + 离屏渲染 | **`EasyLive2D/live2d-py`** | 封装 Cubism Native Core，支持逐参数控制、Part 透明度、精确点击。配 GLFW/EGL + FBO 离屏。**训练管线首选** |
| Web 渲染 / 快速原型 | `guansss/pixi-live2d-display` (~1.5k★) + Playwright | 同时支持 Cubism 2.1 与 4/5，事实标准。主仓 2024-08 后靠 fork 延续 |
| 大规模数据生成 | `Live2D/CubismNativeSamples` (C++) | 最高性能 |
| 参考实现（绕开闭源 Core） | `MahouTechnologies/moc3-rs` | 唯一从零实现变形器数学的开源渲染器，`deformers/` 有算法文档。**若要做可微版本，从这里抄** |
| 面捕标注 | `emilianavt/OpenSeeFace`（支持 `-c video.mp4` 离线）+ `adrianiainlam/facial-landmarks-for-cubism`（MIT，开源的 landmark→Cubism 参数换算） | 后者带 `--old-param-id` 做 Cubism 2.1 ID 转换 |
| 现成规则基线 | `shinshin86/live2d-add-motion-sample-web-ui` | 其 `tools/analyze_model.py` 自动分析可用参数、安全范围、识别物理参数；`validate_motions.py` 独立校验器可直接复用 |

---

## 7. Few-shot 适配新模型

三级递进，按成本从低到高：

**L0 · 零样本（5 分钟，无需训练）**
```
解析 moc3/model3/physics3/cdi3 → 参数表 + 三张排除表 + 拓扑图
跑扫掠探针 → 指纹
VLM 标注 → 参数语义描述
→ 直接生成
```
这是 parameter-as-token 的核心红利。AnyTop 的证据表明：**分布内的未见拓扑很稳，强 OOD 会崩**（其 Table 2：OOD 距离 1.70 时 coverage 88.7%，4.04 时降到 16.5%）。Live2D 模型之间的差异远小于"螃蟹 vs 蜈蚣"，所以零样本应该相当可用。

**L1 · 参数嵌入反演（几十步，几分钟）**
给几条该模型自带的 motion 或参考视频，**只训练参数 token 嵌入的残差** `Δe_p`，冻结整个主干。类比 textual inversion：参数量只有 `P × F ≈ 128 × 384 ≈ 5万`，几十步就收敛，**不可能破坏主干**。这应该是默认的 few-shot 方案。

**L2 · LoRA 微调**
主干加 LoRA（rank 8~16），需要几十条数据。只在 L1 效果不够时用。

**L3 · 专用 PCA 头（UniTalker 式）**
对头部 20 个高频使用的角色，额外训一个 PCA 专用解码头做精修（UniTalker, arXiv:2408.00762 用 PCA 解决"输出维度差异巨大导致训练不稳"）。PCA 尤其适合 Live2D：一个角色的 100+ 参数实际有效自由度往往 <20。**代价是失去零样本能力，所以只作为补丁，不作为主干。**

---

## 8. 评测：必须在渲染空间做

**参数空间的 MSE 跨模型不可比**（维度不同、语义不同、量程不同），所以主指标必须在渲染后的像素/顶点空间。

| 维度 | 指标 |
|---|---|
| **视觉保真** | 渲染 GT 与生成 → 逐帧 LPIPS / DINOv2 特征距离、FVD |
| **顶点空间**（更精确，无需渲染） | `csmGetDrawableVertexPositions` 的逐帧 L2，按角色高度归一化 |
| **语义一致性** | VLM-as-judge（渲染视频 + 指令 → 1-5 分）；视频-文本检索 R-Precision |
| **物理合法性** | 参数越界率、jerk（三阶导能量）、**物理排除集泄漏率**（写了不该写的参数） |
| **事件级**（★ 别忘） | 眨眼次数/分钟、间隔分布 KL、峰值幅度分布。**总 MSE 对此完全失明** |
| **多样性** | 同指令多次采样的 inter-diversity（防塌陷）；AnyTop 的 coverage / local diversity / intra-diversity-diff 四件套可以直接搬 |
| **可编辑性** | 段数/秒、贝塞尔段占比、人工可读性抽检 |
| **Loop 质量** | 首尾值与首尾导数的不连续度 |
| **跨模型泛化** | 留出若干模型完全不参与训练，测 L0 零样本 |

**没有现成基线**，需要自建三条：
1. 随机采样 + 平滑（下界）
2. `live2d-add-motion-sample-web-ui` 式规则/Agent 生成（当前工业实践天花板）
3. GT 手工 motion（上界）

---

## 9. 分期路线图（效果优先）

**M0 · 地基（1~2 周）· 完全确定可行**
- moc3 参数抽取、model3/physics3/cdi3 解析、三张排除表
- motion3.json 读写器（**Meta 计数字段自动一致性校验**）
- live2d-py 离屏渲染 + 逐参数扫掠探针
- 曲线拟合器（VW 简化 + 闭式贝塞尔）
- 验收：能把 Hiyori 自带 motion 读进来、渲染成视频、再无损写回去

**M1 · 单模型验证（2~3 周）· 不碰 VLM，不碰跨模型**
- 固定 Hiyori 一个模型，固定维度动作向量
- 数据：自带 ~10 条 motion + 程序化生成几百条 + VLM 标注
- 模型：文本编码器（多语言）+ flow matching DiT，稠密帧 @30fps
- **验收：能生成"点头""摇头""歪头笑"且渲染出来自然**
- ⚠️ **这一步没做出来之前不要往下走。** 它验证的是最核心的假设：扩散能不能学会 Live2D 动作曲线

**M2 · 跨模型（3~4 周）· 核心创新在这里**
- 参数即 token + 四路身份嵌入 + 探针指纹
- 10~20 个模型，跨模型重定向增强
- 通道三路分诊上线
- **验收：留出 3 个模型完全不训练，零样本生成质量不崩**
- ★ **必须做对照**：token 集合 vs 多头 vs padding。这是你的主要 novelty，得有数据支撑

**M3 · 多模态条件（3~4 周）**
- 接 VLM（Qwen2.5-VL 3B 起步），梯度隔离
- 参考视频条件（风格）
- **必须做对照**：VLM vs 纯文本编码器。**如果 VLM 没有显著增益就不要它** —— V1 里说过，很多"VLA"的增益其实来自 chunking 和连续表示

**M4 · 数据飞轮（持续）**
- inverse model（video → params）
- Gauss-Newton 精修
- 面捕弱监督预训练
- 事件通道 / 关键帧直出 / RTC 流式续接

---

## 10. 风险清单

| # | 风险 | 缓解 |
|---|---|---|
| 1 | **数据量根本不够**（合法模型只有几十个） | 数据引擎（§6）是必答题不是加分题。M0 就要把渲染管线跑通 |
| 2 | motion3.json Meta 计数字段写错 → SDK 崩溃 | 序列化器自动计算 + 独立 validator（抄 `validate_motions.py`） |
| 3 | 生成了物理驱动参数 → 被运行时覆盖，白训 | 排除表在**数据清洗阶段**就执行，不是生成阶段 |
| 4 | 眨眼塌陷成"半眯眼直线" | 默认排除交给 SDK；需要时走事件通道，不走高斯扩散 |
| 5 | 归一化在近静态通道上爆炸 | `scale = max(q99-q01, 0.05×量程)`，双重下限 |
| 6 | 时间注意力窗口无法兼顾 3 帧脉冲与 90 帧转头 | 多尺度窗口（层间交替 16 / 64） |
| 7 | 日文参数名被英文编码器打成噪声 | 必须用多语言编码器（mT5 / BGE-M3） |
| 8 | 强 OOD 模型零样本崩溃 | AnyTop 证据显示退化是连续的；准备 L1 参数嵌入反演兜底 |
| 9 | 面捕数据的风格 gap 污染主模型 | 只做预训练，微调阶段必须用手工 motion |
| 10 | 块间跳变（动画里比机器人刺眼得多） | RTC（arXiv:2506.07339），已进 LeRobot，无需重训 |
| 11 | Cubism Core 闭源许可 | 个人项目 OK；若要开源产品需评估。`moc3-rs` 是唯一开源替代但不完整 |
| 12 | 解包模型的版权风险 | 个人研究自用与公开发布是两回事，提前想清楚边界 |
| 13 | 没有基线可比 | 自建三条基线（§8），否则无法证明有效 |

---

## 11. 参考文献（按阅读优先级）

**必读五篇**
1. **AnyTop: Character Animation Diffusion with Any Topology** — arXiv:2502.17327, SIGGRAPH 2025. 代码 `github.com/Anytop2025/Anytop`。**你的主干直接照抄这个**
2. **Identity-Preserving Realistic Talking Face Generation** — arXiv:2005.12318. 其 *Spontaneous Blink Generation* 小节是事件通道的第一参考
3. **CondMDI: Flexible Motion In-betweening with Diffusion Models** — arXiv:2405.11126, SIGGRAPH 2024. 关键帧生成的 mask 机制
4. **Knowledge Insulation for VLA** — arXiv:2505.23705. VLM 梯度隔离（V1 已述）
5. **Real-Time Chunking (RTC)** — arXiv:2506.07339. 块间无缝续接

**跨本体**
- URMA — arXiv:2409.06366, CoRL 2024（**token 集合 > 多头 > padding** 的直接消融）
- CrossFormer — arXiv:2408.11812（readout token 机制）
- Embedding Morphology into Transformers — arXiv:2603.00182, MERL 2026
- MetaMorph — arXiv:2203.11931, ICLR 2022；Universal Morphology Control via Contextual Modulation — arXiv:2302.11070
- SAME: Skeleton-Agnostic Motion Embedding — SIGGRAPH Asia 2023, DOI 10.1145/3610548.3618206（随机关节遮蔽增强）
- Skeleton-Aware Networks for Deep Motion Retargeting — arXiv:2005.05732, SIGGRAPH 2020
- UniTalker — arXiv:2408.00762, ECCV 2024（PCA 多头，异构标注统一）
- RDT-1B — arXiv:2410.07864（128 维统一动作空间，附录 C 表 4）

**关键帧与曲线**
- sMDM: Less is More — arXiv:2503.13859（VW 简化 + Lipschitz MLP）
- DanceNet3D — arXiv:2103.10206（关键姿态 + 运动曲线，工业动机）
- Robust Motion In-betweening — arXiv:2102.04942, SIGGRAPH 2020（time-to-arrival embedding）
- AutoKeyframe — SIGGRAPH 2025, DOI 10.1145/3721238.3730664

**本体辨识**
- Active Embodiment Identification with RL for Legged Robots — arXiv:2605.08020（主动探测学本体表征）
- UPESI — arXiv:2109.13438

**眨眼与面部**
- DFA-NeRF — arXiv:2201.00785；CP-EB — arXiv:2311.08673；MoDiT — arXiv:2412.09296；VASA-1 — arXiv:2404.10667（反方证据）

**2026 新工作（编号较新，引用前建议二次核实）**
- SkelMo: Universal Skeletal Motion Generation for 3D Rigged Shapes — arXiv:2606.01518（2D 视频引导 + 异构骨骼，与你的组合最贴近）
- RigMo: Unifying Rig and Motion Learning — arXiv:2601.06378（从形变数据自监督学 rig 语义）

**Live2D 官方规范**
- CubismSpecs（JSON Schema 权威）— `github.com/Live2D/CubismSpecs/tree/master/FileFormats`
- 标准参数列表 — `docs.live2d.com/en/cubism-editor-manual/standard-parameter-list/`
- LipSync / MotionSync — `docs.live2d.com/en/cubism-sdk-manual/lipsync`
- Drawable 顶点校验 — `docs.live2d.com/4.2/zh-CHS/cubism-sdk-manual/drawablevertexposition-checking`

**现有实践（找天花板）**
- Textoon（文本→Live2D 外观）— arXiv:2501.10020
- `shinshin86/live2d-add-motion-sample-web-ui`（规则+Agent 生成 motion3.json，当前工业天花板）
- `nanlingyin/SoulLink_Live2D`（LLM 直出参数值，0.5fps + 插值）
