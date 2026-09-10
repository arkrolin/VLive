# VLA / 具身智能模型深度调研 —— 兼论「VLM + 扩散」驱动 2D 机体连续动作参数生成

> 调研截止：2026 年 8 月
> 目标读者：准备构建「自然语言指令 → 2D 机体连续动作参数序列」生成模型的研发者
> 证据标注约定：**〔一手〕**= 论文正文/官方技术报告可核；**〔自报〕**= 项目方自述、无第三方复现；**〔未确认〕**= 仅见于二手来源

---

## 0. 结论先行（TL;DR）

如果你只读一页，读这一页。

**（1）VLA 三年演进的净结论**：真正带来性能跃迁的不是"用了扩散"，而是三件事的组合 —— **动作分块（action chunking）+ 并行/一次性产出 + 连续动作表示**。「扩散 vs L1 回归」的争论，在窄分布基准上是平局，在**多模态动作分布**上扩散/流匹配明显占优。你的场景（同一句指令对应多种合理动作）**天然是多模态的，所以应该选流匹配，而不是 L1**。

**（2）动作头形式的行业收敛点**：2025 下半年至今，工业界（π0/π0.5、GR00T N1.x、GR-3、SmolVLA、GO-1、FLOWER）几乎一致选择 **rectified flow matching + DiT 类动作专家**，而非 DDPM。原因很实际：线性路径、4~10 步 Euler 就能收敛、不需要复杂噪声调度、也不需要额外蒸馏。

**（3）VLM 与动作头的接口是最关键的工程决策**。有六种范式（见 §2.4），对你最有价值的是两条：
- **π0 式权重分离 action expert**（同一 Transformer、两组权重、blockwise causal attention），配合 **Knowledge Insulation 的 attention stop-gradient** —— 这是目前最成熟、有开源实现（openpi）、有配套实时方案（RTC）的路线；
- **FLOWER 式中间层融合**（取 VLM 中间层 hidden state → Linear+RMSNorm → cross-attention 注入 flow transformer，砍掉 VLM 后 30~50% 层），训练成本低一个数量级，且有明确消融证据：**中间层融合 93.4% vs 早融合 33.4% vs 晚融合 73%**。

**（4）跨领域最关键的发现**：机器人 VLA 这条线解决的是「架构与接口」，而 **文本→动作生成（Text-to-Motion）+ blendshape 系数扩散** 这条线解决的才是你真正的问题形态 —— **低维连续参数序列（几十~几百维 × 几十~几百帧）的生成**。这两条线在 2026 年已经出现合流：**EgoMotion（arXiv:2604.19105）** 明确提出「VLM 出离散运动基元 → 冻结/断梯度 → 扩散出连续参数」的两阶段范式，几乎可以原样搬到 2D 机体上。

**（5）你的领域存在真实的研究空白**：多轮检索**未找到任何直接生成 Live2D `.motion3.json` / Spine 骨骼动画参数曲线的同行评审论文**。当前工程实践（如 Open-LLM-VTuber）的天花板仍是「LLM 输出情绪标签 → 映射到预制动作索引」的**离散查表**。这既是风险（无基线、无数据集、无 benchmark），也是你的机会窗口。

**（6）最短落地路径**（详见 §4）：
```
表示：分部位分组 + 分位数归一化 + 根变换用增量  →  连续向量（首选）
架构：Qwen2.5-VL-3B（冻结主体）+ 0.3B Flow Action Expert（DiT）
接口：blockwise causal attention + KI stop-gradient + Global AdaLN-Zero
损失：L_flow + 速度正则 + jerk 正则 + 锚点约束 + 文本 next-token（保语义）
推理：4~10 步 Euler + RTC inpainting 做块间无缝续接
```

---

# 第一部分：VLA 与具身智能模型全景（2022 → 2026）

## 1.1 三条演进主线

VLA 的发展可以拆成三条并行主线，理解它们的分歧点比记住模型名字重要得多：

| 主线 | 核心主张 | 代表 | 现状 |
|---|---|---|---|
| **A. 离散动作 token + 自回归** | 把动作离散化，复用 LLM 词表和 next-token 目标 | RT-1 → RT-2 → OpenVLA → π0-FAST → SpatialVLA | 训练友好、语义保持好，但**推理慢、长程退化**；已退居"预训练阶段"角色 |
| **B. 连续动作 + 扩散/流匹配专家** | 动作是连续多模态分布，用生成模型建模 | Octo → Diffusion Policy → RDT-1B → π0/π0.5 → GR00T N1.x → GR-3 → FLOWER | **当前工业主流** |
| **C. 连续动作 + 直接回归** | 大主干 + chunking 已足够，扩散是过度设计 | OpenVLA-OFT / VLA-Adapter | 速度最快，但**牺牲多模态建模** |

三条线在 2025 年出现融合：**π0.5 的做法最值得抄** —— 预训练阶段用离散 FAST token 走 next-token 目标（训练快、能和文本/bbox 共享损失、语言跟随好），后训练/部署阶段启用 flow matching action expert 输出连续动作，靠 attention mask 隔离两种表示。

## 1.2 代表模型逐条拆解

### RT-1（2022.12，Google，arXiv:2212.06817）〔一手〕
- **创新**：首次证明"连续控制量离散成 token + Transformer 自回归"可行，是后续所有自回归 VLA 的地基。
- **架构**：FiLM 条件化 EfficientNet-B3 + TokenLearner 压缩 + Transformer 解码器，约 35M 参数；每个动作维度离散为 **256 bin**；单步动作、约 3Hz。
- **数据**：13 台机器人、17 个月、约 13 万条演示 / 700+ 任务。

### RT-2（2023.07，Google DeepMind，arXiv:2307.15818）〔一手〕
- **创新一**：直接把预训练 VLM 变成策略，动作 token **复用 VLM 词表中最少用的 256 个 token**。
- **创新二**：**co-fine-tuning** —— 训练时按比例混合网络 VQA/caption 数据与机器人数据（RT-2-PaLI-X 机器人数据占 50%，RT-2-PaLM-E 占 66%）。这是"不让动作微调冲掉语义"的第一个系统性答案，后续所有工作都在沿用。
- **结果**：未见物体成功率 62% vs RT-1 的 32%；出现符号推理、多语言等涌现能力；**在动作 token 前插入 reasoning token 能进一步提升 —— 这是 ECoT 的前身**。
- **代价**：55B 参数、约 1Hz 推理，几乎不可实时。

### Octo（2024.05，UC Berkeley 等，RSS 2024，arXiv:2405.12213）〔一手〕
- **创新**：开源、**模块化输入输出** —— block-wise attention + readout token，使新观测（腕部相机、本体状态）或新动作头能在微调时"插拔"，不必重训。
- **架构**：T5-base 编码语言，Transformer 主干 27M(S)/93M(B)；**扩散动作头（DDPM）输出 chunk=4 的动作块**；OXE 约 80 万条轨迹。
- **意义**：确立了"Transformer 主干 + 扩散头 + 动作分块"的开源基线。

### OpenVLA（2024.06，Stanford/TRI/UW，arXiv:2406.09246）〔一手〕
- **架构**：Prismatic-7B = Llama-2-7B + **DINOv2 + SigLIP 双视觉编码器融合** + 2 层 MLP projector；动作离散 256 bin，**bin 边界用 1%–99% 分位数**（而非 min-max，抗离群值）。
- **关键消融（对你有直接价值）**：
  - LoRA(r=32) 只训 1.4% 参数即达到全量微调水平（68.2% vs 69.7%），显存 60GB vs 163GB；
  - **只训最后一层会崩（30.3%）**；
  - **冻结视觉编码器明显掉点（47.0% vs 69.7%）** —— 与 VLM 常规做法相反，机器人任务必须微调视觉塔；
  - 训练 27 个 epoch（远超 LLM 的 1-2 epoch）。
- **痛点**：自回归逐维解码 → 4.2Hz。

### OpenVLA-OFT / OFT+（2025.02，arXiv:2502.19645，RSS 2025）〔一手〕
**这是整份报告里对设计决策最有用的一篇消融**，它把三个正交维度拆开做对照：

| 变体 | LIBERO 均值 | LIBERO-Long | 吞吐 | 延迟 |
|---|---|---|---|---|
| OpenVLA 原始（自回归 256-bin） | 76.5% | 53.7% | 4.2 Hz | 0.240 s |
| 并行解码 + chunk + 连续 + **L1** | **95.3%** | 90.7% | **109.7 Hz** | 0.073 s |
| 并行解码 + chunk + 连续 + **扩散(50步)** | 95.4% | 91.1% | 4.2 Hz | 1.907 s |

- 并行解码 + chunk（K=8）→ 绝对 +14%；连续表示比离散再 +5%；L1 与扩散**成功率打平**但快 26×。
- DDIM 减步：50 步 91.1% → 5 步 90.0% → **1 步 0.0%**。
- **FiLM 的必要性**：ALOHA 多视角（含腕部相机）下不加 FiLM，语言跟随退化到随机水平（33%）。做法是对指令 embedding 取均值 → 仿射投影出 γ、β，在 SigLIP/DINOv2 每个 block 的 self-attn 之后、FFN 之前做 `F̂=(1+γ)⊙F+β`，**空间无关调制**（逐 patch 调制反而更弱），γ/β 近零初始化。
- **作者自己的告诫（务必记住）**：他们**不主张 L1 普遍优于扩散**。若训练集动作分布真的多模态，L1 会收敛到"中位数模态"。他们的解释是现实相机噪声让确定性策略也表现出类多模态行为，且 LIBERO 分布较窄。**这个前提在动画生成上不成立。**

### π0（2024.10，Physical Intelligence，arXiv:2410.24164，RSS 2025）〔一手〕
**这是接口设计的参考实现，值得逐条抄。**

- **架构**：PaliGemma 3B（SigLIP-So400m + Gemma-2B）+ **独立参数化的 300M action expert**（width 1024 / mlp_dim 4096，从零初始化），总计约 3.3B。
- **目标函数（条件流匹配）**：
  ```
  A_t^τ = τ·A_t + (1-τ)·ε,   ε ~ N(0, I)
  L = E ‖ v_θ(A_t^τ, o_t) − (ε − A_t) ‖²
  τ ~ Beta((s−τ)/s; 1.5, 1),  s = 0.999      # 偏重高噪声端
  ```
  时间步用 **Beta(1.5,1)** 而非均匀分布，因为动作维度低、低噪声端太容易学。
- **权重分离**：单一 Transformer，**两组权重（两个 expert）**，token 按类型路由，**只在 self-attention 层交互**。作者试过共享权重（Transfusion 思路），发现机器人 token 用独立权重更好。
- **注意力规则（blockwise causal，三块）**：
  ```
  [图像 + 语言 tokens]  →  [state q_t]  →  [噪声动作 a^τ_{t..t+H-1}]
       块内全双向             块内双向          块内双向
       不看后续块          只看块1          可看块1、块2
  ```
  设计动机很具体：块 1 不看后续以**减少 VLM 分布漂移**；块 2 不看块 3 使 **state 的 K/V 可缓存**；采样 10 步时只需重算动作后缀。
- **超参**：H=50（1 秒 @50Hz）；推理 **10 步 Euler，δ=0.1**；4090 单机延迟 **73 ms**（图像编码 14ms + 观测前向 32ms + 10 步动作前向 27ms）。
- **归一化**：openpi 仓库确认使用 `norm_stats.json` 的 **q01/q99 分位 + std**，并在 troubleshooting 中明确警告：**某些维度使用率低会导致 q01/q99 或 std 极小 → 归一化后数值爆炸 → loss 发散**。这个坑在你的场景（很多 Live2D 参数常年不动）**极易踩到**。

### Knowledge Insulation（2025.05，arXiv:2505.23705）〔一手〕
**接口设计上最重要的一篇，必读。**

- **诊断**：给预训练 VLM 外挂随机初始化的连续动作专家，其**梯度会污染主干的预训练表征** —— 训练慢、语言跟随与泛化变差；但完全冻结主干也不行。
- **解法三件套**：
  1. **注意力层里的 stop-gradient**：注意力块矩阵写成
     ```
     P_ab = softmax( Q_a(X_a) · sg(K_b(X_b))^T + A )
     out_a = P_ab · sg(V_b(X_b)) + P_aa · V_a(X_a)
     ```
     **前向信息照常从主干流向专家，反向梯度不回写主干**。
  2. **主干改用 FAST 离散 token 做 next-token 预测**，保证主干激活里确实含有足够的动作信息；注意力掩码强制离散 FAST token 与连续动作 token 互不可见。
  3. **与通用 VL 数据 co-training**。
- **副产品**：因为梯度被隔离，**流匹配损失权重 α 可以直接取 1**，不用小心调小。

### π0-FAST（2025.01，arXiv:2501.09747）〔一手〕
- **FAST tokenizer**：先对动作块做 **DCT（离散余弦变换）**，再做 BPE 压缩，把高频动作块压成很短的离散 token 序列，使自回归 VLA 也能训高频灵巧任务。同时发布通用 tokenizer FAST+。
- **对你的启发**：DCT 是把「时序连续曲线」压成「少量低频系数」的经典手段，**对 2D 动画参数曲线同样适用**（动画曲线本身就以低频为主）。

### π0.5（2025.04，CoRL 2025，arXiv:2504.16054）〔一手〕
- **离散+连续双目标混合训练**：`L = L_CE + λ·L_FM`。预训练阶段动作用 FAST 离散 token 走 next-token（还包括文本、子任务、bbox），后训练/推理阶段启用 flow matching action expert（300M）输出连续动作块，靠 attention mask 隔离。
- **分层推理**：同一个模型先自回归输出高层语言子任务 ŝ（"pick up the plate"），再以 (o_t, l, ŝ) 为条件生成 50 步连续动作块。**单模型两段式**。
- **数据消融（OOD 评测，follow rate / success rate）**：
  | 配置 | follow | success |
  |---|---|---|
  | 完整 | 94% | 94% |
  | 去掉网络多模态数据 | 80% | 74% |
  | 去掉跨本体数据 | 67% | 49% |
  | 去掉多环境机器人数据 | 33% | 31% |

  **结论：泛化不是靠单一数据源，每一类数据补的是不同的洞。** 其中"去掉网络数据掉 20 点"是 co-training 保语义的最强证据之一。

### π*0.6 / RECAP（2025.11，arXiv:2511.14759）〔一手〕
- **用真实部署经验做 RL 后训练**。RECAP = 演示 + 自主 rollout + 专家介入纠错三类异构数据统一进 offline RL：训练大规模多任务**价值函数**，由它算 advantage，把**优势指示符作为 prefix 条件**注入 VLA（advantage-conditioned policy extraction），迭代。
- **结果**：最难任务上吞吐翻倍、失败率降低 2× 以上。
- **局限（作者承认）**：依赖人工标 reward 与介入；探索是贪心的；是迭代 offline 更新而非真在线 RL。

### NVIDIA GR00T N 系列〔N1/N1.5 一手，N1.6/N1.7 部分未确认〕

| 版本 | System 2（VLM） | System 1（动作头） | 关键变化 |
|---|---|---|---|
| **N1**（arXiv:2503.14734，2025.03） | Eagle-2（~1.34B），10Hz | **DiT + flow matching，120Hz**，chunk=16 | 首个开源人形双系统 VLA，总参 2.2B；数据金字塔（网络视频/仿真/真机）；**语言跟随仅 46.6%** |
| **N1.5**（2025.05-06） | Eagle 2.5，**预训练与微调全程冻结 VLM** | DiT 16 层 | 冻结 VLM → **语言跟随 46.6%→93.3%，总成功率 43.3%→83.0%**；新增 FLARE（从动作去噪网络隐状态预测未来观测表示，解锁无动作标签人类视频）与 DreamGen（视频世界模型造合成轨迹） |
| **N1.6**（2025.12，官方页） | Cosmos-2B VLM 变体 | **DiT 16→32 层** | **移除 VLM↔DiT 之间的 4 层 adapter，改为预训练时解冻 VLM 顶部 4 层**；改为预测 **state-relative 动作块**（更平滑更准，但小数据易误差累积） |
| **N1.7**（2026.03 GTC） | Cosmos-Reason2-2B | 32 层 DiT | **EgoScale：20,854 小时人类第一视角视频**预训练；报告"灵巧性 scaling law"（人类视频 1k→20k 小时，后训练平均任务完成度翻倍以上）〔自报〕 |

**N1.5 → N1.6 这两次反向操作极值得玩味**：先"全冻结 VLM"保语义，再"只解冻顶部 4 层、删掉 adapter"换取跨模态特征损耗更低。**说明「冻结」不是终点，而是数据量不足时的正则手段** —— 这条经验对你（小数据起步）尤其重要。

### Figure Helix（2025.02，公司博客）〔官方博客〕
- **S2**：开源 VLM，7B 级，7-9Hz；**S1**：约 80M transformer visuomotor policy，200Hz 输出连续动作。
- **通信接口（关键）**：S2 产生的**潜在向量投影到 S1 的 token 空间，并在序列维度上与 S1 视觉特征拼接**；**梯度可从 S1 反传到 S2**，端到端联合优化。
- **两个工程技巧（对你直接可用）**：
  1. 训练时**人为插入 S1/S2 时延**，校准到部署时的推理延迟差，消除 train/inference 分布差；
  2. 动作空间里附加一个合成的**"任务完成百分比"维度**，让模型能预测自己的终止条件，便于行为串联 —— **这是动画序列"什么时候结束"的现成解法**。
- **Sport Mode**：把 `[T × action_dim]` 的动作块**线性重采样成 0.8T 再按原频率播放 → 直接提速 20%**。极便宜的速度控制手段，对动画的"快放/慢放"需求是白送的。

### 智元 GO-1 / ViLLA（2025.03 发布，2025.09 开源，arXiv:2503.06669）〔一手 + 官方〕
- **ViLLA = Vision-Language-Latent-Action**：在 VLM 与动作之间插入 **latent action tokens** 作为中间语义层。结构 = InternVL-2B + MoE，MoE 含 **Latent Planner**（预测 latent action token，作为 Chain-of-Planning）与 **Action Expert**（扩散生成高频连续动作）。
- **动机**：低层动作数据太少、跨本体动作空间不可迁移 → 用 latent 动作空间吸收人类视频与跨本体数据。
- **消融**：加 Latent Planner 复杂任务平均完成分 +0.12；人工校验数据比未校验 +0.18。

### 字节 GR-3（2025.07，arXiv:2507.15493）〔一手〕
- **架构**：mixture-of-transformers；Qwen2.5-VL-3B-Instruct + **action DiT（flow matching）**，总 4B；动作块表示为 k 个 token 与 state token 拼接进 DiT，flow 时间步用 **AdaLN** 注入，DiT 内用**因果注意力**建模块内时序；**DiT 层数是 VLM 的一半，且只用 VLM 后半段层的 KV cache**。
- **稳定性发现（极实用）**：在 DiT 的 attention 与 FFN 线性层后**额外加 RMSNorm**（受 QK-norm 启发），"drastically" 改善训练稳定性，**并显著提升下游语言跟随能力**。
- 每个新物体只需 **10 条人类 VR 轨迹**即可适配。

### RoboVLMs（arXiv:2412.14058，2025 发表于 Nature Machine Intelligence）〔一手〕
**唯一一篇大规模对照实验的"VLA 设计空间指南"**：8 个 VLM 主干 × 4 种策略架构 × 600+ 组实验。

主要结论：
1. 从大规模 VLM 微调的 VLA 在鲁棒性、泛化、数据效率上优于其他通才策略范式；
2. **"policy head + 连续动作"是最佳结构组合**，且趋势在多个主干上一致复现；
3. **离散动作在长程任务上严重退化**；
4. 其"in-domain 数据比跨本体数据更有效"的结论与 π0.5 的跨本体消融**存在张力**，属当前分歧点。

### 2025-2026 的四个新方向

**(a) Latent action pretraining**
- **LAPA（arXiv:2410.11758，ICLR 2025）**：用 VQ-VAE 从**无动作标签视频**里量化出 latent action（latent 序列长 4、码本 8，即动作空间 8⁴），先用 latent action 预训练，再换真实动作头微调。**预训练效率：272 H100-hours vs OpenVLA 的 21,500 A100-hours，约 30-40× 更省，且性能反超。**
- **对你的意义极大**：你缺的正是「带精确参数标注的指令-动作对」，但**不缺无标注的 2D 动画视频**。LAPA 路线让你能用这些视频做预训练。

**(b) World-model 增强的 VLA**
- **V-JEPA 2 / V-JEPA 2-AC（arXiv:2506.09985）**：阶段一在 100 万+ 小时互联网视频上做掩码**潜空间**特征预测（不生成像素）；阶段二**冻结编码器**，只训 300M 的动作条件预测器，**仅用 62 小时无标注机器人视频**；部署时在潜空间做 MPC。
- **WorldVLA（arXiv:2506.21539）**：动作、图像、文本用三个 tokenizer 统一到同一词表，单模型同时做世界模型与动作模型，互相增益。论文指出自回归连续生成动作序列会受**前序错误动作传染**，需专门的注意力掩码策略。

**(c) RL 后训练**
- **SimpleVLA-RL（ICLR 2026）**：作用于 OpenVLA-OFT，LIBERO 91.0→**99.1**；纯仿真训练迁移到真机 17.5%→38.5%；**单条轨迹 SFT(48.9) + RL → 96.9，超过全量演示 SFT(91.0)**。
- **关键失败模式**：若 SFT 初始成功率为 0，RL 也是 0 —— outcome-reward RL 需要"能偶尔成功"的门槛。
- **RIPT-VLA（arXiv:2505.17016）**：稀疏二值成功奖励；把 OpenVLA-OFT 推到 97.5%；1 条演示 SFT（4%）在 15 次迭代内到 97%。

**(d) 具身思维链**
- **ECoT（arXiv:2407.08693）**：训练 VLA 在动作前生成 `task → plan → subtask → move primitive → gripper 像素位置 → 物体 bbox` 的 **visually grounded** 推理链。OpenVLA 绝对成功率 **+28%（66% vs 44%）**。
- **决定性消融**：只做语义 CoT（task→subtask）仅 48%，几乎无收益；**加上 bbox + 夹爪像素 + 低层运动原语才到 66%**。→ **"grounded" 才是关键，而非"更长的思维链"**。这条对你的启示：如果要让 VLM 先"想"，想的内容必须是**可落到参数空间的具体量**（如"左手抬到画布 (0.3, 0.2)"），而不是抽象描述。

**(e) 轻量化 VLA**
- **SmolVLA（arXiv:2506.01844，450M）**：SmolVLM-2 主干；**pixel-shuffle 到每帧仅 64 个视觉 token**；**只用 VLM 前 N=L/2 层的特征**喂动作专家（省一半算力）；动作专家 = **交错的 cross-attention（看 VLM 特征）与因果 self-attention** 层，hidden dim 取 VLM 的 0.75×，flow matching；仅 481 个社区数据集、约 23k episodes 预训练。配套**异步推理栈**（PolicyServer / RobotClient 解耦，队列低于阈值才触发新块预测）。
- **VLA-Adapter**：0.5B 主干 + 97M policy，用 **Bridge Attention（可学习门控 g，tanh(g) 控制原始 VLM 特征注入强度，初始化为 0）** 把 VLM 多层特征注入策略；**即使完全冻结 VLM 主干仍有 86.4%**〔自报〕。
- 注意 **GR-3 用后半段层、SmolVLA 用前半段层，结论相反** —— 说明这仍是经验选择，需要在你的数据上实测。

## 1.3 VLA 设计空间：共识与分歧

### 共识（可以直接照抄）
1. **动作分块是必选项**。典型长度：Octo 4；OpenVLA-OFT 8（LIBERO）/25（ALOHA 25Hz）；GR00T N1 16；π0/π0.5 **50 步 = 1 秒 @50Hz**。经验规律：**块 ≈ 0.5-1 秒的行为片段**。
2. **连续表示优于 256-bin 离散**（OFT：+5% 绝对值），差距来源是数值精度。
3. **归一化用 1%-99% 分位数**，不用 min-max（抗离群）。
4. **必须做 co-training 保语义**。π0.5 消融：去掉网络多模态数据，OOD 成功率 94%→74%。
5. **块边界必须处理**。开环执行整块会在块交界处跳变，需 RTC / temporal ensembling / 训练期时延对齐。

### 分歧（需要在你的数据上实测）
1. **L1 vs flow matching**：OFT 说打平，π/GR00T/GR-3/SmolVLA/GO-1/FLOWER 全选 flow。判据应是「你的数据里同一条件下是否真存在多模态动作」。
2. **用 VLM 的哪一半层**：GR-3 用后半段 KV cache，SmolVLA 用前半段，FLOWER 用中间层且砍掉后 30-50%。
3. **冻结程度**：GR00T N1.5 全冻（语言跟随 46.6→93.3），N1.6 改为解冻顶 4 层；OpenVLA 发现**必须微调视觉编码器**（冻结掉 22 个点）。→ 折中结论：**冻语言塔、放视觉塔**。
4. **相对 vs 绝对动作**：GR00T N1.6 改为 state-relative（更平滑），但**小数据下相对动作易误差累积、纠错能力下降**。
5. **跨本体适配四法**：① 统一低层空间 + zero-pad（π0 的 18 维）；② 每本体独立编解码头 + 共享主干（GR00T CategorySpecificMLP）；③ latent action 共享空间（LAPA/ViLLA）；④ Motion Transfer（Gemini 1.5）。

---

# 第二部分：扩散 / 流匹配动作头深潜

## 2.1 Diffusion Policy —— 所有动作扩散头的模板（arXiv:2303.04137，RSS 2023）〔一手〕

**Receding horizon control**：三个视界 `To`（观测）/ `Tp`（预测）/ `Ta`（执行）。每次预测 `Tp` 步、只执行 `Ta` 步后重规划。**消融显示 Ta=8 在多数任务最优** —— 太长损失反应性，太短则出现 chunk 间抖动。

**两种骨干**：
- **CNN-UNet 版**：1D 时间卷积，**FiLM 把观测特征和去噪步 k 同时注入每个卷积层的通道维**（scale-shift）。开箱即用、超参不敏感；但有**低频归纳偏置，对高频/速度型动作会过平滑**。
- **Transformer 版**：噪声动作作为 token 序列，扩散步 k 的正弦嵌入作为首 token，观测经 MLP 成 token 后**在每个 block 用 cross-attention 注入**。缓解过平滑，但超参更敏感。
- **官方建议：新任务先用 CNN 版，性能不足再换 Transformer 版。**

> 对 2D 动画的取舍：动画参数曲线**以低频为主**，CNN-UNet 的低频偏置反而是优势；但如果需要打斗、快速抖动等高频动作，必须上 Transformer。建议**双头对比实验**。

**为什么能建模多模态**：扩散学的是条件动作分布的 score，两处随机性协同 —— 从 N(0,I) 随机初始化决定收敛到哪个 basin；迭代中注入的高斯扰动使样本可在模态间迁移。Push-T 中它能学到"左绕/右绕"两种模式并**单次 commit 一种**（而不是取平均）。

**步数**：DDPM 训练 100 步，DDIM 推理 10 步，3080 上约 0.1s。噪声调度用 **iDDPM 的 square-cosine**（实测优于 linear/cosine）。**位置控制显著优于速度控制**。

## 2.2 少步化 / 一步化

| 方法 | 机制 | 关键数字 |
|---|---|---|
| **DP3**（arXiv:2403.03954） | 点云 → FPS 采样 → MLP+MaxPool+LayerNorm → **64 维紧凑 3D 特征**作条件 | 72 任务，多数仅 10 条示教，相对提升 24.2% |
| **ManiCM**（arXiv:2406.01586） | 对 DP3 做**一致性蒸馏**，且**直接预测动作样本而非噪声**（低维动作流形收敛更快） | 1 步 0.05s/80% vs DP3 0.15s/60%，约 10× 加速 |
| **OneDP**（arXiv:2410.21257，ICML 2025） | 反向 KL / score-difference 蒸馏（DMD 思路） | 1 NFE；动作频率 **1.5Hz → 62Hz**；真机 0.98 vs DP-DDIM(10步) 0.83 |
| **Flow2One**（arXiv:2603.09415，2026.03） | **IMLE + 双向 Chamfer 距离**做分布级蒸馏，显式对抗单步蒸馏的 mode collapse | RLBench 单步 68.6% @123.5Hz（教师 50 步 74.1%）；比 2.9Hz 教师快 43× |

**趋势判断**：2024 年是"多步扩散 + 蒸馏"，2025-2026 主流是**直接用 rectified flow 训练，让 4-10 步 Euler 就够用**，蒸馏退化为可选项。一步化的核心风险是**模态坍塌**，最新工作都在用集合级损失显式保模态。

> **对你的建议：直接从 rectified flow 起步，不要走 DDPM + 蒸馏的老路。** 4-10 步在 30fps 动画上完全够用（每块 1 秒的预算下，10 步去噪只占几十毫秒）。

## 2.3 RDT-1B 的三处 DiT 改造（arXiv:2410.07864，ICLR 2025）〔一手〕

这三条是把 DiT 用在**物理量数值范围不稳**的动作上时的必备补丁，**你会遇到完全一样的问题**（Live2D 参数有的是 [-30,30] 角度，有的是 [0,1] 开合度）：

1. **QKNorm** —— 防注意力溢出；
2. **RMSNorm 替换 LayerNorm** —— 去 centering，防 token shift / attention shift，**去掉会训练爆炸**；
3. **MLP decoder 替换线性 decoder** —— 去掉则精细任务做不了。

**ACI（Alternating Condition Injection）**：图像/文本 token 数量差异大，若每层同时 cross-attend 会让**图像 token 淹没文本**，故**在相邻层间交替注入图像与文本 token**，实测显著改善指令跟随。

**编码器全冻结**（SigLIP + T5-XXL）；低维输入（本体状态、动作块、控制频率）用**带 Fourier 特征的 MLP** 编码后作为去噪网络的**直接输入**（不是 cross-attn 条件）。训练时对各模态做**随机独立 masking 防走捷径**。

推理用 **DPM-Solver++ 把 100 步降到 5 步**，4090 上 6Hz/chunk、平均 381Hz/action。消融：去掉扩散建模改回归，明显更差。

## 2.4 六种接口范式对比（核心决策表）

| 范式 | 代表 | 条件注入 | 参数开销 | 训练稳定性 | 伤 VLM 语义? | 延迟 | 可扩展性 |
|---|---|---|---|---|---|---|---|
| **A. 后置浅头**（单条件向量 → MLP/小 DiT） | CogACT、TinyVLA、Diffusion-VLA、DexVLA、Octo | cognition token / 最后层 hidden（reasoning 走 FiLM） | 最低（3M-1B） | 高，可分阶段单独预训 head | 中：梯度仍回流主干 | 低 | 一般：**信息瓶颈在单个向量** |
| **B. 独立 DiT + cross-attention** | RDT-1B、GR00T N1/N1.5 | 每层（或交替层）cross-attend 图文 token | 中高（0.3-1.2B） | 需 QKNorm+RMSNorm | **低**：VLM 可整体冻结 | 中 | 高：条件粒度细、支持异构动作空间 |
| **C. 权重分离 action expert（MoE 式）** | **π0 / π0.5 / KI / SmolVLA** | 同一 Transformer、两组权重、blockwise causal attention | +10%（3B+300M） | 中→高：**朴素做法会污染主干**，需 KI 的 stop-gradient | 默认会伤；**加 sg 后不伤** | 低（73ms） | 高：K/V 可缓存，工业主流 |
| **D. In-context 单序列** | Dita、HybridVLA | 图文与噪声动作 token 拼在同一因果序列 | 无额外 head | 中：需 marker token + 掩码隔离 | 高风险：动作梯度直写主干 | 偏高（每去噪步跑全主干） | 最好但算力贵 |
| **E. 中间层融合** | **FLOWER** | VLM **中间层** hidden → Linear+RMSNorm → cross-attn；Global-AdaLN-Zero 注入时间步与本体类型 | 950M 全模型 | 高（零初始化 AdaLN） | 低（砍掉后 30-50% 层） | 最低 | 好且**极省算力**（200 H100 小时预训练） |
| **F. 无扩散：并行解码 + L1** | OpenVLA-OFT | 最后层 hidden → 4 层 MLP 直出 chunk | 极低 | 最高 | 低（需 FiLM 保语言接地） | 最低（109.7Hz） | **牺牲多模态建模** |

### 关键消融证据汇总

- **CogACT（arXiv:2411.19650）** —— 同参数下 **DiT ≫ MLP**：
  | 头 | 参数 | 成功率 |
  |---|---|---|
  | MLP-3层 | 3M | 50.6 |
  | MLP-7层 | 89M | 52.5 |
  | DiT-S | 13M | 58.5 |
  | DiT-B | 89M | **62.5** |
  | DiT-L | 308M | **64.8** |

  **同为 89M，DiT 比 MLP 高 10 个点。** chunk N=15 最优；DDIM 10 步，**CFG=1.5**。
  条件注入方式：在 LLM 序列里插入**可学习 cognition token `c`**，取其输出特征作为 DiT 的条件 token，**去噪步 i 的正弦嵌入加到该特征上**。

- **Dita（arXiv:2503.19757）** —— in-context 全序列去噪 vs "同主干 + 3 层 MLP 扩散头"：CALVIN ABC→D **3.61 vs 3.16**；ManiSkill2 **65.8% vs 58.6%**。DDPM 训练 1000 步，DDIM 评测 20 步；**降到 10 步无损（85.3 vs 85.5），2 步仍有 70.4%**。

- **FLOWER（arXiv:2509.04996，CoRL 2025）** —— 最完整的一组消融：
  | 配置 | CALVIN ABC 分数 |
  |---|---|
  | 完整 | **4.44** |
  | 标准逐层 AdaLN（而非 Global-AdaLN） | 4.43（即 Global-AdaLN 白省 20% 参数） |
  | 去掉 Flow head | 3.33 |
  | 小 head | 2.60 |
  | **冻结 VLM** | **2.65** |
  | **换离散 token** | **1.12** |

  LIBERO-Long 融合位置消融：**中间层 93.4% vs 早融合 33.4% vs 晚融合 73%**。

- **HybridVLA（arXiv:2503.10631，ICLR 2026）** —— 不外挂头，把去噪塞进 LLM 的 next-token 流，用 `<BOD>/<EOD>` 分隔。10 个 RLBench 任务：**纯 AR 62% / 纯扩散 66% / 集成 74%**。结论：**扩散擅长精细操作、AR 擅长场景语义推理**，二者互补。

- **DexVLA（arXiv:2502.05855）** —— `L = L_diff + α·L_ntp`，**α=1**；reasoning token 用 **FiLM 去 scale/shift** projection 层。三阶段 embodied curriculum：**阶段 1 先脱离 VLM 单独预训练扩散专家**，再对齐 embodiment，再 post-train。
- **Diffusion-VLA（arXiv:2412.03293）** —— 实测 `L_ntp ≈ 0.1·L_diff`，故取 **α=10**〔未确认，来自二手解读〕；自生成 reasoning 文本经 FiLM 注入 policy head（非拼接，**推理期零额外开销**）。
- **MoDE（arXiv:2412.12953，ICLR 2025）** —— DiT 内部按**噪声水平 σ_t 路由**的稀疏 MoE（非按内容路由）。因为调度确定，**专家可预先融合缓存**：激活参数 −40%、FLOPs −90%、推理约 2×。load-balancing 系数 γ≈0.01，否则专家坍塌。

## 2.5 「扩散头是否必要」之争的裁决

OFT 的结论（L1 ≈ 扩散）不能简单推广，理由有四：

1. **OFT 的"扩散"基线偏弱** —— 是 4 层 MLP 噪声预测头 + 训练 50 步 DDPM。而 Dita 与 CogACT 的消融恰好说明**浅 MLP 头就是瓶颈**（Dita 3.61 vs 3.16；CogACT 同参数 DiT-B 62.5 vs MLP-89M 52.5）。
2. **FLOWER 给了反向证据**：去掉 flow head 4.44→3.33，换离散 token →1.12，缩小 head →2.60。
3. **L1 本质给出条件均值/中位数，会抹平多模态**。LIBERO 类基准不敏感，但**人类演示天然多模态**的数据上风险大。
4. **工程上真正的分水岭是 chunk + 并行解码 + 连续表示**，而非"扩散 vs 回归"本身。

> **对你的判决**：动画生成中，「跳一下」可以是原地小跳、蓄力大跳、侧身跳……**同一指令对应多种合理动作是常态**。L1 回归会输出一个"平均的、僵硬的、幅度偏小的"动作 —— 这在动画上是致命的观感缺陷。**用 flow matching。**

## 2.6 工程细节清单（可直接落地）

**噪声调度 / 时间步**
- 首选 **rectified flow 线性路径** `x_τ = τ·x_1 + (1−τ)·ε`，4-10 步 Euler。
- 时间步采样：图像扩散常用均匀/logit-normal，**π0 用 Beta(1.5,1) 偏重高噪声端**，因为动作维度低、低噪声端太容易 —— **低维参数序列请照抄这条**。
- 若走 DDPM：用 iDDPM 的 square-cosine 调度。

**动作归一化**
- **q01/q99 分位 + std**（π0/openpi 标准）。
- ⚠️ **明确的坑**：某些维度使用率低会导致 q01/q99 或 std 极小 → 归一化后数值爆炸 → loss 发散。openpi 的 troubleshooting 专门写了这条。**你的 Live2D 参数里必然有大量常年不动的维度，上线前必须做每维统计量的异常检查与下限截断。**
- 跨本体统一空间：**zero-pad + mask**（π0 pad 到 18 维；RDT 用物理可解释的统一槽位）。

**chunk 内时间维建模**
- 1D 时间卷积 UNet（平滑、超参鲁棒、低频偏置）vs 每时间步一个 token 的 Transformer（可扩展、跨本体友好，2025 后主流）。
- **三方独立证据（Dita/CogACT/FLOWER）都指向：动作头容量与条件粒度比去噪算法本身更重要。**

**CFG**
- 动作生成中**不是标配**，收益证据弱于图像域。CogACT 用 CFG=1.5 + DDIM 10 步；π0、OFT、RDT-1B 均未报告使用。
- 更常见的替代：RDT 的**模态随机独立 masking**（防走捷径）、KI 的 co-training。
- 但在**文本→动作生成**（更接近你的场景）里 CFG 是标配：MDM/EMDM 训练时 10% 概率丢条件，百度 2D dance 用 30%；MotionLCM 把 w∈[5,15] 均匀采样**蒸馏进模型**。注意**高 guidance 会产生过饱和 artifact**（arXiv:2410.02416），在参数上表现为**打到边界值 / 表情僵硬**。

**采样期时间平滑（块间衔接）**
1. **ACT 式 temporal ensembling**（重叠 chunk 指数加权）—— π0 表示试过后**反而掉性能**〔未确认，二手转述〕。
2. **CogACT 的自适应集成 ADE**：`â_t = Σ_k w_k · a_{t|o_{t−k}}`，`w_k = exp(α · cos_sim(a_{t|o_t}, a_{t|o_{t−k}}))`，**α=0.1** —— 只对"方向一致"的历史预测加权，**避免把不同模态平均掉**。这比朴素平均聪明得多。
3. **RTC / Real-Time Chunking（arXiv:2506.07339，已进 LeRobot）** —— 当前最佳实践：
   - 把新 chunk 的前缀当 **inpainting 问题**，在流匹配去噪中加 guidance 项，让重叠时间步贴合已执行动作；
   - "保证会执行"的前 d 步**冻结**为硬约束，重叠区用**指数衰减软掩码**引导；
   - **无需重训任何 flow/diffusion 模型**；
   - LeRobot 官方默认值：`execution_horizon` 8-12、`max_guidance_weight=10.0`（对 10 步流匹配最优）、`prefix_attention_schedule=EXP`、`inference_delay=⌊δ/Δt⌋`；
   - 实测对 +200ms 注入延迟完全鲁棒，比 temporal ensembling 更平滑，动作快 20%。
   - 后续：**Training-Time RTC（arXiv:2512.05964）** 把前缀条件化搬进训练，去掉推理时的引导开销；**VLASH（arXiv:2512.01031）** 让模型以"新块真正开始执行时的未来状态"为条件。

---

# 第三部分：文本 → 连续参数序列生成（与你的目标最同构的一侧）

## 3.1 Text-to-Motion 经典基线技术切片

| 方法 | 动作表示 | 扩散空间 | 条件注入 | 关键损失 | 长度处理 |
|---|---|---|---|---|---|
| **MDM**（arXiv:2209.14916） | 原始逐帧姿态（HumanML3D 263 维） | raw | CLIP 文本嵌入作为**额外 token** 加到 encoder 输入 | **预测 x₀ 而非 ε** + L_pos(FK) + L_vel + L_foot | Transformer encoder 天然变长，padding mask |
| **MotionDiffuse** | 原始姿态 | raw | 文本 **cross-attention** | 标准 ε 预测 | 支持 body-part 级独立文本 |
| **MLD** | VAE 潜码 | **latent** | CLIP + cross-attn | 潜空间 MSE | 潜码固定长度，推理 ~0.2s |
| **MoMask**（arXiv:2312.00063） | **残差 VQ（RVQ）多层 token** | 离散 token 掩码生成 | 文本条件 Masked Transformer | 掩码交叉熵 | 迭代填充，**天然支持 temporal inpainting** |
| **MotionLCM**（arXiv:2404.19759） | MLD 潜码 | latent + 一致性蒸馏 | CFG 尺度 w∈[5,15] **蒸馏进模型** | LCD Huber + 解码到动作空间的显式控制损失 | 1-4 步，**~30ms/序列** |
| **OmniControl**（arXiv:2310.08580） | raw | raw | 文本 + **解析式空间引导 + realism guidance** | 推理时梯度引导 | 任意关节任意时刻控制 |

**MDM 的"预测 x₀"是关键设计**，因为只有直接输出干净动作，才能施加几何损失：
```
L = L_simple + λ_pos · L_pos(FK) + λ_vel · L_vel + λ_foot · L_foot
```
**这条对你必须照抄** —— 你需要在参数空间之外，对**变形后的实际关键点位置**施加损失。

**性能参考**：MoMask HumanML3D FID **0.045**（T2M-GPT 0.141）。生成耗时：MDM ~24s、MLD ~0.2s、OmniControl ~81s、MotionLCM ~30ms。EMDM（arXiv:2312.02256）明确指出 **1 步采样会退化为纯 GAN（FID 5.64），10 步是甜点**。

**评测指标（五件套）**：FID、R-Precision(Top-1/2/3)、Diversity（**越接近真实越好，不是越大越好**）、MM-Dist、MModality。长序列组合另有 FlowMDM 的 **Peak Jerk (PJ) / Area Under the Jerk (AUJ)**，专测突变过渡 —— **这两个指标对 2D 参数序列直接可用**。

## 3.2 流式 / 长序列生成（对"连续动作参数流"最关键）

**MotionStreamer（ICCV 2025）** —— **连续因果潜空间 + AR + diffusion head**：
- 1D 因果卷积 Causal TAE，时间下采样率 l=4，潜维 d_c=16；Transformer（因果 mask）输出条件特征，喂给轻量 MLP diffusion head 预测下一个潜码。
- **Two-Forward 训练策略**缓解 exposure bias：第一遍用 GT 潜码，第二遍按 cosine 调度替换为自身预测，**只回传第二遍梯度**。
- **连续停止条件**：把"不可能姿态"（全零向量）编码为参考终止潜码，生成潜码与它距离低于阈值即停 —— **自动决定长度**。
- **⚠️ 极重要的消融**：纯 AE 重建最好（MPJPE 1.7mm）但**生成最差（FID 43.8）**；Causal TAE 正则后重建 FID 0.661 / 生成 FID 11.79。**"重建好 ≠ 生成好"是低维参数序列上的真实陷阱**，潜空间必须正则。

**FlowMDM（CVPR 2024，arXiv:2402.15509）** —— **Blended Positional Encodings**：
- 去噪**早期**用绝对位置编码 + 按文本段限制的全局注意力，恢复全局连贯性；
- 去噪**晚期**切到相对位置编码 + 无限制窗口注意力，构建平滑过渡；
- 训练时随机交替两种编码。
- **Pose-Centric Cross-Attention**：条件只喂 query，使得单描述数据训练也能推理时多描述拼接。
- **无需后处理、无冗余去噪步** —— 这是"多段指令无缝拼接"最优雅的方案。

**DART / DartControl（ICLR 2025）**：扩散自回归 motion primitive，实时文本控制；局限是固定窗口 primitive。

## 3.3 规模化与数据质量的两个警示

**Being-M0 / MotionLib（ICML 2025，arXiv:2410.03311）**：首个百万级动作数据集（120 万序列 / 248 万文本，比现有大 15×）。首次验证动作生成的 scaling law：**数据 0.02M→1.2M，FID 从 ~30 降到 ~6；数据规模影响远大于模型规模**。提出 2D-LFQ（2D 无查找量化），码本从 512 扩到 65536+，利用率近 100%。

**OpenT2M（CVPR 2026，arXiv:2603.18623）** —— **最有冲击力的发现**：
- 现有 T2M 基准**训练/验证集逐字重叠 10.6%-17.0%**，清洗后多数 SOTA 崩盘。**这意味着文献中报告的 FID 提升有相当部分是过拟合假象。**
- 用**在 AMASS 上训练的 RL 跟踪策略"能否复现该动作"作为物理可行性过滤器**（很聪明的数据清洗思路，2D 上可用"能否被绑定系统真实播放且不穿模"替代）。
- 其 tokenizer **2D-PRQ**（按生物学部位分块 + 残差量化）在大数据下优势显著，但**明确警告：收益与数据规模强耦合，小数据集上 token 更多反而掉点**。

> **对你的两条硬约束**：① 自建 2D 数据集时**必须做训练/验证泄漏审计**；② **小数据起步时不要用太大的码本 / 太细的分部位量化**。

## 3.4 VLM + 扩散的分层架构（与你的目标同构度最高）

**EgoMotion（arXiv:2604.19105，2026 预印本）** —— **这篇几乎就是你要做的事的 3D 版本**：
- 明确提出 **reasoning-generation entanglement**：同时优化语义推理与运动学建模会产生**梯度冲突**。
- 解法两阶段解耦：
  1. **Cognitive Reasoning 阶段**：VLM 把多模态输入投影到**离散运动基元（motion primitive）空间**；
  2. **Motion Generation 阶段**：**冻结 / detach VLM**，把其表征作为条件送入连续潜空间的扩散生成器。
- 报告 FID 0.0018，并显式汇报 **acceleration / jerk / foot sliding / foot contact error**。
- **这套范式与 π0.5、Knowledge Insulation 的结论完全一致，三方独立验证 —— 强烈建议采纳。**

**MotiMotion（ICML 2026 预印本）**：训练无关的 VLM 作"物理推理器"，把稀疏用户轨迹补成密集控制信号，并给每条轨迹**置信度 s∈[0,1]**，高置信强约束、低置信只做粗引导。**这个"置信度加权条件"思路对"指令模糊时不要死板执行"很有价值**。

## 3.5 2D 特有的动作参数生成

### ⚠️ 一个重要的负面结论

多轮检索**未找到任何直接生成 Live2D `.motion3.json` 参数曲线 / Spine 骨骼动画参数序列的同行评审论文**。当前工程实践的天花板是**离散映射**：以 Open-LLM-VTuber 为例，LLM 在文本流中输出 `[joy]`/`[sad]` 标签，正则抽取后通过 `emotionMap` 映射到模型中**预制的表情索引**，前端播放预录动作；口型由 TTS 音频驱动。

也就是说，**"指令 → 连续参数曲线"这一段目前是真空**。Live2D Cubism 官方的"自动生成脸部动作"是几何变形器工具，不是时序生成模型。

### 最同构的替代技术线：blendshape 系数序列扩散

「几十~几百维连续低维参数 × 数十~数百帧」，这正是语音驱动面部动画在做的事，且方法成熟。

**SAiD（arXiv:2401.08655）** —— 语音 → **32 维 ARKit blendshape 系数序列**的扩散模型：
- 轻量 Transformer-based UNet1D；
- **L1 损失**（比 L2 更好保留感知距离）；
- **noise-level velocity loss** —— **直接在扩散空间惩罚高频时序抖动，无需每步解码**。这是低维参数抗抖动的最低成本方案；
- cross-attention 中加 **alignment bias**（时序邻近性偏置，等价于局部注意力掩码）保证同步；
- **支持 inpainting 式编辑**，给定二值 mask m：
  ```
  ũ_t = (1−m) ⊙ u_t + m ⊙ ( √ᾱ_t · u_ref + √(1−ᾱ_t) · ε )
  ```
  **这个公式可以一字不改地用于你的 2D 参数序列续接 / 局部重生成。**
- 数据集 BlendVOCA（VOCASET 经变形传递得到 32 blendshape，12 说话人，60fps）。

**DiffusionTalker（arXiv:2311.16565）**：在 **BEAT 数据集（ARKit 52 维 blendshape，30 人，约 32 小时，切成 11,427 条 10s 序列）** 上做扩散 + 知识蒸馏，采样步数 256 → 8 步。指标：MBE / LBE / FDD / ITF。

**PMMTalk（arXiv:2312.02781）**：32 维 blendshape + **64 维个人风格向量作为独立条件** —— 风格/内容解耦的现成做法。

### 2D pose sequence 生成的两篇关键工程论文

**X-Dancer（ICCV 2025，arXiv:2502.17414，字节）** —— **目前最值得抄的 2D 参数生成工程**：
- **组合式、置信度感知的 2D pose tokenization**：60 个全身关键点（含手、头）+ 置信度分数，按 **5 个部位（上身/下身/左手/右手/头）各自独立** 1D 卷积编码与量化，**每部位 6 个 token、512 条目码本、6 维嵌入**，量化后拼接送共享解码器重建全身。
- **动机很明确**：单一 VQ 抓不住手指/头部倾斜这类高频细节，分部位可让不同频率的动作独立表征；置信度让模型能处理运动模糊与遮挡。
- GPT-2 初始化的自回归 Transformer 预测 token，音乐条件双路注入（全局 start token 给曲风 + 逐帧嵌入给同步）。
- 训练数据 **107,546 段单目舞蹈视频**（30s 均长，896×512，30fps）。
- 推理用 **64 帧滑窗、12 帧重叠**，另取历史中均匀采样的 8 帧作为全局运动上下文。

**Reframing Music-Driven 2D Dance Pose Generation as Multi-Channel Image Generation（arXiv:2512.11720，百度）** —— **表示设计的强证据**：
- 把 2D pose 序列**重构成多通道图像**用 DiT 生成。放弃 raw 坐标回归，改用 **SimCC 式 one-hot 分箱**（x、y 各离散化为长度 W=512 的稀疏向量，热值处填置信度），K 个关键点堆成 C=2K 通道，T 帧堆成 C×W×T 张量。
- 冻结图像 VAE 8× 压缩、Lumina（2.6B）DiT 从头训练、JukeBox 音乐 token 做 cross-attention、**CFG 训练时 30% 概率丢条件**。
- **⚠️ 关键消融：raw 坐标模型有明显 jitter，one-hot 变体显著更平滑。** —— 这对你选表示是强证据。
- 长序列用 **256 帧段、16 帧重叠**拼接；首段用 shape-only 参考姿态，后续段用**前段末 16 帧作 pose-aware 参考**。
- **time-shared temporal indexing**：把音乐编码器 hop size 调到 token 长度等于潜时间维，让姿态潜变量 (0,w,t) 与音乐 token (0,0,t) **共享时间坐标 t**，节拍对齐 BAS 提升 14.6%。
- 指标：2D 关键点 kinetic feature 上的 FID、DIV、BAS + 渲染后 FVD/FID-VID + 人工 pairwise win rate。

**Text-to-Skeleton Cascades（arXiv:2603.08028，2026 预印本）**：**自回归 text-to-2D-skeleton**（逐关节、条件于已生成姿态）+ pose-conditioned 视频扩散。在 Motion-X Fitness 上用 FID / R-precision / diversity 评估 **2D 骨架生成** —— 说明 3D 那套指标可以直接搬到 2D。

## 3.6 可用的 2D 动画数据集

| 数据集 | 规模 | 标注 | 对你的价值 |
|---|---|---|---|
| **AniDiffusion**（含于 arXiv:2503.15586） | 135 个卡通角色，Adobe Character Animator 生成 | **逐帧精确关键点 + alpha matte，Motion Library 含 140+ 动作**，关节定义与 OpenPose 对齐 | ⭐ 最契合。且其**"程序化枚举关节角度组合"**思路可直接用于合成训练数据。母论文本身是"任意拓扑角色的扩散式自动绑定"，**支持昆虫、海洋生物、机械、玩具等非人拓扑** —— 与"2D 机械体"高度契合 |
| **MagicAnime**（arXiv:2507.20368） | 40 万 I2V 片段 | **5 万对视频+全身关键点（133 关节 / 68 面部）**、1.2 万对视频驱动面部、2.9 千对音频驱动面部 | 配 MagicAnime-Bench |
| **AnimeCeleb** | 3,613 个 3D 动漫头模、240 万图像 | **带 pose/expression 向量与 23 个 morph** | ⭐ 最接近"Live2D 参数"的公开标注（morph 系数 = 形变系数） |
| **LinkTo-Anime**（arXiv:2506.02733） | 395 序列 / 24,230 训练帧 | 前后向光流、遮挡掩码、**Mixamo 骨架** | 光流可用于动作平滑性监督 |
| **BEAT** | 30 人、约 32 小时 | ARKit 52 维 blendshape | 表情参数序列预训练 |
| 其他 | CoNR（70 万图/22K 角色，超密集 pose map）、PaintBucket-Character、Avatar Anime-Character | | |

## 3.7 从视频抽参数的标准管线

综合 X-Dancer、百度 2D dance、Being-M0 的做法：

```
2D 姿态估计（DWPose / OpenPose，whole-body 60+ 点带置信度）
  → 镜头切换检测与过滤
  → 帧率重采样统一（20-60fps → 固定 fps）
  → 坐标归一化
       方案 A（百度）：直接除图像宽高归一到 [0,1]，不做根节点归一化，
                       改用"参考姿态"约束体型比例与屏占比
       方案 B（AIST++2D）：固定体型投影，显式标准化比例/尺度
  → 置信度加权 / 低置信过滤
  → 时序平滑（低通滤波 / 双边滤波，保边缘）
  → LLM 生成分层文本标注
       Being-M0：整体语义 + 手臂/腿部等局部描述
       OpenT2M：秒级标注
  → 质量 / 物理可行性过滤
  → 去重审计（防泄漏）
```

## 3.8 低维参数序列生成的通用坑清单

1. **不要直接回归 raw 坐标** —— arXiv:2512.11720 的消融显示 raw 坐标扩散有明显 jitter，one-hot/SimCC 分箱明显更平滑。若参数有界（Live2D 参数都有 min/max），**离散分箱 + 分类式监督**是低成本抗抖动手段。
2. **根节点解耦** —— 3D 侧标准做法是根节点用**速度/角速度增量**、其余关节用局部量。2D 对应：**画布位置/整体缩放/旋转作为"根"用增量表示，形变系数用局部绝对值**。
3. **潜空间必须正则** —— MotionStreamer 消融：纯 AE 生成 FID 43.8，正则后 11.79。
4. **分部位量化，但要看数据量** —— X-Dancer（5 部位各 6 token）、2D-PRQ、2D-LFQ 都指向"低维参数不要用单个大码本"。但 OpenT2M 警告：**小数据集上 token 更多反而掉点**。
5. **抖动 / 脚滑 / 穿模的 2D 对应**：
   - L_pos → 前向变形后的顶点/关键点位置误差；
   - L_vel → 参数一阶差分；
   - L_foot → **接触/锚定掩码**（脚与地面、手与道具、机械部件的铰接约束）：`L = Σ f_i · ‖Δp_i‖²`；
   - **穿模的 2D 对应是图层顺序 / 部件重叠冲突** —— 检索中未找到现成方法〔未确认〕。可参考 contact loss 形式 `ReLU(v_pred − δ_v) + ReLU(−h_pred − δ_h)`，把"穿透深度"改成图层间的有向距离。
6. **评估抖动用 Peak Jerk / AUJ**，以及 acceleration / jerk / sliding。

---

# 第四部分：面向「2D 机体连续动作参数生成」的架构设计方案

> **前置说明**：本节假设"2D 机体"指**参数化绑定的 2D 角色或机械体**（Live2D / Spine / 自研骨骼绑定），其动作可由一组有界连续参数 `a ∈ R^D` 完全描述（D 约 30-200），动画即该参数向量随时间的轨迹。若你的定义不同（例如指 2D 物理仿真中的刚体），§4.2 的表示层需要调整，但架构骨架不变。

## 4.1 问题形式化

**输入**
- `c`：自然语言指令（"生气地叉腰，然后转身走开"）
- `I`：机体外观与绑定信息（立绘 / T-pose 渲染图 + 参数表 schema），可选
- `s_t`：当前参数状态向量 `∈ R^D`
- `h`：历史上下文（前 k 帧参数 / 已生成前缀）

**输出**
- `A_t = [a_t, a_{t+1}, ..., a_{t+H−1}] ∈ R^{H×D}`

**建议超参**
| 项 | 值 | 依据 |
|---|---|---|
| fps | 30 | 动画行业标准 |
| H（chunk 长度） | **24-32 帧（0.8-1.07 秒）** | π0 的 50 步@50Hz=1s；经验规律"块≈0.5-1 秒行为片段" |
| 执行步长 Ta | **8-12 帧** | Diffusion Policy 消融 Ta=8 最优；RTC 默认 execution_horizon 8-12 |
| 去噪步数 | **训练连续 τ，推理 4-10 步 Euler** | π0 用 10 步；Dita 10 步无损；EMDM 警告 1 步退化 |
| D（参数维度） | 按机体，pad 到 D_max（如 256）+ mask | π0 zero-pad 到 18 维的做法 |

## 4.2 表示层设计（最容易出错的一层）

**（a）参数分组** —— 照抄 X-Dancer 的分部位思路，但**分组数按数据量决定**：

| 组 | 内容 | 表示 |
|---|---|---|
| **root** | 画布位置 x/y、整体旋转、缩放 | **一阶差分（增量）** |
| **head** | 头部角度 XYZ、眼球、眉毛、嘴 | 局部绝对值 |
| **torso** | 身体角度 XYZ、呼吸 | 局部绝对值 |
| **arm_L / arm_R** | 手臂各关节 | 局部绝对值 |
| **leg / lower** | 下半身 | 局部绝对值 |
| **extra** | 机械部件、特效层、物理摆件 | 局部绝对值 |

> ⚠️ **相对 vs 绝对的取舍**：GR00T N1.6 改用 state-relative 后更平滑更准，但**小数据下相对量易误差累积、纠错能力下降**。建议：**只有 root 用增量，其余用绝对值**，这是最稳的折中。

**（b）归一化**：每维 **q01/q99 分位数 + std**，并**强制加下限**：
```python
scale = max(q99 - q01, eps_floor)   # eps_floor 按参数量纲设，例如 0.05 * 全局中位数
```
这一行代码能避免 openpi troubleshooting 里记录的"低使用率维度导致 loss 发散"。**你的参数表里必然有大量常年不动的维度，这条是必需品。**

**（c）连续 vs 分箱**：
- **主路线：连续向量 + flow matching**（架构简单、能出精细数值、和 π0/GR-3 一致）；
- **抗抖动靠损失而非表示**：SAiD 的 noise-level velocity loss + jerk 正则；
- **备选路线（若抖动严重）**：SimCC 式分箱（W=256）+ 分类监督，有 arXiv:2512.11720 的消融支持。**建议作为 M1 阶段的 A/B 实验，不要一开始就做复杂化。**

**（d）跨机体**：
- 统一到 `D_max` 维槽位 + **语义槽位对齐**（RDT 的做法：按物理含义填入对应槽位，其余补零）而非任意 pad；
- 每个机体类别一套 **CategorySpecificMLP** 输入/输出头，共享主干（GR00T 做法）；
- 机体类型 ID 通过 **AdaLN** 注入（FLOWER 的 Action-Space Global-AdaLN-Zero）。

## 4.3 架构：三个候选与推荐

### 候选 A（最小可行，1-2 周出结果）
```
CLIP/T5 文本编码器  →  cross-attention  →  Transformer Flow Policy (~100M)  →  A_t
```
不含 VLM。等价于「MDM + rectified flow + 2D 参数」。**必须先做这个作为基线**，否则你无法判断 VLM 到底带来了多少增益。

### 候选 B（推荐主线）—— π0 式权重分离 + KI 隔离
```
                     ┌──────────────────────────────────────┐
   指令 c  ──────────►│  VLM 主干 (Qwen2.5-VL-3B)            │
   参考图 I ─────────►│  语言塔冻结 / 视觉塔可训              │
                     │  额外目标: 输出离散运动基元 token      │
                     └───────────┬──────────────────────────┘
                                 │ blockwise causal attention
                                 │ + stop-gradient on K,V (KI)
                                 ▼
   状态 s_t ────────►┌──────────────────────────────────────┐
   噪声 A^τ ────────►│  Flow Action Expert (~0.3B DiT)      │──► v_θ ──► A_t
   时间步 τ ────────►│  QKNorm + RMSNorm + Global AdaLN-Zero│
   机体类型 ────────►└──────────────────────────────────────┘
```

**关键实现细节**：
1. **注意力分块（照抄 π0）**：
   ```
   块1 [图像 + 语言 + 基元 token]  块内双向，看不到后续
   块2 [状态 s_t]                  只看块1（K/V 可缓存）
   块3 [噪声动作 a^τ_{t..t+H-1}]   看块1、块2、块内双向
   ```
2. **KI stop-gradient**：动作专家 attend 到主干 token 时，对主干的 K、V 施加 `sg(·)`。前向照常，反向不回写。**加了之后 flow 损失权重可直接取 1**。
3. **权重分离**：动作专家用独立权重（width 1024，depth 12-16，mlp_dim 4096），从零初始化。
4. **只用 VLM 后半段层的 KV cache**（GR-3 做法）或**中间层 hidden**（FLOWER 做法）—— **这两个必须在你的数据上 A/B**，因为 GR-3 与 SmolVLA 结论相反。
5. **稳定性三件套**：QKNorm、attention/FFN 后额外 RMSNorm（GR-3 明确说"drastically 改善稳定性并提升语言跟随"）、MLP decoder 而非线性 decoder（RDT）。
6. **Global AdaLN-Zero**：时间步 τ + 机体类型共同生成调制信号，**所有层共享一套 modulation 权重、零初始化**，比逐层 AdaLN 省 20% 参数且效果相同（FLOWER 消融 4.44 vs 4.43）。

### 候选 C（低成本变体）—— FLOWER 式中间层融合
砍掉 VLM 后 30-50% 层（反正你不需要它生成长文本），取中间层 hidden → Linear+RMSNorm → cross-attention 注入 Flow Transformer。**训练成本低一个数量级（FLOWER 只用 200 H100 小时预训练）**，且有"中间层 93.4% vs 晚融合 73%"的消融支持。

> **推荐路径：A（基线）→ C（低成本验证 VLM 增益）→ B（规模化主线）**。
> 如果算力紧张，C 可能就是终点 —— FLOWER 950M 全模型在 CALVIN ABC 上拿了 4.53 SOTA。

## 4.4 损失函数

```
L = L_flow
  + λ_geo  · L_geo        # 变形后关键点位置误差（需可微前向变形）
  + λ_vel  · L_vel        # 参数一阶差分一致性
  + λ_jerk · L_jerk       # 三阶差分（抗抖动）
  + λ_anc  · L_anchor     # 锚定约束：脚/铰接点/道具接触
  + λ_bnd  · L_bound      # 参数越界惩罚 ReLU(|a| - 1)
  + α      · L_ntp        # 文本/基元 next-token（保语义，α=1 若已用 KI）
```

**要点**：
- `L_flow`：`‖ v_θ(A^τ, cond) − (ε − A) ‖²`，τ ~ **Beta(1.5, 1)**；
- `L_vel` / `L_jerk` 用 **SAiD 的 noise-level 形式**（直接在扩散空间算，不用每步解码到干净动作），成本几乎为零；
- `L_geo` 需要一个**可微的前向变形函数** `p = FK(a)` —— 对 Live2D 是变形器链，对骨骼绑定是标准 FK。**如果暂时做不到可微，退化为对关键点子集做代理监督**；
- `L_bound` 很重要：CFG 强度高时参数容易打到边界，这项能显著改善"表情僵硬"的观感；
- **λ 调参起点**：`λ_vel=1.0, λ_jerk=0.1, λ_geo=1.0, λ_anc=0.5, λ_bnd=0.1`（参考 MDM 的量级，需实测）。

## 4.5 数据方案

这是你最大的瓶颈。四条路并行：

**路 1：现成资产（最快）**
- 从已有的 Live2D / Spine 工程文件里导出所有 `.motion3.json` / 动画剪辑 → 直接就是「参数序列 GT」；
- 用 LLM/VLM 对渲染出的动画片段做**分层文本标注**（整体语义 + 部位描述 + 情绪 + 速度），参考 Being-M0 的分层标注与 OpenT2M 的秒级标注。
- **这是唯一能给出"真参数"的数据源，质量最高，优先榨干。**

**路 2：程序化合成（规模最大）**
- 照抄 **AniDiffusion** 的做法：让每个关节相对父关节**按等间隔角度摆放并做全组合枚举**，渲染 + 自动标注；
- 加上关键帧插值与噪声扰动生成大量"物理合理但语义弱"的序列；
- 用途：**动作专家的阶段 1 单独预训练**（DexVLA 的 curriculum），学会"什么是平滑合理的参数轨迹"。

**路 3：视频反演（覆盖真实分布）**
- 从 2D 动画视频用 DWPose 抽关键点 → **拟合到你的绑定参数**（可微渲染或优化求解 `argmin_a ‖FK(a) − p_detected‖`）；
- 可用 AniDiffusion（135 角色 + 逐帧关键点）、MagicAnime（5 万对全身关键点）、AnimeCeleb（23 morph 参数向量）作为公开补充。

**路 4：Latent action pretraining（无标注视频，LAPA 路线）**
- 用 VQ-VAE 从**无参数标注**的 2D 动画视频里量化出 latent action（LAPA 用 latent 长 4、码本 8）；
- 先用 latent action 预训练 VLM→latent 的映射，再换成真实参数头微调；
- **LAPA 报告 30-40× 的预训练效率提升，且性能反超** —— 对数据稀缺场景是最高杠杆的一招。

**⚠️ 数据纪律**：
- 建立**训练/验证泄漏审计**（OpenT2M 发现现有基准 10.6-17% 逐字重叠）；
- 建立**可播放性过滤器**：生成的参数序列是否能被绑定系统真实播放、是否穿模、图层是否冲突 —— 这是 OpenT2M "RL 跟踪策略做物理可行性过滤"的 2D 版本。

## 4.6 训练配方（四阶段）

| 阶段 | 内容 | 数据 | 冻结策略 |
|---|---|---|---|
| **S0 · 表示层** | 统计归一化参数、（可选）训练参数序列 VAE/RVQ | 全部参数序列 | — |
| **S1 · 动作专家单独预训练** | 只训 Flow Expert，条件用弱信号（动作类别 / latent action） | 程序化合成 + 视频反演 + 无标注视频 latent | 无 VLM |
| **S2 · 联合预训练** | 接入 VLM，KI stop-gradient；**co-train 图文数据 + 文本 next-token** | 全部 + 通用 VL 数据 | 语言塔冻结，视觉塔可训 |
| **S3 · 指令后训练** | 高质量「指令 ↔ 参数序列」对；启用分层推理（先出基元，再出参数） | 人工精标小集 | 视情况解冻顶部 4 层（GR00T N1.6） |
| **S4 · 可选 RL** | 以「人评分 / 可播放性 / 指令跟随率」为 outcome reward | 自生成 rollout | — |

**关键纪律**：
- **S2 的 co-training 不能省**。π0.5 消融显示去掉网络多模态数据，OOD 成功率 94%→74%。你的对应做法是混入通用图文 caption/VQA 数据，防止 VLM 退化成"只会输出动作的哑巴"。
- **S1 先单独预训练动作专家**是 DexVLA 的做法，能显著降低 S2 的不稳定性。
- **S4 的前提**：SFT 基线必须已有非零成功率，否则 outcome-reward RL 恒为 0（SimpleVLA-RL 的明确失败模式）。

## 4.7 推理与流式生成

**块间无缝续接** —— 三个方案，按实现成本排序：

1. **Inpainting 式（首选 baseline，零训练成本）**：照抄 SAiD 的 mask 公式，把前一块的末尾 `d` 帧作为 `u_ref`，在每步采样时约束：
   ```
   ũ_τ = (1−m) ⊙ u_τ + m ⊙ ( τ·u_ref + (1−τ)·ε )
   ```
2. **RTC（推荐生产方案）**：把新块前缀当 inpainting，加 guidance 项。LeRobot 默认值：`execution_horizon=8~12`、`max_guidance_weight=10.0`、`prefix_attention_schedule=EXP`。**无需重训模型**，且对推理延迟鲁棒。
3. **FlowMDM 的 Blended Positional Encodings**（若要做**多段指令拼接**）：去噪早期用绝对 PE + 段内全局注意力，晚期切相对 PE + 无限制窗口。**无需后处理**，是多指令组合最优雅的方案。

**自动决定长度**：两个现成方案 ——
- Helix 的**"任务完成百分比"合成维度**（在动作向量里加一维，模型自己预测何时结束）；
- MotionStreamer 的**参考终止潜码 + 距离阈值**。
- 建议用前者，实现更简单。

**速度控制**：Helix 的 **Sport Mode** —— 对 `[T × D]` 动作块做线性重采样成 `0.8T` 再按原帧率播放。免费获得"快放/慢放"控制。

**CFG**：训练时 **10-30% 概率丢文本条件**；推理时 w 做网格搜索（1.5-4），**同时看 FID 与 Diversity**，注意高 w 会导致参数打到边界（这也是 `L_bound` 的价值）。

**延迟预算参考**（4090 量级）：π0 的 73ms = 图像编码 14ms + 观测前向 32ms + 10 步动作前向 27ms。你的动作维度更低、序列更短，**应能压到 30-50ms**，足够 30fps 实时。

## 4.8 评测方案

| 类别 | 指标 | 说明 |
|---|---|---|
| **生成质量** | FID（在参数序列的 kinetic feature 空间） | 需先训一个动作特征提取器，参考 HumanML3D 的做法 |
| **指令对齐** | R-Precision (Top-1/2/3)、MM-Dist | 需训文本-动作对比编码器 |
| **多样性** | Diversity（**越接近真实越好**）、MModality | 检验 flow matching 是否真的保住了多模态 |
| **平滑性** | **Peak Jerk (PJ)、AUJ**、acceleration、参数一阶差分方差 | FlowMDM 与 EgoMotion 的标准做法 |
| **物理/绑定合法性** | 参数越界率、图层冲突率、锚点漂移（脚滑的 2D 对应） | 自定义，需接绑定系统 |
| **块间连续性** | 拼接处的 jerk 峰值、参数跳变幅度 | 专门评估 RTC 效果 |
| **效率** | 单块延迟、吞吐 FPS、去噪步数敏感性曲线 | 报告 1/2/4/10/20 步的质量-速度曲线 |
| **语义保持** | VLM 在通用 VQA 基准上的退化幅度 | 验证 KI + co-training 是否有效 |
| **人评** | pairwise win rate | 动画质量最终还是主观的，参考百度 2D dance 的做法 |

## 4.9 风险与坑清单（每条都有出处）

| 风险 | 现象 | 对策 | 出处 |
|---|---|---|---|
| **低使用率维度归一化爆炸** | loss 突然发散 | q01/q99 加下限截断 | openpi troubleshooting |
| **L1 抹平多模态** | 动作僵硬、幅度偏小、"平均感" | 用 flow matching 而非 L1 | OpenVLA-OFT 作者自述 |
| **动作梯度污染 VLM** | 训练慢、语言跟随退化 | KI 的 attention stop-gradient + co-training | arXiv:2505.23705 |
| **潜空间"重建好但生成差"** | AE 重建 MPJPE 极低，生成 FID 极高 | 潜空间必须正则（VAE/σ-VAE） | MotionStreamer FID 43.8 → 11.79 |
| **一步蒸馏 mode collapse** | 多样性骤降 | 不做 1 步；用 4-10 步 flow；若必须蒸馏用分布级损失 | EMDM（1 步 FID 5.64）、Flow2One |
| **raw 坐标回归抖动** | 高频 jitter | velocity/jerk 正则；或改 SimCC 分箱 | arXiv:2512.11720 消融 |
| **小数据 + 大码本掉点** | 分部位量化反而更差 | 数据量不足时减少分组与码本 | OpenT2M 的 2D-PRQ 警告 |
| **相对动作误差累积** | 长序列漂移 | 只让 root 用增量，其余用绝对值 | GR00T N1.6 |
| **块边界跳变** | 动画每 1 秒抖一下 | RTC / inpainting / FlowMDM BPE | arXiv:2506.07339 |
| **训练/验证泄漏** | 指标虚高 | 逐字去重审计 | OpenT2M 发现 10.6-17% 重叠 |
| **高 CFG 过饱和** | 参数打到边界、表情僵硬 | w 网格搜索 + L_bound | arXiv:2410.02416 |
| **只做语义 CoT 无收益** | 加了思维链但不涨点 | 思考内容必须 grounded 到具体参数/坐标 | ECoT 消融（48% vs 66%） |
| **RL 冷启动失效** | RL 后成功率仍为 0 | 先确保 SFT 有非零成功率 | SimpleVLA-RL |

## 4.10 里程碑路线图

| 里程碑 | 内容 | 判定标准 |
|---|---|---|
| **M0** | 数据管线 + 表示层 + 归一化统计；无条件 flow 模型 | 能生成平滑、合法、可播放的参数序列；PJ/AUJ 接近真实 |
| **M1** | 文本条件基线（候选 A，不含 VLM） | R-Precision 明显高于随机；能区分"跳/走/挥手" |
| **M2** | 接入 VLM（候选 C 中间层融合），对比 M1 | **VLM 必须带来可测增益**，否则不要上 B |
| **M3** | 候选 B + KI 隔离 + co-training | 通用 VQA 退化 < 3 点；指令泛化到未见组合 |
| **M4** | RTC 流式续接 + 少步推理 | 30fps 实时；拼接处 jerk 峰值与块内同量级 |
| **M5** | 分层指令（先出基元再出参数）+ 风格解耦 | 支持"生气地慢慢走"这类复合指令 |
| **M6** | 可选：RL 后训练 / 人评优化 | 人评 win rate 显著提升 |

**最重要的一条纪律**：**M1 → M2 必须做严格对照。** 很多"VLA"工作的增益其实来自 chunking 和连续表示，而不是 VLM。如果你的指令空间不大（几百种动作描述），一个 CLIP 文本编码器 + flow policy 可能就够了，上 3B VLM 只是烧钱。**VLM 的价值在于处理开放词表、组合指令、以及理解机体的视觉外观 —— 确认你真的需要这些再上。**

---

## 附录：核心参考文献

**VLA 基础模型**
- RT-1: arXiv:2212.06817 · RT-2: arXiv:2307.15818
- Octo: arXiv:2405.12213 · OpenVLA: arXiv:2406.09246 · OpenVLA-OFT: arXiv:2502.19645
- π0: arXiv:2410.24164 · π0-FAST: arXiv:2501.09747 · π0.5: arXiv:2504.16054 · π*0.6/RECAP: arXiv:2511.14759
- Knowledge Insulation: arXiv:2505.23705
- GR00T N1: arXiv:2503.14734 · GR00T N1.5/N1.6: research.nvidia.com/labs/gear/
- Gemini Robotics 1.5: arXiv:2510.03342
- GO-1/ViLLA: arXiv:2503.06669 · GR-3: arXiv:2507.15493 · GR-2: arXiv:2410.06158
- RoboVLMs: arXiv:2412.14058 / Nature MI s42256-025-01168-7
- Helix: figure.ai/news/helix, figure.ai/news/helix-logistics

**扩散 / 流匹配动作头**
- Diffusion Policy: arXiv:2303.04137 · DP3: arXiv:2403.03954
- RDT-1B: arXiv:2410.07864 · CogACT: arXiv:2411.19650 · Dita: arXiv:2503.19757
- FLOWER: arXiv:2509.04996 · DexVLA: arXiv:2502.05855 · Diffusion-VLA: arXiv:2412.03293
- HybridVLA: arXiv:2503.10631 · MoDE: arXiv:2412.12953 · MDT: arXiv:2407.05996
- ManiCM: arXiv:2406.01586 · OneDP: arXiv:2410.21257 · SDM Policy: arXiv:2412.09265 · Flow2One: arXiv:2603.09415
- RTC: arXiv:2506.07339 · Training-Time RTC: arXiv:2512.05964 · VLASH: arXiv:2512.01031

**轻量化 / 预训练 / 后训练**
- SmolVLA: arXiv:2506.01844 · TinyVLA: arXiv:2409.12514 · SpatialVLA: arXiv:2501.15830
- LAPA: arXiv:2410.11758 · V-JEPA 2: arXiv:2506.09985 · WorldVLA: arXiv:2506.21539
- ECoT: arXiv:2407.08693 · RIPT-VLA: arXiv:2505.17016 · SimpleVLA-RL (ICLR 2026)

**文本 → 动作 / 参数序列生成**
- MDM: arXiv:2209.14916 · MoMask: arXiv:2312.00063 · MotionLCM: arXiv:2404.19759
- OmniControl: arXiv:2310.08580 · FlowMDM: arXiv:2402.15509 · EMDM: arXiv:2312.02256
- MotionStreamer (ICCV 2025) · DART (ICLR 2025)
- Being-M0/MotionLib: arXiv:2410.03311 · OpenT2M: arXiv:2603.18623 · Motion-X: arXiv:2307.00818
- EgoMotion: arXiv:2604.19105 · MotionLLM: arXiv:2405.20340

**2D / 低维参数序列生成**
- SAiD: arXiv:2401.08655 · DiffusionTalker: arXiv:2311.16565 · PMMTalk: arXiv:2312.02781
- X-Dancer: arXiv:2502.17414 · 2D Dance as Multi-Channel Image: arXiv:2512.11720
- Text-to-Skeleton Cascades: arXiv:2603.08028
- AniDiffusion / How to Train Your Dragon: arXiv:2503.15586
- MagicAnime: arXiv:2507.20368 · LinkTo-Anime: arXiv:2506.02733

---

*本报告的部分 2026 年条目为 arXiv 预印本，未经同行评审，数字应视为作者自报。凡标注〔未确认〕的内容，引用前请核对一手来源。*
