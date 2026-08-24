# 同来源自动检索模块 (rig_retrieval)

部署侧模块：给定**目标 rig**（其 `.moc3` 参数 schema）和**期望动作**，从语料中自动挑出 K 条「同动作 + 最接近来源」的参考动画，作为生成模型的 few-shot 上下文。**零人工来源标签、零重训**。

这是 V6.2「auto3」供体策略的产品化形态——已用数据证明：参数集 Jaccard 相似度是「同一生产来源」的免标签代理（自动挑选命中同来源家族率 **75.8%**）。

---

## 为什么有效（证据链见 V6 / V6.1 / V6.2）

- **B1 路径已实证**：零素材目标用 3 条「别的 Live2D 角色同动作」示例即可，action-R² +0.138（达 K=8 的 84%），且完全域内（避开了真人视频→Live2D 的跨体态/无下肢/量级问题）。
- **来源可全自动**：用目标静态 `.moc3` 参数集对其他角色参数集算 Jaccard，自动挑选 Top-3。纯自动 `auto3` 的 action-R² +0.144，离手动同源 +0.165 仅差 0.021，远高于异源 +0.092；其中 **75.8%** 的自动挑选恰好命中同一生产家族。即「挑同来源示例」这一步**不需要任何 curated 标签**。
- **运行时零三方依赖**：检索只依赖参数集(字符串集合)的 Jaccard，纯标准库即可；重活（特征编码、缓存）都在离线 build 阶段。

## 模块结构

```
deploy/rig_retrieval/
  moc3.py          # 从 .moc3 提取参数集（复刻 read_model_ids 正则，与语料缓存同源）
  similarity.py    # jaccard()
  schema.py        # ReferenceSpec / RetrievalResult（可 JSON 序列化）
  index.py         # CorpusIndex：构建/加载/查询（运行时纯标准库）
  retriever.py     # Retriever：retrieve / retrieve_for_model / retrieve_for_pack
  build_index.py   # 离线构建索引（导入 tools/，仅构建期依赖）
  validate.py      # 等价性回归：复现 V6.2 auto3 数字
  cli.py           # build / retrieve / validate 子命令
```

## 构建索引（离线，一次性）

```bash
uv run python -m deploy.rig_retrieval.cli build
# -> outputs/retrieval_index.jsonl  (每条语料 clip 一行)
```

索引字段：`pack, char, family, action, motion_path, params(参数集)`。参数集优先取自
`outputs/_paramset_cache.json`（与 V6.2 实验同源）；缺失的 pack 回退到实时 moc3 扫描。

## 检索（部署侧 API）

```python
from deploy.rig_retrieval import CorpusIndex, Retriever, extract_param_set

idx = CorpusIndex.load("outputs/retrieval_index.jsonl")
ret = Retriever(idx)

# 目标是一个全新模型目录（部署最典型场景）
res = ret.retrieve_for_model("/path/to/new_model_dir", "wave", k=3)
print(res.summary())

# 或先自行提取参数集再检索
ps = extract_param_set("/path/to/new_model_dir")
res = ret.retrieve(ps, "wave", k=3)
for r in res.references:
    print(r.pack, r.action, r.similarity, r.motion_path)
```

`RetrievalResult.references` 是 `ReferenceSpec` 列表，含 `pack / char / family /
action / motion_path / similarity / near_cluster / same_family`，可直接作为生成模型的
few-shot 参考输入。

### 评分规则
- **字符级打分**：每个来源角色只保留相似度最高的一条代表 clip，再取 Top-K 角色——避免同一过度代表的 rig 返回 3 条近似重复。
- **near_cluster 标记**：`similarity >= 0.30`（对齐 V6 的 0.30 Jaccard 邻域阈值）即视为「同来源近邻」，可信任为同源示例。
- 目标为语料内 pack 时（如「目标在语料里有兄弟」），自动排除其自身角色。

## 命令行

```bash
# 构建
uv run python -m deploy.rig_retrieval.cli build

# 检索：目标 = 语料内 pack
uv run python -m deploy.rig_retrieval.cli retrieve --target l2d22.ugirl06 --action haixiu --k 3

# 检索：目标 = 全新模型目录（真实部署场景）
uv run python -m deploy.rig_retrieval.cli retrieve --target /path/to/new_model --action wave --k 3 --json out.json

# 等价性回归：复现 V6.2 auto3（action-R²≈+0.144 / family-match≈75.8%）
uv run python -m deploy.rig_retrieval.cli validate
```

## 与生成管线的集成位置

```
目标模型 (.moc3) ──extract_param_set──▶ 参数集
                                      │
检索索引 ◀──build_index（离线）──────┘
   │  retrieve(target_ps, action, k=3)
   ▼
3 条同来源同动作参考 (.motion3.json) ──▶ 生成模型 few-shot 上下文
目标静态 moc3 + 扫掠探针 ──────────────▶ rig 项重建（零片段可用）
目标后续补 ≥8 条自有动画 ──────────────▶ 升级到 A 路径 (total-R²≈0.66)
```

## 已知边界
- 检索只解决「选哪几条参考」；动作先验的上限仍受 V4/V6 的 20 通道 CANON 编码与 24.2% 不可约画师内方差约束。
- 目标若是语料内 pack，检索排除了其自身角色（避免近重复）；纯新模型无此约束。
- `validate` 需要研究特征缓存 `outputs/_genprior_cache.npz`（由 `tools/audit_generative_prior.py` 产出，V4–V6 系列共用）。
