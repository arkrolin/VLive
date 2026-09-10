"""V11 可行性验证：角色自己的动作曲线，是否比跨角色 exem 平均更好的锚点？

动机（用户假设）：
  exem = 同动作跨角色均值 -> 抹掉了角色个性，很多通道退化成无意义横线。
  rig 96d = 跨动作聚合 -> 抹掉了动作个性。
  真正该参考的是「该角色自己的某个动作」。

对每个 val 样本 (角色 M, 动作 A, 参数集 P)，构造角色自身的参考曲线库
  bank[M][a][p]  for all a != A        (严格排除目标动作，避免泄露)
然后比较原始单位 abs_mae：

  1  exem            跨角色 (action,param) 均值              —— 当前先验
  2  model           R 臂 best ckpt 预测                      —— 当前 SOTA
  3  ref_idle        角色自己的 IDLE 类动作同参数曲线
  4  ref_charmean    角色自己其它所有动作的同参数均值          —— 角色幅度先验
  5  ref_oracle_pp   角色其它动作中逐参数最优                  —— 检索上界
  6  ref_oracle_act  角色其它动作中样本级最优(单个动作)        —— 实际检索上界
  7  delta           ref_idle + [exem(A) - exem(IDLE)]        —— 动作增量迁移
  8  delta_cm        ref_charmean + [exem(A) - exem(charmean)]
  9  scale           exem(A) * (角色幅度 / exem 幅度)          —— 已证伪的路线，复核

输出 abs_mae（越低越好）与 vs exem / vs model 的相对提升。
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "live2d_vla"
sys.path.insert(0, str(SRC))

from schema import load_whitelist, load_gen_mask      # noqa: E402
from io_motion import (list_motions, load_target_curves,  # noqa: E402
                       set_action_map)

T = 48
FPS = 30.0
IDLE_LIKE = {"IDLE"}

cfg = SimpleNamespace(
    whitelist_path=ROOT / "outputs" / "model_whitelist_all.json",
    gen_mask_path=ROOT / "outputs" / "gen_mask_all.json",
    subset_models=580,
)

# ---- 动作名规范化（与训练完全一致） ----
amap = json.loads((ROOT / "outputs" / "action_semantic_map.json").read_text(encoding="utf-8"))
rev = {}
for grp, names in amap.get("groups", {}).items():
    for n in names:
        rev[str(n).lower()] = grp
set_action_map(rev)

whitelist = load_whitelist(cfg)
gen_mask = load_gen_mask(cfg)
subset = whitelist[: cfg.subset_models]
targets = {m: gen_mask.get(m, []) for m in subset if gen_mask.get(m)}

blob = pickle.load(open(ROOT / "outputs" / "render_data" / "abl_R_all_v9_best.pkl", "rb"))
items = blob["items"]
val_models = sorted({it["model"] for it in items})
print(f"[setup] {len(items)} val samples, {len(val_models)} held-out characters")

# ---- 构建角色自身曲线库（只取 val 角色，快） ----
bank: dict[str, dict[str, dict[str, np.ndarray]]] = {}
for m in val_models:
    if m not in targets:
        continue
    mdir = ROOT / "data" / "all" / m
    per_action = {}
    for action in list_motions(mdir):
        c = load_target_curves(mdir, action, targets[m], fps=FPS, T=T)
        if c:
            per_action[action] = {p: np.asarray(v, np.float64) for p, v in c.items()}
    bank[m] = per_action
n_act = np.array([len(v) for v in bank.values()])
print(f"[bank] per-character motions: mean {n_act.mean():.1f} "
      f"min {n_act.min()} max {n_act.max()}")

# ---- exem 先验缓存（跨角色 (action,param) 均值） ----
exem_cache = pickle.load(
    open(ROOT / "outputs" / "exem_cache_580_v9all.pkl", "rb"))


def mae(a, b):
    return float(np.abs(a - b).mean())


res = {k: [] for k in (
    "exem", "model",
    "ref_idle", "ref_charmean", "ref_oracle_pp", "ref_oracle_act",
    "delta_idle", "delta_cm", "scale",
    "delta_oracle", "exem_oracle",
)}
cov = {"idle": 0, "any": 0, "total": 0}

for it in items:
    m, act = it["model"], it["action"]
    names = list(it["names"])
    tgt = np.asarray(it["target"], np.float64)
    exm = np.asarray(it["exem"], np.float64)
    prd = np.asarray(it["pred"], np.float64)
    P = tgt.shape[0]
    cov["total"] += 1

    res["exem"].append(mae(prd * 0 + exm, tgt))
    res["model"].append(mae(prd, tgt))
    # oracle: 若 exem/模型二选一（逐点），上界参考
    res["exem_oracle"].append(mae(np.where(np.abs(prd - tgt) < np.abs(exm - tgt), prd, exm), tgt))

    others = {a: c for a, c in bank.get(m, {}).items() if a != act}
    if others:
        cov["any"] += 1

    # 逐参数参考矩阵
    def ref_of(a):
        d = others.get(a, {})
        return np.stack([d[p] if p in d else np.zeros(T) for p in names], 0)

    # 3) IDLE 类参考
    idle_acts = [a for a in others if a in IDLE_LIKE]
    if idle_acts:
        cov["idle"] += 1
        R_idle = ref_of(idle_acts[0])
        res["ref_idle"].append(mae(R_idle, tgt))
        # 7) 动作增量迁移: ref_idle + (exem(A) - exem(IDLE))
        exm_idle = np.stack([
            np.asarray(exem_cache.get((idle_acts[0], p), np.zeros(T)), np.float64)
            for p in names], 0)
        res["delta_idle"].append(mae(R_idle + (exm - exm_idle), tgt))
        # 10) delta 但用 oracle 选参考动作
        best, bestv = None, np.inf
        for a in others:
            exm_a = np.stack([
                np.asarray(exem_cache.get((a, p), np.zeros(T)), np.float64)
                for p in names], 0)
            v = mae(ref_of(a) + (exm - exm_a), tgt)
            if v < bestv:
                bestv, best = v, a
        res["delta_oracle"].append(bestv)
    else:
        res["ref_idle"].append(mae(exm, tgt))
        res["delta_idle"].append(mae(exm, tgt))
        res["delta_oracle"].append(mae(exm, tgt))

    if not others:
        for k in ("ref_charmean", "ref_oracle_pp", "ref_oracle_act",
                  "delta_cm", "scale"):
            res[k].append(mae(exm, tgt))
        continue

    # 4) 角色其它动作均值
    R_cm = np.mean(np.stack([ref_of(a) for a in others], 0), 0)
    res["ref_charmean"].append(mae(R_cm, tgt))

    # 5) 逐参数 oracle
    stack = np.stack([ref_of(a) for a in others], 0)          # (nA, P, T)
    err = np.abs(stack - tgt[None]).mean(2)                    # (nA, P)
    bidx = err.argmin(0)
    R_pp = stack[bidx, np.arange(P)]
    res["ref_oracle_pp"].append(mae(R_pp, tgt))

    # 6) 样本级 oracle（整段用同一个参考动作）
    j = int(err.mean(1).argmin())
    res["ref_oracle_act"].append(mae(stack[j], tgt))

    # 8) delta with charmean anchor
    exm_cm = np.mean(np.stack([
        np.stack([np.asarray(exem_cache.get((a, p), np.zeros(T)), np.float64)
                  for p in names], 0) for a in others], 0), 0)
    res["delta_cm"].append(mae(R_cm + (exm - exm_cm), tgt))

    # 9) 幅度缩放（复核已证伪路线）
    num = (tgt * exm).sum(1)
    den = (exm * exm).sum(1)
    g = np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), 1.0)
    # 用角色其它动作估计幅度比（不用目标，防泄露）
    rc = np.abs(R_cm).mean(1)
    ec = np.abs(exm_cm).mean(1)
    s = np.where(ec > 1e-9, rc / np.where(ec > 1e-9, ec, 1.0), 1.0)
    res["scale"].append(mae(exm * s[:, None], tgt))

print(f"\n[coverage] 有 >=1 个其它动作: {cov['any']}/{cov['total']} "
      f"({cov['any']/cov['total']*100:.1f}%)；有 IDLE 类动作: "
      f"{cov['idle']}/{cov['total']} ({cov['idle']/cov['total']*100:.1f}%)")

base_exem = float(np.mean(res["exem"]))
base_model = float(np.mean(res["model"]))
print(f"\n=== 原始单位 abs_mae（越低越好）===")
print(f"{'方案':22s} {'abs_mae':>9s} {'vs exem':>9s} {'vs model':>9s}")
print("-" * 54)
LABEL = {
    "exem": "1 exem 先验(当前)",
    "model": "2 R模型(best)",
    "ref_idle": "3 角色IDLE曲线",
    "ref_charmean": "4 角色其它动作均值",
    "ref_oracle_pp": "5 角色逐参数oracle",
    "ref_oracle_act": "6 角色单动作oracle",
    "delta_idle": "7 delta(IDLE锚)",
    "delta_cm": "8 delta(charmean锚)",
    "delta_oracle": "9 delta(oracle锚)",
    "scale": "10 exem×角色幅度",
    "exem_oracle": "11 model/exem oracle门",
}
for k, lab in LABEL.items():
    v = float(np.mean(res[k]))
    print(f"{lab:22s} {v:9.4f} {(1-v/base_exem)*100:+8.1f}% "
          f"{(1-v/base_model)*100:+8.1f}%")

# 逐参数相关性：参考曲线是否携带形状信息
print("\n=== 形状（一阶差分）MAE：参考曲线是否带来 exem 没有的形状信息 ===")


def d1(x):
    return np.diff(x, axis=1)


shape = {k: [] for k in ("exem", "model", "ref_charmean", "delta_cm")}
for it in items:
    m, act = it["model"], it["action"]
    names = list(it["names"])
    tgt = np.asarray(it["target"], np.float64)
    exm = np.asarray(it["exem"], np.float64)
    prd = np.asarray(it["pred"], np.float64)
    others = {a: c for a, c in bank.get(m, {}).items() if a != act}
    shape["exem"].append(mae(d1(exm), d1(tgt)))
    shape["model"].append(mae(d1(prd), d1(tgt)))
    if not others:
        shape["ref_charmean"].append(mae(d1(exm), d1(tgt)))
        shape["delta_cm"].append(mae(d1(exm), d1(tgt)))
        continue
    R_cm = np.mean(np.stack([
        np.stack([others[a].get(p, np.zeros(T)) for p in names], 0) for a in others], 0), 0)
    shape["ref_charmean"].append(mae(d1(R_cm), d1(tgt)))
    exm_cm = np.mean(np.stack([
        np.stack([np.asarray(exem_cache.get((a, p), np.zeros(T)), np.float64)
                  for p in names], 0) for a in others], 0), 0)
    shape["delta_cm"].append(mae(d1(R_cm + (exm - exm_cm)), d1(tgt)))
sb = float(np.mean(shape["exem"]))
for k in ("exem", "model", "ref_charmean", "delta_cm"):
    v = float(np.mean(shape[k]))
    print(f"  {k:16s}: {v:.4f}  ({(1-v/sb)*100:+.1f}% vs exem)")
