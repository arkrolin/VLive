"""拆解「角色自身动作库」增益的来源，并核查 rig 特征是否泄露目标幅度。

Q1 角色库 oracle (0.5031) 里有多少来自「平凡通道」？
   把通道按目标曲线是否近似常数（std < 0.01）分组，分别算 oracle 增益。
   若增益主要来自非常数通道 -> 真信号；若主要在常数通道 -> 只是「静息偏置」，
   那用一个 per-param 角色静息值就能拿到，不需要整套检索架构。

Q2 rig 96d 是否泄露目标幅度？
   build_model_identity 用角色「全部动作」的 (min,max,mean) 聚合，包含目标动作本身。
   统计：目标曲线的 range 占角色全局 range 的比例分布；
   以及「目标独占该参数动画」的通道占比（= rig 直接暴露目标 range）。

Q3 如果增益是静息偏置，最简单的方案 per-param 静息值 oracle 能拿多少？
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

from schema import load_whitelist, load_gen_mask                    # noqa: E402
from io_motion import (list_motions, load_target_curves,            # noqa: E402
                       set_action_map)

T, FPS = 48, 30.0
cfg = SimpleNamespace(
    whitelist_path=ROOT / "outputs" / "model_whitelist_all.json",
    gen_mask_path=ROOT / "outputs" / "gen_mask_all.json",
    subset_models=580,
)
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

bank = {}
for m in val_models:
    if m not in targets:
        continue
    mdir = ROOT / "data" / "all" / m
    per = {}
    for action in list_motions(mdir):
        c = load_target_curves(mdir, action, targets[m], fps=FPS, T=T)
        if c:
            per[action] = {p: np.asarray(v, np.float32) for p, v in c.items()}
    bank[m] = per


def mae(a, b):
    return float(np.abs(a - b).mean())


flat = {"exem": [], "model": [], "self_pp": [], "self_act": [], "rest": []}
nonflat = {k: [] for k in flat}
ratios, exclusive = [], []
n_ch = n_flat = 0

for it in items:
    m, act = it["model"], it["action"]
    names = list(it["names"])
    tgt = np.asarray(it["target"], np.float64)
    exm = np.asarray(it["exem"], np.float64)
    prd = np.asarray(it["pred"], np.float64)
    P = tgt.shape[0]
    ar = np.arange(P)
    others = [a for a in bank.get(m, {}) if a != act]
    if not others:
        continue
    S = np.stack([np.stack([bank[m][a].get(p, np.zeros(T)) for p in names], 0)
                  for a in others], 0)
    err = np.abs(S - tgt[None]).mean(2)
    best_pp = S[err.argmin(0), ar]
    best_act = S[int(err.mean(1).argmin())]

    # Q3 角色静息值 oracle：取其它动作在该参数的「最常见静息值」= 中位数
    rest = np.median(S, axis=0)                       # (P,T) 逐帧中位数
    rest_const = np.median(S.reshape(-1, P, T), axis=0).mean(1, keepdims=True)

    tgt_std = tgt.std(1)
    is_flat = tgt_std < 0.01
    n_ch += P
    n_flat += int(is_flat.sum())

    for i in range(P):
        bucket = flat if is_flat[i] else nonflat
        bucket["exem"].append(mae(exm[i], tgt[i]))
        bucket["model"].append(mae(prd[i], tgt[i]))
        bucket["self_pp"].append(mae(best_pp[i], tgt[i]))
        bucket["self_act"].append(mae(best_act[i], tgt[i]))
        bucket["rest"].append(mae(rest_const[i] * np.ones(T), tgt[i]))

    # Q2 rig 泄露：目标是该参数在角色全部动作里的「唯一/最极端」动画吗
    allc = np.stack([np.stack([bank[m][a].get(p, np.zeros(T)) for p in names], 0)
                     for a in bank[m]], 0)            # (nA_all, P, T)
    g_rng = allc.max(axis=(0, 2)) - allc.min(axis=(0, 2))
    t_rng = tgt.max(1) - tgt.min(1)
    keep = g_rng > 1e-6
    if keep.any():
        ratios.extend((t_rng[keep] / g_rng[keep]).tolist())
        exclusive.extend((t_rng[keep] >= 0.999 * g_rng[keep]).tolist())

print(f"\n=== Q1 通道构成 ===")
print(f"  总通道 {n_ch}，目标近似常数通道 {n_flat} ({n_flat/n_ch*100:.1f}%)")

print(f"\n=== Q1 分组 abs_mae（逐通道平均，原始单位）===")
print(f"{'方案':16s} {'常数通道':>12s} {'非常数通道':>12s} {'全部':>10s}")
print("-" * 54)
allc = {k: flat[k] + nonflat[k] for k in flat}
for k, lab in (("exem", "exem 先验"), ("model", "R模型"),
               ("self_pp", "角色库逐参数oracle"), ("self_act", "角色库单动作oracle"),
               ("rest", "角色静息值oracle")):
    f_ = float(np.mean(flat[k])) if flat[k] else float("nan")
    n_ = float(np.mean(nonflat[k])) if nonflat[k] else float("nan")
    a_ = float(np.mean(allc[k]))
    print(f"{lab:16s} {f_:12.4f} {n_:12.4f} {a_:10.4f}")

print(f"\n=== Q1 结论：非常数通道上 oracle vs model ===")
mo, so = float(np.mean(nonflat["model"])), float(np.mean(nonflat["self_pp"]))
print(f"  非常数通道 model {mo:.4f} -> 角色库oracle {so:.4f}  "
      f"({(1-so/mo)*100:+.1f}%)，这部分才是真正的形状/幅度信息")

print(f"\n=== Q2 rig 幅度泄露 ===")
r = np.array(ratios)
e = np.array(exclusive)
print(f"  目标range / 角色全局range: 中位 {np.median(r):.3f}  "
      f"均值 {r.mean():.3f}")
print(f"  目标独占该参数动画（>=99.9% 全局range）的通道占比: {e.mean()*100:.1f}%")
print(f"  目标贡献 >=50% 全局range 的通道占比: {(r>=0.5).mean()*100:.1f}%")
print("  -> 该比例越高，rig 96d（含 min/max）泄露目标幅度越严重")
