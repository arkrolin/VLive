"""diag_ref_prior 的诚实性复核 + 可部署检索器。

上一轮发现（可能只是 min-of-many 偏差，必须排除）：
  角色逐参数 oracle 0.5031 / 角色单动作 oracle 0.8234  vs  R模型 0.9778
但从 ~21 条候选里挑最优，即使候选全是随机曲线也会偏低。

本脚本做三件事：
  A) 对照组：候选池换成「其它角色」的曲线（同数量），分离 min-of-many 偏差
     - xchar_oracle_pp / xchar_oracle_act
  B) 可部署检索器（不偷看目标）：用动作语义检索
     - retr_nn_act : 样本级, a* = argmin_a ||exem(A) - exem(a)||  (跨字符动作均值做键)
     - retr_nn_pp  : 逐参数, 同上但每个参数独立检索
     - retr_nn_top3: 取最相似的 3 个参考动作求平均
  C) 检索结果与现有模型/先验融合
     - blend: 0.5*model + 0.5*retr
     - oracle blend 系数
  D) 形状（一阶差分）口径下重复 A/B，看检索是否真的修「形状」
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

from schema import load_whitelist, load_gen_mask                  # noqa: E402
from io_motion import (list_motions, load_target_curves,          # noqa: E402
                       set_action_map)

T = 48
FPS = 30.0
rng = np.random.default_rng(1234)

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

# 对照组必须真的加载「其它角色」的曲线库，否则 oracle 数字全是 min-of-many 偏差
POOL_N = 90
_pool = [m for m in subset if m in targets and m not in set(val_models)]
pool_models = list(rng.choice(_pool, size=min(POOL_N, len(_pool)), replace=False))
print(f"[setup] 对照池: {len(pool_models)} 个非 val 角色")


def build_bank(models):
    out = {}
    for m in models:
        mdir = ROOT / "data" / "all" / m
        per_action = {}
        for action in list_motions(mdir):
            c = load_target_curves(mdir, action, targets[m], fps=FPS, T=T)
            if c:
                per_action[action] = {p: np.asarray(v, np.float32) for p, v in c.items()}
        out[m] = per_action
    return out


bank = build_bank(val_models)
xbank_all = build_bank([str(x) for x in pool_models])

exem_cache = pickle.load(open(ROOT / "outputs" / "exem_cache_580_v9all.pkl", "rb"))


def mae(a, b):
    return float(np.abs(a - b).mean())


def d1(x):
    return np.diff(x, axis=1)


# 跨角色候选池（对照组）：每个 val 角色固定配一个非 val 角色，取其动作曲线
xbank: dict[str, dict[str, dict[str, np.ndarray]]] = {}
pool_list = list(xbank_all.keys())
for m in val_models:
    xbank[m] = xbank_all[str(rng.choice(pool_list))]
assert all(len(v) > 0 for v in xbank.values()), "对照池为空，oracle 对照无效"

# 跨角色·同动作曲线池（对照池 90 个角色里所有做过动作 A 的曲线）
act_pool: dict[str, list[dict]] = {}
for mm, per in xbank_all.items():
    for a, d in per.items():
        act_pool.setdefault(a, []).append(d)

KEYS = ("exem", "model",
        "self_pp", "self_act",
        "xchar_pp", "xchar_act",
        "xchar_same_act_pp", "xchar_same_act_oracle",
        "retr_nn_pp", "retr_nn_act", "retr_top3",
        "blend_model_retr")
res = {k: [] for k in KEYS}
res_d1 = {k: [] for k in ("exem", "model", "self_act", "xchar_act", "retr_nn_act")}

for it in items:
    m, act = it["model"], it["action"]
    names = list(it["names"])
    tgt = np.asarray(it["target"], np.float64)
    exm = np.asarray(it["exem"], np.float64)
    prd = np.asarray(it["pred"], np.float64)
    P = tgt.shape[0]
    ar = np.arange(P)

    res["exem"].append(mae(exm, tgt))
    res["model"].append(mae(prd, tgt))
    res_d1["exem"].append(mae(d1(exm), d1(tgt)))
    res_d1["model"].append(mae(d1(prd), d1(tgt)))

    def stack_of(src, actions):
        return np.stack([
            np.stack([src[a].get(p, np.zeros(T)) for p in names], 0)
            for a in actions], 0) if actions else np.zeros((0, P, T))

    others = [a for a in bank.get(m, {}) if a != act]
    xothers = [a for a in xbank.get(m, {}) if a != act]

    S = stack_of(bank.get(m, {}), others)
    X = stack_of(xbank.get(m, {}), xothers)
    if S.shape[0] == 0:
        S = exm[None].copy()
        others = ["<none>"]
    if X.shape[0] == 0:
        X = S.copy()
        xothers = others

    for tag, Sx in (("self", S), ("xchar", X)):
        err = np.abs(Sx - tgt[None]).mean(2)                 # (nA, P)
        res[f"{tag}_pp"].append(mae(Sx[err.argmin(0), ar], tgt))
        res[f"{tag}_act"].append(mae(Sx[int(err.mean(1).argmin())], tgt))
    res_d1["self_act"].append(
        mae(d1(S[np.abs(S - tgt[None]).mean(2).mean(1).argmin()]), d1(tgt)))
    res_d1["xchar_act"].append(
        mae(d1(X[np.abs(X - tgt[None]).mean(2).mean(1).argmin()]), d1(tgt)))

    # ---- 可部署检索：用 exem(目标动作) 与 exem(候选动作) 的相似度排序 ----
    exm_a = np.stack([
        np.stack([np.asarray(exem_cache.get((a, p), np.zeros(T)), np.float64)
                  for p in names], 0) for a in others], 0)        # (nA, P, T)
    dist = ((exm_a - exm[None]) ** 2).mean(axis=(1, 2))           # (nA,)
    order = np.argsort(dist)
    res["retr_nn_act"].append(mae(S[order[0]], tgt))
    # 逐参数检索
    distp = ((exm_a - exm[None]) ** 2).mean(axis=2)               # (nA, P)
    res["retr_nn_pp"].append(mae(S[distp.argmin(0), ar], tgt))
    k = min(3, len(order))
    res["retr_top3"].append(mae(S[order[:k]].mean(0), tgt))
    res_d1["retr_nn_act"].append(mae(d1(S[order[0]]), d1(tgt)))
    res["blend_model_retr"].append(mae(0.5 * prd + 0.5 * S[order[0]], tgt))

    # 跨角色·同动作池（oracle）
    cands = act_pool.get(act, [])
    if cands:
        A = np.stack([np.stack([d.get(p, np.zeros(T)) for p in names], 0)
                      for d in cands], 0)
        eA = np.abs(A - tgt[None]).mean(2)
        res["xchar_same_act_pp"].append(mae(A[eA.argmin(0), ar], tgt))
        res["xchar_same_act_oracle"].append(mae(A[int(eA.mean(1).argmin())], tgt))
    else:
        res["xchar_same_act_pp"].append(mae(exm, tgt))
        res["xchar_same_act_oracle"].append(mae(exm, tgt))

base_exem = float(np.mean(res["exem"]))
base_model = float(np.mean(res["model"]))
print("\n=== 原始单位 abs_mae（越低越好）===")
print(f"{'方案':26s} {'abs_mae':>9s} {'vs exem':>9s} {'vs model':>9s}")
print("-" * 58)
LABEL = {
    "exem": "exem 先验(当前)",
    "model": "R模型 best(当前SOTA)",
    "self_pp": "A 角色库·逐参数oracle",
    "self_act": "A 角色库·单动作oracle",
    "xchar_pp": "B 它角色库·逐参数oracle(对照)",
    "xchar_act": "B 它角色库·单动作oracle(对照)",
    "xchar_same_act_pp": "B2 跨角色同动作·逐参数oracle",
    "xchar_same_act_oracle": "B2 跨角色同动作·单条oracle",
    "retr_nn_pp": "C 可部署检索·逐参数",
    "retr_nn_act": "C 可部署检索·单动作",
    "retr_top3": "C 可部署检索·top3均值",
    "blend_model_retr": "D 0.5*model + 0.5*检索",
}
for k, lab in LABEL.items():
    v = float(np.mean(res[k]))
    print(f"{lab:26s} {v:9.4f} {(1-v/base_exem)*100:+8.1f}% {(1-v/base_model)*100:+8.1f}%")

print("\n=== min-of-many 偏差剥离 ===")
sa, xa = float(np.mean(res["self_act"])), float(np.mean(res["xchar_act"]))
sp, xp = float(np.mean(res["self_pp"])), float(np.mean(res["xchar_pp"]))
print(f"  单动作 oracle: 自己 {sa:.4f} vs 它角色 {xa:.4f}  -> 真实信号 "
      f"{(1-sa/xa)*100:+.1f}%")
print(f"  逐参数 oracle: 自己 {sp:.4f} vs 它角色 {xp:.4f}  -> 真实信号 "
      f"{(1-sp/xp)*100:+.1f}%")
r = float(np.mean(res["retr_nn_act"]))
print(f"  可部署检索 {r:.4f}：捕获了 oracle 增益的 "
      f"{(base_model-r)/(base_model-sa)*100:.0f}% "
      f"(0%=无增益, 100%=达到oracle)")

print("\n=== 形状（一阶差分）MAE ===")
sb = float(np.mean(res_d1["exem"]))
for k in ("exem", "model", "self_act", "xchar_act", "retr_nn_act"):
    v = float(np.mean(res_d1[k]))
    print(f"  {k:14s}: {v:.4f}  ({(1-v/sb)*100:+.1f}% vs exem)")
