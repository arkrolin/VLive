"""Training dataset for the Live2D VLA (V7.1): real corpus -> tensors.

Each sample = one (target model, action) pair:
    * target dense curves for the model's generation-target params (excludes
      physics / blink / lipsync / nonmotion via gen_mask)
    * static rig signature (moc3 bag-of-hash)   -> rig branch
    * action id                                  -> action branch (text/gesture proxy)
    * B1 cross-character exemplar mean curves    -> action branch (few-shot prior)
"""
from __future__ import annotations

import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from schema import load_whitelist, load_gen_mask, build_vocab  # noqa: E402
from io_motion import load_target_curves                                       # noqa: E402


def _build_action_vocab(samples):
    acts = sorted({a for _, a in samples})
    return {a: i for i, a in enumerate(acts)}, acts


class Live2DDataset(Dataset):
    def __init__(self, cfg, action_vocab=None, split: str = None):
        """split: None (all) | "train" | "val". Vocab is always built over the
        full subset so train/val share embeddings; samples are then filtered by
        a deterministic whole-model holdout (val_frac of models)."""
        self.cfg = cfg
        whitelist = load_whitelist(cfg)
        gen_mask = load_gen_mask(cfg)
        self.subset = whitelist[: cfg.subset_models]
        self.targets = {m: gen_mask.get(m, []) for m in self.subset if gen_mask.get(m)}

        # param vocabulary (name -> idx) over the subset
        self.param2idx, self.word2idx = build_vocab(self.targets, cap=2048)

        # all (model, action) samples (used to build a consistent action vocab)
        from io_motion import list_motions

        all_samples = []
        for m in self.subset:
            if m not in self.targets:
                continue
            for action in list_motions(ROOT / cfg.data_root / m):
                all_samples.append((m, action))
        self.action2idx, _ = _build_action_vocab(all_samples) if action_vocab is None \
            else (action_vocab, list(action_vocab))

        # How many models share each action. Needed to build an HONEST val split:
        # 72% of actions occur in only one model, and for those the exem prior
        # (the (action,param) mean) is literally that model's own curve, so the
        # sample measures memorisation, not generalisation.
        self.action_nmodels: dict[str, int] = {}
        for _, a in all_samples:
            self.action_nmodels[a] = self.action_nmodels.get(a, 0) + 1

        # deterministic whole-model holdout
        if cfg.val_frac and cfg.val_frac > 0:
            rng = random.Random(cfg.seed)
            n_val = max(1, int(round(len(self.subset) * cfg.val_frac)))
            self.holdout = set(rng.sample(self.subset, n_val))
        else:
            self.holdout = set()

        if split == "val":
            self.samples = [(m, a) for (m, a) in all_samples if m in self.holdout]
            # P0: keep only actions shared by >= N models. This is the real
            # deployment task ("known action, new character") and the only honest
            # generalisation test. Training still uses every sample.
            k = getattr(cfg, "eval_shared_min_models", 0)
            if k and k > 0:
                kept = [s for s in self.samples if self.action_nmodels[s[1]] >= k]
                if kept:
                    self.samples = kept
        elif split == "train":
            self.samples = [(m, a) for (m, a) in all_samples if m not in self.holdout]
        else:
            self.samples = all_samples

        # In-corpus same-action exemplar: per-(action, param) mean curve across
        # ALL whitelisted models. This is the strong action prior the model was
        # missing (replaces the broken B1 retrieval index). Cached on disk.
        self.action_param_mean = build_action_param_mean(cfg, self.subset, self.targets)

        # Transferable character-identity feature: a fixed-length STRUCTURED
        # vector per model built from its STATIC motion profile (per-canonical-
        # param z-scored range+mean for the top-K most identity-informative
        # params). Unlike the old bag-of-hash rig_sig / coarse 2-histogram, this
        # preserves per-param structural identity and is computable for ANY model
        # (incl. held-out) from its curves, so the learned mapping transfers.
        # Cached on disk. Consumed by the learnable IdentityEncoder in model.py.
        self.model_ident = build_model_identity(cfg, self.subset, self.targets)

    def __len__(self):
        return len(self.samples)

    def _exemplar_mean(self, model: str, action: str, names: list[str]) -> np.ndarray:
        """In-corpus same-action mean curve for this sample's params (the prior)."""
        T = self.cfg.T
        exem = np.zeros((len(names), T), np.float32)
        for i, nm in enumerate(names):
            c = self.action_param_mean.get((action, nm))
            if c is not None:
                exem[i] = c
        return exem

    def __getitem__(self, idx):
        model, action = self.samples[idx]
        T = self.cfg.T
        target_params = self.targets[model]
        curves = load_target_curves(
            ROOT / self.cfg.data_root / model, action, target_params,
            fps=self.cfg.fps, T=T)
        # active names present in this clip AND in the vocabulary
        names = [p for p in target_params if p in curves and p in self.param2idx]
        if not names:
            names = target_params[:1]
        target = np.stack([curves[n] for n in names], 0).astype(np.float32)  # (n,T)
        # transferable character-identity fingerprint (replaces bag-of-hash rig_sig)
        rig = self.model_ident.get(model, np.zeros(self.cfg.rig_sig_dim, np.float32))
        exem = self._exemplar_mean(model, action, names)
        return {
            "names": names,
            "target": target,
            "rig": rig.astype(np.float32),
            "action_id": self.action2idx.get(action, 0),
            "exem": exem,
        }


def collate(batch, max_tokens: int):
    """Pad variable token counts to max_tokens; build masks."""
    B = len(batch)
    T = batch[0]["target"].shape[1]
    target = np.zeros((B, max_tokens, T), np.float32)
    exem = np.zeros((B, max_tokens, T), np.float32)
    rig = np.zeros((B, batch[0]["rig"].shape[0]), np.float32)
    action_id = np.zeros(B, np.int64)
    token_mask = np.zeros((B, max_tokens), np.float32)  # 1 = active
    names = []
    for i, s in enumerate(batch):
        n = min(len(s["names"]), max_tokens)
        token_mask[i, :n] = 1.0
        target[i, :n] = s["target"][:n]
        exem[i, :n] = s["exem"][:n]
        rig[i] = s["rig"]
        action_id[i] = s["action_id"]
        names.append(s["names"][:n])
    out = {
        "names": names,
        "target": torch.from_numpy(target),
        "exem": torch.from_numpy(exem),
        "rig": torch.from_numpy(rig),
        "action_id": torch.from_numpy(action_id),
        "token_mask": torch.from_numpy(token_mask),
    }
    return out


def build_action_param_mean(cfg, subset, targets) -> dict[tuple, np.ndarray]:
    """Per-(action, param) mean curve across all `subset` models (the action prior).

    Cached to outputs/exem_cache_{subset_models}.pkl so train/val share it and
    reruns are instant.
    """
    from io_motion import list_motions, load_target_curves

    cache = ROOT / "outputs" / f"exem_cache_{cfg.subset_models}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    T = cfg.T
    acc: dict[tuple, list] = {}
    for m in subset:
        if m not in targets:
            continue
        for action in list_motions(ROOT / cfg.data_root / m):
            curves = load_target_curves(
                ROOT / cfg.data_root / m, action, targets[m],
                fps=cfg.fps, T=T)
            for p, c in curves.items():
                acc.setdefault((action, p), []).append(np.asarray(c, np.float32))
    mean: dict[tuple, np.ndarray] = {}
    for k, lst in acc.items():
        mean[k] = np.mean(np.stack(lst, 0), 0).astype(np.float32)
    cache.write_bytes(pickle.dumps(mean))
    return mean


def build_model_identity(cfg, subset, targets, K: int = 48) -> dict[str, np.ndarray]:
    """Per-model transferable identity FEATURE (the `rig` branch input).

    Replaces the old bag-of-hash / coarse 2-histogram signature with a STRUCTURED
    per-canonical-param feature: rank all params by how much they vary across
    characters (global range-variance), keep the top-K most identity-informative
    ones, and for each emit a z-scored (range, mean) pair. Output is a fixed
    `cfg.rig_sig_dim`-dim vector (= 2*K, padded/truncated).

    Crucially this is still a STATIC property computable from a model's own curves
    alone (using the global normalizers cached here), so it transfers to held-out
    characters; and being per-param (not a single histogram) it preserves the
    character's structural identity far better. Cached to disk (v2 key).
    """
    from io_motion import list_motions, load_target_curves

    cache = ROOT / "outputs" / f"ident_cache_{cfg.subset_models}_v2.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())

    T = cfg.T
    # 1) aggregate per-model per-param (min, max, sum, cnt) over all its curves
    per_model: dict[str, dict[str, list]] = {}
    for m in subset:
        if m not in targets:
            continue
        per_model[m] = {}
        for action in list_motions(ROOT / cfg.data_root / m):
            curves = load_target_curves(
                ROOT / cfg.data_root / m, action, targets[m], fps=cfg.fps, T=T)
            for p, c in curves.items():
                a = np.asarray(c, np.float32).ravel()
                if a.size == 0:
                    continue
                d = per_model[m].setdefault(p, [np.inf, -np.inf, 0.0, 0])
                d[0] = min(d[0], float(a.min()))
                d[1] = max(d[1], float(a.max()))
                d[2] += float(a.mean())
                d[3] += 1

    # 2) global per-param stats (over models) + select top-K by range-variance
    per_param: dict[str, list] = {}
    for m, pmap in per_model.items():
        for p, d in pmap.items():
            per_param.setdefault(p, []).append(
                (d[0], d[1], d[2] / d[3] if d[3] > 0 else 0.0))
    global_stat: dict[str, tuple] = {}
    for p, vals in per_param.items():
        rngs = np.array([v[1] - v[0] for v in vals], np.float32)
        means = np.array([v[2] for v in vals], np.float32)
        global_stat[p] = (rngs.mean(), rngs.std() + 1e-6,
                          means.mean(), means.std() + 1e-6)
    order = sorted(global_stat.keys(),
                   key=lambda p: global_stat[p][1], reverse=True)
    K = min(K, len(order))
    selected = order[:K]
    sel_idx = {p: i for i, p in enumerate(selected)}
    feat_dim = cfg.rig_sig_dim
    half = feat_dim // 2

    # 3) build the fixed-dim structured feature per model
    ident: dict[str, np.ndarray] = {}
    for m, pmap in per_model.items():
        vec = np.zeros(feat_dim, np.float32)
        for p, d in pmap.items():
            i = sel_idx.get(p)
            if i is None or i >= half:
                continue
            rng = d[1] - d[0]
            mn = d[2] / d[3] if d[3] > 0 else 0.0
            rm, rs, mm, ms = global_stat[p]
            vec[2 * i]     = float(np.clip((rng - rm) / rs, -3.0, 3.0))   # z-range
            vec[2 * i + 1] = float(np.clip((mn - mm) / ms, -3.0, 3.0))    # z-mean
        ident[m] = vec
    cache.write_bytes(pickle.dumps(ident))
    return ident
