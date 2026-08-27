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

        # deterministic whole-model holdout
        if cfg.val_frac and cfg.val_frac > 0:
            rng = random.Random(cfg.seed)
            n_val = max(1, int(round(len(self.subset) * cfg.val_frac)))
            self.holdout = set(rng.sample(self.subset, n_val))
        else:
            self.holdout = set()

        if split == "val":
            self.samples = [(m, a) for (m, a) in all_samples if m in self.holdout]
        elif split == "train":
            self.samples = [(m, a) for (m, a) in all_samples if m not in self.holdout]
        else:
            self.samples = all_samples

        # In-corpus same-action exemplar: per-(action, param) mean curve across
        # ALL whitelisted models. This is the strong action prior the model was
        # missing (replaces the broken B1 retrieval index). Cached on disk.
        self.action_param_mean = build_action_param_mean(cfg, self.subset, self.targets)

        # Transferable character-identity fingerprint: a fixed-length vector per
        # model built from its STATIC motion profile (per-param range/mean
        # distributions). Unlike the old bag-of-hash rig_sig, this is computable
        # for ANY model (incl. held-out) from its curves, so the learned
        # mapping actually transfers to unseen characters. Cached on disk.
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


def _ident_fingerprint(ranges: np.ndarray, means: np.ndarray, dim: int) -> np.ndarray:
    """Fixed-length, transferable model-identity fingerprint from per-param
    range/mean distributions.

    Two normalized histograms (range-profile + mean-profile), each of dim/2 bins.
    Ranges/means are normalized by their own global stats so the fingerprint is
    comparable across models in the shared CANON param space. This is the
    principled replacement for the bag-of-hash rig_sig: it is a STATIC property
    of the model (its overall motion profile) that can be computed for held-out
    characters too, so a mapping learned on train models transfers.
    """
    out = np.zeros(dim, np.float32)
    half = dim // 2
    if ranges.size == 0:
        return out
    r = ranges / (ranges.max() + 1e-6)                       # normalized range profile
    m = means / (np.abs(means).max() + 1e-6)                # normalized mean profile
    rh, _ = np.histogram(r, bins=half, range=(0.0, 1.0))
    mh, _ = np.histogram(m, bins=dim - half, range=(-1.0, 1.0))
    if rh.sum() > 0:
        rh = rh / rh.sum()
    if mh.sum() > 0:
        mh = mh / mh.sum()
    out[:half] = rh.astype(np.float32)
    out[half:] = mh.astype(np.float32)
    return out


def build_model_identity(cfg, subset, targets) -> dict[str, np.ndarray]:
    """Per-model transferable identity fingerprint (the new `rig` branch input).

    For each model, accumulate per-param (min, max, mean) over ALL its motion
    curves, then summarise into a fixed-length fingerprint via _ident_fingerprint.
    Cached to outputs/ident_cache_{subset_models}.pkl so train/val share it and
    reruns are instant.
    """
    from io_motion import list_motions, load_target_curves

    cache = ROOT / "outputs" / f"ident_cache_{cfg.subset_models}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    T = cfg.T
    per_model: dict[str, dict[str, list]] = {}
    for m in subset:
        if m not in targets:
            continue
        per_model[m] = {}
        for action in list_motions(ROOT / cfg.data_root / m):
            curves = load_target_curves(
                ROOT / cfg.data_root / m, action, targets[m],
                fps=cfg.fps, T=T)
            for p, c in curves.items():
                a = np.asarray(c, np.float32).ravel()
                if a.size == 0:
                    continue
                d = per_model[m].setdefault(p, [np.inf, -np.inf, 0.0, 0])
                d[0] = min(d[0], float(a.min()))
                d[1] = max(d[1], float(a.max()))
                d[2] += float(a.mean())
                d[3] += 1
    ident: dict[str, np.ndarray] = {}
    for m, pmap in per_model.items():
        ranges_l, means_l = [], []
        for p, d in pmap.items():
            if d[3] > 0:
                ranges_l.append(d[1] - d[0])
                means_l.append(d[2] / d[3])
        ident[m] = _ident_fingerprint(
            np.asarray(ranges_l, np.float32), np.asarray(means_l, np.float32),
            cfg.rig_sig_dim)
    cache.write_bytes(pickle.dumps(ident))
    return ident
