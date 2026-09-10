"""Training dataset for the Live2D VLA (V7.1): real corpus -> tensors.

Each sample = one (target model, action) pair:
    * target dense curves for the model's generation-target params (excludes
      physics / blink / lipsync / nonmotion via gen_mask)
    * static rig signature (moc3 bag-of-hash)   -> rig branch
    * action id                                  -> action branch (text/gesture proxy)
    * B1 cross-character exemplar mean curves    -> action branch (few-shot prior)
"""
from __future__ import annotations

import json

import pickle
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from schema import (load_whitelist, load_gen_mask, build_vocab,              # noqa: E402
                    build_action_char_vocab, encode_action_name)
from io_motion import load_target_curves                                       # noqa: E402


def _cache_suffix(cfg) -> str:
    """V9d: namespace on-disk caches so different corpora never collide."""
    t = getattr(cfg, "cache_tag", "") or ""
    return f"_{t}" if t else ""


def _load_action_map(path) -> dict[str, str]:
    """outputs/action_semantic_map.json -> {core name: canonical group}."""
    import orjson
    d = orjson.loads(Path(path).read_bytes())
    rev = {}
    for group, names in d.get("groups", {}).items():
        for n in names:
            rev[str(n).lower()] = group
    print(f"[dataset] action map: {len(rev)} names -> "
          f"{len(d.get('groups', {}))} groups", flush=True)
    return rev


def _load_dedup_skip(path) -> dict[str, set]:
    """{model: {action, ...}} to drop as (body, action) duplicates."""
    if not path:
        return {}
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise SystemExit(f"dedup_skip_path set but missing: {p}")
    import json as _json
    return {k: set(v) for k, v in _json.loads(p.read_text()).items()}


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

        # V9a: canonical action naming. MUST be installed before any
        # list_motions() call, because it changes the action vocabulary that
        # the exem / identity caches below are keyed on.
        if getattr(cfg, "action_map_path", ""):
            from io_motion import set_action_map
            set_action_map(_load_action_map(ROOT / cfg.action_map_path))
        # V9b: sample-level (body, action) de-duplication.
        self._dedup = _load_dedup_skip(getattr(cfg, "dedup_skip_path", ""))

        # param vocabulary (name -> idx) over the subset
        self.param2idx, self.word2idx = build_vocab(self.targets, cap=2048)

        # all (model, action) samples (used to build a consistent action vocab)
        from io_motion import list_motions

        all_samples = []
        n_dropped = 0
        n_empty = 0
        for m in self.subset:
            if m not in self.targets:
                continue
            skip = self._dedup.get(m)
            mdir = ROOT / cfg.data_root / m
            for action in list_motions(mdir):
                if skip and action in skip:
                    n_dropped += 1
                    continue
                # Live2d-model-master ships empty / zero-duration motion3.json
                # (scan_motion rejects Duration <= 0). They used to reach
                # __getitem__, where `names` fell back to target_params[:1] and
                # then KeyError'd on curves[n]. Drop them here instead: a
                # zero-filled sample would teach the model that some clips are
                # silent, and would pollute the exem prior.
                if not load_target_curves(mdir, action, self.targets[m],
                                          fps=cfg.fps, T=cfg.T):
                    n_empty += 1
                    continue
                all_samples.append((m, action))
        if n_dropped:
            print(f"[dataset] dedup: dropped {n_dropped} duplicate "
                  f"(body, action) samples", flush=True)
        if n_empty:
            print(f"[dataset] dropped {n_empty} empty/zero-duration motions",
                  flush=True)
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
        pinned = getattr(cfg, "val_holdout_path", "")
        if pinned:
            # V9c: a FIXED holdout, so runs with different corpus sizes are
            # scored on the same characters. Paths are resolved relative to
            # the project root when they are not absolute.
            p = Path(pinned)
            if not p.is_absolute():
                p = ROOT / p
            try:
                self.holdout = {m for m in json.loads(p.read_text()) if m in self.subset}
            except Exception as e:            # noqa: BLE001
                raise SystemExit(f"cannot read val_holdout_path {p}: {e}")
            print(f"[dataset] pinned holdout: {len(self.holdout)} characters "
                  f"from {p.name}", flush=True)
        elif cfg.val_frac and cfg.val_frac > 0:
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
        # V11a: raw material (see build_model_identity); the vector is assembled
        # per sample in __getitem__ so cfg.rig_loo can drop the target motion.
        self.model_ident = build_model_identity(cfg, self.subset, self.targets)

        # V11a: per-(model, action, param) leave-one-out statistics of THIS
        # character's other motions. Only built when the feature is enabled -
        # the one-off cache build is a full corpus pass.
        self.char_stats = getattr(cfg, "char_stats", "none")
        self.char_stat_tab = (build_char_param_stats(cfg, self.subset, self.targets)
                              if self.char_stats != "none" else {})

        # V11b: the character's own motion-bank curve table. Only the (model,
        # action) entries are stored; the K reference actions are sampled per
        # sample in __getitem__ (deterministically - see _bank_refs).
        self.bank_cond = getattr(cfg, "bank_cond", "none")
        self.bank_k = int(getattr(cfg, "bank_k", 8))
        self.bank_tab = (build_motion_bank(cfg, self.subset, self.targets)
                         if self.bank_cond != "none" else {})
        if self.bank_tab:
            self._acts_by_model: dict[str, list] = {}
            for (m, a) in self.bank_tab:
                self._acts_by_model.setdefault(m, []).append(a)

        # P2: char-level vocab over action names, for the compositional action
        # encoder that replaces the degenerate ID embedding (~4.4 samples per
        # action makes nn.Embedding a memorisation table).
        self.action_char2idx = build_action_char_vocab(sorted(self.action2idx))

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

    def _rig_vec(self, model: str, curves: dict) -> np.ndarray:
        """Assemble the `rig_sig_dim`-dim identity vector, optionally leave-one-out.

        rig_loo=True removes the target motion's own contribution to every
        per-param (min, max, mean), which is what deployment looks like (the
        target motion does not exist yet). Params with no remaining support are
        left at zero, i.e. "unknown", instead of falling back to a leaky value.
        """
        blob = self.model_ident
        gstat = blob["global_stat"]
        selected = blob["selected"]
        pm = blob["per_model"].get(model, {})
        feat_dim = self.cfg.rig_sig_dim
        half = feat_dim // 2
        loo = bool(getattr(self.cfg, "rig_loo", False))
        vec = np.zeros(feat_dim, np.float32)
        for i, p in enumerate(selected[:half]):
            d = pm.get(p)
            if d is None:
                continue
            lo, hi, s, n = d[0], d[2], d[4], d[5]
            if loo:
                a = np.asarray(curves.get(p), np.float32).ravel() \
                    if curves.get(p) is not None else None
                if a is not None and a.size:
                    t_lo, t_hi = float(a.min()), float(a.max())
                    lo = d[1] if (t_lo <= d[0] and np.isfinite(d[1])) else d[0]
                    hi = d[3] if (t_hi >= d[2] and np.isfinite(d[3])) else d[2]
                    s, n = d[4] - float(a.mean()), d[5] - 1
                else:
                    s, n = d[4], d[5]
                if n <= 0:
                    continue                      # no other motion -> unknown
            rng = hi - lo
            mn = s / max(n, 1)
            rm, rs, mm, ms = gstat[p]
            vec[2 * i] = float(np.clip((rng - rm) / rs, -3.0, 3.0))     # z-range
            vec[2 * i + 1] = float(np.clip((mn - mm) / ms, -3.0, 3.0))  # z-mean
        return vec

    def _char_stat_vec(self, model: str, action: str, names: list) -> np.ndarray:
        """(n, 5) leave-one-out character statistics for `names`, raw param units."""
        out = np.zeros((len(names), CHAR_STATS_DIM), np.float32)
        entry = self.char_stat_tab.get((model, action))
        if not entry:
            return out
        ps, vals = entry
        idx = {p: i for i, p in enumerate(ps)}
        for j, nm in enumerate(names):
            i = idx.get(nm)
            if i is not None:
                out[j] = vals[i]
        return out

    def _bank_refs(self, model: str, action: str) -> list:
        """Deterministically sample up to K reference actions from this character.

        Seeded on (model, action) with crc32 - NOT hash(), which is salted per
        process and would give every DDP rank a different bank. The target
        action is always excluded: leaving it in would let the model copy the
        answer, and at deployment the target does not exist yet.
        """
        others = [a for a in self._acts_by_model.get(model, []) if a != action]
        if not others:
            return []
        k = min(self.bank_k, len(others))
        seed = zlib.crc32(f"{model}\x00{action}".encode("utf-8"))
        rng = np.random.default_rng(seed)
        idxs = np.sort(rng.choice(len(others), size=k, replace=False))
        return [others[i] for i in idxs]

    def _bank_tensor(self, model: str, action: str, names: list):
        """(K, n, T) reference curves + (K,) mask + (K,) action ids."""
        K, T = self.bank_k, self.cfg.T
        n = len(names)
        bank = np.zeros((K, n, T), np.float32)
        mask = np.zeros((K,), np.float32)
        acts = np.zeros((K,), np.int64)
        refs = self._bank_refs(model, action)
        for k, a in enumerate(refs):
            entry = self.bank_tab.get((model, a))
            if not entry:
                continue
            ps, vals = entry
            idx = {p: i for i, p in enumerate(ps)}
            for j, nm in enumerate(names):
                i = idx.get(nm)
                if i is not None:
                    bank[k, j] = vals[i]
            mask[k] = 1.0
            acts[k] = self.action2idx.get(a, 0)
        return bank, mask, acts

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
            # Every channel this clip animates fell outside the CAPPED param
            # vocabulary (2048 entries, full at both 285 and 580 models).
            # Zero-fill instead of KeyError-ing on curves[n] - the sample still
            # carries identity + action signal. Rare: ~1e-4 of samples.
            names = target_params[:1]
            target = np.zeros((1, T), np.float32)
        else:
            target = np.stack([curves[n] for n in names], 0).astype(np.float32)  # (n,T)
        # transferable character-identity fingerprint (replaces bag-of-hash rig_sig)
        rig = self._rig_vec(model, curves).astype(np.float32)
        exem = self._exemplar_mean(model, action, names)
        out = {
            "names": names,
            "target": target,
            "rig": rig,
            "action_id": self.action2idx.get(action, 0),
            "action_chars": encode_action_name(
                action, self.action_char2idx, self.cfg.action_name_max_len),
            "exem": exem,
        }
        if self.char_stat_tab:
            out["cstats"] = self._char_stat_vec(model, action, names)
        if self.bank_tab:
            b, bm, ba = self._bank_tensor(model, action, names)
            out["bank"] = b
            out["bank_mask"] = bm
            out["bank_act"] = ba
        return out


def collate(batch, max_tokens: int):
    """Pad variable token counts to max_tokens; build masks."""
    B = len(batch)
    T = batch[0]["target"].shape[1]
    has_cs = "cstats" in batch[0]
    has_bk = "bank" in batch[0]
    K = batch[0]["bank"].shape[0] if has_bk else 0
    bank = np.zeros((B, K, max_tokens, T), np.float32) if has_bk else None
    bank_mask = np.zeros((B, K), np.float32) if has_bk else None
    bank_act = np.zeros((B, K), np.int64) if has_bk else None
    target = np.zeros((B, max_tokens, T), np.float32)
    exem = np.zeros((B, max_tokens, T), np.float32)
    cstats = np.zeros((B, max_tokens, CHAR_STATS_DIM), np.float32) if has_cs else None
    rig = np.zeros((B, batch[0]["rig"].shape[0]), np.float32)
    action_id = np.zeros(B, np.int64)
    token_mask = np.zeros((B, max_tokens), np.float32)  # 1 = active
    max_len = max((len(s.get("action_chars", [1])) for s in batch), default=1)
    action_chars = np.zeros((B, max_len), np.int64)
    names = []
    for i, s in enumerate(batch):
        n = min(len(s["names"]), max_tokens)
        token_mask[i, :n] = 1.0
        if has_bk:
            bank[i, :, :n] = s["bank"][:, :n]
            bank_mask[i] = s["bank_mask"]
            bank_act[i] = s["bank_act"]
        target[i, :n] = s["target"][:n]
        exem[i, :n] = s["exem"][:n]
        if has_cs:
            cs = np.asarray(s["cstats"], np.float32)
            cstats[i, :n] = cs[:n] * token_mask[i, :n][:, None]
        rig[i] = s["rig"]
        action_id[i] = s["action_id"]
        ac = s.get("action_chars", [1])
        action_chars[i, : len(ac)] = ac
        names.append(s["names"][:n])
    out = {
        "names": names,
        "target": torch.from_numpy(target),
        "exem": torch.from_numpy(exem),
        "rig": torch.from_numpy(rig),
        "action_id": torch.from_numpy(action_id),
        "action_chars": torch.from_numpy(action_chars),
        "token_mask": torch.from_numpy(token_mask),
    }
    if has_cs:
        out["cstats"] = torch.from_numpy(cstats)
    if has_bk:
        out["bank"] = torch.from_numpy(bank)
        out["bank_mask"] = torch.from_numpy(bank_mask)
        out["bank_act"] = torch.from_numpy(bank_act)
    return out


def build_action_param_mean(cfg, subset, targets) -> dict[tuple, np.ndarray]:
    """Per-(action, param) mean curve across all `subset` models (the action prior).

    Cached to outputs/exem_cache_{subset_models}.pkl so train/val share it and
    reruns are instant.
    """
    from io_motion import list_motions, load_target_curves

    cache = ROOT / "outputs" / f"exem_cache_{cfg.subset_models}{_cache_suffix(cfg)}.pkl"
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


def build_model_identity(cfg, subset, targets, K: int = 48) -> dict:
    """Per-model transferable identity FEATURE (the `rig` branch input).

    Replaces the old bag-of-hash / coarse 2-histogram signature with a STRUCTURED
    per-canonical-param feature: rank all params by how much they vary across
    characters (global range-variance), keep the top-K most identity-informative
    ones, and for each emit a z-scored (range, mean) pair. Output is a fixed
    `cfg.rig_sig_dim`-dim vector (= 2*K, padded/truncated).

    Crucially this is still a STATIC property computable from a model's own curves
    alone (using the global normalizers cached here), so it transfers to held-out
    characters; and being per-param (not a single histogram) it preserves the
    character's structural identity far better. Cached to disk (v3 key).

    V11a change: the returned object is no longer the finished {model: vec} map
    but the raw material needed to build it LEAVE-ONE-OUT:
        {
          "global_stat": {p: (range_mean, range_std, mean_mean, mean_std)},
          "selected":    [p, ...]                       # top-K, ordered
          "per_model":   {m: {p: (min1, min2, max1, max2, sum, cnt)}}
        }
    min2/max2 are the second-smallest / second-largest values, which is what
    makes exact leave-one-out possible: if the target motion owns the extreme,
    the LOO extreme is the runner-up. Without this, rig leaks the target's
    amplitude (measured: 29.2% of channels have the target contributing >=50%
    of the global range -> see cfg.rig_loo).
    """
    from io_motion import list_motions, load_target_curves

    cache = ROOT / "outputs" / f"ident_cache_{cfg.subset_models}_v3{_cache_suffix(cfg)}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())

    T = cfg.T
    # 1) aggregate per-model per-param over all its curves. Keep the two smallest
    #    and two largest values so leave-one-out extremes are exact.
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
                lo, hi = float(a.min()), float(a.max())
                d = per_model[m].setdefault(p, [np.inf, np.inf, -np.inf, -np.inf, 0.0, 0])
                # (min1, min2, max1, max2, sum, cnt)
                if lo < d[0]:
                    d[1] = d[0]
                    d[0] = lo
                elif lo < d[1]:
                    d[1] = lo
                if hi > d[2]:
                    d[3] = d[2]
                    d[2] = hi
                elif hi > d[3]:
                    d[3] = hi
                d[4] += float(a.mean())
                d[5] += 1

    # 2) global per-param stats (over models) + select top-K by range-variance
    per_param: dict[str, list] = {}
    for m, pmap in per_model.items():
        for p, d in pmap.items():
            per_param.setdefault(p, []).append(
                (d[0], d[2], d[4] / d[5] if d[5] > 0 else 0.0))
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

    blob = {"global_stat": global_stat, "selected": selected, "per_model": per_model}
    cache.write_bytes(pickle.dumps(blob))
    return blob


def build_motion_bank(cfg, subset, targets) -> dict:
    """Per-(model, action) curve table = that character's RAW motion bank.

    Returns {(model, action): (params, vals)} with `vals` a (nP, T) float32
    array. V11b conditions on a sample of the OTHER entries of the same model,
    turning "generate from a label" into "translate from the character's own
    existing motion", which is the task the oracle says is learnable.

    Measured (outputs/diag_ref_prior2.py), 614 val samples:
      cross-character, same action, per-param ORACLE over a 90-character pool:
        1.7673   (and the plain cross-character MEAN beats it: 1.6242)
      THIS character's other motions, per-param ORACLE over ~22 candidates:
        0.5031   (and the within-character MEAN is much worse: 1.8510)
    i.e. across characters averaging wins (candidates are noise), within a
    character SELECTION wins by 3.7x (candidates are different actions). So the
    bank must be attended over, never pooled. Cached to disk.
    """
    from io_motion import list_motions, load_target_curves

    cache = ROOT / "outputs" / f"motionbank_{cfg.subset_models}{_cache_suffix(cfg)}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())

    T = cfg.T
    out: dict = {}
    for m in subset:
        if m not in targets:
            continue
        mdir = ROOT / cfg.data_root / m
        for action in list_motions(mdir):
            curves = load_target_curves(mdir, action, targets[m], fps=cfg.fps, T=T)
            if not curves:
                continue
            ps = [p for p in curves if p in targets[m]]
            if not ps:
                continue
            vals = np.stack([np.asarray(curves[p], np.float32).ravel()[:T]
                             for p in ps], 0)
            if vals.shape[1] != T:
                continue
            out[(m, action)] = (ps, vals)
    cache.write_bytes(pickle.dumps(out))
    return out


CHAR_STATS_DIM = 5
# column order of the per-token character feature (raw param units; the caller
# normalises by the same per-param (lo, span) used for the curves):
#   0 rest : median over the character's OTHER motions of that motion's mean
#   1 amp  : median over OTHER motions of (max - min)   [typical amplitude]
#   2 lo   : min over OTHER motions of that motion's min
#   3 hi   : max over OTHER motions of that motion's max
#   4 cnt  : log1p(number of OTHER motions that animate this param)
CHAR_STATS_NAMES = ("rest", "amp", "lo", "hi", "log1p_cnt")


def build_char_param_stats(cfg, subset, targets) -> dict:
    """Per-(model, action, param) LEAVE-ONE-OUT character statistics.

    Returns {(model, action): (params, vals)} with `params` a list of param
    names and `vals` a (nP, 5) float32 array - the summary of THIS character's
    OTHER motions for that param, with the target motion's own row removed.

    Motivation (outputs/diag_ref_prior2.py / diag_ref_prior3.py): a character's
    OWN other motions are 3.5-6.7x more informative than any other character's
    curves (same-size candidate pool: 0.5031 vs 3.3719 abs_mae), and the exact
    per-param rest value alone is worth 1.31 -> 0.03 abs_mae on the 43.9% of
    channels whose target is a constant. Precomputed (not done in __getitem__)
    because the per-epoch quantile cost was ~35 s. Cached to disk.
    """
    from io_motion import list_motions, load_target_curves

    cache = ROOT / "outputs" / f"charstats_loo_{cfg.subset_models}{_cache_suffix(cfg)}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())

    T = cfg.T
    out: dict = {}
    for m in subset:
        if m not in targets:
            continue
        mdir = ROOT / cfg.data_root / m
        acts, rows = [], {}
        for action in list_motions(mdir):
            curves = load_target_curves(mdir, action, targets[m], fps=cfg.fps, T=T)
            if not curves:
                continue
            acts.append(action)
            for p, c in curves.items():
                a = np.asarray(c, np.float32).ravel()
                rows.setdefault(p, []).append(
                    (float(a.mean()), float(a.min()), float(a.max())))
        nA = len(acts)
        # per (action, param) leave-one-out summary
        for k, action in enumerate(acts):
            ps, vs = [], []
            for p, r in rows.items():
                a = np.asarray(r, np.float32).reshape(len(r), 3)
                if a.shape[0] == nA:
                    a = np.delete(a, k, axis=0)
                a = a[~np.isnan(a).any(axis=1)]
                if a.shape[0] == 0:
                    continue
                means, mins, maxs = a[:, 0], a[:, 1], a[:, 2]
                ps.append(p)
                vs.append((float(np.median(means)),
                           float(np.median(maxs - mins)),
                           float(mins.min()),
                           float(maxs.max()),
                           float(np.log1p(a.shape[0]))))
            if ps:
                out[(m, action)] = (ps, np.asarray(vs, np.float32))
    cache.write_bytes(pickle.dumps(out))
    return out
