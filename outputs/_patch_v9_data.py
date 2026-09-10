"""V9 data-hygiene patch (idempotent).

Four changes, all OFF by default so existing checkpoints reproduce exactly:

1. io_motion.list_motions: canonical action naming.
   Old rule had two defects:
     (a) Path("x.motion3.json").stem == "x.motion3", so EVERY action name in
         the vocabulary ended in ".motion3".
     (b) splitting on the FIRST underscore turns "Anim_1" -> "1" and
         "touch_idle_01" -> "idle_01"; 1651/26115 samples (6.3%) landed on
         meaningless numeric keys that merged unrelated motions.
   Canonicalisation strips the character prefix and the variant tail, then
   merges true synonyms through outputs/action_semantic_map.json.
   Collisions WITHIN one model (stand_a vs stand_c of the same character) are
   kept as separate samples via a "~k" suffix: collapsing them would silently
   delete 26% of the corpus (measured 7420 -> 5505 on std285).

2. dataset: sample-level de-duplication of (body, action) pairs via
   cfg.dedup_skip_path. Under the agreed definition a duplicate is the same
   .moc3 bytes AND the same .motion3.json bytes.

3. dataset: cfg.val_holdout_path pins the validation characters, so an
   expansion run with 580 models is scored on the SAME 34 characters as the
   285-model baseline instead of a different, larger holdout.

4. cfg.cache_tag namespaces the on-disk exem/ident/range caches. Without it a
   285-model run with different data hygiene would silently reuse
   outputs/exem_cache_285.pkl built under the old rules.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "live2d_vla"

# --------------------------------------------------------------------------- #
# 1. config.py - new fields
# --------------------------------------------------------------------------- #
CFG_ANCHOR = '    retrieval_index: Path = ROOT / "outputs" / "retrieval_index.jsonl"\n'
CFG_ADD = '''    # ---- V9 data hygiene (all OFF by default = old behaviour) -------------- #
    # Pinned validation characters (JSON list of model names). Needed because
    # val_frac draws a DIFFERENT holdout as soon as the whitelist grows, which
    # makes abs_mae incomparable across data scales.
    val_holdout_path: str = ""
    # {model: [action, ...]} to drop: the same (moc3 md5, motion md5) pair
    # already occurs earlier in the corpus. See outputs/prep_v9.py.
    dedup_skip_path: str = ""
    # Semantic action consolidation map (outputs/action_semantic_map.json).
    action_map_path: str = ""
    # Namespaces outputs/exem_cache_*.pkl, ident_cache_*.pkl, range_cache_*.pkl
    # so runs that share subset_models but not the corpus never share a cache.
    cache_tag: str = ""
'''

# --------------------------------------------------------------------------- #
# 2. io_motion.py - canonical action naming
# --------------------------------------------------------------------------- #
IO_OLD = '''def list_motions(model_dir) -> dict[str, Path]:
    """action-name -> motion3.json path for one model folder."""
    model_dir = Path(model_dir)
    out = {}
    for f in sorted(model_dir.glob("motions/*.motion3.json")):
        # Mgirl02_weixiao -> weixiao (strip the MgirlNN_ prefix)
        stem = f.stem
        if "_" in stem:
            action = stem.split("_", 1)[1]
        else:
            action = stem
        out[action] = f
    return out
'''

IO_NEW = '''# ---- V9: canonical action naming ------------------------------------------ #
# Enabled by set_action_map(); disabled (legacy behaviour) when the map is {}.
_VARIANT_TAIL = re.compile(r"(_[a-z]|\\d)+$")
_ACTION_MAP: dict[str, str] = {}
_ACTION_MAP_ON = False


def set_action_map(rev_map):
    """rev_map: normalised core name -> canonical group name. {} disables."""
    global _ACTION_MAP, _ACTION_MAP_ON
    _ACTION_MAP = dict(rev_map or {})
    _ACTION_MAP_ON = bool(rev_map)


def _canonical(stem: str) -> str:
    s = stem
    for suf in (".motion3", ".exp3"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    parts = s.split("_", 1)
    # strip the character prefix only when it actually looks like one (Mgirl02_)
    body = parts[1] if len(parts) > 1 and re.search(r"\\d", parts[0]) else s
    body = body.lower().strip()
    core = _VARIANT_TAIL.sub("", body).rstrip("_.-")
    if not core:
        core = body.rstrip("_.-") or s.lower()
    return _ACTION_MAP.get(core, core)


def list_motions(model_dir) -> dict[str, Path]:
    """action-name -> motion3.json path for one model folder."""
    model_dir = Path(model_dir)
    out: dict[str, Path] = {}
    for f in sorted(model_dir.glob("motions/*.motion3.json")):
        stem = f.stem
        if _ACTION_MAP_ON:
            action = _canonical(stem)
        else:
            # Mgirl02_weixiao -> weixiao (strip the MgirlNN_ prefix)
            action = stem.split("_", 1)[1] if "_" in stem else stem
        if action in out:
            # Same canonical name, different clip (stand_a vs stand_c). Keep
            # BOTH: collapsing them dropped 26% of the corpus (7420 -> 5505).
            k = 2
            while f"{action}~{k}" in out:
                k += 1
            action = f"{action}~{k}"
        out[action] = f
    return out
'''

# --------------------------------------------------------------------------- #
# 3. dataset.py
# --------------------------------------------------------------------------- #
DS_OLD_HEAD = '''        self.cfg = cfg
        whitelist = load_whitelist(cfg)
        gen_mask = load_gen_mask(cfg)
        self.subset = whitelist[: cfg.subset_models]
        self.targets = {m: gen_mask.get(m, []) for m in self.subset if gen_mask.get(m)}
'''
DS_NEW_HEAD = '''        self.cfg = cfg
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
'''

DS_OLD_SAMPLES = '''        all_samples = []
        for m in self.subset:
            if m not in self.targets:
                continue
            for action in list_motions(ROOT / cfg.data_root / m):
                all_samples.append((m, action))
'''
DS_NEW_SAMPLES = '''        all_samples = []
        n_dropped = 0
        for m in self.subset:
            if m not in self.targets:
                continue
            skip = self._dedup.get(m)
            for action in list_motions(ROOT / cfg.data_root / m):
                if skip and action in skip:
                    n_dropped += 1
                    continue
                all_samples.append((m, action))
        if n_dropped:
            print(f"[dataset] dedup: dropped {n_dropped} duplicate "
                  f"(body, action) samples", flush=True)
'''

DS_OLD_HOLDOUT = '''        # deterministic whole-model holdout
        if cfg.val_frac and cfg.val_frac > 0:
            rng = random.Random(cfg.seed)
            n_val = max(1, int(round(len(self.subset) * cfg.val_frac)))
            self.holdout = set(rng.sample(self.subset, n_val))
        else:
            self.holdout = set()
'''
DS_NEW_HOLDOUT = '''        # deterministic whole-model holdout
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
'''

DS_OLD_IMPORT = '''def _build_action_vocab(samples):'''
DS_NEW_IMPORT = '''def _cache_suffix(cfg) -> str:
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


def _build_action_vocab(samples):'''

DS_OLD_EXEM = '''    cache = ROOT / "outputs" / f"exem_cache_{cfg.subset_models}.pkl"'''
DS_NEW_EXEM = '''    cache = ROOT / "outputs" / f"exem_cache_{cfg.subset_models}{_cache_suffix(cfg)}.pkl"'''

DS_OLD_IDENT = '''    cache = ROOT / "outputs" / f"ident_cache_{cfg.subset_models}_v2.pkl"'''
DS_NEW_IDENT = '''    cache = ROOT / "outputs" / f"ident_cache_{cfg.subset_models}_v2{_cache_suffix(cfg)}.pkl"'''

# --------------------------------------------------------------------------- #
# 4. train.py - range cache namespace
# --------------------------------------------------------------------------- #
TR_OLD = '''    subset = int(getattr(cfg, "subset_models", 0))
    tag = "all" if n_req is None else str(int(n_req))
    return (ROOT / "outputs"
            / f"range_cache_{subset}_{tag}_{n_samples}.pkl")'''
TR_NEW = '''    subset = int(getattr(cfg, "subset_models", 0))
    tag = "all" if n_req is None else str(int(n_req))
    ct = getattr(cfg, "cache_tag", "") or ""
    if ct:
        tag = f"{tag}_{ct}"
    return (ROOT / "outputs"
            / f"range_cache_{subset}_{tag}_{n_samples}.pkl")'''

# --------------------------------------------------------------------------- #
# 5. eval_ckpt.py - corpus cache key
# --------------------------------------------------------------------------- #
EV_OLD = '''                "range_stats_n", "fb_span_q")'''
EV_NEW = '''                "range_stats_n", "fb_span_q",
                # V9: corpus-defining settings. Two runs that differ in any of
                # these build a different train/val split, so they must never
                # share a cached corpus.
                "data_root", "val_holdout_path", "dedup_skip_path",
                "action_map_path", "cache_tag")'''

EDITS = [
    ("config.py", CFG_ANCHOR, CFG_ANCHOR + CFG_ADD),
    ("io_motion.py", IO_OLD, IO_NEW),
    ("dataset.py", DS_OLD_HEAD, DS_NEW_HEAD),
    ("dataset.py", DS_OLD_SAMPLES, DS_NEW_SAMPLES),
    ("dataset.py", DS_OLD_HOLDOUT, DS_NEW_HOLDOUT),
    ("dataset.py", DS_OLD_IMPORT, DS_NEW_IMPORT),
    ("dataset.py", DS_OLD_EXEM, DS_NEW_EXEM),
    ("dataset.py", DS_OLD_IDENT, DS_NEW_IDENT),
    ("train.py", TR_OLD, TR_NEW),
]


def main() -> None:
    # io_motion needs `re`
    io = SRC / "io_motion.py"
    s = io.read_text(encoding="utf-8")
    if "\nimport re\n" not in s:
        s = s.replace("import sys\nfrom pathlib import Path\n",
                      "import re\nimport sys\nfrom pathlib import Path\n", 1)
        io.write_text(s, encoding="utf-8")
        print("io_motion.py: added `import re`")
    # dataset needs `json`
    ds = SRC / "dataset.py"
    s = ds.read_text(encoding="utf-8")
    if "import json" not in s:
        s = s.replace("from __future__ import annotations\n",
                      "from __future__ import annotations\n\nimport json\n", 1)
        ds.write_text(s, encoding="utf-8")
        print("dataset.py: added `import json`")

    for fname, old, new in EDITS:
        p = SRC / fname
        s = p.read_text(encoding="utf-8")
        if new in s:
            print(f"{fname}: already patched ({old[:38]!r}...)")
            continue
        if old not in s:
            raise SystemExit(f"ANCHOR NOT FOUND in {fname}:\n{old[:200]}")
        p.write_text(s.replace(old, new, 1), encoding="utf-8")
        print(f"{fname}: patched ({old[:38]!r}...)")

    ev = ROOT / "outputs" / "eval_ckpt.py"
    if ev.exists():
        s = ev.read_text(encoding="utf-8")
        if EV_NEW not in s and EV_OLD in s:
            ev.write_text(s.replace(EV_OLD, EV_NEW, 1), encoding="utf-8")
            print("eval_ckpt.py: corpus cache key extended")
        else:
            print("eval_ckpt.py: already patched")
    print("V9 data patch OK")


if __name__ == "__main__":
    main()
