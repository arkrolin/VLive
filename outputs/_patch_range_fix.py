"""Fix the per-param range statistics: full-corpus coverage + sane fallback.

Root cause (measured locally, see outputs/diag_range_coverage_local.py)
-----------------------------------------------------------------------
``mean_range_vecs`` seeds every lo/hi with the GLOBAL range and overwrites only
params present in ``per_lo``/``per_hi``. Those dicts come from
``compute_range(train_ds, n=400)`` - and because samples are ordered by
character, those 400 samples cover only **29 of 251 train characters (11.6%)**.

Every param name absent from that 11.6% silently gets

    span = g_hi - g_lo = 1130.203

instead of its own range. Measured consequences on the val split:

* 5315 / 35778 val param-instances (14.9%) fall back to that 1130.203,
  and **100% of them DO have a real range** in the full corpus.
* their TRUE span: median 1.70, 89.5% below 22.6 -> the fallback inflates
  the median case by **665x**.
* they therefore carry **87.8%** of ``abs_mae``'s total span weighting
  (abs_mae = E[|err_norm| * span]); with full-corpus stats it would be 10.6%.

This is exactly the "decile 10" of report §20.2: the run reported
``span 1130.20, model -102.7%, win rate 1.9%`` - the fallback span reproduces
1130.203 digit for digit. So that decile is NOT a semantic class of parameter;
it is a statistics-coverage artefact.

It also explains the training side: with span 1130 the normalised error of
those params is ~0.004, so they contribute essentially no gradient - the model
never learns them - while at eval time the same 1130 multiplies their (untrained)
error back up and dominates abs_mae.

What this patch does
--------------------
1. ``compute_range(ds, n=None)`` scans the WHOLE train corpus (``cfg.range_stats_n``,
   default None = all). One pass costs ~3.5 min, so the result is cached to
   ``outputs/range_cache_{subset}_{n}_{len(ds)}.pkl`` with an atomic rename
   (safe when several runs are launched in parallel).
2. Returns a 5th value ``fb_span``: the fallback span for params still absent
   from the stats. It is the q-quantile of the per-param span distribution
   (``cfg.fb_span_q``, default 0.5 = median) instead of the global range.
   Pass ``fb_span_q < 0`` for the old global-range behaviour.
3. ``mean_range_vecs`` gains ``fb_span=None``; when given, unknown params get
   ``lo=0, hi=fb_span`` instead of ``(g_lo, g_hi)``. Default None = unchanged.

Purely additive: with ``range_stats_n=None`` and ``fb_span_q=0.5`` the numbers
change (that is the point), but every call site degrades gracefully - old
checkpoints carry their own cfg, and any missing key falls back to legacy.

Idempotent. Run from the project root on the SERVER:
    .venv/bin/python outputs/_patch_range_fix.py
"""
from __future__ import annotations

import re
from pathlib import Path

# --------------------------------------------------------------------------- #
# train.py
# --------------------------------------------------------------------------- #
tp = Path("src/live2d_vla/train.py")
src = tp.read_text(encoding="utf-8")
orig = src


def sub1(src, old, new, done_marker, what, must=True):
    """Replace `old` with `new` once. Skip silently if `done_marker` present.

    Idempotency is a hard requirement for these patch scripts - they get scp'd
    to the server and re-run after any doubt about whether they were applied.
    """
    if done_marker in src:
        print(f"  train.py: {what} already applied")
        return src
    c = src.count(old)
    if c != 1:
        if not must and c == 0:
            print(f"  train.py: {what} - anchor not found (skipped)")
            return src
        raise AssertionError(f"train.py: {what} anchor count = {c}, expected 1")
    print(f"  train.py: {what}")
    return src.replace(old, new)


# 1. helper functions before compute_range ------------------------------------
A = "def compute_range(ds, n: int = None):\n"
HELPERS = '''def _range_cache_path(cfg, n_req, n_samples):
    """Disk cache for the one-off full-corpus range scan (atomic, parallel-safe)."""
    if cfg is None:
        return None
    subset = int(getattr(cfg, "subset_models", 0))
    tag = "all" if n_req is None else str(int(n_req))
    return (ROOT / "outputs"
            / f"range_cache_{subset}_{tag}_{n_samples}.pkl")


def _fallback_span(cfg, per_lo, per_hi, g_lo, g_hi):
    """Span used for params ABSENT from the stats.

    Historically this was the GLOBAL range (g_hi - g_lo = 1130.2 here), which is
    ~665x the true median span of the params it was applied to, and (a) handed
    them ~88% of abs_mae's weighting and (b) shrank their normalised error to
    ~0.004 so the model got no gradient on them. Default is now the MEDIAN
    per-param span; cfg.fb_span_q < 0 restores the legacy global range.
    """
    q = getattr(cfg, "fb_span_q", 0.5)
    if q is None or float(q) < 0 or not per_lo:
        return float(max(g_hi - g_lo, 1.0))
    spans = np.array([per_hi[p] - per_lo[p] for p in per_lo], dtype=np.float64)
    return float(np.quantile(spans, float(q)))


'''
src = sub1(src, A, HELPERS + A, "def _range_cache_path(", "helpers inserted")


# 2. cache load at the top of compute_range -----------------------------------
A = ('    Returns (g_lo, g_hi, per_lo, per_hi). per_* are dicts keyed by param name;\n'
     '    g_lo/g_hi are global fallbacks for params unseen in the stats.\n'
     '    """\n'
     '    per_lo, per_hi = {}, {}\n')
B = ('    Returns (g_lo, g_hi, per_lo, per_hi, fb_span). per_* are dicts keyed by\n'
     '    param name; fb_span is the span given to params unseen in the stats\n'
     '    (median per-param span, NOT the global range - see _fallback_span).\n'
     '    """\n'
     '    import pickle as _pk\n'
     '\n'
     '    cfg = getattr(ds, "cfg", None)\n'
     '    n_req = None if n is None else int(n)\n'
     '    cache = _range_cache_path(cfg, n_req, len(ds))\n'
     '    if cache is not None and cache.exists():\n'
     '        try:\n'
     '            blob = _pk.loads(cache.read_bytes())\n'
     '            print(f"[range stats] loaded {cache.name} "\n'
     '                  f"({len(blob[\'per_lo\'])} params)", flush=True)\n'
     '            return (blob["g_lo"], blob["g_hi"], blob["per_lo"],\n'
     '                    blob["per_hi"], blob["fb_span"])\n'
     '        except Exception as exc:\n'
     '            print(f"[range stats] cache unreadable ({exc}) - recomputing",\n'
     '                  flush=True)\n'
     '    per_lo, per_hi = {}, {}\n')
src = sub1(src, A, B, "cache = _range_cache_path(", "compute_range cache load")


# 3. cache write + new return value ------------------------------------------
A = ("    g_lo = min(per_lo.values()) if per_lo else 0.0\n"
     "    g_hi = max(per_hi.values()) if per_hi else 1.0\n"
     "    return g_lo, g_hi, per_lo, per_hi\n")
B = ("    g_lo = min(per_lo.values()) if per_lo else 0.0\n"
     "    g_hi = max(per_hi.values()) if per_hi else 1.0\n"
     "    fb_span = _fallback_span(cfg, per_lo, per_hi, g_lo, g_hi)\n"
     "    if cache is not None:\n"
     "        try:\n"
     "            import os as _os\n"
     "\n"
     "            tmp = cache.with_suffix('.tmp')\n"
     "            tmp.write_bytes(_pk.dumps({'g_lo': g_lo, 'g_hi': g_hi,\n"
     "                                       'per_lo': per_lo, 'per_hi': per_hi,\n"
     "                                       'fb_span': fb_span}))\n"
     "            _os.replace(tmp, cache)      # atomic -> safe under parallel launches\n"
     "            print(f'[range stats] cached {cache.name} '\n"
     "                  f'({len(per_lo)} params, fallback span {fb_span:.4f})',\n"
     "                  flush=True)\n"
     "        except Exception as exc:\n"
     "            print(f'[range stats] cache write failed ({exc})', flush=True)\n"
     "    return g_lo, g_hi, per_lo, per_hi, fb_span\n")
src = sub1(src, A, B, "    return g_lo, g_hi, per_lo, per_hi, fb_span",
           "compute_range cache write + fb_span return")


# 4. mean_range_vecs: optional fb_span ---------------------------------------
A = ("def mean_range_vecs(names_b, per_lo, per_hi, g_lo, g_hi, max_tokens):\n"
     '    """(B, max_tokens) lo/hi arrays aligned to token rows (for range norm)."""\n'
     "    B = len(names_b)\n"
     "    lo_v = np.full((B, max_tokens), g_lo, np.float32)\n"
     "    hi_v = np.full((B, max_tokens), g_hi, np.float32)\n")
B = ("def mean_range_vecs(names_b, per_lo, per_hi, g_lo, g_hi, max_tokens,\n"
     "                    fb_span=None):\n"
     '    """(B, max_tokens) lo/hi arrays aligned to token rows (for range norm).\n'
     "\n"
     "    fb_span: span handed to params ABSENT from per_lo/per_hi. None (legacy)\n"
     "    fills with the global range (g_lo, g_hi); a float fills (0, fb_span).\n"
     '    """\n'
     "    B = len(names_b)\n"
     "    fill_lo, fill_hi = (0.0, float(fb_span)) if fb_span is not None \\\n"
     "        else (g_lo, g_hi)\n"
     "    lo_v = np.full((B, max_tokens), fill_lo, np.float32)\n"
     "    hi_v = np.full((B, max_tokens), fill_hi, np.float32)\n")
src = sub1(src, A, B, "                    fb_span=None):",
           "mean_range_vecs fb_span parameter")


# 5. the two mean_range_vecs call sites --------------------------------------
# The two call sites (noise_loss, recon_metrics) sit at different indents, so
# match any leading whitespace and re-emit with a consistent +4 continuation.
_RX_CALL = re.compile(
    r"^(\s*)lo_v, hi_v = mean_range_vecs\(names, per_lo, per_hi, g_lo, g_hi, "
    r"cfg\.max_tokens\)$", re.M)


def _rewire(mm):
    ind = mm.group(1)
    return (f"{ind}lo_v, hi_v = mean_range_vecs(names, per_lo, per_hi, g_lo, g_hi,\n"
            f"{ind}    cfg.max_tokens,\n"
            f"{ind}    fb_span=getattr(cfg, 'fb_span', None))")


if "fb_span=getattr(cfg, 'fb_span', None))" in src:
    print("  train.py: mean_range_vecs call sites already rewired")
else:
    src, n_call = _RX_CALL.subn(_rewire, src)
    if n_call < 1:
        raise AssertionError("no mean_range_vecs call site found in train.py")
    print(f"  train.py: rewired {n_call} mean_range_vecs call site(s)")


# 6. main(): full-corpus stats + publish fb_span on cfg ----------------------
A = "    g_lo, g_hi, per_lo, per_hi = compute_range(train_ds, n=400)\n"
B = ("    # Full-corpus range stats. n=400 covered only 29/251 train characters,\n"
     "    # so 14.9% of val param-instances silently got the 1130-wide global range.\n"
     "    g_lo, g_hi, per_lo, per_hi, fb_span = compute_range(\n"
     "        train_ds, n=getattr(cfg, 'range_stats_n', None))\n"
     "    cfg.fb_span = fb_span      # travels with the checkpoint via cfg.__dict__\n")
src = sub1(src, A, B, "    cfg.fb_span = fb_span",
           "main(): full-corpus stats + cfg.fb_span")


# 7. CLI flags (lets a run opt back into the legacy stats for a clean A/B) ----
A = ('    p.add_argument("--span_w", type=float, default=None,\n')
B = ('    p.add_argument("--range_stats_n", type=int, default=None,\n'
     '                   help="samples scanned for the per-param range stats; "\n'
     '                        "None/-1 = whole train corpus (fixed), 400 = legacy"\n'
     '                        " (covers only 29/251 characters)")\n'
     '    p.add_argument("--fb_span_q", type=float, default=None,\n'
     '                   help="quantile of the per-param span distribution used "\n'
     '                        "for params absent from the stats; <0 = legacy "\n'
     '                        "global range")\n') + A
src = sub1(src, A, B, '"--range_stats_n"', "CLI --range_stats_n / --fb_span_q",
           must=False)

A = "    if args.span_w is not None:\n        cfg.span_w = args.span_w\n"
B = (A + "    if args.range_stats_n is not None:\n"
     "        cfg.range_stats_n = (None if args.range_stats_n < 0\n"
     "                             else args.range_stats_n)\n"
     "    if args.fb_span_q is not None:\n"
     "        cfg.fb_span_q = args.fb_span_q\n")
src = sub1(src, A, B, "if args.range_stats_n is not None:",
           "CLI overrides applied", must=False)

# 8. header echo (every log line must be self-attributing) -------------------
A = '              f"span_w={cfg.span_w} span_w_cap={cfg.span_w_cap} "\n'
B = (A + '              f"range_stats_n={cfg.range_stats_n} '
     'fb_span_q={cfg.fb_span_q} "\n')
src = sub1(src, A, B, "range_stats_n={cfg.range_stats_n}", "header echo",
           must=False)

if src != orig:
    tp.write_text(src, encoding="utf-8")
    print("patched train.py (full-corpus range stats + median fallback)")
else:
    print("train.py already up to date")

# --------------------------------------------------------------------------- #
# config.py
# --------------------------------------------------------------------------- #
cp = Path("src/live2d_vla/config.py")
src = cp.read_text(encoding="utf-8")
orig = src
A = "    out_dir: Path = ROOT / \"outputs\" / \"train_runs\"\n"
B = ("    # ---- per-param range statistics (see outputs/_patch_range_fix.py) ----\n"
     "    # None = scan the whole train corpus (one pass, ~3.5 min, cached to\n"
     "    # outputs/range_cache_*.pkl). 400 = the old behaviour, which covered only\n"
     "    # 29/251 train characters and handed 14.9% of val instances a 1130-wide\n"
     "    # span (665x their true median span) -> 88% of abs_mae's weighting.\n"
     "    range_stats_n: int = None\n"
     "    # Span given to params still absent from the stats: the q-quantile of the\n"
     "    # per-param span distribution. <0 restores the legacy global range.\n"
     "    fb_span_q: float = 0.5\n"
     "\n") + A
if "range_stats_n" in src:
    print("config.py already up to date")
elif src.count(A) == 1:
    src = src.replace(A, B)
    cp.write_text(src, encoding="utf-8")
    print("patched config.py (range_stats_n / fb_span_q)")
else:
    raise AssertionError(
        f"config.py out_dir anchor count = {src.count(A)}, expected 1")

# --------------------------------------------------------------------------- #
# eval_ckpt.py  (unpack the 5th value, publish fb_span)
# --------------------------------------------------------------------------- #
ep = Path("outputs/eval_ckpt.py")
if ep.exists():
    src = ep.read_text(encoding="utf-8")
    orig = src
    A = "        g_lo, g_hi, per_lo, per_hi = compute_range(train_ds, n=400)\n"
    B = ("        g_lo, g_hi, per_lo, per_hi, fb_span = compute_range(\n"
         "            train_ds, n=getattr(cfg, 'range_stats_n', None))\n"
         "        # Old checkpoints carry no fb_span -> they keep the legacy global-\n"
         "        # range fill, which reproduces the normalisation they were trained\n"
         "        # under. New checkpoints restore their own value from cfg.\n"
         "        if getattr(cfg, 'fb_span', None) is None:\n"
         "            cfg.fb_span = fb_span\n")
    if src.count(A) != 1:
        print(f"  !! eval_ckpt.py: compute_range anchor count = {src.count(A)} (skipped)")
    else:
        src = src.replace(A, B)
        ep.write_text(src, encoding="utf-8")
        print("patched eval_ckpt.py (range stats)")

# --------------------------------------------------------------------------- #
# diag scripts (would crash on the 5-tuple otherwise)
# --------------------------------------------------------------------------- #
for name, extra in (("outputs/diag_per_character.py", False),
                    ("outputs/diag_span_buckets.py", True),
                    ("outputs/floor_check.py", True)):
    p = Path(name)
    if not p.exists():
        continue
    src = p.read_text(encoding="utf-8")
    orig = src
    A = "    g_lo, g_hi, per_lo, per_hi = compute_range(train_ds, n=400)\n"
    if src.count(A) == 1:
        src = src.replace(A, "    g_lo, g_hi, per_lo, per_hi, fb_span = compute_range(\n"
                             "        train_ds, n=getattr(cfg, 'range_stats_n', None))\n"
                             "    if getattr(cfg, 'fb_span', None) is None:\n"
                             "        cfg.fb_span = fb_span\n")
    if extra:
        # handles both the single-line and the wrapped call layout
        _RX = re.compile(r"mean_range_vecs\(\s*names,\s*per_lo,\s*per_hi,\s*"
                         r"g_lo,\s*g_hi,\s*cfg\.max_tokens\s*\)")
        src, k = _RX.subn(
            "mean_range_vecs(names, per_lo, per_hi, g_lo, g_hi, cfg.max_tokens,\n"
            "                             fb_span=getattr(cfg, 'fb_span', None))",
            src)
        if k:
            print(f"  {name}: rewired {k} mean_range_vecs call(s)")
    if src != orig:
        p.write_text(src, encoding="utf-8")
        print(f"patched {name} (range stats)")
    else:
        print(f"  {name}: no change (anchor not found - check manually)")

print("\ndone. verify with:")
print("  .venv/bin/python -c \"import sys; sys.path.insert(0,'src/live2d_vla');"
      " import train; print(train.mean_range_vecs.__defaults__)\"")
