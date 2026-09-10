"""V9 data preparation: everything the two training arms need.

Produces, on the server under outputs/:

  val_holdout_std.json   34 model names = the CURRENT validation holdout, so a
                         methodology-only arm is scored on exactly the same
                         characters as arm M (abs_mae 0.9640).
  val_holdout_all.json   the same 34 characters under their data/all link names
                         ("std__l2d01.ugirl04", ...) so the expansion arm is
                         scored on the same characters too. Without this,
                         val_frac on a 580-model whitelist would draw a
                         completely different holdout and abs_mae would stop
                         being comparable.
  dedup_skip_std.json    {model: [action, ...]} to drop from standrad-live-2d
  dedup_skip_all.json    same for data/all

De-duplication rule (agreed with the user): a sample is a (body, action) pair
and is a DUPLICATE when both the .moc3 bytes and the .motion3.json bytes match.
Body-only overlap is NOT leakage. Only one copy of a duplicate group survives.

Ordering: HELD-OUT models are hashed FIRST, so when a duplicate group contains a
validation sample the copy that survives is the validation one. Both orders are
leak-free (the other copy is dropped either way), but this one keeps the
validation set bit-identical to the baseline.

Also prints the action-consolidation ledger: sample counts before/after
canonical naming, and how many samples sit on actions shared by >= 5 bodies
(the only regime where the exem prior carries cross-character information).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/root/work/nlp/xjzhao13/lijie_llama/VLive")
sys.path.insert(0, str(ROOT / "src" / "live2d_vla"))
OUT = ROOT / "outputs"
ALL = ROOT / "data" / "all"
STD = ROOT / "standrad-live-2d"

SEED = 1234
VAL_FRAC = 0.12
K = 5

VARIANT_TAIL = re.compile(r"(_[a-z]|\d)+$")


# --------------------------------------------------------------------------- #
def canon_map() -> dict[str, str]:
    m = json.loads((OUT / "action_semantic_map.json").read_text(encoding="utf-8"))
    rev = {}
    for g, names in m["groups"].items():
        for n in names:
            rev[n.lower()] = g
    return rev


CANON = canon_map()


def raw_action(fname: str) -> str:
    """Current io_motion.list_motions rule (kept for the 'before' ledger)."""
    stem = fname[: -len(".json")] if fname.endswith(".json") else fname
    return stem.split("_", 1)[1] if "_" in stem else stem


def canon_action(fname: str) -> str:
    s = fname
    for suf in (".motion3.json", ".motion3", ".exp3.json", ".exp3", ".json"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    parts = s.split("_", 1)
    body = parts[1] if len(parts) > 1 and re.search(r"\d", parts[0]) else s
    body = body.lower().strip()
    core = VARIANT_TAIL.sub("", body).rstrip("_.-")
    if not core:
        core = body.rstrip("_.-") or s.lower()
    return CANON.get(core, core)


def list_motion_files(mdir: Path) -> list[Path]:
    return sorted((mdir / "motions").glob("*.motion3.json"))


# --------------------------------------------------------------------------- #
def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def body_id(mdir: Path) -> str:
    c = sorted(mdir.glob("*.moc3")) or sorted(mdir.glob("*.moc"))
    return md5(c[0]) if c else "nobody::" + mdir.name


# --------------------------------------------------------------------------- #
def ledger(models: list[str], root: Path, label: str):
    """Sample counts under the raw and the canonical action vocabulary."""
    for tag, fn in (("raw ", raw_action), ("canon", lambda f: canon_action(f))):
        act_bodies = defaultdict(set)
        act_files = Counter()
        for m in models:
            for f in list_motion_files(root / m):
                a = fn(f.name)
                act_bodies[a].add(m)
                act_files[a] += 1
        tot = sum(act_files.values())
        nb = {a: len(s) for a, s in act_bodies.items()}
        shared = {a for a, n in nb.items() if n >= K}
        cov = sum(v for a, v in act_files.items() if a in shared)
        print(f"  {label:26s} [{tag}] samples={tot:6d} actions={len(act_files):5d} "
              f">={K}-bodies={len(shared):4d} covered={cov / max(tot,1):5.1%} "
              f"per-action={tot / max(len(act_files),1):5.2f}")
    # how much does canonical naming shrink the corpus?
    return tot


def build_dedup(models: list[str], root: Path, holdout: set[str], out_path: Path):
    """Hash (body md5, motion md5); keep the first copy, drop the rest.

    Uses the trainer's OWN list_motions() (canonical naming + "~k" collision
    suffix) so the action keys written here match what the dataset will see.
    """
    from io_motion import list_motions, set_action_map
    set_action_map(CANON)
    order = sorted(models, key=lambda m: (0 if m in holdout else 1, m))
    seen: dict[tuple[str, str], str] = {}
    skip: dict[str, list[str]] = defaultdict(list)
    n = 0
    for m in order:
        mdir = root / m
        bid = body_id(mdir)
        for a, f in list_motions(mdir).items():
            n += 1
            key = (bid, md5(f))
            if key in seen:
                skip[m].append(a)
            else:
                seen[key] = f"{m}::{a}"
    n_skip = sum(len(v) for v in skip.values())
    out_path.write_text(json.dumps({k: v for k, v in sorted(skip.items())}, indent=1),
                        encoding="utf-8")
    print(f"  dedup {out_path.name}: samples={n} distinct={len(seen)} "
          f"dropped={n_skip} ({n_skip / max(n,1):.1%}) models_affected={len(skip)}")
    return n, n_skip


# --------------------------------------------------------------------------- #
def main() -> None:
    wl_std = json.loads((OUT / "model_whitelist.json").read_bytes())["kept"]
    wl_all = json.loads((OUT / "model_whitelist_all.json").read_bytes())["kept"]
    print(f"whitelist std={len(wl_std)} all={len(wl_all)}")

    # ---- the CURRENT holdout: deterministic rng.sample over the std whitelist
    rng = random.Random(SEED)
    n_val = max(1, int(round(len(wl_std) * VAL_FRAC)))
    holdout_std = sorted(rng.sample(wl_std, n_val))
    (OUT / "val_holdout_std.json").write_text(
        json.dumps(holdout_std, indent=1), encoding="utf-8")
    (OUT / "val_holdout_all.json").write_text(
        json.dumps(["std__" + m for m in holdout_std], indent=1), encoding="utf-8")
    print(f"holdout: {n_val} characters -> val_holdout_std.json / val_holdout_all.json")

    std_all = [m for m in wl_all if m.startswith("std__")]
    print(f"std models inside the merged whitelist: {len(std_all)} "
          f"(of {len(wl_std)} in the std whitelist)")
    missing = sorted(set("std__" + m for m in wl_std) - set(std_all))
    if missing:
        print(f"  !! {len(missing)} std whitelist models dropped by the merged gate: "
              f"{missing[:10]}")

    print("\n=== action-consolidation ledger ===")
    ledger(wl_std, STD, "std285 (arm1)")
    ledger(wl_all, ALL, "all580 (arm2)")

    print("\n=== sample-level de-duplication ===")
    build_dedup(wl_std, STD, set(holdout_std), OUT / "dedup_skip_std.json")
    ho_all = {"std__" + m for m in holdout_std}
    build_dedup(wl_all, ALL, ho_all, OUT / "dedup_skip_all.json")


if __name__ == "__main__":
    main()
