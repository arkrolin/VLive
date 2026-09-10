"""V9 CLI: expose the corpus-defining config fields on train.py (idempotent).

Without these, the two expansion arms cannot be launched: data_root,
whitelist_path, gen_mask_path, val_holdout_path, dedup_skip_path,
action_map_path and cache_tag all had no command-line flag.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "src" / "live2d_vla" / "train.py"

ARG_ANCHOR = '''    p.add_argument("--fresh", action="store_true",
                   help="ignore existing checkpoint and retrain")
    return p.parse_args()'''

ARG_ADD = '''    # ---- V9: which corpus, and how it is cleaned -------------------------- #
    p.add_argument("--data_root", type=str, default=None,
                   help="model root (one level of model dirs). "
                        "'data/all' = standrad + Live2d-model-master flattened.")
    p.add_argument("--whitelist_path", type=str, default=None)
    p.add_argument("--gen_mask_path", type=str, default=None)
    p.add_argument("--val_holdout_path", type=str, default=None,
                   help="JSON list pinning the validation characters")
    p.add_argument("--dedup_skip_path", type=str, default=None,
                   help="JSON {model: [action,...]} of (body,action) duplicates")
    p.add_argument("--action_map_path", type=str, default=None,
                   help="semantic action-consolidation map")
    p.add_argument("--cache_tag", type=str, default=None,
                   help="namespace for exem/ident/range caches")
    p.add_argument("--fresh", action="store_true",
                   help="ignore existing checkpoint and retrain")
    return p.parse_args()'''

SET_ANCHOR = '''    if args.select_metric is not None:
        cfg.select_metric = args.select_metric
'''

SET_ADD = '''    if args.select_metric is not None:
        cfg.select_metric = args.select_metric
    # ---- V9 corpus selection ---------------------------------------------- #
    if args.data_root is not None:
        p_ = Path(args.data_root)
        cfg.data_root = p_ if p_.is_absolute() else (ROOT / p_)
    if args.whitelist_path is not None:
        p_ = Path(args.whitelist_path)
        cfg.whitelist_path = p_ if p_.is_absolute() else (ROOT / p_)
    if args.gen_mask_path is not None:
        p_ = Path(args.gen_mask_path)
        cfg.gen_mask_path = p_ if p_.is_absolute() else (ROOT / p_)
    if args.val_holdout_path is not None:
        cfg.val_holdout_path = args.val_holdout_path
    if args.dedup_skip_path is not None:
        cfg.dedup_skip_path = args.dedup_skip_path
    if args.action_map_path is not None:
        cfg.action_map_path = args.action_map_path
    if args.cache_tag is not None:
        cfg.cache_tag = args.cache_tag
'''

LOG_ANCHOR = '''              f"residual_gate={cfg.residual_gate} select_metric={cfg.select_metric} "
              f"span_cond={cfg.span_cond}")'''
LOG_ADD = '''              f"residual_gate={cfg.residual_gate} select_metric={cfg.select_metric} "
              f"span_cond={cfg.span_cond}")
        print(f"[rank{rank}] V9 corpus: data_root={cfg.data_root.name} "
              f"whitelist={Path(cfg.whitelist_path).name} "
              f"gen_mask={Path(cfg.gen_mask_path).name} "
              f"cache_tag={cfg.cache_tag or '-'} "
              f"holdout={Path(cfg.val_holdout_path).name if cfg.val_holdout_path else '-'} "
              f"dedup={Path(cfg.dedup_skip_path).name if cfg.dedup_skip_path else '-'} "
              f"action_map={Path(cfg.action_map_path).name if cfg.action_map_path else '-'}")'''


def main() -> None:
    s = TRAIN.read_text(encoding="utf-8")
    for name, old, new in (("args", ARG_ANCHOR, ARG_ADD),
                           ("setattr", SET_ANCHOR, SET_ADD),
                           ("log", LOG_ANCHOR, LOG_ADD)):
        if new in s:
            print(f"train.py: {name} already patched")
            continue
        if old not in s:
            raise SystemExit(f"ANCHOR NOT FOUND ({name}):\n{old[:160]}")
        s = s.replace(old, new, 1)
        print(f"train.py: {name} patched")
    TRAIN.write_text(s, encoding="utf-8")
    print("V9 CLI patch OK")


if __name__ == "__main__":
    main()
