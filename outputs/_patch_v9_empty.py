"""V9 fix: drop empty / zero-duration motions at corpus-build time.

R (data/all) died in __getitem__ with

    KeyError: 'Param1'
    dataset.py:200   target = np.stack([curves[n] for n in names], 0)

`names` is `target_params` filtered to what the clip actually animates; when a
motion animates none of them (empty motion3.json, or Meta.Duration <= 0, which
motion_curves.scan_motion rejects) `names` is empty and the fallback
`names = target_params[:1]` indexes a curve that does not exist.

standrad-live-2d never shipped such files, so this stayed hidden for 8 arms;
Live2d-model-master has them. Filtering at build time is better than zero-
filling: a zero-filled sample would teach the model that some clips are silent
and would also pollute the exem prior.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "src" / "live2d_vla" / "dataset.py"

OLD = '''        all_samples = []
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

NEW = '''        all_samples = []
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
'''


def main() -> None:
    s = DS.read_text(encoding="utf-8")
    if NEW in s:
        print("dataset.py: already patched")
        return
    if OLD not in s:
        raise SystemExit("ANCHOR NOT FOUND in dataset.py")
    DS.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")
    print("dataset.py: empty-motion filter installed")


if __name__ == "__main__":
    main()
