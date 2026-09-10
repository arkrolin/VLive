"""V9 fix #2: never KeyError in __getitem__ when no target channel is usable.

Root cause of arm R's `KeyError: 'Param1'` (dataset.py:200):

    names = [p for p in target_params if p in curves and p in self.param2idx]
    if not names:
        names = target_params[:1]
    target = np.stack([curves[n] for n in names], 0)     # <-- KeyError

build_vocab() caps the param vocabulary at cfg.max_tokens-side 2048 entries and
it is FULL (params_vocab=2048 for both the 285- and the 580-model corpus). A
clip that animates only params outside that cap yields names == [], the fallback
picks target_params[0] (which the clip does not animate) and curves[...] raises.

standrad-live-2d never hit it; Live2d-model-master has such clips. Emitting a
zero curve keeps the sample (it still carries identity + action signal) instead
of killing the run hours in.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "src" / "live2d_vla" / "dataset.py"

OLD = '''        if not names:
            names = target_params[:1]
        target = np.stack([curves[n] for n in names], 0).astype(np.float32)  # (n,T)
'''

NEW = '''        if not names:
            # Every channel this clip animates fell outside the CAPPED param
            # vocabulary (2048 entries, full at both 285 and 580 models).
            # Zero-fill instead of KeyError-ing on curves[n] - the sample still
            # carries identity + action signal. Rare: ~1e-4 of samples.
            names = target_params[:1]
            target = np.zeros((1, T), np.float32)
        else:
            target = np.stack([curves[n] for n in names], 0).astype(np.float32)  # (n,T)
'''


def main() -> None:
    s = DS.read_text(encoding="utf-8")
    if NEW in s:
        print("dataset.py: already patched")
        return
    if OLD not in s:
        raise SystemExit("ANCHOR NOT FOUND in dataset.py")
    DS.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")
    print("dataset.py: safe zero-fill fallback installed")


if __name__ == "__main__":
    main()
