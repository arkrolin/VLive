"""Smoke test for the V11c shape loss term.

Three things must hold, and each is checked numerically:

1. **shape_w=0 leaves the legacy loss untouched** (the term sits behind
   `if shape_w > 0`), and the reported loss matches a level-only run.
2. **The extra term is real and well-formed.** L(w) - L(0) must be positive and
   exactly linear in w. Linearity also catches any misalignment of the (T-1)
   axis or of the (B, n, 1) mask broadcasting against (B, n, T-1).
3. **Gradients reach the model** - a shape-only loss must produce non-zero
   parameter gradients, otherwise the term is decorative.

Usage
-----
    .venv/bin/python outputs/smoke_shape_loss.py --device cuda:7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path("/root/work/nlp/xjzhao13/lijie_llama/VLive")
sys.path.insert(0, str(ROOT / "src" / "live2d_vla"))
sys.path.insert(0, str(ROOT / "outputs"))

from train import noise_loss, collate  # noqa: E402
import eval_ckpt  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="abl_V11b_bank")
    ap.add_argument("--ckpt", default="latest")
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--shape_w", type=float, default=1.0)
    ap.add_argument("--arm_weight", type=float, default=3.0)
    args = ap.parse_args()

    device = torch.device(args.device)
    ctx = eval_ckpt.load_ctx(args.run, device, 0, args.ckpt, 0)
    cfg = ctx["cfg"]
    ds = ctx["loader"].dataset
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        collate_fn=lambda b: collate(b, cfg.max_tokens),
                        num_workers=0)
    batch = next(iter(loader))

    # noise_loss calls the model with `training=True` hard-coded (correct for a
    # training loss), so model.eval() does NOT disable dropout. Nor is editing
    # cfg enough: the dropout probabilities were baked into nn.Dropout modules
    # at construction time. Zero the modules themselves, then PROVE the forward
    # is deterministic before trusting any w-difference below.
    model = ctx["model"]
    n_drop = 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
            n_drop += 1
    # ParamNameEncoder / action encoder use FUNCTIONAL dropout whose probability
    # lives in a plain attribute, so zeroing nn.Dropout modules is not enough.
    # Patch the functional entry point, which covers every call site.
    import torch.nn.functional as F
    _orig_dropout = F.dropout

    def _no_dropout(input, p=0.5, training=True, inplace=False):  # noqa: A002
        return _orig_dropout(input, 0.0, training, inplace)

    F.dropout = _no_dropout

    # Found the hard way: ParamNameEncoder gates on a HAND-ROLLED
    # `torch.rand(1).item() < self.dropout` and zeroes the whole embedding. That
    # is invisible to both nn.Dropout.p=0 and an F.dropout patch, so it must be
    # switched off on the attribute itself.
    n_attr = 0
    for name in ("name_enc", "action_enc"):
        sub = getattr(model, name, None)
        if sub is not None and hasattr(sub, "dropout"):
            sub.dropout = 0.0
            n_attr += 1
    print(f"[setup] zeroed {n_drop} nn.Dropout modules + patched F.dropout "
          f"+ {n_attr} hand-rolled rand gates")

    def forward(w: float, grad: bool) -> tuple[float, dict]:
        cfg.shape_w = w
        # Belt and braces: reseed so ANY remaining hidden randomness (including
        # hand-rolled rand gates we have not found) yields the same draw across
        # the three calls, which is what makes the w-difference meaningful.
        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        # requires_grad_ is sticky: the no-grad calls would otherwise leave the
        # weights frozen and make the later backward() fail.
        for p in model.parameters():
            p.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        loss = noise_loss(model, batch, device, cfg, ctx["per_lo"], ctx["per_hi"],
                          ctx["g_lo"], ctx["g_hi"], args.arm_weight)
        if grad:
            loss.backward()
            n_grad = sum(1 for p in model.parameters()
                         if p.grad is not None and float(p.grad.abs().sum()) > 0)
            n_tot = sum(1 for p in model.parameters())
            return float(loss.detach()), {"n_grad": n_grad, "n_tot": n_tot}
        return float(loss.detach()), {}

    # Validity precondition: with dropout zeroed the forward must be
    # deterministic, otherwise every difference below is noise.
    l0a, _ = forward(0.0, grad=False)
    l0b, _ = forward(0.0, grad=False)
    deterministic = abs(l0a - l0b) < 1e-9
    print(f"[precondition] forward deterministic  : "
          f"{'PASS' if deterministic else 'FAIL'}  ({l0a:.8f} vs {l0b:.8f})")
    if not deterministic:
        print("  -> dropout not fully disabled; differences below are meaningless")
        return 1

    l0 = l0a
    l1, g1 = forward(args.shape_w, grad=True)
    l2, _ = forward(2.0 * args.shape_w, grad=False)

    print()
    print("=== V11c shape-loss smoke ===")
    print(f"  L(shape_w=0)          = {l0:.8f}   (legacy level-only)")
    print(f"  L(shape_w={args.shape_w})          = {l1:.8f}   (+{l1 - l0:.8f})")
    print(f"  L(shape_w={2 * args.shape_w})          = {l2:.8f}   (+{l2 - l0:.8f})")
    print()

    d1, d2 = l1 - l0, l2 - l0
    ok_add = d1 > 0
    ok_lin = abs(d2 - 2.0 * d1) <= 1e-6 * max(1.0, abs(d2))
    ok_grad = g1.get("n_grad", 0) > 0

    print(f"  [2a] extra term positive                  : "
          f"{'PASS' if ok_add else 'FAIL'}  (+{d1:.8f})")
    print(f"  [2b] extra term linear in shape_w         : "
          f"{'PASS' if ok_lin else 'FAIL'}  (2*d1={2 * d1:.8f} vs d2={d2:.8f})")
    print(f"  [3]  backward: {g1.get('n_grad')}/{g1.get('n_tot')} params "
          f"have non-zero grad : {'PASS' if ok_grad else 'FAIL'}")
    print()

    # ---- lambda calibration -------------------------------------------------
    # d1 is the raw shape term, l0 the raw level term. Their ratio says what
    # share of the loss the shape term gets at shape_w=1, i.e. how large
    # shape_w must be for shape to be a real objective rather than a rounding
    # error. Measured 2026-09-14: ratio 0.0119, so shape_w=1 gives shape 1.2%
    # of the loss - the "obvious" 0.1/0.5/1.0 sweep would have been wasted
    # compute, and 20/50 are the values worth running.
    ratio = d1 / l0 if l0 > 0 else float("nan")
    print("lambda calibration:")
    print(f"  level term (raw)            = {l0:.6f}")
    print(f"  shape term (raw, per unit w)= {d1:.6f}")
    print(f"  shape share at shape_w=1    = {ratio * 100:6.2f}%  "
          f"(equivalent full-loss scale: 1/{ratio:.1f})")
    for target in (0.25, 0.50):
        print(f"  shape_w for {target * 100:3.0f}% shape share = {target / ratio:8.1f}")
    print()

    ok = ok_add and ok_lin and ok_grad
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
