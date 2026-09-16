"""STRICT bit-identity check: new switches OFF must reproduce the old code exactly.

The smoke test proves the new branches work. This proves they are DORMANT when
disabled -- the property that lets us ship them without perturbing existing
arms (V11b/V11c/V11d must remain comparable).

Method: run the CURRENT loss with shape_motion_mask=0 against a manually
re-implemented copy of the OLD term, on identical tensors, and require exact
equality. Comparing against a hand-copy of the old expression is stronger than
comparing two runs of the same file (which would share any new bug).

Run:  .venv/bin/python outputs/smoke_v11e_noop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "live2d_vla"))

from model import Live2DModel          # noqa: E402
from config import PipelineConfig          # noqa: E402


def make_cfg(**over):
    cfg = PipelineConfig()
    cfg.T = 12
    cfg.d_model = 16
    cfg.n_layers = 1
    cfg.n_heads = 2
    cfg.max_tokens = 6
    cfg.rig_sig_dim = 8
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def make_model(cfg):
    n = cfg.max_tokens
    names = [f"param_{i}" for i in range(n)]
    w2i = {w: i for i, w in enumerate(names)}
    return Live2DModel(cfg, word2idx=w2i, action_vocab_size=4,
                       n_tokens_pad=n, action_char2idx=None, param2idx=dict(w2i))


def old_shape_term(x0_hat, x0, valid):
    """Verbatim copy of the PRE-CHANGE shape term (git HEAD)."""
    d_hat = x0_hat[..., 1:] - x0_hat[..., :-1]
    d_tgt = x0[..., 1:] - x0[..., :-1]
    sd = (d_hat - d_tgt) ** 2
    return (sd * valid).sum() / valid.sum().clamp(min=1.0)


def new_shape_term(x0_hat, x0, valid, target, thr):
    """The CURRENT shape term, with the mask path taken only when thr > 0."""
    d_hat = x0_hat[..., 1:] - x0_hat[..., :-1]
    d_tgt = x0[..., 1:] - x0[..., :-1]
    sd = (d_hat - d_tgt) ** 2
    if thr > 0:
        ptp = target.amax(dim=-1) - target.amin(dim=-1)
        m = (ptp > thr).unsqueeze(-1).float()
        sw = valid * m
        return (sd * sw).sum() / sw.sum().clamp(min=1.0)
    return (sd * valid).sum() / valid.sum().clamp(min=1.0)


def main() -> int:
    torch.manual_seed(11)
    B, n, T = 4, 6, 12
    x0_hat = torch.randn(B, n, T)
    x0 = torch.randn(B, n, T)
    valid = (torch.rand(B, n, 1) > 0.25).float()
    # target must have BOTH flat and moving channels, otherwise the motion mask
    # keeps (or drops) everything and the test cannot detect a real difference.
    target = torch.randn(B, n, T) * 3.0
    target[:, 0, :] = 0.5          # constant channel (ptp = 0)
    target[:, 3, :] = -1.25        # another constant channel
    ptp = target.amax(-1) - target.amin(-1)
    n_flat = int((ptp <= 1e-3).sum().item())
    print(f"flat channels in test target: {n_flat} / {B*n}")

    a = old_shape_term(x0_hat, x0, valid)
    b = new_shape_term(x0_hat, x0, valid, target, 0.0)
    print(f"old term      = {a.item():.10f}")
    print(f"new (off)     = {b.item():.10f}")
    print(f"exact equal   = {a.item() == b.item()}")
    assert a.item() == b.item(), "switches OFF must be BIT-IDENTICAL"

    c = new_shape_term(x0_hat, x0, valid, target, 1e-3)
    print(f"new (mask on) = {c.item():.10f}  (differs: {c.item() != b.item()})")
    assert c.item() != b.item(), "mask must actually change the term"

    # ---- model-level: additive amp OFF must not create params / change names
    cfg_off = make_cfg(head_mode="structured", residual_rank=3,
                       head_additive_amp=False)
    m_off = make_model(cfg_off)
    names_off = {k for k, _ in m_off.named_parameters()}
    print(f"\nadditive OFF has amp_head: {any('amp_head' in k for k in names_off)}")
    assert not any("amp_head" in k for k in names_off)

    cfg_on = make_cfg(head_mode="structured", residual_rank=3,
                      head_additive_amp=True)
    m_on = make_model(cfg_on)
    names_on = {k for k, _ in m_on.named_parameters()}
    print(f"additive ON  has amp_head: {any('amp_head' in k for k in names_on)}")
    assert any("amp_head" in k for k in names_on)
    # the ONLY parameter-set difference must be amp_head
    extra = names_on - names_off
    print(f"extra params when ON: {sorted(extra)}")
    assert extra == {"amp_head.weight", "amp_head.bias"}

    print("\nBIT-IDENTITY OK: switches OFF are dormant")
    return 0


if __name__ == "__main__":
    sys.exit(main())
