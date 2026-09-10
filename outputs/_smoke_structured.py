"""Smoke test: construct the structured-head model and run forward+backward
on synthetic data. Runs on the server (has torch). Verifies:
  - head_mode="structured" builds, forward produces (B,n,T), backward flows
  - zero-init => x0_hat == exem at step 0 (start from the prior)
  - dense head still works (backward compat)
"""
import sys, os
sys.path.insert(0, "/root/work/nlp/xjzhao13/lijie_llama/VLive")
import torch
from config import PipelineConfig
from model import Live2DModel

torch.manual_seed(0)
cfg = PipelineConfig()
cfg.head_mode = "structured"
cfg.residual_rank = 2
cfg.d_model = 64
cfg.n_layers = 2
cfg.rig_sig_dim = 96
cfg.max_tokens = 64

word2idx = {"<pad>": 0, "<unk>": 1, "angle": 2, "arm": 3, "brow": 4}
param2idx = {"<pad>": 0, "<unk>": 1, "PARAM_ANGLE_X": 2, "PARAM_ARM_L": 3}
act_vocab = 10
char2idx = {"<pad>": 0, "<unk>": 1, "a": 2, "b": 3}

m = Live2DModel(cfg, word2idx, act_vocab, cfg.max_tokens, char2idx, param2idx)
print("structured model built; head =", m.head, "gain_head =", type(m.gain_head).__name__,
      "coef_head =", type(m.coef_head).__name__, "basis", tuple(m.basis.shape))

B, n, T = 2, cfg.max_tokens, cfg.T
x_t = torch.randn(B, n, T)
t = torch.zeros(B, dtype=torch.long)
names = [["PARAM_ANGLE_X", "PARAM_ARM_L"] * (n // 2) for _ in range(B)]
rig = torch.randn(B, cfg.rig_sig_dim)
action_id = torch.zeros(B, dtype=torch.long)
token_mask = torch.zeros(B, n); token_mask[:, :6] = 1.0
exem = torch.randn(B, n, T)
action_chars = torch.zeros(B, 4, dtype=torch.long) + 2

out = m(x_t, t, names, rig, action_id, token_mask, exem, training=True,
        action_chars=action_chars)
print("out shape", tuple(out.shape))
assert out.shape == (B, n, T), out.shape

# zero-init => out == exem * (1+0) + 0 == exem (on masked positions)
diff = (out - exem)[token_mask.bool()].abs().max().item()
print("init diff from exem (should be ~0):", diff)
assert diff < 1e-4, f"structured head not zero-init! diff={diff}"

loss = ((out - exem) ** 2 * token_mask.unsqueeze(-1)).sum() / token_mask.sum()
loss.backward()
grads = [p.grad for p in m.parameters() if p.grad is not None]
print(f"backward OK: {len(grads)} params have grad; loss={loss.item():.4f}")

# dense backward-compat
cfg2 = PipelineConfig()
cfg2.head_mode = "dense"
cfg2.d_model = 64; cfg2.n_layers = 2; cfg2.rig_sig_dim = 96; cfg2.max_tokens = 64
m2 = Live2DModel(cfg2, word2idx, act_vocab, cfg2.max_tokens, char2idx, param2idx)
out2 = m2(x_t, t, names, rig, action_id, token_mask, exem, training=True,
          action_chars=action_chars)
diff2 = (out2 - exem)[token_mask.bool()].abs().max().item()
print("dense init diff from exem (should be ~0):", diff2)
assert diff2 < 1e-4
print("SMOKE TEST PASS")
