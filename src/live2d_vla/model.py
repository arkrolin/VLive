"""Live2D VLA generative model (V7.1): param-as-token + additive decomposition
+ stable DiT + DDPM.

Factorised design keeps it cheap and runnable:
    * TemporalDiT  : transformer over the T axis, per token, conditioned by the
                     token's 4-way embedding + timestep (AdaLN). Uses QKNorm +
                     RMSNorm + SwiGLU ("stable DiT" patches from RDT).
    * TokenMixer   : transformer over the token axis (lets e.g. left/right arms
                     share), masked by active tokens.
The 4-way identity embedding realises f(char,action)=mu+rig+act as an *additive*
token feature; rig comes from the learned IdentityEncoder over the static
per-param motion-profile feature (transferable to held-out models), action from
the id + B1 exemplar mean fed into the loss as a target prior.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from schema import ParamNameEncoder


# --------------------------------------------------------------------------- #
# stable-DiT primitives
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


class QKNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = RMSNorm(d)
        self.k = RMSNorm(d)

    def __call__(self, q, k):
        return self.q(q), self.k(k)


class TimestepEmbed(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.SiLU(), nn.Linear(d * 4, d))

    def forward(self, t: torch.Tensor):
        half = self.mlp[0].in_features // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, dtype=torch.float32)
                          / max(half, 1)).to(t.device)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], -1)
        return self.mlp(emb)


def _adaln(cond, scale_shift):
    s, sh = scale_shift.chunk(2, -1)
    return (1.0 + s) * cond + sh


class TemporalBlock(nn.Module):
    def __init__(self, d, n_heads, mlp_ratio, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.qk = QKNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ad1 = nn.Linear(d, 2 * d)
        self.norm2 = RMSNorm(d)
        self.ad2 = nn.Linear(d, 2 * d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d * mlp_ratio), nn.SiLU(),
            nn.Linear(d * mlp_ratio, d * mlp_ratio), nn.SiLU(),
            nn.Linear(d * mlp_ratio, d))
        # zero-init AdaLN so the block is identity at init (stable DiT start)
        nn.init.zeros_(self.ad1.weight)
        nn.init.zeros_(self.ad1.bias)
        nn.init.zeros_(self.ad2.weight)
        nn.init.zeros_(self.ad2.bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cond):
        # x: (Bn, T, d); cond: (Bn, d)
        a = _adaln(self.norm1(x), self.ad1(cond))
        q, k = self.qk(a, a)
        att, _ = self.attn(q, k, a)
        x = x + att
        m = _adaln(self.norm2(x), self.ad2(cond))
        x = x + self.drop(self.mlp(m))
        return x


class TokenMixBlock(nn.Module):
    def __init__(self, d, n_heads, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.qk = QKNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm2 = RMSNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.SiLU(), nn.Linear(d * 4, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: (Bt, n, d); mask: (Bt, n) 1=active
        a = self.norm1(x)
        q, k = self.qk(a, a)
        att, _ = self.attn(q, k, a)
        x = x + att
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x * mask.unsqueeze(-1)


# --------------------------------------------------------------------------- #
# diffusion model
# --------------------------------------------------------------------------- #
class Live2DModel(nn.Module):
    def __init__(self, cfg, word2idx, action_vocab_size, n_tokens_pad):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.n_tokens_pad = n_tokens_pad
        self.name_enc = ParamNameEncoder(word2idx, d, cfg.name_dropout)
        self.identity_enc = nn.Sequential(
            nn.Linear(cfg.rig_sig_dim, d), nn.SiLU(), nn.LayerNorm(d),
            nn.Linear(d, d), nn.SiLU(), nn.LayerNorm(d),
            nn.Linear(d, d))
        self.deformer = nn.Parameter(torch.zeros(d))
        self.action_emb = nn.Embedding(action_vocab_size, d)
        self.time_emb = TimestepEmbed(d)
        self.in_proj = nn.Linear(2, d)   # x_t channel + in-corpus exem channel
        self.temporal = nn.ModuleList(
            [TemporalBlock(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
             for _ in range(cfg.n_layers)])
        self.tok_mix = nn.ModuleList(
            [TokenMixBlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)])
        self.head = nn.Linear(d, 1)
        # zero-init head: x0_hat = exem (the prior) at init -> loss starts at floor
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _token_emb(self, names, rig, action_id, training):
        B = len(names)
        embs = []
        dev = rig.device
        for i in range(B):
            ne = self.name_enc(names[i], training=training).to(dev)  # (n_i, d)
            if ne.shape[0] == 0:
                ne = torch.zeros(1, self.cfg.d_model, device=dev)
            embs.append(ne)
        # pad to n_tokens_pad
        tok = torch.zeros(B, self.n_tokens_pad, self.cfg.d_model, device=rig.device)
        for i in range(B):
            n = embs[i].shape[0]
            tok[i, :n] = embs[i]
        rig_e = self.identity_enc(rig).unsqueeze(1)        # (B,1,d) learned identity embedding
        act_e = self.action_emb(action_id).unsqueeze(1)   # (B,1,d)
        tok = tok + rig_e + act_e + self.deformer
        return tok  # (B, n_pad, d)

    def forward(self, x_t, t, names, rig, action_id, token_mask, exem, training=True):
        B, n, T = x_t.shape
        d = self.cfg.d_model
        tok = self._token_emb(names, rig, action_id, training)        # (B,n,d)
        t_e = self.time_emb(t).unsqueeze(1)                           # (B,1,d)
        cond = (tok + t_e).reshape(B * n, 1, d)                      # (Bn, 1, d)

        h = self.in_proj(torch.cat([x_t.unsqueeze(-1), exem.unsqueeze(-1)], -1))  # (B,n,T,d)
        # temporal over T
        h = h.reshape(B * n, T, d)
        for blk in self.temporal:
            h = blk(h, cond)
        h = h.reshape(B, n, T, d)
        # token mixer over n (per timestep)
        h = h.permute(0, 2, 1, 3).reshape(B * T, n, d)               # (BT, n, d)
        m = token_mask.unsqueeze(1).expand(B, T, n).reshape(B * T, n)
        for blk in self.tok_mix:
            h = blk(h, m)
        h = h.reshape(B, T, n, d).permute(0, 2, 1, 3)                # (B,n,T,d)
        out = self.head(h).squeeze(-1)                               # (B,n,T)
        out = exem + out                                            # x0_hat = prior(exem) + residual
        return out * token_mask.unsqueeze(-1)

    # --------------------------- DDPM --------------------------- #
    def ddpm_schedule(self, steps, cosine: bool = False):
        if cosine:
            # Nichol & Dhariwal cosine schedule (lower loss floor, smoother).
            s = 0.008
            x = torch.linspace(0.0, 1.0, steps + 1)
            f = torch.cos((x + s) / (1.0 + s) * math.pi / 2.0) ** 2
            alphabar = f / f[0]
            betas = torch.clamp(1.0 - alphabar[1:] / alphabar[:-1], 1e-5, 0.999)
            alphas = 1.0 - betas
            return betas, alphas, alphabar[1:]
        betas = torch.linspace(1e-4, 2e-2, steps)
        alphas = 1 - betas
        alphabar = torch.cumprod(alphas, 0)
        return betas, alphas, alphabar

    def q_sample(self, x0, t, noise, alphabar):
        a = alphabar[t].sqrt().view(-1, 1, 1)
        return a * x0 + (1 - a).sqrt().view(-1, 1, 1) * noise
