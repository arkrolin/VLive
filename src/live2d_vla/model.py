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

from schema import ParamNameEncoder, ActionNameEncoder
from dataset import CHAR_STATS_DIM


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
    def __init__(self, cfg, word2idx, action_vocab_size, n_tokens_pad,
                 action_char2idx=None, param2idx=None):
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
        # ---- per-token span conditioning (V8.3) ----
        # Inject log(span) so the model SEES each param's absolute range instead
        # of having to infer it from the name. Zero-init -> a "log" run starts
        # bit-identical to "none" and only learns to use the span as it trains.
        self.span_cond = getattr(cfg, "span_cond", "none")
        if self.span_cond == "log":
            self.span_enc = nn.Linear(1, d)
            nn.init.zeros_(self.span_enc.weight)
            nn.init.zeros_(self.span_enc.bias)
        else:
            self.span_enc = None
        # ---- V11a: per-token character conditioning ----
        # Leave-one-out statistics of THIS character's OTHER motions for the
        # same param (rest value / typical amplitude / p0-p100 envelope /
        # support count), computed in dataset.build_char_param_stats. Measured:
        # the character's own bank is 3.5-6.7x more informative than any other
        # character's curves, and its rest value is worth 1.31 -> 0.03 abs_mae
        # on the 43.9% of channels whose target is a constant. Zero-init so a
        # char_stats run starts bit-identical to the un-conditioned baseline.
        self.char_stats = getattr(cfg, "char_stats", "none")
        if self.char_stats == "mlp":
            self.char_stats_enc = nn.Linear(CHAR_STATS_DIM, d)
            nn.init.zeros_(self.char_stats_enc.weight)
            nn.init.zeros_(self.char_stats_enc.bias)
        else:
            self.char_stats_enc = None
        # ---- action conditioning (P2) ----
        # "id"   : plain lookup - degenerates into a memorisation table at
        #          ~4.4 samples per action (1679 actions / 7382 samples).
        # "name" : compositional char-level encoder over the semantic pinyin
        #          action name - shares statistics, handles unseen names.
        self.action_cond = getattr(cfg, "action_cond", "id")
        if self.action_cond in ("id", "both"):
            self.action_emb = nn.Embedding(action_vocab_size, d)
        if self.action_cond in ("name", "both"):
            if action_char2idx is None:
                raise ValueError("action_cond='name' requires action_char2idx")
            self.action_name_enc = ActionNameEncoder(
                action_char2idx, d, getattr(cfg, "action_name_max_len", 32),
                getattr(cfg, "action_dropout", 0.0))
        self.time_emb = TimestepEmbed(d)
        # ---- V11b: character motion-bank cross-attention ----
        # query = the param token; keys/values = the K reference curves of THIS
        # character (its other motions), one key per (reference action, param).
        # Selection happens PER PARAM, which is the only regime the oracle says
        # works: within a character, per-param selection (0.5031) beats the mean
        # (1.8510) by 3.7x, while across characters the mean wins - so the bank
        # must be attended over, never pooled.
        # bank_cond != "none" adds a 3rd input channel (the attended reference
        # curve) whose weights are ZERO-INIT, so the run starts bit-identical to
        # the 2-channel baseline.
        self.bank_cond = getattr(cfg, "bank_cond", "none")
        if self.bank_cond == "attn":
            self.bank_k = int(getattr(cfg, "bank_k", 8))
            self.bank_curve_enc = nn.Linear(cfg.T, d)
            self.bank_q = nn.Linear(d, d)
            self.bank_gate = nn.Parameter(torch.zeros(1))
            self.in_proj = nn.Linear(3, d)   # x_t + exem + attended bank curve
            nn.init.zeros_(self.in_proj.weight[:, 2:3])
        else:
            self.bank_k = 0
            self.in_proj = nn.Linear(2, d)   # x_t channel + in-corpus exem channel
        self.temporal = nn.ModuleList(
            [TemporalBlock(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
             for _ in range(cfg.n_layers)])
        self.tok_mix = nn.ModuleList(
            [TokenMixBlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)])
        self.head_mode = getattr(cfg, "head_mode", "dense")
        if self.head_mode == "structured":
            self.residual_rank = int(getattr(cfg, "residual_rank", 2))
            # per-param amplitude gain alpha in R^P  ->  exem * (1 + alpha)
            self.gain_head = nn.Linear(d, 1)
            # per-param coefficients over r global time bases -> coef @ basis
            self.coef_head = nn.Linear(d, self.residual_rank)
            # global low-frequency time bases B in R^(r, T), fixed at DCT-II.
            # Fixed (not learned) so the shape correction stays a LINEAR map in
            # coef space, exactly matching the oracle's rank-r decomposition.
            self.register_buffer("basis", self._dct_basis(self.residual_rank, self.cfg.T))
            # zero-init: x0_hat = exem*(1+0) + 0 = exem at init (same start as dense)
            nn.init.zeros_(self.gain_head.weight)
            nn.init.zeros_(self.gain_head.bias)
            nn.init.zeros_(self.coef_head.weight)
            nn.init.zeros_(self.coef_head.bias)
            # V11e(b2): ADDITIVE amplitude correction in R^P. The multiplicative
            # gain above cannot move a near-zero prior or fix a sign error; the
            # prior under-shoots in 28.4% of moving channels, so this term is
            # required for those. Zero-init -> init stays x0_hat == exem.
            self.additive_amp = bool(getattr(cfg, "head_additive_amp", False))
            if self.additive_amp:
                self.amp_head = nn.Linear(d, 1)
                nn.init.zeros_(self.amp_head.weight)
                nn.init.zeros_(self.amp_head.bias)
            else:
                self.amp_head = None
            self.head = None
        else:
            self.gain_head = None
            self.coef_head = None
            self.amp_head = None
            self.additive_amp = False
            self.head = nn.Linear(d, 1)
            # zero-init head: x0_hat = exem (the prior) at init -> loss starts at floor
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        # ---- learnable residual gate (V8.2) ----
        # g = sigmoid(bias + Emb(param_name)) applied to the residual, so the
        # model can learn to fall back to the exem prior per param KIND. The
        # gate reads only the static param name (never the instance's residual)
        # so it cannot collapse into "always predict nothing". bias=4 -> g~0.98
        # at init, i.e. it starts as an exact no-op on the ungated model.
        self.residual_gate = getattr(cfg, "residual_gate", "none")
        if self.residual_gate == "name":
            if param2idx is None:
                raise ValueError("residual_gate='name' requires param2idx")
            # plain dict is NOT a module attribute -> it is not a parameter and
            # is not touched by .to(); safe to store (unlike an nn.Embedding,
            # which MUST be assigned on an nn.Module - see the ParamNameEncoder
            # bug: a plain class holding an Embedding never reached the optimiser).
            self.param2idx = dict(param2idx)
            self.gate_emb = nn.Embedding(len(self.param2idx), 1)
            nn.init.zeros_(self.gate_emb.weight)
            self.gate_bias = nn.Parameter(torch.tensor(4.0))
        else:
            self.gate_emb = None

    def _dct_basis(self, r: int, T: int) -> torch.Tensor:
        """First `r` DCT-II bases over T time steps, orthonormal (r, T).

        Basis 0 is the constant (DC) component; bases 1..r-1 are increasingly
        high-frequency. Using only the first r gives a smooth, low-frequency
        shape correction -- the structural prior that kills the 2x jitter the
        dense head exhibited (diag_quality2.py H1).
        """
        n = torch.arange(T, dtype=torch.float32).unsqueeze(0)   # (1, T)
        k = torch.arange(r, dtype=torch.float32).unsqueeze(1)   # (r, 1)
        B = torch.cos(math.pi * k * (n + 0.5) / T)              # (r, T)
        # orthonormalise (DCT-II is orthogonal; scale to unit rows)
        B = B / B.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return B

    def _gate_ids(self, names, device):
        """(B, n_pad) LongTensor of param-name ids for the residual gate."""
        B = len(names)
        n = self.n_tokens_pad
        gi = torch.zeros(B, n, dtype=torch.long, device=device)
        get = self.param2idx.get
        for i in range(B):
            for j, nm in enumerate(names[i]):
                if j >= n:
                    break
                gi[i, j] = get(nm, 1)
        return gi

    def _action_emb(self, action_id, action_chars, training):
        """Action conditioning: ID lookup and/or compositional name encoding."""
        parts = []
        if self.action_cond in ("id", "both"):
            parts.append(self.action_emb(action_id))
        if self.action_cond in ("name", "both"):
            if action_chars is None:
                parts.append(torch.zeros(action_id.shape[0], self.cfg.d_model,
                                         device=action_id.device))
            else:
                parts.append(self.action_name_enc(
                    action_chars.to(action_id.device), training=training))
        if not parts:
            return torch.zeros(action_id.shape[0], self.cfg.d_model,
                               device=action_id.device)
        out = parts[0]
        for p in parts[1:]:
            out = out + p
        return out                                                # (B, d)

    def _token_emb(self, names, rig, action_id, action_chars, training, span=None,
                   cstats=None):
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
        act_e = self._action_emb(action_id, action_chars, training).unsqueeze(1)  # (B,1,d)
        tok = tok + rig_e + act_e + self.deformer
        if self.span_enc is not None and span is not None:
            ls = torch.log(span.clamp(min=1e-6)).unsqueeze(-1)   # (B, n, 1)
            tok = tok + self.span_enc(ls)                        # (B, n, d)
        if self.char_stats_enc is not None and cstats is not None:
            tok = tok + self.char_stats_enc(cstats.to(tok.dtype))   # (B, n, d)
        return tok  # (B, n_pad, d)

    def _bank_attend(self, tok, bank, bank_mask, bank_act, action_emb_fn):
        """Per-param soft selection over the character's motion bank.

        tok       (B, n, d)   current token embeddings -> queries
        bank      (B, K, n, T) reference curves, ALREADY normalised like x0
        bank_mask (B, K)      1 = a real reference action, 0 = padding
        bank_act  (B, K)      action ids of the references
        Returns (att_curve (B,n,T), att_emb (B,n,d)).
        """
        B, K, n, T = bank.shape
        d = tok.shape[-1]
        kv = self.bank_curve_enc(bank)                       # (B,K,n,d)
        kv = kv + action_emb_fn(bank_act).unsqueeze(2)       # (B,K,n,d) + (B,K,1,d)
        q = self.bank_q(tok)                                 # (B,n,d)
        # scores[b,p,k] = <q[b,p], kv[b,k,p]>
        scores = torch.einsum("bnd,bknd->bnk", q, kv) / math.sqrt(d)
        valid = bank_mask > 0.5                              # (B,K)
        scores = scores.masked_fill(~valid.unsqueeze(1), -1e4)
        w = torch.softmax(scores, dim=-1) * valid.unsqueeze(1).to(scores.dtype)
        att_curve = torch.einsum("bnk,bknt->bnt", w, bank)   # (B,n,T)
        att_emb = torch.einsum("bnk,bknd->bnd", w, kv)       # (B,n,d)
        return att_curve, att_emb

    def forward(self, x_t, t, names, rig, action_id, token_mask, exem, training=True,
                action_chars=None, span=None, cstats=None, bank=None, bank_mask=None,
                bank_act=None):
        B, n, T = x_t.shape
        d = self.cfg.d_model
        tok = self._token_emb(names, rig, action_id, action_chars, training, span,
                              cstats)                                           # (B,n,d)
        if self.bank_cond == "attn" and bank is not None:
            att_curve, att_emb = self._bank_attend(
                tok, bank, bank_mask, bank_act,
                lambda a: self._action_emb(a, None, False))
            tok = tok + self.bank_gate * att_emb
        else:
            att_curve = None
        t_e = self.time_emb(t).unsqueeze(1)                           # (B,1,d)
        cond = (tok + t_e).reshape(B * n, 1, d)                      # (Bn, 1, d)

        chans = [x_t.unsqueeze(-1), exem.unsqueeze(-1)]
        if att_curve is not None:
            chans.append(att_curve.unsqueeze(-1))
        h = self.in_proj(torch.cat(chans, -1))                                  # (B,n,T,d)
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
        if self.head_mode == "structured":
            # pool over T -> per-token feature (B,n,d)
            h_pool = h.mean(dim=2)
            alpha = self.gain_head(h_pool)          # (B,n,1) per-param amplitude gain
            coef = self.coef_head(h_pool)           # (B,n,r) per-param basis coeffs
            shape = coef @ self.basis              # (B,n,T) low-rank shape correction
            out = exem * (1.0 + alpha) + shape      # x0_hat = prior * (1+gain) + shape
            if self.amp_head is not None:
                # V11e(b2): additive amplitude correction, for the 28.4% of
                # moving channels where the prior under-shoots (and any sign
                # error, which the multiplicative form cannot express).
                out = out + self.amp_head(h_pool)
        else:
            out = self.head(h).squeeze(-1)                               # (B,n,T)
            if self.gate_emb is not None:
                g = torch.sigmoid(
                    self.gate_bias + self.gate_emb(self._gate_ids(names, out.device)))
                out = exem + g * out        # x0_hat = prior(exem) + g * residual
            else:
                out = exem + out                                        # x0_hat = prior(exem) + residual
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
