"""Schema layer for Live2D VLA: whitelist/gen_mask loaders, param-name
vocabulary + 4-way identity encoders (V7.1).

Four identity signals per parameter token:
    1. name      -> word-piece embedding (mean-pooled), with 10-30% dropout
                     so unnamed params (Param14) fall back to rig signature
    2. rig       -> static moc3 bag-of-hash signature (proxy for sweep probe)
    3. range     -> [min, max, std] of the param across the corpus
    4. deformer  -> constant/learned placeholder (deformer-tree; upgraded later)
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORD_RE = re.compile(r"[A-Za-z]+|\d+")


# --------------------------------------------------------------------------- #
# whitelist / gen_mask
# --------------------------------------------------------------------------- #
def load_whitelist(cfg) -> list[str]:
    import orjson

    d = orjson.loads(Path(cfg.whitelist_path).read_bytes())
    return d.get("kept", [])


def load_gen_mask(cfg) -> dict[str, list[str]]:
    import orjson

    d = orjson.loads(Path(cfg.gen_mask_path).read_bytes())
    return {m: rec["target"] for m, rec in d.get("models", {}).items()}


# --------------------------------------------------------------------------- #
# vocabularies
# --------------------------------------------------------------------------- #
def build_vocab(targets_per_model: dict[str, list[str]], cap: int = 2048):
    """Return (param2idx, word2idx). <pad>=0, <unk>=1."""
    freq = Counter()
    for params in targets_per_model.values():
        for p in set(params):
            freq[p] += 1
    words: Counter = Counter()
    for p in freq:
        for w in WORD_RE.findall(p):
            words[w.lower()] += 1

    top = [p for p, _ in freq.most_common(cap - 2)]
    param2idx = {"<pad>": 0, "<unk>": 1}
    for p in top:
        param2idx.setdefault(p, len(param2idx))
    word2idx = {"<pad>": 0, "<unk>": 1}
    for w, _ in words.most_common(2000):
        word2idx.setdefault(w, len(word2idx))
    return param2idx, word2idx


def tokenize_name(name: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(name)]


# --------------------------------------------------------------------------- #
# 4-way identity encoders
# --------------------------------------------------------------------------- #
class ParamNameEncoder(nn.Module):
    """Learned word-piece -> name embedding with dropout (V7.1).

    MUST subclass nn.Module. As a plain class its parameters were never
    registered in Live2DModel.parameters(), so the name branch stayed at random
    initialisation and never trained (found 2026-08-31: the "name" identity
    signal had been inert for the entire project).
    """

    def __init__(self, word2idx: dict[str, int], d_model: int, dropout: float = 0.2):
        super().__init__()
        self.word2idx = word2idx
        self.dropout = dropout
        self.d_w = 64
        n = len(word2idx)
        self.emb = nn.Embedding(n, self.d_w, padding_idx=0)
        self.proj = nn.Linear(self.d_w, d_model)

    def forward(self, names: list[str], training: bool = True):
        dev = self.emb.weight.device
        out = []
        for name in names:
            toks = tokenize_name(name)
            idx = [self.word2idx.get(t, 1) for t in toks] or [1]
            w = self.emb(torch.tensor(idx, dtype=torch.long, device=dev))
            v = w.mean(0)                       # mean-pool word pieces
            v = self.proj(v)
            if training and self.dropout > 0 and torch.rand(1).item() < self.dropout:
                v = torch.zeros_like(v)        # force rig-signature fallback
            out.append(v)
        if not out:
            return torch.zeros(0, self.proj.out_features, device=dev)
        return torch.stack(out)                # (n_tokens, d)


# --------------------------------------------------------------------------- #
# P2: compositional action conditioning
# --------------------------------------------------------------------------- #
ACTION_SUFFIX_RE = re.compile(r"\.motion3$")


def normalize_action_name(a: str) -> str:
    """Strip the constant `.motion3` suffix so the encoder sees the semantics."""
    return ACTION_SUFFIX_RE.sub("", a).lower()


def build_action_char_vocab(actions) -> dict[str, int]:
    """Char-level vocab over action names. <pad>=0, <unk>=1."""
    chars: Counter = Counter()
    for a in actions:
        chars.update(normalize_action_name(a))
    c2i = {"<pad>": 0, "<unk>": 1}
    for c, _ in chars.most_common(256):
        c2i.setdefault(c, len(c2i))
    return c2i


def encode_action_name(name: str, c2i: dict[str, int], max_len: int = 32) -> list[int]:
    s = normalize_action_name(name)
    return [c2i.get(ch, 1) for ch in s[:max_len]] or [1]


class ActionNameEncoder(nn.Module):
    """Compositional action-name encoder (P2).

    Action names are semantic pinyin (`weixiao`=smile, `shengqi`=angry,
    `jiandao`/`shitou`/`bu`=rock-paper-scissors, `stand`), so a char-level
    encoder lets related actions share statistics and generalises to unseen
    names. This matters because the corpus has only ~4.4 samples per action
    (1679 actions / 7382 samples), which turns a plain Embedding into a
    memorisation table that cannot transfer to held-out characters.
    """

    def __init__(self, char2idx, d_model, max_len: int = 32, dropout: float = 0.0):
        super().__init__()
        self.max_len = max_len
        self.dropout = dropout
        self.emb = nn.Embedding(len(char2idx), d_model, padding_idx=0)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
            nn.SiLU(), nn.Linear(d_model, d_model))

    def forward(self, name_ids, training: bool = True):
        """name_ids: (B, L) LongTensor, 0 = pad."""
        w = self.emb(name_ids)
        mask = (name_ids != 0).float().unsqueeze(-1)
        v = (w * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        v = self.proj(v)
        if training and self.dropout > 0:
            keep = (torch.rand(v.shape[0], 1, device=v.device) > self.dropout).float()
            v = v * keep                      # action-condition dropout (CFG prep)
        return v


def rig_signature(model_dir, dim: int = 64) -> np.ndarray:
    """Static moc3 param-set -> bag-of-hash-buckets vector (proxy for sweep probe).

    Production upgrade (V7.1): replace with Cubism Core vertex-position probe.
    """
    model_dir = Path(model_dir)
    try:
        from deploy.rig_retrieval.moc3 import extract_param_set

        params = extract_param_set(model_dir)
    except Exception:
        params = set()
    vec = np.zeros(dim, dtype=np.float32)
    for p in params:
        h = (hash(p) % dim + dim) % dim
        vec[h] += 1.0
    if vec.sum() > 0:
        vec /= vec.sum()
    return vec


def range_stats(curves_per_param: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """min/max/std per param across its curves -> (n,3) array keyed by param."""
    stats = {}
    for p, arr in curves_per_param.items():
        a = np.asarray(arr, dtype=np.float32).ravel()
        if a.size == 0:
            stats[p] = np.array([0.0, 0.0, 0.0], np.float32)
        else:
            stats[p] = np.array([a.min(), a.max(), a.std() + 1e-6], np.float32)
    return stats
