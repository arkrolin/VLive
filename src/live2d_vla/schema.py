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
class ParamNameEncoder:
    """Learned word-piece -> name embedding with dropout (V7.1, unnamed-param fallback)."""

    def __init__(self, word2idx: dict[str, int], d_model: int, dropout: float = 0.2):
        import torch

        self.word2idx = word2idx
        self.dropout = dropout
        self.d_w = 64
        n = len(word2idx)
        self.emb = torch.nn.Embedding(n, self.d_w, padding_idx=0)
        self.proj = torch.nn.Linear(self.d_w, d_model)

    def __call__(self, names: list[str], training: bool = True):
        import torch

        out = []
        for name in names:
            toks = tokenize_name(name)
            idx = [self.word2idx.get(t, 1) for t in toks] or [1]
            w = self.emb(torch.tensor(idx, dtype=torch.long))
            v = w.mean(0)                       # mean-pool word pieces
            v = self.proj(v)
            if training and self.dropout > 0 and torch.rand(1).item() < self.dropout:
                v = torch.zeros_like(v)        # force rig-signature fallback
            out.append(v)
        if not out:
            return torch.zeros(0, self.proj.out_features)
        return torch.stack(out)                # (n_tokens, d)


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
