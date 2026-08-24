"""Training dataset for the Live2D VLA (V7.1): real corpus -> tensors.

Each sample = one (target model, action) pair:
    * target dense curves for the model's generation-target params (excludes
      physics / blink / lipsync / nonmotion via gen_mask)
    * static rig signature (moc3 bag-of-hash)   -> rig branch
    * action id                                  -> action branch (text/gesture proxy)
    * B1 cross-character exemplar mean curves    -> action branch (few-shot prior)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from schema import load_whitelist, load_gen_mask, build_vocab, rig_signature  # noqa: E402
from io_motion import load_target_curves                                       # noqa: E402


def _build_action_vocab(samples):
    acts = sorted({a for _, a in samples})
    return {a: i for i, a in enumerate(acts)}, acts


class Live2DDataset(Dataset):
    def __init__(self, cfg, action_vocab=None):
        self.cfg = cfg
        whitelist = load_whitelist(cfg)
        gen_mask = load_gen_mask(cfg)
        self.subset = whitelist[: cfg.subset_models]
        self.targets = {m: gen_mask.get(m, []) for m in self.subset if gen_mask.get(m)}

        # param vocabulary (name -> idx) over the subset
        self.param2idx, self.word2idx = build_vocab(self.targets, cap=2048)

        # samples = (model, action) with at least one motion
        from io_motion import list_motions

        self.samples = []
        for m in self.subset:
            if m not in self.targets:
                continue
            for action in list_motions(ROOT / cfg.data_root / m):
                self.samples.append((m, action))
        self.action2idx, _ = _build_action_vocab(self.samples) if action_vocab is None \
            else (action_vocab, list(action_vocab))

        # B1 retriever (lazy, cached per (model,action))
        from deploy.rig_retrieval import CorpusIndex, Retriever

        idx = CorpusIndex.load(str(cfg.retrieval_index))
        self.retriever = Retriever(idx)
        self._exem_cache: dict[tuple, np.ndarray] = {}

    def __len__(self):
        return len(self.samples)

    def _exemplar_mean(self, model: str, action: str, names: list[str]) -> np.ndarray:
        key = (model, action)
        if key in self._exem_cache:
            return self._exem_cache[key]
        T = self.cfg.T
        k = self.cfg.k_exemplars
        acc = {n: [] for n in names}
        try:
            refs = self.retriever.retrieve_for_pack(model, action, k=k).references
        except Exception:
            refs = []
        for r in refs:
            curves = load_target_curves(
                ROOT / self.cfg.data_root / r.pack, action, names,
                fps=self.cfg.fps, T=T)
            for n in names:
                if n in curves:
                    acc[n].append(curves[n])
        mean = np.zeros((len(names), T), np.float32)
        for i, n in enumerate(names):
            if acc[n]:
                mean[i] = np.mean(np.stack(acc[n]), 0)
        self._exem_cache[key] = mean
        return mean

    def __getitem__(self, idx):
        model, action = self.samples[idx]
        T = self.cfg.T
        target_params = self.targets[model]
        curves = load_target_curves(
            ROOT / self.cfg.data_root / model, action, target_params,
            fps=self.cfg.fps, T=T)
        # active names present in this clip AND in the vocabulary
        names = [p for p in target_params if p in curves and p in self.param2idx]
        if not names:
            names = target_params[:1]
        target = np.stack([curves[n] for n in names], 0).astype(np.float32)  # (n,T)
        rig = rig_signature(ROOT / self.cfg.data_root / model, dim=self.cfg.rig_sig_dim)
        exem = self._exemplar_mean(model, action, names)
        return {
            "names": names,
            "target": target,
            "rig": rig.astype(np.float32),
            "action_id": self.action2idx.get(action, 0),
            "exem": exem,
        }


def collate(batch, max_tokens: int):
    """Pad variable token counts to max_tokens; build masks."""
    B = len(batch)
    T = batch[0]["target"].shape[1]
    target = np.zeros((B, max_tokens, T), np.float32)
    exem = np.zeros((B, max_tokens, T), np.float32)
    rig = np.zeros((B, batch[0]["rig"].shape[0]), np.float32)
    action_id = np.zeros(B, np.int64)
    token_mask = np.zeros((B, max_tokens), np.float32)  # 1 = active
    names = []
    for i, s in enumerate(batch):
        n = min(len(s["names"]), max_tokens)
        token_mask[i, :n] = 1.0
        target[i, :n] = s["target"][:n]
        exem[i, :n] = s["exem"][:n]
        rig[i] = s["rig"]
        action_id[i] = s["action_id"]
        names.append(s["names"][:n])
    out = {
        "names": names,
        "target": torch.from_numpy(target),
        "exem": torch.from_numpy(exem),
        "rig": torch.from_numpy(rig),
        "action_id": torch.from_numpy(action_id),
        "token_mask": torch.from_numpy(token_mask),
    }
    return out
