"""Data shapes returned by the retriever.

Kept as plain dataclasses so the deployment pipeline can serialise them to JSON,
feed them to the generation model as few-shot context, or log them for review --
no numpy / Live2D tooling required on the consumer side.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class ReferenceSpec:
    """One retrieved same-action reference clip for the target rig."""

    pack: str                 # corpus pack, e.g. "l2d22.ugirl06"
    char: str                 # character id, e.g. "ugirl06"
    family: str               # production family, e.g. "l2d22"
    action: str               # normalised action key, e.g. "haixiu"
    motion_path: str          # path to the .motion3.json on disk
    similarity: float         # Jaccard(param_set_target, param_set_this_rig)
    rank: int = 0
    param_count: int = 0
    near_cluster: bool = False
    # None when the target has no known family (a brand-new model).
    same_family: Optional[bool] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    """Full answer to one retrieval request."""

    target: str               # label of the target (pack / model path)
    target_param_count: int   # size of the target rig's parameter schema
    requested_action: str
    matched_action: str       # resolved corpus action (may differ under --fuzzy)
    references: list[ReferenceSpec] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    excluded_chars: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["references"] = [r.to_dict() for r in self.references]
        return d

    def summary(self) -> str:
        lines = [f"target={self.target} (params={self.target_param_count}) "
                 f"action={self.requested_action!r} -> {self.matched_action!r}"]
        if self.notes:
            lines += [f"  note: {n}" for n in self.notes]
        if not self.references:
            lines.append("  (no references found)")
        for r in self.references:
            fam = "" if r.same_family is None else \
                f" same_src={'Y' if r.same_family else 'n'}"
            lines.append(
                f"  #{r.rank} {r.pack}/{r.action}  sim={r.similarity:.3f}"
                f" near={'*' if r.near_cluster else ' '}{fam}  -> {r.motion_path}")
        return "\n".join(lines)
