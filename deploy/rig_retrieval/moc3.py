"""Parameter-id extraction from Live2D Cubism ``.moc3`` files.

This is the deployment-side, dependency-free twin of
``tools/audit_pack_identity.read_model_ids``: it harvests the rig's
parameter-id schema straight out of the moc3 binary (the ids are NUL-padded
ASCII records). The *exact* same regex produces the corpus cache
(``outputs/_paramset_cache.json``), so a brand-new target model's extracted set
is directly comparable -- Jaccard similarity is meaningful with **no retraining
and no manual labels**.
"""
from __future__ import annotations

import re
from pathlib import Path

# moc3 stores ids as NUL-padded ASCII; match whole identifier records only.
_PARAM_RE = re.compile(rb"(?<![A-Za-z0-9_])([Pp][Aa][Rr][Aa][Mm][A-Za-z0-9_]{0,58})\x00")


def read_param_ids(moc3_path) -> set[str]:
    """Parameter-id set of a single ``.moc3`` file (empty if unreadable)."""
    p = Path(moc3_path)
    if not p.exists():
        return set()
    blob = p.read_bytes()
    return {m.decode("ascii", "ignore") for m in _PARAM_RE.findall(blob)}


def extract_param_set(model_dir) -> set[str]:
    """Union of parameter ids declared by every ``.moc3`` in *model_dir*.

    A Live2D model directory usually holds one ``.moc3`` (the rig schema). If it
    holds several (e.g. expression variants), the union is the full schema, which
    is exactly what we want for source-matching. A bare ``.moc3`` file path is
    also accepted.
    """
    d = Path(model_dir)
    files = [d] if d.is_file() and d.suffix.lower() == ".moc3" else []
    if not files:
        files = sorted(d.glob("*.moc3"))
    if not files:                       # maybe nested one level down
        files = sorted(d.rglob("*.moc3"))
    out: set[str] = set()
    for f in files:
        out |= read_param_ids(f)
    return out
