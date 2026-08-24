"""In-memory corpus index for same-source retrieval.

Runtime use is intentionally dependency-free (stdlib + the two pure modules in
this package). The index is built offline by ``build_index.py`` and serialised to
a single JSON-Lines file, one corpus clip per line. Loading it needs only ``json``.

Entry per clip:
    pack        corpus pack id, e.g. "l2d22.ugirl06"
    char        character id,   e.g. "ugirl06"   (family stripped)
    family      production family, e.g. "l2d22"
    action      normalised action key, e.g. "haixiu"
    motion_path absolute path to the .motion3.json
    params      the rig's parameter-id set (for Jaccard at query time)
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .similarity import jaccard


def family_of(pack: str) -> str:
    """Production family of a pack id. Mirrors tools.audit_rig_source_split."""
    return pack.split(".")[0]


@dataclass
class _Entry:
    pack: str
    char: str
    family: str
    action: str
    motion_path: str
    params: set


class CorpusIndex:
    def __init__(self) -> None:
        self.entries: list[_Entry] = []
        self.by_action: dict[str, list[int]] = defaultdict(list)
        # character id -> union of that character's rig parameter ids (char-level
        # schema, the unit the V6.2 experiment matched on).
        self.char_params: dict[str, set] = {}
        # character id -> family (all packs of a char share the prefix).
        self.char_family: dict[str, str] = {}

    # ------------------------------------------------------------------ build
    @classmethod
    def from_entries(cls, entries: list[_Entry]) -> "CorpusIndex":
        idx = cls()
        for e in entries:
            idx._add(e)
        return idx

    def _add(self, e: _Entry) -> None:
        self.entries.append(e)
        self.by_action[e.action].append(len(self.entries) - 1)
        self.char_params.setdefault(e.char, set()).update(e.params)
        self.char_family.setdefault(e.char, e.family)

    # --------------------------------------------------------------- persist
    def save(self, path) -> None:
        rows = [
            {
                "pack": e.pack, "char": e.char, "family": e.family,
                "action": e.action, "motion_path": e.motion_path,
                "params": sorted(e.params),
            }
            for e in self.entries
        ]
        Path(path).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path) -> "CorpusIndex":
        idx = cls()
        raw = Path(path).read_text(encoding="utf-8")
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idx._add(_Entry(
                pack=r["pack"], char=r["char"], family=r["family"],
                action=r["action"], motion_path=r["motion_path"],
                params=set(r["params"]),
            ))
        return idx

    # ----------------------------------------------------------------- query
    def actions(self) -> list[str]:
        return sorted(self.by_action)

    def char_param_set(self, char: str) -> set:
        return self.char_params.get(char, set())

    def match_actions(self, query: str, fuzzy: bool = False) -> list[str]:
        """Resolve a requested action to corpus action keys.

        Exact match first (this is what the V6.2 experiment used). ``fuzzy``
        enables substring / shared-4-char-token fallback for free-text targets.
        """
        if query in self.by_action:
            return [query]
        q = query.lower()
        hits = [a for a in self.by_action if q in a or a in q]
        if hits:
            return hits
        if fuzzy:
            out = [
                a for a in self.by_action
                if len(a) >= 4 and any(
                    a[i:i + 4] in q or q[i:i + 4] in a
                    for i in range(len(a) - 3)
                )
            ]
            if out:
                return out
        return []
