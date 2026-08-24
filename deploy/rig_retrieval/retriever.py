"""Same-source automatic reference retriever (deployment entry point).

Given a target rig's parameter-id set and a desired action, return the K
same-action clips from the corpus whose rig schema is most Jaccard-similar to
the target -- i.e. the closest production source. No manual source labels, no
retraining. This is the productised form of the V6.2 "auto3" donor strategy.

Design notes
------------
* Scoring is **char-level**: we keep at most one representative clip per source
  character, then take the top-K characters. This mirrors the experiment (which
  picked 3 *other characters*) and avoids returning 3 near-identical clips from
  one over-represented rig.
* A "near-cluster" flag (``similarity >= NEAR_THR``) marks references that are
  schema-close enough to trust as same-source exemplars, the deployment-side
  analogue of the V6 experiment's 0.30 Jaccard neighbourhood.
"""
from __future__ import annotations

from pathlib import Path

from .index import CorpusIndex, family_of
from .moc3 import extract_param_set
from .schema import ReferenceSpec, RetrievalResult
from .similarity import jaccard

NEAR_THR = 0.30  # schema-near neighbourhood, matching V6's 0.30 cluster cut


class Retriever:
    def __init__(self, index: CorpusIndex) -> None:
        self.index = index

    def retrieve(
        self,
        target_param_set: set,
        action: str,
        k: int = 3,
        *,
        min_similarity: float = 0.0,
        exclude_chars: set | None = None,
        fuzzy: bool = False,
        target_family: str | None = None,
        target_label: str = "<target>",
    ) -> RetrievalResult:
        exclude = set(exclude_chars or [])
        matched = self.index.match_actions(action, fuzzy=fuzzy)
        result = RetrievalResult(
            target=target_label,
            target_param_count=len(target_param_set),
            requested_action=action,
            matched_action=matched[0] if matched else "",
            excluded_chars=sorted(exclude),
        )
        if not matched:
            result.notes.append(
                f"no corpus action matches {action!r}"
                + (" (try fuzzy=True)" if not fuzzy else "")
            )
            return result

        # One best candidate per source character (char-level scoring).
        best_per_char: dict[str, tuple[float, _EntryLike, bool | None]] = {}
        for a in matched:
            for ei in self.index.by_action[a]:
                e = self.index.entries[ei]
                if e.char in exclude:
                    continue
                sim = jaccard(target_param_set, self.index.char_param_set(e.char))
                if sim < min_similarity:
                    continue
                same_fam = None
                if target_family is not None:
                    same_fam = (e.family == target_family)
                if e.char not in best_per_char or sim > best_per_char[e.char][0]:
                    best_per_char[e.char] = (sim, e, same_fam)

        ranked = sorted(best_per_char.values(), key=lambda x: -x[0])[:k]
        if not ranked:
            result.notes.append("no candidate passed the similarity filter")
            return result

        for rank, (sim, e, same_fam) in enumerate(ranked, 1):
            result.references.append(ReferenceSpec(
                pack=e.pack, char=e.char, family=e.family, action=e.action,
                motion_path=e.motion_path, similarity=round(sim, 4),
                param_count=len(e.params), near_cluster=sim >= NEAR_THR,
                same_family=same_fam, rank=rank,
            ))
        return result

    # -------------------------------------------------- convenience wrappers
    def retrieve_for_pack(self, pack: str, action: str, k: int = 3, **kw) \
            -> RetrievalResult:
        """Target is a corpus pack we already indexed (used for offline eval and
        for the 'has a sibling in corpus' case). Its own character is excluded."""
        e = next((x for x in self.index.entries if x.pack == pack), None)
        if e is None:
            raise KeyError(f"pack {pack!r} not in index")
        return self.retrieve(
            self.index.char_param_set(e.char), action, k=k,
            exclude_chars=[e.char], target_family=e.family,
            target_label=pack, **kw,
        )

    def retrieve_for_model(self, model_dir, action: str, k: int = 3, **kw) \
            -> RetrievalResult:
        """Target is a brand-new model directory (or a bare .moc3 file). Its rig
        schema is read straight from the moc3 -- no corpus membership needed."""
        ps = extract_param_set(Path(model_dir))
        return self.retrieve(
            ps, action, k=k, target_label=str(model_dir), **kw,
        )


# Local type alias so we don't import the private _Entry name into the API.
from .index import _Entry as _EntryLike  # noqa: E402
