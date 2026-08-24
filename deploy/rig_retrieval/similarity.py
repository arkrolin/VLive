"""Set-similarity primitives for rig-schema matching.

The whole "same-source automatic retrieval" idea rests on one fact measured in
V6.2: the Jaccard overlap of two rigs' parameter-id sets is a strong, label-free
proxy for "same production source". See ``deploy`` README for the evidence.
"""
from __future__ import annotations


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity of two id sets; 0.0 when both empty."""
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def param_similarity(a: set, b: set) -> float:
    """Alias used by the retriever; rig-schema similarity == Jaccard."""
    return jaccard(a, b)
