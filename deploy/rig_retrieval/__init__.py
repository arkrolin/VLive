"""Same-source automatic reference retrieval for Live2D motion generation.

Deployment-side module that, given a *target* rig (its ``.moc3`` parameter
schema) and a desired action, returns the K closest-source same-action reference
clips from the corpus -- no manual source labels, no retraining. This is the
productised form of the V6.2 "auto3" donor strategy, which proved parameter-set
Jaccard is a label-free proxy for production source (family-match 75.8%).

Runtime API (zero third-party dependencies):
    Retriever(index).retrieve(target_param_set, action, k=3)
    Retriever(index).retrieve_for_model(model_dir, action, k=3)
    Retriever(index).retrieve_for_pack(pack, action, k=3)

Offline tooling:
    build_index.build_corpus_index(...)  -> CorpusIndex  (imports tools/)
    validate.run(index_path)             -> equivalence regression vs V6.2
"""
from __future__ import annotations

from .index import CorpusIndex, family_of
from .moc3 import extract_param_set, read_param_ids
from .retriever import Retriever
from .schema import ReferenceSpec, RetrievalResult
from .similarity import jaccard, param_similarity

__all__ = [
    "CorpusIndex", "Retriever", "ReferenceSpec", "RetrievalResult",
    "family_of", "extract_param_set", "read_param_ids",
    "jaccard", "param_similarity",
]
