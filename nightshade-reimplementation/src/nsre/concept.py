"""Concept (poison tag) detection and target-concept selection.

Mirrors utils.ImageAnalyzer.get_current_concept / concept_candidate_filter / get_target_style
(evidence E-050..E-056). The original target-concept list and its embeddings live in an
encrypted proprietary asset (full_asset3.p); this module takes a caller-supplied list instead,
so the *selection rule* is recovered but the *candidate pool* is an approximation (L-001).
"""
from __future__ import annotations

import random
from typing import Callable, Optional, Sequence

import numpy as np

from .params import (
    BLIP_PREFIX,
    CONCEPT_BLOCKLIST,
    CONCEPT_SEED,
    FALLBACK_CONCEPT,
    TARGET_RANK_SLICE,
)


def extract_nouns(caption: str, pos_tag: Callable[[list], list]) -> list[str]:
    """Split on single spaces, POS-tag, drop the first 3 tokens, keep tags starting with 'NN'."""
    tagged = pos_tag(caption.split(" "))[3:]
    return [w for (w, tag) in tagged if tag[:2] in ("NN",)]


def filter_candidates(nouns: Sequence[str]) -> list[str]:
    return [n for n in nouns if n not in CONCEPT_BLOCKLIST]


def pick_concept(nouns: Sequence[str], clip_scores: Optional[Callable[[list[str]], np.ndarray]] = None) -> str:
    """0 candidates -> 'painting'; 1 -> it; more -> argmax CLIP zero-shot over 'a photo of {n}'."""
    cands = filter_candidates(nouns)
    if not cands:
        return FALLBACK_CONCEPT
    if len(cands) == 1 or clip_scores is None:
        return cands[0]
    probs = np.asarray(clip_scores([f"a photo of {k}" for k in cands]))
    return cands[int(np.argmax(probs))]


def pick_target(
    concept: str,
    target_concepts: Sequence[str],
    similarity: np.ndarray,
    seed: int = CONCEPT_SEED,
    mapping: Optional[dict] = None,
) -> str:
    """similarity: cosine similarity of `concept` to each entry of target_concepts.

    Sort descending, take ranks [4:10], choose uniformly with a *freshly seeded* global RNG
    (random.seed(678121) then random.choice) so the choice is deterministic per concept.
    NOTE: reseeding the global RNG also fixes the subsequent crop sequence in the original.
    """
    if mapping and concept in mapping:  # original: precompute_mapping wins (no reseed on this path)
        return mapping[concept]
    lo, hi = TARGET_RANK_SLICE
    order = np.asarray(similarity).reshape(-1).argsort()[::-1][lo:hi]
    cands = [target_concepts[i] for i in order]
    random.seed(seed)
    return random.choice(cands)

