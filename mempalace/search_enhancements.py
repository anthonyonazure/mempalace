"""Opt-in search enhancements, ported from the techempower-org/mempalace fork.

Every feature here is OFF by default, so a palace that sets nothing searches
byte-for-byte the way upstream MemPalace does. Each one is switched on by an
environment variable or a ``config.json`` key (env wins):

==========================  ==============================  ===================
Feature                     Env var                         config.json key
==========================  ==============================  ===================
RRF fusion (vs. convex)     MEMPALACE_FUSION_MODE=rrf       "fusion_mode"
Recency boost               MEMPALACE_RECENCY_BOOST=1       "recency_boost"
  half-life (days)          MEMPALACE_RECENCY_HALFLIFE_DAYS "recency_halflife_days"
Cross-encoder rerank        MEMPALACE_RERANK_CROSS_ENCODER  "cross_encoder_rerank"
  model / top-N             MEMPALACE_RERANK_CROSS_ENCODER_MODEL / _TOP_N
Calibrated confidence       MEMPALACE_CALIBRATION_PATH      "calibration_path"
Source provenance labels    MEMPALACE_PROVENANCE=1          "provenance"
Curated-first ordering      MEMPALACE_PREFER_CURATED=1      "prefer_curated"
==========================  ==============================  ===================

The searcher calls exactly two functions from here, so keeping this fork in
step with upstream means re-attaching two call sites, nothing more:

* :func:`rank_pool` replaces the hybrid re-rank over the FULL candidate pool
  (before dedupe and the ``n_results`` cut), so fusion, recency and the
  cross-encoder can change which hits make the cut, not just their order.
* :func:`finish_hits` runs on the final hits (after the cut): provenance
  labels, curated-first ordering, and calibrated confidence.

Nothing here may break a search: every stage catches its own errors, logs
them, and hands back the input order unchanged.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

logger = logging.getLogger("mempalace_mcp")

_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off")

FUSION_MODES = ("convex", "rrf")


# ── settings ─────────────────────────────────────────────────────────────────


def _file_config() -> dict:
    try:
        from .config import MempalaceConfig

        cfg = getattr(MempalaceConfig(), "_file_config", None)
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        logger.debug("search_enhancements: config unreadable", exc_info=True)
        return {}


def _flag(env_name: str, config_key: str, file_config: Optional[dict] = None) -> bool:
    raw = os.environ.get(env_name, "").strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    cfg = _file_config() if file_config is None else file_config
    value = cfg.get(config_key, False)
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY
    return bool(value)


def fusion_mode(file_config: Optional[dict] = None) -> str:
    raw = os.environ.get("MEMPALACE_FUSION_MODE", "").strip().lower()
    if not raw:
        cfg = _file_config() if file_config is None else file_config
        raw = str(cfg.get("fusion_mode", "convex") or "convex").strip().lower()
    if raw not in FUSION_MODES:
        logger.warning("search_enhancements: unknown fusion_mode %r, using convex", raw)
        return "convex"
    return raw


def recency_enabled(file_config: Optional[dict] = None) -> bool:
    return _flag("MEMPALACE_RECENCY_BOOST", "recency_boost", file_config)


def recency_halflife_days(file_config: Optional[dict] = None) -> float:
    from .recency import RECENCY_HALFLIFE_DAYS

    raw = os.environ.get("MEMPALACE_RECENCY_HALFLIFE_DAYS", "").strip()
    if not raw:
        cfg = _file_config() if file_config is None else file_config
        raw = str(cfg.get("recency_halflife_days", "") or "").strip()
    if not raw:
        return RECENCY_HALFLIFE_DAYS
    try:
        return float(raw)
    except ValueError:
        logger.warning("search_enhancements: bad recency half-life %r, using default", raw)
        return RECENCY_HALFLIFE_DAYS


def provenance_enabled(file_config: Optional[dict] = None) -> bool:
    return _flag("MEMPALACE_PROVENANCE", "provenance", file_config)


def prefer_curated_enabled(file_config: Optional[dict] = None) -> bool:
    return _flag("MEMPALACE_PREFER_CURATED", "prefer_curated", file_config)


def calibration_path(file_config: Optional[dict] = None) -> Optional[str]:
    raw = os.environ.get("MEMPALACE_CALIBRATION_PATH", "").strip()
    if not raw:
        cfg = _file_config() if file_config is None else file_config
        raw = str(cfg.get("calibration_path", "") or "").strip()
    return os.path.abspath(os.path.expanduser(raw)) if raw else None


def any_enabled() -> bool:
    """True when at least one enhancement is switched on."""
    from . import cross_encoder_rerank as cer

    cfg = _file_config()
    return (
        fusion_mode(cfg) != "convex"
        or recency_enabled(cfg)
        or cer.is_enabled(cfg)
        or provenance_enabled(cfg)
        or prefer_curated_enabled(cfg)
        or calibration_path(cfg) is not None
    )


# ── stage 1: ranking the full candidate pool ─────────────────────────────────


def _hit_identity(hit: dict):
    return hit.get("drawer_id") or id(hit)


def _rrf_order(hits: list) -> list:
    """Reciprocal Rank Fusion of the vector order and the BM25 order.

    Expects ``bm25_score`` already set on every hit (the convex ranker sets it).
    Hits without a vector distance (BM25-only candidates) appear only in the
    BM25 list, so they fuse on that signal alone.
    """
    from .rrf import DEFAULT_K, rrf_fuse

    vector_list = sorted(
        (h for h in hits if h.get("distance") is not None), key=lambda h: h["distance"]
    )
    bm25_list = sorted(hits, key=lambda h: h.get("bm25_score") or 0.0, reverse=True)
    fused = rrf_fuse([vector_list, bm25_list], key=_hit_identity, k=DEFAULT_K)
    for _ident, score, rep in fused:
        rep["rrf_score"] = round(score, 6)
    return [rep for _ident, _score, rep in fused]


def _relevance_scores(hits: list) -> tuple:
    """Each hit's relevance on the scale the active ranker used.

    After RRF, that is ``rrf_score``. Otherwise it is upstream's convex blend
    recomputed from the fields the convex ranker left on each hit
    (``distance`` and ``bm25_score``), with the same weights. Returns the
    scores and the recency bonus ceiling suited to that scale.
    """
    from .recency import RECENCY_DISTANCE_MAX

    if all("rrf_score" in h for h in hits):
        # Adjacent RRF ranks near the top differ by ~1/(k+r)^2, about 0.0003
        # at k=60; this ceiling lets a fresh drawer pass one or two neighbors.
        return [h["rrf_score"] for h in hits], 0.0006
    from . import searcher

    vector_weight, bm25_weight = searcher._resolve_hybrid_rank_weights()
    bm25 = [float(h.get("bm25_score") or 0.0) for h in hits]
    top = max(bm25) if bm25 else 0.0
    scores = []
    for hit, raw in zip(hits, bm25):
        sim = searcher._distance_to_similarity(hit.get("distance"), "cosine")
        scores.append(vector_weight * sim + bm25_weight * (raw / top if top > 0 else 0.0))
    # Same ceiling the fork used: 0.03 in similarity units.
    return scores, RECENCY_DISTANCE_MAX


def _apply_recency(hits: list, halflife_days: float) -> list:
    """Nudge newer drawers up without letting age override relevance.

    A fresh drawer gets a small bonus on its relevance score that halves
    every ``halflife_days``; the ceiling is small enough to reorder near
    neighbors, never to lift a weak match over a strong one. Undated hits
    get no bonus and no penalty; ties keep the incoming order.
    """
    from .recency import RECENCY_DISTANCE_MAX, recency_distance_adjustment

    if len(hits) < 2:
        return hits
    scores, ceiling = _relevance_scores(hits)
    scored = []
    for position, (hit, base) in enumerate(zip(hits, scores)):
        # search_memories hits carry filed_at at the top level; CLI hits
        # nest it under "metadata".
        meta = hit if "filed_at" in hit else (hit.get("metadata") or {})
        fraction = -recency_distance_adjustment(meta, halflife_days=halflife_days)
        fraction = fraction / RECENCY_DISTANCE_MAX if RECENCY_DISTANCE_MAX else 0.0
        bonus = ceiling * fraction
        if bonus > 0:
            hit["recency_boost"] = round(bonus, 6)
        scored.append((base + bonus, -position, hit))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [h for _s, _p, h in scored]


def rank_pool(
    hits: list,
    query: str,
    convex_rank: Callable[[list], list],
    *,
    scorer=None,
) -> list:
    """Rank the full candidate pool. Upstream's convex rank always runs first
    (it sets ``bm25_score``); each enabled stage then reorders the result."""
    ranked = convex_rank(hits)
    if not ranked:
        return ranked
    convex_order = list(ranked)
    try:
        from . import cross_encoder_rerank as cer

        cfg = _file_config()
        if fusion_mode(cfg) == "rrf":
            ranked = _rrf_order(ranked)
        if recency_enabled(cfg):
            ranked = _apply_recency(ranked, recency_halflife_days(cfg))
        if cer.is_enabled(cfg):
            ranked = cer.rerank(
                query,
                ranked,
                model_name=cer.get_model_name(cfg),
                top_n=cer.get_top_n(cfg),
                scorer=scorer,
            )
    except Exception:
        logger.warning(
            "search_enhancements: ranking stage failed; using convex order", exc_info=True
        )
        return convex_order
    return ranked


# ── stage 2: finishing the returned hits ─────────────────────────────────────

_CALIBRATOR_CACHE: dict = {}


def _load_calibrator(path: str):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    cached = _CALIBRATOR_CACHE.get(path)
    if cached is not None and cached[1] == mtime:
        return cached[0]
    from .calibration import Calibrator

    cal = Calibrator.load(path)
    _CALIBRATOR_CACHE[path] = (cal, mtime)
    return cal


def finish_hits(hits: list) -> list:
    """Label, reorder and score the final hits. Returns the (possibly
    reordered) list; never drops a hit."""
    if not hits:
        return hits
    cfg = _file_config()
    try:
        if provenance_enabled(cfg) or prefer_curated_enabled(cfg):
            from .provenance import annotate

            annotate(hits)
        if prefer_curated_enabled(cfg):
            from .result_ordering import collapse_identical_text, prefer_curated

            collapse_identical_text(hits)
            prefer_curated(hits)
        path = calibration_path(cfg)
        if path:
            cal = _load_calibrator(path)
            if cal is not None:
                from .calibration import apply_calibrator

                for hit in hits:
                    conf = apply_calibrator(cal, hit.get("similarity"))
                    if conf is not None:
                        hit["confidence"] = conf
    except Exception:
        logger.warning("search_enhancements: finishing stage failed", exc_info=True)
    return hits
