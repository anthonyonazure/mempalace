"""Tests for recency weighting + the prune CLI (#158).

Covers four layers:
  * pure age parsing + bounded recency adjustment (mempalace.recency)
  * the searcher integration (recency reorders but never excludes — recall)
  * the `mempalace prune --stale-days` CLI (dry-run default, --confirm gate,
    undated drawers never deleted)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mempalace import recency


def _iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


# ---------------------------------------------------------------------------
# Pure age parsing
# ---------------------------------------------------------------------------


class TestAgeDays:
    def test_none_and_empty(self):
        assert recency.age_days(None) is None
        assert recency.age_days({}) is None
        assert recency.age_days({recency.FILED_AT_KEY: ""}) is None
        assert recency.age_days({recency.FILED_AT_KEY: "not-a-date"}) is None

    def test_parses_naive_isoformat(self):
        meta = {recency.FILED_AT_KEY: _iso(10)}
        age = recency.age_days(meta)
        assert age == pytest.approx(10.0, abs=0.1)

    def test_parses_z_suffix(self):
        # A 'Z' suffix (which 3.9 fromisoformat rejects) must still parse.
        now = datetime.now(timezone.utc).replace(microsecond=0)
        meta = {recency.FILED_AT_KEY: now.isoformat().replace("+00:00", "Z")}
        age = recency.age_days(meta)
        assert age is not None and age >= 0.0

    def test_future_date_clamps_to_zero(self):
        meta = {recency.FILED_AT_KEY: _iso(-5)}  # 5 days in the future
        assert recency.age_days(meta) == 0.0


# ---------------------------------------------------------------------------
# Recency distance adjustment
# ---------------------------------------------------------------------------


class TestRecencyAdjustment:
    def test_undated_is_noop(self):
        assert recency.recency_distance_adjustment({}) == 0.0
        assert recency.recency_distance_adjustment(None) == 0.0

    def test_fresh_drawer_gets_near_max_boost(self):
        meta = {recency.FILED_AT_KEY: _iso(0)}
        adj = recency.recency_distance_adjustment(meta)
        # Negative (pulls distance down); close to the full max for age ~0.
        assert adj == pytest.approx(-recency.RECENCY_DISTANCE_MAX, abs=1e-3)

    def test_one_halflife_is_half_boost(self):
        meta = {recency.FILED_AT_KEY: _iso(recency.RECENCY_HALFLIFE_DAYS)}
        adj = recency.recency_distance_adjustment(meta)
        assert adj == pytest.approx(-recency.RECENCY_DISTANCE_MAX / 2, abs=1e-3)

    def test_two_halflives_is_quarter_boost(self):
        meta = {recency.FILED_AT_KEY: _iso(2 * recency.RECENCY_HALFLIFE_DAYS)}
        adj = recency.recency_distance_adjustment(meta)
        assert adj == pytest.approx(-recency.RECENCY_DISTANCE_MAX / 4, abs=1e-3)

    def test_always_within_bounds(self):
        for d in (0, 1, 30, 365, 10000):
            adj = recency.recency_distance_adjustment({recency.FILED_AT_KEY: _iso(d)})
            assert -recency.RECENCY_DISTANCE_MAX <= adj <= 0.0

    def test_nonpositive_halflife_disables(self):
        meta = {recency.FILED_AT_KEY: _iso(1)}
        assert recency.recency_distance_adjustment(meta, halflife_days=0) == 0.0
        assert recency.recency_distance_adjustment(meta, halflife_days=-5) == 0.0

    def test_nonpositive_max_disables(self):
        meta = {recency.FILED_AT_KEY: _iso(1)}
        assert recency.recency_distance_adjustment(meta, max_shift=0.0) == 0.0


# ---------------------------------------------------------------------------
# Searcher integration
# ---------------------------------------------------------------------------
