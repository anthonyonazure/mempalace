"""Integration tests for the opt-in search enhancements hook layer."""

import json

import pytest

from mempalace import cross_encoder_rerank as cer
from mempalace import search_enhancements as enh
from mempalace.calibration import fit_calibrator
from mempalace.searcher import search_memories

_ENV_VARS = (
    "MEMPALACE_FUSION_MODE",
    "MEMPALACE_RECENCY_BOOST",
    "MEMPALACE_RECENCY_HALFLIFE_DAYS",
    "MEMPALACE_RERANK_CROSS_ENCODER",
    "MEMPALACE_RERANK_CROSS_ENCODER_MODEL",
    "MEMPALACE_RERANK_TOP_N",
    "MEMPALACE_CALIBRATION_PATH",
    "MEMPALACE_PROVENANCE",
    "MEMPALACE_PREFER_CURATED",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(enh, "_file_config", lambda: {})
    cer.reset_model_cache()
    enh._CALIBRATOR_CACHE.clear()
    yield
    cer.reset_model_cache()


def _ids(result):
    return [h["drawer_id"] for h in result["results"]]


def _plain_upstream(monkeypatch):
    """Bypass the hook layer entirely: pure upstream behavior."""
    monkeypatch.setattr(enh, "rank_pool", lambda hits, query, convex_rank, **kw: convex_rank(hits))
    monkeypatch.setattr(enh, "finish_hits", lambda hits: hits)


class TestOffByDefault:
    def test_nothing_enabled_without_settings(self):
        assert enh.any_enabled() is False

    def test_results_identical_to_upstream_when_off(
        self, palace_path, seeded_collection, monkeypatch
    ):
        with_hooks = search_memories("JWT authentication tokens", palace_path, n_results=4)
        _plain_upstream(monkeypatch)
        upstream = search_memories("JWT authentication tokens", palace_path, n_results=4)
        assert with_hooks == upstream

    def test_no_extra_fields_when_off(self, palace_path, seeded_collection):
        result = search_memories("database migrations", palace_path, n_results=4)
        for hit in result["results"]:
            for key in (
                "rrf_score",
                "recency_boost",
                "cross_encoder_score",
                "source_kind",
                "confidence",
            ):
                assert key not in hit


class TestFusion:
    def test_rrf_scores_every_hit_and_keeps_the_set(
        self, palace_path, seeded_collection, monkeypatch
    ):
        convex = search_memories("PostgreSQL connection pooling", palace_path, n_results=4)
        monkeypatch.setenv("MEMPALACE_FUSION_MODE", "rrf")
        fused = search_memories("PostgreSQL connection pooling", palace_path, n_results=4)
        assert set(_ids(fused)) == set(_ids(convex))
        assert all("rrf_score" in h for h in fused["results"])
        assert _ids(fused)[0] == "drawer_proj_backend_bbb"

    def test_unknown_mode_falls_back_to_convex(self, monkeypatch):
        monkeypatch.setenv("MEMPALACE_FUSION_MODE", "bogus")
        assert enh.fusion_mode() == "convex"


class TestRecency:
    @staticmethod
    def _days_ago(days):
        from datetime import datetime, timedelta, timezone

        return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    def test_fresh_hit_climbs_past_an_equally_relevant_stale_one(self):
        hits = [
            {
                "drawer_id": "old",
                "distance": 0.5,
                "bm25_score": 1.0,
                "filed_at": self._days_ago(400),
            },
            {"drawer_id": "new", "distance": 0.5, "bm25_score": 1.0, "filed_at": self._days_ago(1)},
        ]
        ordered = enh._apply_recency(hits, halflife_days=30)
        assert [h["drawer_id"] for h in ordered] == ["new", "old"]
        assert ordered[0]["recency_boost"] > 0

    def test_recency_never_lifts_a_weak_match_over_a_strong_one(self):
        hits = [
            {
                "drawer_id": "strong",
                "distance": 0.2,
                "bm25_score": 3.0,
                "filed_at": self._days_ago(900),
            },
            {
                "drawer_id": "weak",
                "distance": 0.9,
                "bm25_score": 0.2,
                "filed_at": self._days_ago(0),
            },
        ]
        ordered = enh._apply_recency(hits, halflife_days=30)
        assert [h["drawer_id"] for h in ordered] == ["strong", "weak"]

    def test_works_on_rrf_scores(self):
        hits = [
            {"drawer_id": "old", "rrf_score": 0.0325, "filed_at": self._days_ago(400)},
            {"drawer_id": "new", "rrf_score": 0.0322, "filed_at": self._days_ago(1)},
        ]
        assert [h["drawer_id"] for h in enh._apply_recency(hits, 30)] == ["new", "old"]

    def test_undated_hits_are_never_penalized(self):
        hits = [
            {"drawer_id": "a", "distance": 0.5, "bm25_score": 1.0},
            {"drawer_id": "b", "distance": 0.5, "bm25_score": 1.0},
        ]
        ordered = enh._apply_recency(hits, 30)
        assert [h["drawer_id"] for h in ordered] == ["a", "b"]
        assert all("recency_boost" not in h for h in ordered)

    def test_cli_shaped_hits_read_nested_metadata(self):
        hits = [
            {
                "text": "x",
                "distance": 0.5,
                "bm25_score": 1.0,
                "metadata": {"filed_at": self._days_ago(400)},
            },
            {
                "text": "y",
                "distance": 0.5,
                "bm25_score": 1.0,
                "metadata": {"filed_at": self._days_ago(1)},
            },
        ]
        assert [h["text"] for h in enh._apply_recency(hits, 30)] == ["y", "x"]

    def test_enabled_search_still_returns_everything(
        self, palace_path, seeded_collection, monkeypatch
    ):
        baseline = search_memories("tokens", palace_path, n_results=4)
        monkeypatch.setenv("MEMPALACE_RECENCY_BOOST", "1")
        boosted = search_memories("tokens", palace_path, n_results=4)
        assert set(_ids(boosted)) == set(_ids(baseline))


class TestCrossEncoder:
    def test_scorer_reorders_results(self, palace_path, seeded_collection, monkeypatch):
        def fake_scorer(pairs):
            return [10.0 if "React" in doc else 0.0 for _q, doc in pairs]

        monkeypatch.setattr(cer, "get_scorer", lambda model_name: fake_scorer)
        monkeypatch.setenv("MEMPALACE_RERANK_CROSS_ENCODER", "1")
        result = search_memories("JWT authentication", palace_path, n_results=4)
        assert result["results"][0]["drawer_id"] == "drawer_proj_frontend_ccc"
        assert "cross_encoder_score" in result["results"][0]

    def test_broken_model_keeps_the_search_alive(self, palace_path, seeded_collection, monkeypatch):
        def boom(model_name):
            raise ImportError("no sentence-transformers here")

        monkeypatch.setattr(cer, "get_scorer", boom)
        monkeypatch.setenv("MEMPALACE_RERANK_CROSS_ENCODER", "1")
        result = search_memories("JWT authentication", palace_path, n_results=4)
        assert len(result["results"]) > 0


class TestStageFailureIsContained:
    def test_failing_fusion_returns_convex_order(self, palace_path, seeded_collection, monkeypatch):
        convex = search_memories("JWT authentication", palace_path, n_results=4)

        def explode(hits):
            raise RuntimeError("fusion bug")

        monkeypatch.setattr(enh, "_rrf_order", explode)
        monkeypatch.setenv("MEMPALACE_FUSION_MODE", "rrf")
        result = search_memories("JWT authentication", palace_path, n_results=4)
        assert _ids(result) == _ids(convex)


class TestFinishing:
    def test_provenance_labels_every_hit(self, palace_path, seeded_collection, monkeypatch):
        monkeypatch.setenv("MEMPALACE_PROVENANCE", "1")
        result = search_memories("JWT authentication", palace_path, n_results=4)
        assert result["results"]
        assert all(h["source_kind"] == "file" for h in result["results"])

    def test_prefer_curated_puts_memory_above_the_transcript_quoting_it(self, monkeypatch):
        monkeypatch.setenv("MEMPALACE_PREFER_CURATED", "1")
        text = "Lighthouse manages every client tenant; separate admin accounts were retired."
        hits = [
            {"drawer_id": "t", "text": "user said: " + text, "source_file": "session.jsonl"},
            {"drawer_id": "m", "text": text, "source_file": "/home/a/project/memory/tenants.md"},
        ]
        out = enh.finish_hits(hits)
        assert [h["drawer_id"] for h in out] == ["m", "t"]

    def test_calibrated_confidence(self, palace_path, seeded_collection, monkeypatch, tmp_path):
        cal = fit_calibrator([(0.1, 0), (0.2, 0), (0.6, 1), (0.9, 1)], source="test")
        path = tmp_path / "cal.json"
        cal.save(path)
        assert json.loads(path.read_text())
        monkeypatch.setenv("MEMPALACE_CALIBRATION_PATH", str(path))
        result = search_memories("JWT authentication", palace_path, n_results=4)
        assert all(0.0 <= h["confidence"] <= 1.0 for h in result["results"])

    def test_missing_calibrator_file_adds_nothing(
        self, palace_path, seeded_collection, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("MEMPALACE_CALIBRATION_PATH", str(tmp_path / "nope.json"))
        result = search_memories("JWT authentication", palace_path, n_results=4)
        assert all("confidence" not in h for h in result["results"])


class TestConfigFile:
    def test_config_json_switches_features_on(self, monkeypatch):
        cfg = {"fusion_mode": "rrf", "recency_boost": True, "provenance": "yes"}
        monkeypatch.setattr(enh, "_file_config", lambda: cfg)
        assert enh.fusion_mode() == "rrf"
        assert enh.recency_enabled() is True
        assert enh.provenance_enabled() is True
        assert enh.any_enabled() is True

    def test_env_off_beats_config_on(self, monkeypatch):
        monkeypatch.setattr(enh, "_file_config", lambda: {"recency_boost": True})
        monkeypatch.setenv("MEMPALACE_RECENCY_BOOST", "0")
        assert enh.recency_enabled() is False
