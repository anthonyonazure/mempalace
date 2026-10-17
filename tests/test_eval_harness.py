"""Tests for the search evaluation harness."""

import json

import pytest

from mempalace import eval_harness as ev


def _write(tmp_path, data, name="q.yaml"):
    path = tmp_path / name
    if name.endswith(".json"):
        path.write_text(json.dumps(data))
    else:
        import yaml

        path.write_text(yaml.safe_dump(data))
    return str(path)


def test_load_accepts_strings_and_lists(tmp_path):
    path = _write(
        tmp_path,
        [
            {"question": "a", "expect_file": "Auth.py"},
            {"question": "b", "expect_text": ["Five GB", "5 GB"], "wing": "azure"},
        ],
    )
    qs = ev.load_questions(path)
    assert qs[0]["expect_file"] == ["auth.py"]
    assert qs[1]["expect_text"] == ["five gb", "5 gb"]
    assert qs[1]["wing"] == "azure"


@pytest.mark.parametrize("bad", [[], [{"question": ""}], [{"question": "x"}], {"question": "x"}])
def test_load_rejects_bad_files(tmp_path, bad):
    with pytest.raises(ValueError):
        ev.load_questions(_write(tmp_path, bad, "q.json"))


def test_correctness_by_path_or_text():
    q = {"expect_file": ["decisions.md"], "expect_text": ["graphql"]}
    assert ev.is_correct({"source_path": "/x/notes/decisions.md"}, q)
    assert ev.is_correct({"text": "We moved to GraphQL in March"}, q)
    assert not ev.is_correct({"source_file": "clients.md", "text": "Intune"}, q)


def test_metrics_with_a_fake_search(tmp_path):
    path = _write(
        tmp_path,
        [
            {"question": "first", "expect_file": "a.md"},
            {"question": "second", "expect_file": "a.md"},
            {"question": "missing", "expect_file": "zzz.md"},
        ],
    )

    def fake_search(question, palace, **kw):
        order = {"first": ["a.md", "b.md"], "second": ["b.md", "a.md"], "missing": ["b.md"]}
        return {"results": [{"source_file": f} for f in order[question]]}

    (report,) = ev.evaluate(path, "/nowhere", profiles=["baseline"], search=fake_search)
    assert report["hit@1"] == pytest.approx(1 / 3)
    assert report["hit@3"] == pytest.approx(2 / 3)
    assert report["mrr"] == pytest.approx((1 + 0.5) / 3)
    assert [m["question"] for m in report["misses"]] == ["missing"]


def test_profiles_restore_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMPALACE_FUSION_MODE", "convex")
    seen = {}

    def fake_search(question, palace, **kw):
        import os

        seen[question] = os.environ.get("MEMPALACE_FUSION_MODE")
        return {"results": []}

    path = _write(tmp_path, [{"question": "q", "expect_text": "x"}])
    ev.evaluate(path, "/nowhere", profiles=["rrf"], search=fake_search)
    assert seen["q"] == "rrf"
    import os

    assert os.environ["MEMPALACE_FUSION_MODE"] == "convex"


def test_unknown_profile(tmp_path):
    path = _write(tmp_path, [{"question": "q", "expect_text": "x"}])
    with pytest.raises(ValueError):
        ev.evaluate(path, "/nowhere", profiles=["magic"], search=lambda *a, **k: {"results": []})


def test_real_palace_end_to_end(tmp_path, palace_path, seeded_collection):
    path = _write(
        tmp_path,
        [
            {"question": "JWT session tokens", "expect_file": "auth.py"},
            {"question": "React server state", "expect_text": "TanStack Query"},
        ],
    )
    reports = ev.evaluate(
        path, palace_path, profiles=["baseline", "rrf", "recency", "curated-first"]
    )
    for r in reports:
        assert r["questions"] == 2
        assert r["hit@5"] == 1.0
    table = ev.format_table(reports)
    assert "baseline" in table and "vs baseline" in table


def test_init_writes_a_template(tmp_path):
    class Args:
        questions = str(tmp_path / "questions.yaml")
        init = True

    assert ev.cmd_eval(Args) == 0
    assert len(ev.load_questions(Args.questions)) == 3
    assert ev.cmd_eval(Args) == 1  # never overwrites
