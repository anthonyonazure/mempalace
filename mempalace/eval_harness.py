"""Measure search quality on YOUR questions, before and after a change.

Write a short file of questions you would really ask, each with what the
right answer looks like, then run::

    mempalace eval questions.yaml

It runs every question against your palace once per settings profile
(upstream baseline, each enhancement alone, everything on) and prints how
often the right note came back first, in the top 3, and in the top 5.
Nothing is written to the palace.

Question file format (YAML or JSON list)::

    - question: why did we switch the client portal to GraphQL
      expect_file: decisions.md          # any part of the source path
    - question: what is the Log Analytics daily cap
      expect_text: 5 GB per day          # a phrase inside the right note
      wing: azure                        # optional filter, as in search

A hit counts as correct when ANY ``expect_file`` value is part of its source
path or ANY ``expect_text`` value appears in its text (case-insensitive).
Both keys also accept a list.

Metrics:

* **hit@1 / hit@3 / hit@5** - share of questions whose first correct hit is
  within the top 1, 3 or 5.
* **MRR** - mean reciprocal rank (1.0 = always first; 0.5 = always second).
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Optional

# Settings profiles: env vars applied for the duration of one pass.
_ALL_OFF = {
    "MEMPALACE_FUSION_MODE": "convex",
    "MEMPALACE_RECENCY_BOOST": "0",
    "MEMPALACE_RERANK_CROSS_ENCODER": "0",
    "MEMPALACE_PROVENANCE": "0",
    "MEMPALACE_PREFER_CURATED": "0",
}

PROFILES = {
    "baseline": {},
    "rrf": {"MEMPALACE_FUSION_MODE": "rrf"},
    "recency": {"MEMPALACE_RECENCY_BOOST": "1"},
    "curated-first": {"MEMPALACE_PREFER_CURATED": "1"},
    "cross-encoder": {"MEMPALACE_RERANK_CROSS_ENCODER": "1"},
    "all": {
        "MEMPALACE_FUSION_MODE": "rrf",
        "MEMPALACE_RECENCY_BOOST": "1",
        "MEMPALACE_PREFER_CURATED": "1",
        "MEMPALACE_RERANK_CROSS_ENCODER": "1",
    },
}

TEMPLATE = """\
# Search test questions. Aim for 30-50 real questions you would ask.
# For each one, say how to recognize the right answer:
#   expect_file: part of the file name or path of the right note
#   expect_text: a phrase that appears in the right note
# Mix easy and hard ones, old and recent topics, and every client/project.

- question: why did we switch the client portal API to GraphQL
  expect_file: decisions.md

- question: what daily cap did we put on the Log Analytics workspace
  expect_text: 5 GB per day

- question: which client needs CMMC Level 2
  expect_text: CMMC Level 2
"""


def load_questions(path: str) -> list:
    text = Path(path).read_text(encoding="utf-8")
    if path.lower().endswith(".json"):
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty list of questions")
    questions = []
    for i, item in enumerate(data, 1):
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            raise ValueError(f"{path}: item {i} needs a 'question'")
        files = item.get("expect_file") or []
        texts = item.get("expect_text") or []
        files = [files] if isinstance(files, str) else list(files)
        texts = [texts] if isinstance(texts, str) else list(texts)
        if not files and not texts:
            raise ValueError(f"{path}: item {i} needs 'expect_file' or 'expect_text'")
        questions.append(
            {
                "question": str(item["question"]).strip(),
                "expect_file": [str(f).lower() for f in files],
                "expect_text": [str(t).lower() for t in texts],
                "wing": item.get("wing"),
                "room": item.get("room"),
            }
        )
    return questions


def is_correct(hit: dict, q: dict) -> bool:
    path = " ".join(str(hit.get(k) or "") for k in ("source_path", "source_file")).lower()
    text = str(hit.get("text") or "").lower()
    return any(f in path for f in q["expect_file"]) or any(t in text for t in q["expect_text"])


def first_correct_rank(hits: list, q: dict) -> Optional[int]:
    for rank, hit in enumerate(hits, 1):
        if is_correct(hit, q):
            return rank
    return None


@contextlib.contextmanager
def _env(overrides: dict):
    saved = {k: os.environ.get(k) for k in set(_ALL_OFF) | set(overrides)}
    try:
        os.environ.update(_ALL_OFF)
        os.environ.update(overrides)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def profile_available(name: str) -> tuple:
    if "cross-encoder" in name or name == "all":
        if importlib.util.find_spec("sentence_transformers") is None:
            return False, "needs: pip install sentence-transformers"
    return True, ""


def run_profile(
    name: str, questions: list, palace_path: str, k: int = 5, search=None, overrides=None
) -> dict:
    if search is None:
        from .searcher import search_memories as search
    ranks = []
    misses = []
    with _env(PROFILES[name] if overrides is None else overrides):
        for q in questions:
            result = search(q["question"], palace_path, wing=q["wing"], room=q["room"], n_results=k)
            hits = result.get("results", []) if isinstance(result, dict) else []
            rank = first_correct_rank(hits, q)
            ranks.append(rank)
            if rank is None:
                misses.append(
                    {
                        "question": q["question"],
                        "got": [h.get("source_file") for h in hits[:3]],
                        "error": result.get("error") if isinstance(result, dict) else None,
                    }
                )
    n = len(questions)
    return {
        "profile": name,
        "questions": n,
        "hit@1": sum(1 for r in ranks if r and r <= 1) / n,
        "hit@3": sum(1 for r in ranks if r and r <= 3) / n,
        "hit@5": sum(1 for r in ranks if r and r <= 5) / n,
        "mrr": sum(1.0 / r for r in ranks if r) / n,
        "misses": misses,
    }


def evaluate(questions_path: str, palace_path: str, profiles=None, search=None) -> list:
    questions = load_questions(questions_path)
    reports = []
    for name in profiles or list(PROFILES):
        if name not in PROFILES:
            raise ValueError(f"unknown profile {name!r} (known: {', '.join(PROFILES)})")
        ok, why = profile_available(name) if search is None else (True, "")
        if not ok and name == "all":
            # Still worth knowing what everything else does together.
            overrides = dict(PROFILES["all"], MEMPALACE_RERANK_CROSS_ENCODER="0")
            report = run_profile(name, questions, palace_path, search=search, overrides=overrides)
            report["profile"] = "all-but-CE"
            reports.append(report)
            continue
        if not ok:
            reports.append({"profile": name, "skipped": why})
            continue
        reports.append(run_profile(name, questions, palace_path, search=search))
    return reports


def format_table(reports: list) -> str:
    lines = [f"  {'profile':<15} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'MRR':>6}"]
    lines.append("  " + "-" * 43)
    base = next((r for r in reports if r.get("profile") == "baseline" and "mrr" in r), None)
    for r in reports:
        if "skipped" in r:
            lines.append(f"  {r['profile']:<15} skipped ({r['skipped']})")
            continue
        delta = ""
        if base is not None and r is not base:
            diff = r["mrr"] - base["mrr"]
            delta = f"   {'+' if diff >= 0 else ''}{diff:.3f} vs baseline"
        lines.append(
            f"  {r['profile']:<15} {r['hit@1']:>6.0%} {r['hit@3']:>6.0%} "
            f"{r['hit@5']:>6.0%} {r['mrr']:>6.3f}{delta}"
        )
    return "\n".join(lines)


def cmd_eval(args) -> int:
    from .config import MempalaceConfig

    if getattr(args, "init", False):
        target = Path(args.questions)
        if target.exists():
            print(f"{target} already exists; not overwriting.")
            return 1
        target.write_text(TEMPLATE, encoding="utf-8")
        print(
            f"Wrote a starter question file to {target}. Edit it, then run: mempalace eval {target}"
        )
        return 0
    palace = (
        os.path.expanduser(args.palace)
        if getattr(args, "palace", None)
        else MempalaceConfig().palace_path
    )
    profiles = args.profiles.split(",") if getattr(args, "profiles", None) else None
    reports = evaluate(args.questions, palace, profiles=profiles)
    print(f"\n  Search evaluation: {args.questions} against {palace}\n")
    print(format_table(reports))
    base = next((r for r in reports if r.get("profile") == "baseline" and "misses" in r), None)
    if base and base["misses"]:
        print(f"\n  Baseline misses ({len(base['misses'])}):")
        for m in base["misses"][:15]:
            print(
                f"    - {m['question']}  (got: {', '.join(str(g) for g in m['got']) or 'nothing'})"
            )
    if getattr(args, "json", None):
        Path(args.json).write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\n  Full report: {args.json}")
    print()
    return 0
