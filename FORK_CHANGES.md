# What this fork adds to MemPalace v3.11.0

Branch `enhanced/search-v3.11`. Each feature lives in its own file, attached to
upstream code at a few marked "Fork hook" spots, so moving to a new upstream
release means re-attaching those spots rather than untangling a merge.

| Feature | File | Default |
|---|---|---|
| Search add-ons (RRF fusion, recency boost, cross-encoder re-rank, calibrated confidence, source labels, curated-first ordering) | `mempalace/search_enhancements.py` (+ `rrf.py`, `recency.py`, `cross_encoder_rerank.py`, `calibration.py`, `provenance.py`, `result_ordering.py`, ported from techempower-org/mempalace, MIT) | Off |
| Secret scrubbing on save | `mempalace/secret_scrubber.py` | **On** |
| Client walls (one palace per client) | `mempalace/client_walls.py` | Off until configured |
| `mempalace eval` search scoring | `mempalace/eval_harness.py` | n/a |

## Hook points in upstream files

- `mempalace/searcher/candidates.py` and `searcher/cli_search.py`: ranking goes through `search_enhancements.rank_pool`
- `mempalace/searcher/query.py`: results go through `search_enhancements.finish_hits`
- `mempalace/palace/collection.py`: writable collections get `secret_scrubber.install`
- `mempalace/config.py` (`palace_path`): palace choice goes through `client_walls.resolve`
- `mempalace/cli/parser.py`: `clients` and `eval` commands

## Settings (config.json, or the env var in brackets)

```json
{
  "fusion_mode": "rrf",
  "recency_boost": true,
  "recency_halflife_days": 30,
  "cross_encoder_rerank": false,
  "provenance": true,
  "prefer_curated": true,
  "scrub_secrets": true,
  "client_walls": {
    "root": "~/.mempalace/clients",
    "strict": true,
    "clients": {
      "acme-law": ["~/Clients/AcmeLaw"],
      "beltway-defense": ["~/Clients/Beltway"]
    }
  }
}
```

Env equivalents: `MEMPALACE_FUSION_MODE`, `MEMPALACE_RECENCY_BOOST`,
`MEMPALACE_RECENCY_HALFLIFE_DAYS`, `MEMPALACE_RERANK_CROSS_ENCODER`
(needs `pip install sentence-transformers`), `MEMPALACE_CALIBRATION_PATH`,
`MEMPALACE_PROVENANCE`, `MEMPALACE_PREFER_CURATED`, `MEMPALACE_SCRUB_SECRETS`,
`MEMPALACE_CLIENT` (pin a client palace).

## Recommended order

1. `mempalace eval questions.yaml --init`, fill in 30-50 real questions.
2. `mempalace eval questions.yaml` to get a baseline and see which add-ons help.
3. Turn on only the add-ons that beat baseline.

## Known limits

- Secret scrubbing applies to text saved after it is on. Existing drawers and
  the original files on disk are unchanged.
- Client walls do not separate the global entity registry / people map in the
  config folder.
- Ranking add-ons (RRF, recency, cross-encoder) apply to both `mempalace search`
  and the MCP search tool. Source labels, curated-first ordering and confidence
  apply to the MCP search tool only. The media/code search path and the
  keyword-only fallback (used when the vector index is damaged) are unchanged.
