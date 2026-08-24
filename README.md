# kv-allocation

Controlled from-scratch study of KV-cache byte allocation across the head axis (GQA) and the
layer axis (cross-layer sharing), targeting a compute-optimal allocation law and a hazard
analysis of deep sharing. Target venue: ICML 2027.

Read in order:
1. `docs/idea-lock.md` — locked claim, scope guards, competitor map.
2. `docs/prereg-g0a.md` — FROZEN gate design and thresholds (guarded by tests).

Layout: `src/kvalloc/` experiment code · `tests/` (stdlib unittest, no GPU) ·
`data/` gate outputs (merge-only, never overwrite).

Run tests: `python3 -m unittest discover -s tests -v`
