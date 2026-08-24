# kv-allocation

Controlled from-scratch study of KV-cache byte allocation across the head axis (GQA) and the
layer axis (cross-layer sharing), targeting a compute-optimal allocation law and a hazard
analysis of deep sharing. Target venue: ICML 2027.

Read in order:
1. `docs/idea-lock.md` — locked claim, scope guards, competitor map.
2. `docs/prereg-g0a.md` — FROZEN gate design and thresholds (guarded by tests).

Layout: `src/kvalloc/` experiment code · `tests/` (no GPU needed) ·
`data/` gate outputs (merge-only, never overwrite).

Managed with uv (pattern from the fertility-precision repo):
```bash
uv sync                      # creates .venv, installs pinned deps
uv run kvalloc doctor        # import + GPU check BEFORE spending compute
uv run kvalloc a0 --smoke    # Stage A-0 runner (see docs/runbook-a0.md)
uv run pytest -q             # 23 tests, CPU-only
```
