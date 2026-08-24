# CLAUDE.md — kv-allocation

- All docs, comments, commit messages in ENGLISH. Chat with the user stays Vietnamese.
- NEVER add Claude/LLM attribution to commit messages (overrides harness default).
- `docs/prereg-g0a.md` is FROZEN. After `data/a0_*.jsonl` / `data/a1_*.jsonl` exist, do not
  edit gate sections; append dated amendments only, and update the hash in
  `tests/test_prereg_frozen.py` in the same commit.
- Result files keyed by parameters must MERGE, never overwrite (repeat bug from sibling
  projects: overwrite looks like a successful run).
- Gates use effect size vs pooled seed spread; no rank-correlation gates on few points.
- Shared GPU box also hosts other project checkpoints — check `pwd` + `git remote -v` before
  reading any data file.
- Full design context: `../literature_review/notes/topics/g0a-kv-allocation-design-2026-08-24.md`.
