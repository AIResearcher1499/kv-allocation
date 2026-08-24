# A-0 runbook (2×A6000 box)

Setup once (uv-managed, same pattern as fertility-precision):
```bash
cd ~/working_dir && git clone <this-repo> kv-allocation && cd kv-allocation
uv sync
uv run kvalloc doctor    # must print the two A6000s + sdpa ok
uv run pytest -q         # must be green before any run
```
All commands from the repo root. Check `pwd` + `git remote -v` first —
the box hosts other projects with similar file layouts.

## Order of operations

1. **Smoke** (~minutes): `uv run kvalloc a0 --smoke` → writes `data/a0_smoke.jsonl`
   (separate file by design). Confirms end-to-end on CUDA.
2. **Calibrate** (anchor dose only, no gate exposure):
   `uv run kvalloc a0 --calibrate --steps 20000` → `data/a0_calibrate.jsonl`.
   Requirement: diag acc ≥ 0.9 on (512,32) at some LR. If not, retry `--steps 40000`.
   Record the chosen steps in `data/config_lock.json` (`{"a0_steps": <n>}`).
3. **Grid** (796 runs): one shell per card, SEPARATE output files (append races):
   `CUDA_VISIBLE_DEVICES=0 uv run kvalloc a0 --steps <n> --out data/a0_results_gpu0.jsonl`
   `CUDA_VISIBLE_DEVICES=1 uv run kvalloc a0 --steps <n> --out data/a0_results_gpu1.jsonl`
   then `cat data/a0_results_gpu*.jsonl > data/a0_results.jsonl`. The full-field resume
   key makes restarts harmless within each shard.
4. **Analyse / gate**: `uv run kvalloc a0 --analyse --out data/a0_results.jsonl`.
   Gate logic is frozen in `docs/prereg-g0a.md` §4; `analyse()` implements it.

## Budget guardrail (prereg §7)

A-0 cap ≤ 4 GPU-days. At 20k steps a toy run is minutes; 796 runs ≈ 2–3 card-days.
If calibration forces 40k steps, drop nothing silently — recompute the total and if it
exceeds 2× cap, stop and re-plan.

## Known traps (inherited from sibling projects)

- Env vars do NOT cross ssh boundaries — pass explicitly in the remote command string.
- Never `pkill -f` a pattern that matches your own ssh command; kill by PID.
- Any watchdog may only stop work on a POSITIVE signal (results file grown AND synced).
