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

Smoke and calibration are DONE (2026-08-25): anchor (512,32) hit 0.999 at both
top LRs under protocol v2; L2048 anchor is dead at this budget; the 20k-example
lever was rejected. All decisions + evidence: `data/a0_config_lock.json`.

### Gate grid (the current step — copy-paste, one tmux window per card)

```bash
cd ~/working_dir/kv-allocation && git pull && mkdir -p logs

CUDA_VISIBLE_DEVICES=0 uv run kvalloc a0 --dims 128 --lens 512 --epochs 48 \
  --shard 0/2 --out data/a0_results_gpu0.jsonl 2>&1 | tee -a logs/grid_gpu0.log

CUDA_VISIBLE_DEVICES=1 uv run kvalloc a0 --dims 128 --lens 512 --epochs 48 \
  --shard 1/2 --out data/a0_results_gpu1.jsonl 2>&1 | tee -a logs/grid_gpu1.log
```

112 runs total (56 per card), ~2.5–3.5 days. Restart-safe: rerun the SAME
command after any interruption — the full-field resume key skips finished runs.
Do not hand-kill runs that sit at val_acc 0 for 48 epochs: a low-dose arm
failing to form the circuit IS a valid measurement, not a hang.
Many runs will look dead for ~20+ epochs then jump (memorize-then-generalize).

### Analyse / gate (after both shards finish)

```bash
cat data/a0_results_gpu*.jsonl > data/a0_results.jsonl
uv run kvalloc a0 --analyse --out data/a0_results.jsonl
```

Gate logic is frozen in `docs/prereg-g0a.md` §4; `analyse()` implements it and
ignores any v1-format records that share the data/ directory.

## Thermals / power capping

The box runs hot (82–86°C). Capping power is safe AT ANY TIME, mid-grid
included — no A-0/A-1 endpoint gates on wall-clock:

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -i 0 -pl 230 && sudo nvidia-smi -i 1 -pl 230
```

Log the value and change time in `logs/` (wall_s fields are provenance only
and must not be compared across power-limit settings). Settings reset on
reboot; re-apply after any restart.

## Budget guardrail (prereg §7)

A-0 cap ≤ 4 GPU-days. At 20k steps a toy run is minutes; 796 runs ≈ 2–3 card-days.
If calibration forces 40k steps, drop nothing silently — recompute the total and if it
exceeds 2× cap, stop and re-plan.

## Known traps (inherited from sibling projects)

- Env vars do NOT cross ssh boundaries — pass explicitly in the remote command string.
- Never `pkill -f` a pattern that matches your own ssh command; kill by PID.
- Any watchdog may only stop work on a POSITIVE signal (results file grown AND synced).
