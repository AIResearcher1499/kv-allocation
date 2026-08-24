# Pre-registration — Gate G0-A (FROZEN 2026-08-24)

This document is FROZEN as of the commit that adds it. Amendments after any file matching
`data/a0_*.jsonl` or `data/a1_*.jsonl` exists are prohibited except as clearly-marked,
dated amendment sections that do not alter §4/§7 thresholds. `tests/test_prereg_frozen.py`
guards the content hash.

## 1. Question

Is there a measurable, monotone dose-response of recall capability on KV-cache bytes/token at
160M scale, and do byte-matched allocations via the head route (GQA) vs the layer route
(cross-layer sharing) behave differently?

## 2. Stage A-0 — synthetic MQAR dose-response (run first)

- Models: 4-layer transformers, d ∈ {128, 256}, trained directly on MQAR (Zoology protocol).
- Doses: n_kv ∈ {n_h, n_h/4, 1} × KV-layers ∈ {4, 2, 1} (full factorial, 9 cells per d).
- Task grid: seq len {512, 2048, 4096} × KV pairs {32, 64, 128, 256}, vocab 8192; eval on
  longer/denser settings than trained (Based convention).
- LR: 4 values log-spaced in [1e-4, 1e-2]; report max test accuracy per cell (Zoology
  convention). Repeat ONE cell twice to estimate max-over-LR jitter.

## 3. Stage A-1 — from-scratch 160M grid

- Recipe: FineWeb-Edu, fixed tokenizer (chosen once at pilot, recorded in `data/config_lock.json`
  before the grid starts), 12 layers, d=768, 12 q-heads, d_h=64, seq 4096, global batch ~1M
  tokens, AdamW(0.9, 0.95), wd 0.1, warmup 500 steps, cosine to 10%, 3.2B tokens.
- Arms (relative KV bytes/token = L_kv·n_kv/144):

| arm | n_kv | KV layers | rel. bytes |
|---|---|---|---|
| mha | 12 | 12 | 1.000 |
| gqa4 | 3 | 12 | 0.250 |
| mqa | 1 | 12 | 0.0833 |
| cla2 | 12 | 6 | 0.500 |
| cla4 | 12 | 3 | 0.250 |
| cla12 | 12 | 1 | 0.0833 |
| gqa4_cla2 | 3 | 6 | 0.125 |
| gqa2_cla3 | 6 | 4 | 0.1667 |

- 2 seeds per arm. Freed K/V-projection params reallocated to FFN (iso-param); one
  no-reallocation robustness run for the {gqa4, cla4} pair.
- LR protocol: pilot {3e-4, 6e-4, 1.2e-3} on mha and cla12 only; freeze arm-class→LR mapping
  in `data/config_lock.json` before any grid run.
- Endpoints, priority order: (E1) AR-slice log-loss on held-out FineWeb (Zoology definition:
  tokens whose bigram occurred earlier in context); (E2) synthetic battery S-NIAH / MK-NIAH /
  MQ-NIAH at 2K/4K/8K, RULER task definitions shrunk — pilot on mha anchor first, drop tasks
  where the anchor is at floor; (E3) val loss (reported only); (E4, exploratory, ungated)
  retrieval-head census per arXiv:2404.15574 (threshold 0.1, NIAH shrunk to 1–8K).

## 4. Gates (frozen)

**Gate A-0 → proceed to A-1** if: monotone dose-response with range ≥ 15 accuracy points
between largest and smallest byte budget in ≥ 2 of the (len × pairs) stress cells, exceeding
the max-over-LR jitter of the repeated cell. If all arms saturate (> 0.95 everywhere), ONE
pre-authorized stress escalation (pairs ×2 and/or value-vocab ×4), then re-judge. Saturation
after escalation → A-0 INCONCLUSIVE (not a kill); proceed to A-1 with E1/E2 only.

**Gate G0-A (GO/KILL, judged on A-1):** GO iff BOTH
(a) monotone trend of E1 or E2 along the byte ladder with extreme-arm effect size
    ≥ 3× the pooled 2-seed spread (mha vs the 0.0833× arms), AND
(b) instrument sanity: mha anchor off-floor on ≥ 2 battery tasks at 4K.
Route divergence (byte-matched pairs {gqa4, cla4} and {mqa, cla12} differing beyond the pooled
2-seed spread) is RECORDED but NOT required for GO — divergence and fungibility are both live
paper outcomes.
KILL iff (a) fails after the one permitted battery stress escalation.

No rank-correlation statistics in any gate. All comparisons within-recipe; no thresholds may
be compared across different tokenizer or LR choices.

## 5. Exclusion & rerun rules

- A run whose loss diverges (spike > 2× trailing-100-step median that does not recover within
  500 steps) is excluded and rerun with seed+10; the event is logged in `data/exclusions.jsonl`
  (PolyPythias precedent: 2/10 seeds diverged at 410M).
- Result files are keyed by (arm, seed, endpoint, context_len) and MERGED, never overwritten.

## 6. Foreknowledge declaration

At freeze time we know: CLA2 ≈ Pareto-neutral on ppl at 1B (+0.06); MLKV uptraining collapse
at 1 shared KV layer; 2410.14442 found ~2× sharing ≈ baseline from scratch. We therefore
EXPECT loss (E3) to be near-blind and the deep-sharing arms (cla12, mqa) to separate on E2.
The gate is designed to be failable: if recall instruments also fail to separate arms at this
scale, the project dies here.

## 7. Budget cap

A-0 ≤ 4 GPU-days; A-1 ≤ 16 card-days + pilots. If the grid cannot finish within 2× these
caps, stop and re-plan rather than silently shrinking arms.
