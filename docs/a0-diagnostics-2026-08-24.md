# A-0 pipeline diagnostics — 2026-08-24 (local, MPS/CPU)

Findings from bringing up the MQAR pipeline before any gate data. All runs tiny
(dim 128, L=64, N=8 unless noted) — NOT gate evidence, instrument calibration only.

## Bug found and fixed

**Missing GPT-style init.** `nn.Embedding` defaults to N(0,1); with the tied LM head this
produced initial CE ≈ 106 vs the ln(8192)=9.01 chance floor — the model spent hundreds of
steps just shrinking logits, and the first smoke run DIVERGED (loss 20.7 after 200 steps).
Fix: normal(0, 0.02) everywhere + residual-out projections scaled by 1/sqrt(2·n_layers).
After the fix, step-0 loss is exactly at chance.

## Hypotheses tested for the ~30% accuracy plateau (all REJECTED as the cause)

| hypothesis | probe | result |
|---|---|---|
| filler segment breaks attention | L=32 (zero filler) vs L=64 | identical (~29%) |
| LR too low | 3e-3 vs 1e-2 | 1e-2 much better (6%→29%) but still plateaus |
| weight decay 0.1 too strong | wd=0 | WORSE (7.6%) |
| head_dim=16 bottleneck | 2 heads (hd=64) | same (~25%) |
| vocab too large for embeddings | 256-key/value vocab | still ~33% (loss 1.67 ≈ uniform-over-8-values) |
| RoPE vs learned positions | + learned pos emb | same trajectory (~31% @ 3k) |
| batch too small | 512 vs 128 | same trajectory |

## Conclusion

The small-vocab probe is diagnostic: loss ≈ ln(8) means the model learns "the answer is one
of the N values present" quickly, but the key→value MATCHING circuit forms SLOWLY. Constant-LR
runs keep climbing through 6k steps (0.06 → 0.30) with no plateau; every cosine run was cut
by LR decay mid-formation. This matches slow induction-circuit formation, not a bug.
Zoology reports attention ≈ 1.0 at convergence with LR sweeps; our probes never reached
convergence on MPS (minutes/1k steps — wrong hardware for this).

## Protocol consequence (implemented)

1. Default steps raised 8000 → 20000 for the grid; `steps` is in the resume key.
2. New `--calibrate` mode: anchor dose (8,4) ONLY × {(512,32), (2048,64)} × top-2 LRs.
   Run on the A6000 FIRST; pick the smallest steps where anchor diag acc ≥ 0.9, then launch
   the grid at that setting. Calibration trains no low-dose arm, so no dose contrast is
   visible before the gate (prereg hygiene).
3. If the anchor cannot reach 0.9 on (512,32) within 40k steps on the A6000, that is an
   instrument-scale problem to solve BEFORE the grid (options, in order: longer training,
   finite-dataset epoching à la Zoology, easier gate cells) — record whatever is chosen in
   `data/config_lock.json` before any dose-contrast run.
