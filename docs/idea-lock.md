# Idea lock — KV-cache allocation law (locked 2026-08-24)

## One-sentence claim (candidate)

Given a fixed KV-cache byte budget, capability depends on HOW the bytes are allocated across
the head axis (GQA grouping) and the layer axis (cross-layer sharing) — or, if the two routes
prove fungible, KV bytes obey a single simple frontier. Either outcome, established from
scratch with a recall-sensitive instrument, is the paper; the hazard region (sharing ≥3×,
where CLA stopped and MLKV saw collapse) is where no published analysis exists.

## Why this is ours to take

- Cross-layer sharing is deployed in production (Gemma 3n, Apple FM, YOCO lineage) with no
  principled allocation law behind it.
- arXiv:2503.09579 (Cost-Optimal GQA) built the law template for the HEAD axis only.
- arXiv:2405.12981 (CLA) showed only CLA2 Pareto-wins and never tested recall/long-context.
- arXiv:2410.14442 is the closest prior: from-scratch cross-layer topology study — but ~2 dose
  levels, no continuous byte dose, no GQA crossing, no recall battery, no fitted law. We must
  cite it prominently and differentiate on those four axes.
- Our unfair advantage: the MLKV line (arXiv:2406.09297) is our own architecture work; the
  training/eval tooling and intuition transfer directly. 100M–1B from-scratch is exactly
  2×A6000 territory and below the radar of the big labs racing dLLM/RL lanes.

## Design commitments (frozen rationale, see prereg for thresholds)

1. Primary instruments are recall-sensitive (AR-slice loss, MQAR/NIAH battery), NOT aggregate
   val loss — CLA2 vs MHA differ by ~0.06 ppl while recall instruments separate architectures
   at 0-vs-1 magnitudes (Zoology).
2. Byte-matched arm pairs across routes ({GQA-4 vs CLA-4} at 0.25×, {MQA vs CLA-12} at 0.083×)
   are the fungibility test — the scientifically load-bearing comparison.
3. Params freed by removed K/V projections are reallocated to FFN width so arms are iso-param
   and iso-compute (CLA convention); one no-reallocation robustness pair.
4. LR is a registered confound (CLA advantage appears only at tuned/high LR): pilot LR on two
   anchor arms, freeze the arm-class→LR mapping before the grid.
5. Both gate outcomes are live: route divergence → allocation law; route fungibility → single
   frontier ("KV bytes are fungible"). The only KILL is instrument failure / no dose-response.

## Kill history & scope guards

- Do NOT drift into an inference-time compression paper (xKV/CommonKV lane is post-training
  adaptation; ours is from-scratch science). Compression papers are motivation, not venue.
- Do NOT gate on Spearman/rank statistics over few points (Phase-3 lesson from dllm-fertility):
  gates use effect size vs pooled seed spread.
- Before writing the paper intro, skim arXiv:2604.22782 (Stochastic KV Routing) and
  arXiv:2602.03560 (HySparse) — flagged as orthogonal but unread at lock time.

## Venue & schedule

ICML 2027 (abstract ~16 Jan, paper ~22 Jan 2027 AoE). Gate A-0 (synthetic MQAR dose,
2–3 GPU-days) → Gate A-1 (160M from-scratch grid, ~2 calendar weeks) → if GO, scale arm subset
to 410M, fit the frontier, retrieval-head mechanism study. Full design context:
`../../literature_review/notes/topics/g0a-kv-allocation-design-2026-08-24.md`.
