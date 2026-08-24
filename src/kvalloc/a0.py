"""Stage A-0 runner — synthetic MQAR dose-response (prereg-g0a.md §2, §4).

Frozen grids live in this module; the gate is judged by `analyse()`.

Operationalization recorded BEFORE any data exists (clarifies, does not alter,
frozen thresholds): a "stress cell" for the gate is a (seq_len, num_pairs) task
cell evaluated ON ITS OWN diagonal (train setting == eval setting), per dim;
the byte-ladder range is acc(max rel_bytes dose) - acc(min rel_bytes dose),
each maxed over the 4 LRs; jitter is the max over eval cells of
|max-over-LR acc(seed A) - max-over-LR acc(seed B)| on the repeated cell.

Result files are MERGE-only: existing (full-key) records are skipped on
resume, never overwritten. Smoke runs write to a SEPARATE file by default.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import torch

from .model import KVModel, ModelConfig
from .mqar import IGNORE, build_batch, is_valid_cell, query_accuracy

# ---- frozen A-0 grids (prereg §2) -------------------------------------------
DIMS = (128, 256)
DOSES = tuple((n_kv, kv_layers) for n_kv in (8, 2, 1) for kv_layers in (4, 2, 1))
TRAIN_LENS = (512, 2048, 4096)
PAIRS = (32, 64, 128, 256)
LRS = (1e-4, 4.6415888336127824e-4, 2.1544346900318823e-3, 1e-2)
BASE_SEED = 20260824
# The single repeated cell for the jitter estimate (mid-ladder dose).
JITTER_CELL = dict(dim=128, n_kv=2, kv_layers=2, seq_len=2048, num_pairs=128)
JITTER_SEED_OFFSET = 1000
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    dim: int
    n_kv: int
    kv_layers: int
    seq_len: int
    num_pairs: int
    lr: float
    seed: int
    steps: int = 8000
    tokens_per_batch: int = 32768
    warmup_frac: float = 0.05
    weight_decay: float = 0.1
    eval_seqs: int = 256

    def key(self) -> str:
        # Resume key covers EVERY field so no parameter change can silently
        # collide with an old record (repeat bug from sibling projects).
        return json.dumps(asdict(self), sort_keys=True)

    @property
    def batch_size(self) -> int:
        return max(8, self.tokens_per_batch // self.seq_len)


def eval_cells():
    return [(l, n) for l in TRAIN_LENS for n in PAIRS if is_valid_cell(l, n)]


def build_plan(steps: int = 8000, dims=DIMS, lens=TRAIN_LENS, pairs=PAIRS,
               lrs=LRS, doses=DOSES, with_jitter: bool = True):
    plan = []
    for dim in dims:
        for (n_kv, kv_layers) in doses:
            for seq_len in lens:
                for num_pairs in pairs:
                    if not is_valid_cell(seq_len, num_pairs):
                        continue
                    for lr in lrs:
                        plan.append(RunConfig(dim, n_kv, kv_layers, seq_len,
                                              num_pairs, lr, BASE_SEED, steps))
    if with_jitter:
        j = JITTER_CELL
        for lr in lrs:
            plan.append(RunConfig(j["dim"], j["n_kv"], j["kv_layers"],
                                  j["seq_len"], j["num_pairs"], lr,
                                  BASE_SEED + JITTER_SEED_OFFSET, steps))
    return plan


def pick_device(arg: str = "auto") -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_one(rc: RunConfig, device: str):
    torch.manual_seed(rc.seed)
    cfg = ModelConfig(dim=rc.dim, n_kv_heads=rc.n_kv, kv_layers=rc.kv_layers)
    model = KVModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=rc.lr, betas=(0.9, 0.95),
                            weight_decay=rc.weight_decay)
    warmup = max(1, int(rc.steps * rc.warmup_frac))

    def lr_at(step):
        if step < warmup:
            return rc.lr * (step + 1) / warmup
        t = (step - warmup) / max(1, rc.steps - warmup)
        return rc.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))

    gen = torch.Generator().manual_seed(rc.seed)
    model.train()
    first_loss = last_loss = None
    for step in range(rc.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = build_batch(rc.batch_size, rc.seq_len, rc.num_pairs, gen, device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()
    return model, first_loss, last_loss


@torch.no_grad()
def eval_grid(model, rc: RunConfig, device: str):
    model.eval()
    out = {}
    for (l, n) in eval_cells():
        gen = torch.Generator().manual_seed(BASE_SEED + 7 * l + n)  # fixed eval data
        accs, done = [], 0
        bs = max(4, rc.tokens_per_batch // l)
        while done < rc.eval_seqs:
            b = min(bs, rc.eval_seqs - done)
            x, y = build_batch(b, l, n, gen, device)
            accs.append((query_accuracy(model(x), y), b))
            done += b
        out[f"L{l}_N{n}"] = sum(a * w for a, w in accs) / sum(w for _, w in accs)
    return out


def load_done_keys(path: str):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done.add(json.dumps(rec["config"], sort_keys=True))
    return done


def run(plan, out_path: str, device: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = load_done_keys(out_path)
    todo = [rc for rc in plan if rc.key() not in done]
    print(f"plan={len(plan)} done={len(done)} todo={len(todo)} device={device}")
    for i, rc in enumerate(todo):
        t0 = time.time()
        model, first_loss, last_loss = train_one(rc, device)
        evals = eval_grid(model, rc, device)
        rec = {
            "config": asdict(rc),
            "first_loss": first_loss,
            "final_loss": last_loss,
            "evals": evals,
            "wall_s": round(time.time() - t0, 1),
            "device": device,
            "torch": torch.__version__,
        }
        with open(out_path, "a", encoding="utf-8") as f:  # append = merge
            f.write(json.dumps(rec) + "\n")
        diag = evals.get(f"L{rc.seq_len}_N{rc.num_pairs}", float("nan"))
        print(f"[{i + 1}/{len(todo)}] {rc.dim}d nkv={rc.n_kv} kvl={rc.kv_layers} "
              f"L{rc.seq_len} N{rc.num_pairs} lr={rc.lr:.1e} seed={rc.seed} "
              f"diag_acc={diag:.3f} loss={last_loss:.3f} {rec['wall_s']}s",
              flush=True)


# ---- analysis / gate --------------------------------------------------------

def _max_over_lr(records, dim, n_kv, kv_layers, seq_len, num_pairs, seed, cell):
    vals = [r["evals"][cell] for r in records
            if r["config"]["dim"] == dim and r["config"]["n_kv"] == n_kv
            and r["config"]["kv_layers"] == kv_layers
            and r["config"]["seq_len"] == seq_len
            and r["config"]["num_pairs"] == num_pairs
            and r["config"]["seed"] == seed]
    return max(vals) if vals else None


def analyse(path: str):
    with open(path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    hi = max(DOSES, key=lambda d: d[0] * d[1])   # (8, 4) -> rel 1.0
    lo = min(DOSES, key=lambda d: d[0] * d[1])   # (1, 1) -> rel 1/32

    j = JITTER_CELL
    cell = f"L{j['seq_len']}_N{j['num_pairs']}"
    a = _max_over_lr(records, j["dim"], j["n_kv"], j["kv_layers"],
                     j["seq_len"], j["num_pairs"], BASE_SEED, cell)
    b = _max_over_lr(records, j["dim"], j["n_kv"], j["kv_layers"],
                     j["seq_len"], j["num_pairs"],
                     BASE_SEED + JITTER_SEED_OFFSET, cell)
    jitter = abs(a - b) if a is not None and b is not None else None

    rows, passing_cells = [], set()
    for dim in DIMS:
        for (l, n) in eval_cells():
            cell = f"L{l}_N{n}"
            top = _max_over_lr(records, dim, hi[0], hi[1], l, n, BASE_SEED, cell)
            bot = _max_over_lr(records, dim, lo[0], lo[1], l, n, BASE_SEED, cell)
            if top is None or bot is None:
                continue
            rng = top - bot
            ok = (rng >= 0.15) and (jitter is not None and rng > jitter)
            if ok:
                # gate counts DISTINCT (len x pairs) task cells, not dim variants
                passing_cells.add(cell)
            rows.append({"dim": dim, "cell": cell, "acc_hi": top,
                         "acc_lo": bot, "range": rng, "passes": ok})
    saturated = all(r["acc_hi"] > 0.95 and r["acc_lo"] > 0.95 for r in rows) if rows else False
    summary = {"jitter": jitter, "cells_passing": len(passing_cells),
               "gate_a0": len(passing_cells) >= 2, "all_saturated": saturated,
               "rows": rows}
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    for r in sorted(rows, key=lambda r: -r["range"])[:12]:
        print(f"  dim={r['dim']} {r['cell']:12s} hi={r['acc_hi']:.3f} "
              f"lo={r['acc_lo']:.3f} range={r['range']:+.3f} "
              f"{'PASS' if r['passes'] else ''}")
    return summary


def build_calibration_plan(steps: int):
    """Anchor-dose-ONLY runs to pick `steps` such that the full-cache arm
    converges (diag acc >= 0.9) before launching the grid. Uses exclusively
    the top dose (8, 4): no low-dose arm is trained, so no dose contrast can
    be observed before the gate (prereg hygiene)."""
    plan = []
    for (l, n) in ((512, 32), (2048, 64)):
        for lr in LRS[2:]:  # the two highest LRs; low LRs never win on MQAR
            plan.append(RunConfig(128, 8, 4, l, n, lr, BASE_SEED, steps))
    return plan


def main(argv=None):
    p = argparse.ArgumentParser(description="Stage A-0 MQAR dose-response")
    p.add_argument("--out", default="data/a0_results.jsonl")
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--smoke", action="store_true",
                   help="tiny subset, short steps, SEPARATE output file")
    p.add_argument("--calibrate", action="store_true",
                   help="anchor-dose convergence check, SEPARATE output file")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--analyse", action="store_true")
    args = p.parse_args(argv)

    if args.analyse:
        analyse(args.out)
        return

    if args.calibrate:
        plan = build_calibration_plan(steps=args.steps)
        out = args.out if args.out != "data/a0_results.jsonl" else "data/a0_calibrate.jsonl"
        if args.dry_run:
            print(f"{len(plan)} calibration runs -> {out}")
            return
        run(plan, out, pick_device(args.device))
        return

    if args.smoke:
        plan = build_plan(steps=min(args.steps, 200), dims=(128,), lens=(512,),
                          pairs=(32, 64), lrs=LRS[1:3], with_jitter=False)
        out = args.out if args.out != "data/a0_results.jsonl" else "data/a0_smoke.jsonl"
    else:
        plan = build_plan(steps=args.steps)
        out = args.out
    if args.limit:
        plan = plan[:args.limit]
    if args.dry_run:
        print(f"{len(plan)} runs -> {out}")
        for rc in plan[:5]:
            print(" ", rc.key())
        return
    run(plan, out, pick_device(args.device))


if __name__ == "__main__":
    main()
