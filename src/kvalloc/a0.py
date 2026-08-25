"""Stage A-0 runner — synthetic MQAR dose-response (prereg-g0a.md §2, §4).

Training protocol v2 (2026-08-24): faithful to the Zoology reference
(HazyResearch/zoology @1ad20d1) after the fresh-data protocol failed to form
the retrieval circuit (diag acc <= 0.05 at 20k steps, data/a0_calibrate.jsonl):
FIXED dataset (100k train / 3k test), up to 64 epochs, per-EPOCH cosine with
no warmup, AdamW wd=0.1, early stop at val acc > 0.99, learned absolute
position embeddings, dropout 0.1, queries placed by power-law (power_a=0.01,
i.e. near the KV block). Frozen prereg items unchanged: 4 layers / 8 heads,
the 9 (n_kv x kv_layers) doses, the LR sweep, the task grid, the gate.

Operationalization recorded pre-data (clarifies, does not alter, thresholds):
a "stress cell" is a (seq_len, num_pairs) cell judged on its own diagonal
(train == eval); the byte-ladder range is acc(max bytes) - acc(min bytes),
each maxed over LRs; jitter = |max-over-LR acc| gap between the two seeds of
the repeated cell. With learned APE, cross-length eval is meaningless, so the
battery covers same-length cells only (denser pairs = the harder direction).

Result files are MERGE-only. Smoke and calibrate write SEPARATE files.
"""

import argparse
import copy
import json
import os
import time
from dataclasses import asdict, dataclass

import torch

from .model import KVModel, ModelConfig
from .mqar import IGNORE, build_examples, is_valid_cell, query_accuracy

# ---- frozen A-0 grids (prereg §2) -------------------------------------------
DIMS = (128, 256)
DOSES = tuple((n_kv, kv_layers) for n_kv in (8, 2, 1) for kv_layers in (4, 2, 1))
TRAIN_LENS = (512, 2048, 4096)
PAIRS = (32, 64, 128, 256)
LRS = (1e-4, 4.6415888336127824e-4, 2.1544346900318823e-3, 1e-2)
BASE_SEED = 20260824
# Jitter cell moved 2048/128 -> 512/64 (2026-08-25): calibration showed the
# L2048 anchor never converges at this budget, and jitter measured on a dead
# cell estimates nothing. Recorded in data/config_lock.json.
JITTER_CELL = dict(dim=128, n_kv=2, kv_layers=2, seq_len=512, num_pairs=64)
JITTER_SEED_OFFSET = 1000
TEST_SEED_OFFSET = 10_000_019  # train/test example streams must not overlap
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
    max_epochs: int = 64
    num_examples: int = 100_000
    test_examples: int = 3_000
    power_a: float = 0.01
    weight_decay: float = 0.1
    early_stop_acc: float = 0.99
    dropout: float = 0.1
    pos: str = "learned"

    def key(self) -> str:
        # Resume key covers EVERY field so no parameter change can silently
        # collide with an old record (repeat bug from sibling projects).
        return json.dumps(asdict(self), sort_keys=True)

    @property
    def batch_size(self) -> int:
        # Zoology's per-length batch sizes (figure2 configs).
        if self.seq_len <= 128:
            return 512
        if self.seq_len <= 256:
            return 256
        if self.seq_len <= 512:
            return 128
        return 64


def eval_cells():
    return [(l, n) for l in TRAIN_LENS for n in PAIRS if is_valid_cell(l, n)]


def build_plan(max_epochs: int = 64, dims=DIMS, lens=TRAIN_LENS, pairs=PAIRS,
               lrs=LRS, doses=DOSES, with_jitter: bool = True, **overrides):
    plan = []
    for dim in dims:
        for (n_kv, kv_layers) in doses:
            for seq_len in lens:
                for num_pairs in pairs:
                    if not is_valid_cell(seq_len, num_pairs):
                        continue
                    for lr in lrs:
                        plan.append(RunConfig(dim, n_kv, kv_layers, seq_len,
                                              num_pairs, lr, BASE_SEED,
                                              max_epochs, **overrides))
    if with_jitter:
        j = JITTER_CELL
        for lr in lrs:
            plan.append(RunConfig(j["dim"], j["n_kv"], j["kv_layers"],
                                  j["seq_len"], j["num_pairs"], lr,
                                  BASE_SEED + JITTER_SEED_OFFSET,
                                  max_epochs, **overrides))
    return plan


def build_calibration_plan(max_epochs: int = 64,
                           cells=((512, 32), (2048, 64)),
                           lrs=LRS[2:], **overrides):
    """Anchor-dose-ONLY runs to verify the instrument converges before the
    grid. Trains no low-dose arm, so no dose contrast is visible pre-gate."""
    plan = []
    for (l, n) in cells:
        for lr in lrs:
            plan.append(RunConfig(128, 8, 4, l, n, lr, BASE_SEED,
                                  max_epochs, **overrides))
    return plan


def pick_device(arg: str = "auto") -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def _dataset_accuracy(model, x, y, batch: int, device: str) -> float:
    model.eval()
    hits = tot = 0
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i + batch].to(device), y[i:i + batch].to(device)
        pred = model(xb).argmax(dim=-1)
        mask = yb != IGNORE
        hits += (pred[mask] == yb[mask]).sum().item()
        tot += mask.sum().item()
    model.train()
    return hits / max(1, tot)


def train_one(rc: RunConfig, device: str, log_every: int = 0):
    """Zoology fit loop: fixed dataset, per-epoch cosine, no warmup,
    early stop on val acc, best-epoch checkpoint restored at the end."""
    torch.manual_seed(rc.seed)
    gen = torch.Generator().manual_seed(rc.seed)
    xtr, ytr = build_examples(rc.num_examples, rc.seq_len, rc.num_pairs,
                              gen, rc.power_a)
    gen_te = torch.Generator().manual_seed(rc.seed + TEST_SEED_OFFSET)
    xte, yte = build_examples(rc.test_examples, rc.seq_len, rc.num_pairs,
                              gen_te, rc.power_a)

    cfg = ModelConfig(dim=rc.dim, n_kv_heads=rc.n_kv, kv_layers=rc.kv_layers,
                      pos=rc.pos, dropout=rc.dropout,
                      max_seq_len=max(4096, rc.seq_len))
    model = KVModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=rc.lr,
                            weight_decay=rc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=rc.max_epochs, eta_min=0.0)

    hist = {"best_acc": 0.0, "best_epoch": -1, "epochs_run": 0,
            "final_train_loss": None}
    best_state = None
    t0 = time.time()
    model.train()
    for epoch in range(rc.max_epochs):
        perm = torch.randperm(rc.num_examples, generator=gen)
        ep_loss, nb = 0.0, 0
        for i in range(0, rc.num_examples, rc.batch_size):
            idx = perm[i:i + rc.batch_size]
            xb, yb = xtr[idx].to(device), ytr[idx].to(device)
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), yb.view(-1),
                ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        sched.step()
        acc = _dataset_accuracy(model, xte, yte, rc.batch_size, device)
        hist["epochs_run"] = epoch + 1
        hist["final_train_loss"] = ep_loss / max(1, nb)
        if acc > hist["best_acc"]:
            hist["best_acc"], hist["best_epoch"] = acc, epoch
            best_state = copy.deepcopy(model.state_dict())
        if log_every and (epoch % log_every == 0 or epoch == rc.max_epochs - 1):
            print(f"    epoch {epoch:3d}/{rc.max_epochs} "
                  f"train_loss {hist['final_train_loss']:6.3f} "
                  f"val_acc {acc:.3f} best {hist['best_acc']:.3f} "
                  f"{time.time() - t0:6.0f}s", flush=True)
        if acc > rc.early_stop_acc:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def eval_grid(model, rc: RunConfig, device: str, eval_seqs: int = 256):
    """Same-length cells only (learned APE does not extrapolate); denser
    pair counts at the same length are the harder direction."""
    model.eval()
    out = {}
    for (l, n) in eval_cells():
        if l != rc.seq_len:
            continue
        g = torch.Generator().manual_seed(BASE_SEED + 7 * l + n)
        x, y = build_examples(eval_seqs, l, n, g, rc.power_a)
        accs, tot = [], 0
        for i in range(0, eval_seqs, rc.batch_size):
            xb, yb = x[i:i + rc.batch_size].to(device), y[i:i + rc.batch_size].to(device)
            accs.append((query_accuracy(model(xb), yb), xb.size(0)))
            tot += xb.size(0)
        out[f"L{l}_N{n}"] = sum(a * w for a, w in accs) / tot
    return out


def load_done_keys(path: str):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.dumps(json.loads(line)["config"],
                                        sort_keys=True))
    return done


def run(plan, out_path: str, device: str, log_every: int = 0):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = load_done_keys(out_path)
    todo = [rc for rc in plan if rc.key() not in done]
    print(f"plan={len(plan)} done={len(done)} todo={len(todo)} device={device}",
          flush=True)
    for i, rc in enumerate(todo):
        if log_every:
            print(f"[{i + 1}/{len(todo)}] START {rc.dim}d nkv={rc.n_kv} "
                  f"kvl={rc.kv_layers} L{rc.seq_len} N{rc.num_pairs} "
                  f"lr={rc.lr:.1e} epochs<={rc.max_epochs}", flush=True)
        t0 = time.time()
        model, hist = train_one(rc, device, log_every=log_every)
        evals = eval_grid(model, rc, device)
        rec = {"config": asdict(rc), **hist, "evals": evals,
               "wall_s": round(time.time() - t0, 1), "device": device,
               "torch": torch.__version__}
        with open(out_path, "a", encoding="utf-8") as f:  # append = merge
            f.write(json.dumps(rec) + "\n")
        print(f"[{i + 1}/{len(todo)}] {rc.dim}d nkv={rc.n_kv} "
              f"kvl={rc.kv_layers} L{rc.seq_len} N{rc.num_pairs} "
              f"lr={rc.lr:.1e} seed={rc.seed} "
              f"best_acc={hist['best_acc']:.3f}@ep{hist['best_epoch']} "
              f"({hist['epochs_run']} ep, {rec['wall_s']}s)", flush=True)


# ---- analysis / gate --------------------------------------------------------

def _max_over_lr(records, dim, n_kv, kv_layers, seq_len, num_pairs, seed, cell):
    vals = [r["evals"][cell] for r in records
            if r["config"]["dim"] == dim and r["config"]["n_kv"] == n_kv
            and r["config"]["kv_layers"] == kv_layers
            and r["config"]["seq_len"] == seq_len
            and r["config"]["num_pairs"] == num_pairs
            and r["config"]["seed"] == seed
            and cell in r["evals"]]
    return max(vals) if vals else None


def analyse(path: str):
    with open(path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    # v1 (fresh-data/RoPE) records lack the `pos` field — never let them mix
    # into the gate (data/ may hold both formats side by side).
    records = [r for r in records if r["config"].get("pos") == "learned"]
    hi = max(DOSES, key=lambda d: d[0] * d[1])
    lo = min(DOSES, key=lambda d: d[0] * d[1])

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


def main(argv=None):
    p = argparse.ArgumentParser(description="Stage A-0 MQAR dose-response")
    p.add_argument("--out", default="data/a0_results.jsonl")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=64)
    p.add_argument("--num-examples", type=int, default=100_000)
    p.add_argument("--lens", default="512,2048,4096",
                   help="comma list of train seq lens to include")
    p.add_argument("--dims", default="128,256",
                   help="comma list of model dims to include")
    p.add_argument("--smoke", action="store_true",
                   help="tiny subset, SEPARATE output file")
    p.add_argument("--calibrate", action="store_true",
                   help="anchor-dose convergence check, SEPARATE output file")
    p.add_argument("--cal-cells", default="",
                   help="calibrate only: 'L:N,L:N' cells at the top LR")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard", default="",
                   help="'i/n': run every n-th job starting at i (2-GPU split)")
    p.add_argument("--analyse", action="store_true")
    p.add_argument("--log-every", type=int, default=1,
                   help="progress line every N epochs (0 = silent)")
    args = p.parse_args(argv)

    if args.analyse:
        analyse(args.out)
        return

    if args.calibrate:
        kw = {}
        if args.cal_cells:
            kw["cells"] = tuple(tuple(int(v) for v in c.split(":"))
                                for c in args.cal_cells.split(","))
            kw["lrs"] = (LRS[3],)  # fastest-converging LR (1e-2, ep19 on N32)
        plan = build_calibration_plan(max_epochs=args.epochs,
                                      num_examples=args.num_examples, **kw)
        out = args.out if args.out != "data/a0_results.jsonl" else "data/a0_calibrate.jsonl"
    elif args.smoke:
        plan = build_plan(max_epochs=2, dims=(128,), lens=(512,),
                          pairs=(32, 64), lrs=LRS[1:3], with_jitter=False,
                          num_examples=2000, test_examples=500)
        out = args.out if args.out != "data/a0_results.jsonl" else "data/a0_smoke.jsonl"
    else:
        lens = tuple(int(x) for x in args.lens.split(","))
        dims = tuple(int(x) for x in args.dims.split(","))
        plan = build_plan(max_epochs=args.epochs, dims=dims, lens=lens,
                          num_examples=args.num_examples)
        out = args.out
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        plan = plan[i::n]
    if args.limit:
        plan = plan[:args.limit]
    if args.dry_run:
        print(f"{len(plan)} runs -> {out}")
        for rc in plan[:5]:
            print(" ", rc.key())
        return
    run(plan, out, pick_device(args.device), log_every=args.log_every)


if __name__ == "__main__":
    main()
