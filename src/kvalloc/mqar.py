"""MQAR task, faithful to the Zoology reference implementation.

Mirrors zoology/data/multiquery_ar.py (repo HazyResearch/zoology @1ad20d1):

    Key Val Key Val | Query .. Query ..     (length L after next-token shift)
    2   8   4   7   | 4  0  0  2  0  ...
    labels: -100 except AT each query-key position, where the label is the
    paired value (next-token convention after the [:-1]/[1:] shift).

- keys drawn WITHOUT replacement from [1, V/2); values WITHOUT replacement
  from [V/2, V) — both per example.
- KV pairs packed contiguously at the START of the sequence.
- Query gaps drawn from a power law p(g) ∝ (g+1)^(power_a - 1) over the tail
  region; power_a=0.01 (Zoology default) puts queries close after the context.
  power_a=1.0 is uniform — the distance-stress knob for later stages.
- Fillers stay token 0 (random_non_queries=False, the ICLR figure-2 setting).

Valid cells require 4*N <= L (context 2N + tail >= 2N).
"""

import torch

VOCAB = 8192
FILLER = 0
IGNORE = -100
KEY_LO, KEY_HI = 1, VOCAB // 2      # keys: 1..4095
VAL_LO, VAL_HI = VOCAB // 2, VOCAB  # values: 4096..8191
POWER_A = 0.01


def is_valid_cell(seq_len: int, num_pairs: int) -> bool:
    return 4 * num_pairs <= seq_len


def _distinct(batch: int, lo: int, hi: int, n: int, gen: torch.Generator):
    r = torch.rand(batch, hi - lo, generator=gen)
    return r.argsort(dim=1)[:, :n] + lo


def build_examples(num: int, seq_len: int, num_pairs: int,
                   generator: torch.Generator, power_a: float = POWER_A):
    """Returns (inputs, labels), both (num, seq_len) int64 on CPU."""
    if not is_valid_cell(seq_len, num_pairs):
        raise ValueError(f"invalid cell: 4*{num_pairs} > {seq_len}")

    n, ctx = num_pairs, 2 * num_pairs
    keys = _distinct(num, KEY_LO, KEY_HI, n, generator)
    values = _distinct(num, VAL_LO, VAL_HI, n, generator)

    ex = torch.full((num, seq_len + 1), FILLER, dtype=torch.long)
    ex[:, 0:ctx:2] = keys
    ex[:, 1:ctx:2] = values

    space = (seq_len - ctx) // 2
    g = torch.arange(1, space + 1, dtype=torch.float64)
    p = (power_a * g ** (power_a - 1))
    gaps = torch.multinomial((p / p.sum()).expand(num, -1), n,
                             replacement=False, generator=generator)

    labels = torch.full((num, seq_len + 1), IGNORE, dtype=torch.long)
    # query key at tail offset 2*gap; its value is the NEXT-token label
    q_pos = ctx + 2 * gaps          # in [ctx, seq_len-1] since gap < space
    ex.scatter_(1, q_pos, keys)
    labels.scatter_(1, q_pos + 1, values)

    return ex[:, :-1].contiguous(), labels[:, 1:].contiguous()


def query_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of query positions where argmax equals the target value."""
    mask = targets != IGNORE
    if mask.sum() == 0:
        return float("nan")
    pred = logits.argmax(dim=-1)
    return (pred[mask] == targets[mask]).float().mean().item()
