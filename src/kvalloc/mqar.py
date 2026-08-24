"""MQAR task generator (Zoology arXiv:2312.04927 Def 3.1 style).

Sequence layout (length L, N key-value pairs, N queries):

    [k_1 v_1 ... k_N v_N | filler ... | q_(s1) a_(s1) ... q_(sN) a_(sN)]

Keys are distinct within a sequence; queries are the same keys in a random
order; the training target at each query position is the paired value (the
value token also appears at the next input position, i.e. teacher forcing —
causality prevents leakage into its own prediction). Loss/accuracy are
computed on query positions only. The filler gap pushes key→query distance
toward L, which is the capacity/range stress the gate needs.

Valid cells require 4*N <= L (pairs + queries leave a non-negative gap).
"""

import torch

VOCAB = 8192
FILLER = 0
IGNORE = -100
N_KEYS = 4095          # key ids: 1..4095
VALUE_LO = N_KEYS + 1  # value ids: 4096..8191


def is_valid_cell(seq_len: int, num_pairs: int) -> bool:
    return 4 * num_pairs <= seq_len


def build_batch(batch: int, seq_len: int, num_pairs: int,
                generator: torch.Generator, device: str = "cpu"):
    """Returns (inputs, targets), both (batch, seq_len) int64 on `device`.

    targets == IGNORE everywhere except query positions.
    """
    if not is_valid_cell(seq_len, num_pairs):
        raise ValueError(f"invalid cell: 4*{num_pairs} > {seq_len}")

    # Distinct keys per row via argsort-of-uniform.
    r = torch.rand(batch, N_KEYS, generator=generator)
    keys = r.argsort(dim=1)[:, :num_pairs] + 1
    values = torch.randint(VALUE_LO, VOCAB, (batch, num_pairs), generator=generator)

    perm = torch.rand(batch, num_pairs, generator=generator).argsort(dim=1)
    q_keys = torch.gather(keys, 1, perm)
    q_vals = torch.gather(values, 1, perm)

    inputs = torch.full((batch, seq_len), FILLER, dtype=torch.long)
    targets = torch.full((batch, seq_len), IGNORE, dtype=torch.long)

    inputs[:, 0:2 * num_pairs:2] = keys
    inputs[:, 1:2 * num_pairs:2] = values

    qs = seq_len - 2 * num_pairs
    inputs[:, qs::2] = q_keys
    inputs[:, qs + 1::2] = q_vals
    targets[:, qs::2] = q_vals

    return inputs.to(device), targets.to(device)


def query_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of query positions where argmax equals the target value."""
    mask = targets != IGNORE
    if mask.sum() == 0:
        return float("nan")
    pred = logits.argmax(dim=-1)
    return (pred[mask] == targets[mask]).float().mean().item()
