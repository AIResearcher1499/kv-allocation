"""Frozen arm definitions for Gate G0-A (see docs/prereg-g0a.md §3).

The relative KV bytes/token of an arm is (kv_layers * n_kv) / (L * n_h) for the
160M reference config (L=12, n_h=12, d_h=64). Byte-matched pairs across the head
route and the layer route are the fungibility test and must collide exactly.
"""

from dataclasses import dataclass

REF_LAYERS = 12
REF_HEADS = 12
REF_HEAD_DIM = 64


@dataclass(frozen=True)
class Arm:
    name: str
    n_kv: int          # KV heads per attention layer that owns a cache
    kv_layers: int     # number of layers that own a KV cache (others share)

    @property
    def rel_bytes(self) -> float:
        return (self.kv_layers * self.n_kv) / (REF_LAYERS * REF_HEADS)

    @property
    def kv_bytes_per_token_bf16(self) -> int:
        # K and V, 2 bytes each in bf16
        return 2 * 2 * self.kv_layers * self.n_kv * REF_HEAD_DIM


ARMS = (
    Arm("mha", 12, 12),
    Arm("gqa4", 3, 12),
    Arm("mqa", 1, 12),
    Arm("cla2", 12, 6),
    Arm("cla4", 12, 3),
    Arm("cla12", 12, 1),
    Arm("gqa4_cla2", 3, 6),
    Arm("gqa2_cla3", 6, 4),
)

# Byte-matched cross-route pairs (prereg §4): the load-bearing comparison.
FUNGIBILITY_PAIRS = (("gqa4", "cla4"), ("mqa", "cla12"))

SEEDS = (0, 1)


def arm(name: str) -> Arm:
    for a in ARMS:
        if a.name == name:
            return a
    raise KeyError(name)
