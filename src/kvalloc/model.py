"""Small pre-LN transformer with the two KV dose axes of prereg §2.

Head axis: n_kv_heads < n_heads (GQA; KV heads repeated to match Q heads).
Layer axis: kv_layers < n_layers (CLA-style — a layer either owns a KV pair
computed from its own normalized input, or reuses the K/V tensors of the first
layer in its share group; non-owner layers have NO Wk/Wv parameters, matching
the CLA parameter convention). RoPE is applied at KV-computation time, so a
reusing layer consumes position-encoded K from its owner (CLA behaviour).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    dim: int
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 8
    kv_layers: int = 4
    vocab: int = 8192

    def __post_init__(self):
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_kv_heads must divide n_heads")
        if self.n_layers % self.kv_layers:
            raise ValueError("kv_layers must divide n_layers")
        if self.dim % self.n_heads:
            raise ValueError("n_heads must divide dim")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_owner(self):
        g = self.n_layers // self.kv_layers
        return tuple((i // g) * g for i in range(self.n_layers))

    @property
    def rel_bytes(self) -> float:
        """KV bytes/token relative to the MHA/no-sharing reference."""
        return (self.kv_layers * self.n_kv_heads) / (self.n_layers * self.n_heads)


def _rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    # x: (B, H, T, D) with D even
    b, h, t, d = x.shape
    half = d // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * freq[None, :]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.owns_kv = cfg.kv_owner[layer_idx] == layer_idx
        hd, nh, nkv = cfg.head_dim, cfg.n_heads, cfg.n_kv_heads
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.wq = nn.Linear(cfg.dim, nh * hd, bias=False)
        if self.owns_kv:
            self.wk = nn.Linear(cfg.dim, nkv * hd, bias=False)
            self.wv = nn.Linear(cfg.dim, nkv * hd, bias=False)
        self.wo = nn.Linear(nh * hd, cfg.dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, 4 * cfg.dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg.dim, cfg.dim, bias=False),
        )

    def _heads(self, x, n):
        b, t, _ = x.shape
        return x.view(b, t, n, self.cfg.head_dim).transpose(1, 2)

    def forward(self, x, kv_store):
        cfg = self.cfg
        h = self.ln1(x)
        q = _rope(self._heads(self.wq(h), cfg.n_heads))
        if self.owns_kv:
            k = _rope(self._heads(self.wk(h), cfg.n_kv_heads))
            v = self._heads(self.wv(h), cfg.n_kv_heads)
            kv_store[self.layer_idx] = (k, v)
        else:
            k, v = kv_store[cfg.kv_owner[self.layer_idx]]
        rep = cfg.n_heads // cfg.n_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.transpose(1, 2).reshape(x.shape)
        x = x + self.wo(att)
        return x + self.mlp(self.ln2(x))


class KVModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab, bias=False)
        self.head.weight = self.embed.weight  # tied
        self._debug_kv = None
        self.apply(self._init_weights)
        # GPT-2-style scaled init on residual-output projections
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for blk in self.blocks:
            nn.init.normal_(blk.wo.weight, mean=0.0, std=resid_std)
            nn.init.normal_(blk.mlp[2].weight, mean=0.0, std=resid_std)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, debug_kv: bool = False):
        x = self.embed(idx)
        kv_store = {}
        for blk in self.blocks:
            x = blk(x, kv_store)
        if debug_kv:
            self._debug_kv = kv_store
        return self.head(self.ln_f(x))

    def n_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()  # head is tied, counted once
        return n
