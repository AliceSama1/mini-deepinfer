"""Tiny decoder: RMSNorm, rotary causal attention and SwiGLU, CPU float32."""
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


def rotary(x, positions):
    half = x.shape[-1] // 2
    inv = 10000.0 ** (-torch.arange(half, dtype=x.dtype) / half)
    angles = positions[:, None, None] * inv[None, None, :]
    a, b = x.chunk(2, dim=-1)
    return torch.cat((a * angles.cos() - b * angles.sin(),
                      b * angles.cos() + a * angles.sin()), dim=-1)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.dim // config.heads
        self.attn_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)
        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.out = nn.Linear(config.dim, config.dim, bias=False)
        self.gate = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def forward(self, x, positions, layer, cache, request_id, start):
        q, k, v = self.qkv(self.attn_norm(x)).reshape(
            len(x), 3, self.heads, self.head_dim).unbind(1)
        q, k = rotary(q, positions), rotary(k, positions)
        if cache is not None:
            cache.write(layer, request_id, start, k, v)
            k, v = cache.read(layer, request_id, start + len(x))
        scores = q.transpose(0, 1) @ k.permute(1, 2, 0) / self.head_dim**0.5
        # Absolute positions are essential for chunked prefill and cached decode.
        mask = torch.arange(len(k))[None, :] > positions[:, None]
        scores = scores.masked_fill(mask[None], float("-inf"))
        attended = (scores.softmax(-1) @ v.transpose(0, 1)).transpose(0, 1)
        x = x + self.out(attended.reshape(len(x), -1))
        h = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class TinyDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(258, config.dim)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, 258, bias=False)

    def forward(self, token_ids, cache=None, request_id=None, start=0):
        if token_ids.ndim != 1 or not len(token_ids):
            raise ValueError("expected a nonempty 1D token tensor")
        if start < 0 or start + len(token_ids) > self.config.max_seq_len:
            raise ValueError("context length exceeded")
        if cache is None and start != 0:
            raise ValueError("nonzero start requires KV cache")
        positions = torch.arange(start, start + len(token_ids))
        x = self.embedding(token_ids)
        for layer, block in enumerate(self.layers):
            x = block(x, positions, layer, cache, request_id, start)
        return self.lm_head(self.norm(x))
