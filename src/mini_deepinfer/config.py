"""Immutable, deliberately small CPU configurations."""
from dataclasses import dataclass
import math


def integer(name, value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class ModelConfig:
    dim: int = 64
    layers: int = 2
    heads: int = 4
    hidden_dim: int = 128
    max_seq_len: int = 256

    def __post_init__(self):
        for name, value in vars(self).items():
            integer(name, value)
        if self.dim % self.heads or (self.dim // self.heads) % 2:
            raise ValueError("dim must divide into heads with even head dimension")


@dataclass(frozen=True)
class EngineConfig:
    block_size: int = 16
    num_blocks: int = 64
    max_active: int = 4
    token_budget: int = 32

    def __post_init__(self):
        for name, value in vars(self).items():
            integer(name, value)


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 0.0
    seed: int = 0
    ignore_eos: bool = False

    def __post_init__(self):
        integer("max_tokens", self.max_tokens, 0)
        integer("seed", self.seed, 0)
        if self.seed >= 2**63:
            raise ValueError("seed must be < 2**63")
        if (isinstance(self.temperature, bool)
                or not isinstance(self.temperature, (int, float))
                or not math.isfinite(self.temperature) or self.temperature < 0):
            raise ValueError("temperature must be finite and nonnegative")
        if type(self.ignore_eos) is not bool:
            raise ValueError("ignore_eos must be bool")
