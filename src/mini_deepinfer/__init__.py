"""Public API; configuration and scheduler stay usable without importing torch."""
from .config import EngineConfig, ModelConfig, SamplingParams
from .tokenizer import ByteTokenizer

__all__ = ["MiniDeepInfer", "Generation", "EngineConfig", "ModelConfig", "SamplingParams", "ByteTokenizer"]


def __getattr__(name):
    if name in ("MiniDeepInfer", "Generation"):
        from .engine import MiniDeepInfer, Generation
        return {"MiniDeepInfer": MiniDeepInfer, "Generation": Generation}[name]
    raise AttributeError(name)
