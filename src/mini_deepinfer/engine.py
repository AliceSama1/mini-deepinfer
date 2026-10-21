"""A synchronous, offline CPU engine with explicit prefill/decode phases."""
from dataclasses import dataclass
import torch

from .cache import KVCache
from .config import EngineConfig, ModelConfig, SamplingParams
from .model import TinyDecoder
from .scheduler import Request, Scheduler
from .tokenizer import ByteTokenizer


@dataclass(frozen=True)
class Generation:
    text: str
    token_ids: list[int]
    finish_reason: str


class MiniDeepInfer:
    def __init__(self, model_config=None, engine_config=None, seed=0, model=None):
        self.model_config = model_config or ModelConfig()
        self.engine_config = engine_config or EngineConfig()
        if self.model_config.max_seq_len > self.engine_config.block_size * self.engine_config.num_blocks:
            raise ValueError("KV pool capacity is smaller than model max_seq_len")
        SamplingParams(seed=seed)  # Validate without changing the caller's RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = model if model is not None else TinyDecoder(self.model_config)
        self.model.eval()
        self.cache = KVCache(self.model_config, self.engine_config)
        self.tokenizer = ByteTokenizer()
        self._next_id = 0

    def _sample(self, logits, params, step):
        if params.temperature == 0:
            return int(logits.argmax().item())
        generator = torch.Generator().manual_seed((params.seed + step) % 2**63)
        # Subtract first so tiny positive temperatures cannot overflow to +inf.
        scaled = (logits.double() - logits.max().double()) / params.temperature
        return int(torch.multinomial(scaled.softmax(-1), 1, generator=generator).item())

    @torch.inference_mode()
    def generate(self, prompts, sampling_params=None):
        params = sampling_params or SamplingParams()
        if not isinstance(prompts, (list, tuple)):
            raise TypeError("prompts must be a list or tuple of strings")
        requests = []
        for prompt in prompts:
            ids = self.tokenizer.encode(prompt)
            if len(ids) + params.max_tokens > self.model_config.max_seq_len:
                raise ValueError("prompt plus max_tokens exceeds max_seq_len")
            request = Request(self._next_id, prompt, ids, params)
            self._next_id += 1
            requests.append(request)
        scheduler = Scheduler(self.engine_config, self.cache.allocator, requests)
        try:
            while scheduler.pending:
                for request, phase, count in scheduler.schedule():
                    if phase == "prefill":
                        end = request.cached + count
                        logits = self.model(torch.tensor(request.prompt_ids[request.cached:end]),
                                            self.cache, request.request_id, request.cached)
                        request.cached = end
                        # Sampling only from the final prompt position, after all prefill chunks.
                        if request.cached == len(request.prompt_ids):
                            token = self._sample(logits[-1], request.params, 0)
                            request.generated.append(token)
                    else:
                        token_ids = request.prompt_ids + request.generated
                        logits = self.model(torch.tensor([token_ids[-1]]), self.cache,
                                            request.request_id, len(token_ids) - 1)
                        token = self._sample(logits[-1], request.params, len(request.generated))
                        request.generated.append(token)
                    if request.generated and (
                        len(request.generated) >= request.params.max_tokens or
                        (not request.params.ignore_eos and request.generated[-1] == self.tokenizer.eos_id)
                    ):
                        is_eos = not request.params.ignore_eos and request.generated[-1] == self.tokenizer.eos_id
                        scheduler.finish(request, "eos" if is_eos else "length")
            return [Generation(self.tokenizer.decode(r.generated), r.generated, r.finish_reason) for r in requests]
        finally:
            # Ensures errors in model execution never leak physical blocks.
            scheduler.close()
