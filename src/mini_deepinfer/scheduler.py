"""Dependency-free request state and FIFO scheduling."""
from collections import deque
from dataclasses import dataclass, field
from .config import SamplingParams


@dataclass
class Request:
    request_id: int
    prompt: str
    prompt_ids: list[int]
    params: SamplingParams
    generated: list[int] = field(default_factory=list)
    cached: int = 0
    status: str = "waiting"
    finish_reason: str = ""

    @property
    def reservation(self):
        return len(self.prompt_ids) + self.params.max_tokens


class Scheduler:
    def __init__(self, config, allocator, requests):
        self.config = config
        self.allocator = allocator
        self.waiting = deque(requests)
        self.active = []
        for request in requests:
            if request.params.max_tokens and allocator.required(request.reservation) > allocator.num_blocks:
                raise ValueError("request cannot fit in the KV pool, even alone")

    @property
    def pending(self):
        return bool(self.waiting or self.active)

    def finish(self, request, reason):
        request.status, request.finish_reason = "finished", reason
        self.allocator.release(request.request_id)
        self.active.remove(request)

    def schedule(self):
        while self.waiting and len(self.active) < self.config.max_active:
            request = self.waiting[0]
            if request.params.max_tokens == 0:
                self.waiting.popleft()
                request.status, request.finish_reason = "finished", "length"
                continue
            if self.allocator.required(request.reservation) > len(self.allocator.free):
                break  # Strict FIFO admission; later requests cannot jump the queue.
            self.allocator.reserve(request.request_id, request.reservation)
            self.waiting.popleft()
            request.status = "running"
            self.active.append(request)
        budget = self.config.token_budget
        work = []
        for request in self.active:
            if budget == 0:
                break
            remaining = len(request.prompt_ids) - request.cached
            phase = "prefill" if remaining > 0 else "decode"
            count = min(remaining, budget) if remaining > 0 else 1
            work.append((request, phase, count))
            budget -= count
        if not work and self.pending:
            raise RuntimeError("scheduler stalled: no runnable request")
        return work

    def close(self):
        for request in self.active:
            self.allocator.release(request.request_id)
            request.status = "aborted"
        for request in self.waiting:
            request.status = "aborted"
        self.active.clear()
        self.waiting.clear()
