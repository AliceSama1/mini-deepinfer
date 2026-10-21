"""FIFO physical blocks; reserve worst-case length to avoid decode deadlocks."""
from collections import deque
from .config import integer


class BlockAllocator:
    def __init__(self, num_blocks, block_size):
        integer("num_blocks", num_blocks)
        integer("block_size", block_size)
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free = deque(range(num_blocks))
        self.tables = {}

    def required(self, tokens):
        integer("tokens", tokens)
        return (tokens + self.block_size - 1) // self.block_size

    def reserve(self, request_id, tokens):
        if request_id in self.tables:
            raise ValueError("request already owns blocks")
        count = self.required(tokens)
        if count > len(self.free):
            raise MemoryError("insufficient KV blocks")
        table = tuple(self.free.popleft() for _ in range(count))
        self.tables[request_id] = table
        return table

    def release(self, request_id):
        # Idempotent cleanup also works after partially failed engine steps.
        self.free.extend(self.tables.pop(request_id, ()))


class KVCache:
    def __init__(self, model_config, engine_config):
        import torch
        self.allocator = BlockAllocator(engine_config.num_blocks, engine_config.block_size)
        c, e = model_config, engine_config
        # Actual physical storage, not merely accounting for per-request tensors.
        self.data = torch.zeros(2, c.layers, e.num_blocks, e.block_size,
                                c.heads, c.dim // c.heads, dtype=torch.float32)

    def slots(self, request_id, start, end):
        import torch
        table = self.allocator.tables[request_id]
        size = self.allocator.block_size
        if not 0 <= start <= end <= len(table) * size:
            raise ValueError("KV access outside reservation")
        positions = torch.arange(start, end)
        blocks = torch.tensor(table, dtype=torch.long)[positions // size]
        return blocks, positions % size

    def write(self, layer, request_id, start, key, value):
        blocks, offsets = self.slots(request_id, start, start + len(key))
        self.data[0, layer, blocks, offsets] = key
        self.data[1, layer, blocks, offsets] = value

    def read(self, layer, request_id, length):
        blocks, offsets = self.slots(request_id, 0, length)
        return (self.data[0, layer, blocks, offsets],
                self.data[1, layer, blocks, offsets])
