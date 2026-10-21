"""Also runnable with stdlib unittest when torch/pytest are unavailable."""
import unittest
from mini_deepinfer.cache import BlockAllocator
from mini_deepinfer.config import EngineConfig, ModelConfig, SamplingParams
from mini_deepinfer.scheduler import Request, Scheduler
from mini_deepinfer.tokenizer import ByteTokenizer


class CoreTests(unittest.TestCase):
    def test_validation(self):
        for kwargs in ({"token_budget": 0}, {"max_active": True}, {"block_size": -1}):
            with self.assertRaises(ValueError):
                EngineConfig(**kwargs)
        with self.assertRaises(ValueError):
            ModelConfig(dim=15, heads=3)
        for kwargs in ({"temperature": float("nan")}, {"max_tokens": -1}, {"seed": 2**63}):
            with self.assertRaises(ValueError):
                SamplingParams(**kwargs)

    def test_tokenizer(self):
        tokenizer = ByteTokenizer()
        self.assertEqual(tokenizer.decode(tokenizer.encode("你好 hello")), "你好 hello")
        self.assertEqual(tokenizer.encode(""), [256])

    def test_allocator(self):
        pool = BlockAllocator(3, 2)
        self.assertEqual(pool.reserve(1, 3), (0, 1))
        with self.assertRaises(MemoryError):
            pool.reserve(2, 4)
        self.assertEqual(len(pool.free), 1)
        with self.assertRaises(ValueError):
            pool.reserve(1, 1)
        pool.reserve(2, 1)
        pool.release(1)
        pool.release(1)
        self.assertEqual(pool.reserve(3, 4), (0, 1))
        pool.release(2)
        pool.release(3)
        self.assertEqual(set(pool.free), {0, 1, 2})

    def test_fifo_budget_lifecycle(self):
        pool = BlockAllocator(4, 2)
        config = EngineConfig(block_size=2, num_blocks=4, max_active=2, token_budget=2)
        requests = [Request(i, "", [256, 1, 2], SamplingParams(max_tokens=1)) for i in range(3)]
        scheduler = Scheduler(config, pool, requests)
        finished = []
        phases = []
        for _ in range(10):
            if not scheduler.pending:
                break
            work = scheduler.schedule()
            self.assertLessEqual(sum(n for _, _, n in work), 2)
            self.assertLessEqual(len(scheduler.active), 2)
            for request, phase, count in work:
                phases.append(phase)
                request.cached += count
                if request.cached == len(request.prompt_ids):
                    finished.append(request.request_id)
                    scheduler.finish(request, "length")
        self.assertFalse(scheduler.pending)
        self.assertEqual(finished, [0, 1, 2])
        self.assertEqual(set(phases), {"prefill"})
        self.assertEqual(len(pool.free), 4)

    def test_zero_decode_and_cleanup(self):
        pool = BlockAllocator(4, 2)
        zero = Request(0, "", [256], SamplingParams(max_tokens=0))
        run = Request(1, "", [256], SamplingParams(max_tokens=2))
        scheduler = Scheduler(EngineConfig(), pool, [zero, run])
        self.assertEqual(scheduler.schedule()[0][1], "prefill")
        self.assertEqual(zero.status, "finished")
        run.cached = 1
        self.assertEqual(scheduler.schedule()[0][1:], ("decode", 1))
        scheduler.close()
        self.assertEqual(run.status, "aborted")
        self.assertEqual(len(pool.free), 4)

    def test_impossible_request(self):
        pool = BlockAllocator(1, 2)
        request = Request(0, "", [256], SamplingParams(max_tokens=2))
        with self.assertRaises(ValueError):
            Scheduler(EngineConfig(), pool, [request])
        self.assertEqual(len(pool.free), 1)


if __name__ == "__main__":
    unittest.main()
