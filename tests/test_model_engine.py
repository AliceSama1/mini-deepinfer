"""CPU numerical and integration checks; skipped explicitly without torch."""
import unittest
try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from mini_deepinfer import MiniDeepInfer, ModelConfig, EngineConfig, SamplingParams
    from mini_deepinfer.cache import KVCache
    from mini_deepinfer.model import TinyDecoder


@unittest.skipIf(torch is None, "CPU torch is not installed")
class ModelEngineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.mc = ModelConfig(dim=16, heads=2, hidden_dim=24, max_seq_len=32)
        self.ec = EngineConfig(block_size=2, num_blocks=16, token_budget=2, max_active=2)

    def test_cached_logits_match_full_and_causality(self):
        torch.manual_seed(7)
        model = TinyDecoder(self.mc).eval()
        cache = KVCache(self.mc, self.ec)
        # Force a non-contiguous block table by releasing an earlier reservation.
        cache.allocator.reserve(99, 4)
        cache.allocator.reserve(98, 4)
        cache.allocator.release(99)
        cache.allocator.reserve(0, 28)
        ids = torch.tensor([256, 10, 20, 30, 40, 50, 60])
        with torch.inference_mode():
            full = model(ids)
            pieces = [model(ids[:3], cache, 0, 0), model(ids[3:5], cache, 0, 3)]
            pieces.extend(model(ids[i:i+1], cache, 0, i) for i in range(5, 7))
            torch.testing.assert_close(torch.cat(pieces), full, atol=2e-6, rtol=2e-5)
            altered = ids.clone()
            altered[-1] = 100
            torch.testing.assert_close(model(altered)[:-1], full[:-1])
        cache.allocator.release(0)
        cache.allocator.release(98)
        self.assertEqual(len(cache.allocator.free), 16)

    def test_order_determinism_and_reuse(self):
        engine = MiniDeepInfer(self.mc, self.ec, seed=2)
        prompts = ["abcde", "", "xy"]
        for temperature in (0.0, 0.7):
            params = SamplingParams(4, temperature, 11, True)
            together = engine.generate(prompts, params)
            alone = [engine.generate([p], params)[0] for p in prompts]
            self.assertEqual(together, alone)
            self.assertEqual(together, engine.generate(prompts, params))
            self.assertTrue(all(len(x.token_ids) == 4 for x in together))
            self.assertEqual(len(engine.cache.allocator.free), 16)

    def test_eos_zero_and_error_recovery(self):
        engine = MiniDeepInfer(self.mc, self.ec)
        with torch.no_grad():
            engine.model.lm_head.weight.zero_()
        original = engine._sample
        engine._sample = lambda *args: 257
        stopped = engine.generate(["a", ""], SamplingParams(4))
        self.assertTrue(all(x.token_ids == [257] and x.finish_reason == "eos" for x in stopped))
        ignored = engine.generate([""], SamplingParams(3, ignore_eos=True))[0]
        self.assertEqual(ignored.token_ids, [257] * 3)
        self.assertEqual(ignored.finish_reason, "length")
        self.assertEqual(engine.generate([""], SamplingParams(0))[0].token_ids, [])
        self.assertEqual(engine.generate([]), [])
        def fail(*args):
            raise RuntimeError("injected sampling error")
        engine._sample = fail
        with self.assertRaisesRegex(RuntimeError, "injected"):
            engine.generate(["a", "b"])
        self.assertEqual(len(engine.cache.allocator.free), 16)
        engine._sample = original
        self.assertEqual(len(engine.generate([""], SamplingParams(2))[0].token_ids), 2)
        with self.assertRaises(ValueError):
            engine.generate(["x" * 32])


if __name__ == "__main__":
    unittest.main()
