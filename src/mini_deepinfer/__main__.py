"""Run an entirely offline, randomly initialized demonstration."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts", nargs="*", default=["Hello", "你好"])
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        from .engine import MiniDeepInfer
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        parser.exit(2, "PyTorch is required. Install CPU torch in your Python environment first.\n")
    from .config import SamplingParams
    engine = MiniDeepInfer(seed=args.seed)
    print("Random weights, offline byte tokenizer: NO meaningful text guarantee.")
    params = SamplingParams(args.max_tokens, args.temperature, args.seed)
    for prompt, result in zip(args.prompts, engine.generate(args.prompts, params)):
        print(f"prompt={ascii(prompt)} output={ascii(result.text)}")
        print(f"tokens={result.token_ids} stop={result.finish_reason}")
    print(f"Free KV blocks: {len(engine.cache.allocator.free)}")


if __name__ == "__main__":
    main()
