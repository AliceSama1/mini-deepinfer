# mini-deepinfer

一个独立、易读的 CPU 推理教学实现。参考 `../MinivLLM-main/src/myvllm/engine/` 的请求、调度器、块管理器和模型执行分工，不依赖或修改原工程。模型使用随机权重，演示推理机制，**不保证生成有语义的文本**，也不兼容 DeepSeek 权重。

## 运行

需要 Python >= 3.10、PyTorch >= 2.2。建议使用已有 CPU PyTorch 环境，无需 CUDA、Triton、transformers、联网下载模型。首次安装依赖需要网络；安装后演示完全离线。

在本目录的 Git Bash 中：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
PYTHONPATH=src python -m mini_deepinfer 'Hello' '你好' --max-tokens 12 --seed 7
PYTHONPATH=src python -m pytest -q
```

PowerShell 设置路径的方法：

```powershell
$env:PYTHONPATH = 'src'
python -m mini_deepinfer 'Hello' --temperature 0.8 --seed 7
python -m pytest -q
```

`python -m mini_deepinfer` 本身就是离线示例。输出使用转义字符串，避免随机字节中的控制字符干扰终端。UTF-8 无效字节显示为替换字符，实际结果以 `token_ids` 为准。

```python
from mini_deepinfer import MiniDeepInfer, SamplingParams

engine = MiniDeepInfer(seed=7)
results = engine.generate(["Hello", "你好", ""],
                          SamplingParams(max_tokens=8, temperature=0.7, seed=42))
for result in results:
    print(result.token_ids, result.finish_reason, repr(result.text))
```

返回列表保持输入顺序；文本只包含新生成部分。空输入列表返回空列表；空字符串使用 BOS。`max_tokens=0` 不执行模型。EOS 计入生成 token 数，但不显示在文本中；`ignore_eos=True` 时继续生成至长度限制。默认温度为 0（贪心），正温度使用带种子的采样；同一请求在相同运行环境下不受其他请求调度顺序影响。

## 实现与边界

- `config.py`：不可变配置，校验正整数、头维度、有限非负温度等。
- `tokenizer.py`：UTF-8 字节 0–255，BOS=256、EOS=257；提示词总长度包含 BOS，中文通常占多个字节。
- `model.py`：两层小型 decoder-only Transformer，RMSNorm、RoPE、因果注意力、SwiGLU。全序列前向和缓存前向共用模型。
- `cache.py`：真正的固定容量 KV 张量 `[2, layers, blocks, block_size, heads, head_dim]`，通过每个请求的 block table 映射物理槽位。注意力读取时 gather 连续逻辑上下文，属于普通 torch 实现，不是优化的 paged-attention 内核。
- `scheduler.py`：FIFO 准入、活跃请求上限、每轮输入 token 预算。prefill 可以分块，decode 每次输入一个 token；同轮可以包含不同请求的两种阶段，逐请求执行而非向量化 batch。
- `engine.py`：组织生成循环、采样、停止、按原输入顺序收集结果；异常通过 `finally` 释放全部活跃请求块。

为了保持实现简单，请求准入时预留 `prompt_tokens + max_tokens` 的全部块。这会浪费最后一个未写入 token 的空间和部分尾块，但保证所有活跃请求无需追加分配即可完成，避免资源争抢导致死循环。FIFO 队首暂时放不下时等待已有请求释放；不抢占、不换出、不做前缀共享。每个请求的提示词加输出上限必须不超过 `max_seq_len`；KV 池必须至少容纳这个配置长度。超限立即报错，不静默截断。

默认 KV 存储约 1 MiB（float32）；注意力临时矩阵、权重与 Python 请求列表不属于这个池。CPU 单线程调用的同步 API，不支持并发调用同一引擎、GPU、训练、模型权重加载、分布式、流式服务或生产性能保证。

## 测试

测试覆盖缓存/完整 logits 一致性与因果性、非连续 block table、块分配释放、FIFO/预算/活跃限制、零输出、EOS、确定性、输出顺序及异常恢复。

测试也兼容标准库 unittest：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

没有 torch 时，配置/调度/分配器测试仍可运行，模型相关测试明确跳过；这不代表模型数值验证已通过。缺少 pytest 时可使用上述 unittest 命令。安装 CPU torch 后应重新运行完整测试和演示。
