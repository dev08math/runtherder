# Runtherder

A CUDA inference engine for running Llama and Qwen models locally on a single consumer GPU.
Written in C++20 and CUDA with no inference framework underneath.

It loads a Hugging Face checkpoint, generates text, and is fast enough to compare against
llama.cpp.

## Features

- Runs Llama 3 and Qwen 3 checkpoints straight from safetensors, single file or sharded.
- Runs INT8 quantized checkpoints, weights and activations both.
- FP8 KV cache, half the size of BF16.
- Custom flash attention kernels for both prefill and decode.
- CUDA graphs on the decode path.
- Sampling runs on the GPU: greedy, top k, top p.
- Streaming output, and a server mode that keeps the model loaded.

## Performance


| | |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop, 8 GB |
| OS | Windows 11, WSL2, Ubuntu 22.04 |
| CUDA | 13.2 |
| Model | Llama-3.2-3B-Instruct, INT8 W8A8 with a BF16 output head |
| Compared against | llama.cpp, same model at Q8_0 |
| Generation | 128 tokens per run, greedy |
| Method | stock clocks, medians, Release build, warmed up |

![prefill](docs/img/prefill.png)

![decode](docs/img/decode.png)

Prefill, one row per prompt length:

| prompt tokens | prefill tok/s | llama.cpp | ratio | time to first token |
|---|---|---|---|---|
| 35 | 1984 | 1996 | 0.99x | 0.018 s |
| 273 | 8905 | 6592 | 1.35x | 0.031 s |
| 511 | 11274 | 7268 | 1.55x | 0.045 s |
| 1021 | 10742 | 7805 | 1.38x | 0.095 s |
| 2041 | 8345 | 7555 | 1.10x | 0.245 s |
| 4761 | 5863 | 6668 | 0.88x | 0.812 s |
| 8196 | 4438 | 5709 | 0.78x | 1.847 s |

Decode, at matched context depth:

| context depth | decode tok/s | llama.cpp | ratio | memory bandwidth |
|---|---|---|---|---|
| 35 | 61.67 | 68.25 | 0.90x | 87.1% |
| 511 | 62.14 | 66.68 | 0.93x | 88.4% |
| 2041 | 60.38 | 64.05 | 0.94x | 88.0% |
| 8196 | 54.04 | 54.66 | 0.99x | 86.5% |

Decode is bound by memory bandwidth and sustains a near constant fraction of the card's peak as
context grows.

Prefill leads on short and mid length prompts and trails on long ones, where attention cost
grows with the square of the prompt length.

### Memory

Peak usage at 8192 context is 5039 MiB of the 8188 available:

| | |
|---|---|
| Weights | 3441 MiB |
| KV cache at 8192 tokens | 462 MiB |
| CUDA context, activations, scratch | 1136 MiB |

The KV cache is FP8 rather than BF16, which halves its size. Scratch is carved once at load and
scales with the longest prompt accepted. Weights are used as published by the checkpoint author
and are not quantized here. A checkpoint with tied embeddings ships the output head and the
embedding table as separate copies of the same values, so only one is uploaded.

## Build

Needs CUDA 13, CMake 3.24, and a C++20 compiler.

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Builds for Ada (compute 89) by default. Override with `-DRUNTHERDER_CUDA_ARCH=<arch>`.

## Run

```
./build/runtherder ./models/Llama-3.2-3B-Instruct-quantized.w8a8 "What is a KV cache?"
```

```
--prompt-file FILE     read the prompt from a file
--max-tokens N         new tokens to generate
--temperature F        0 is greedy
--top-p F, --top-k N   sampling cutoffs
--stream               emit token ids, one per line
--serve                keep the model loaded, read prompts from stdin
--enforce-eager        disable CUDA graphs
```

Tests: `ctest --test-dir build`.

Benchmarks: `uv run scripts/bench_sweep.py --runs 15`.

## Scope

Single stream on one consumer GPU, for local use on a single machine. Further weight
quantization (INT4), KV cache paging (paged attention), and a scheduler with continuous
batching are planned.

## References

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [andrewkchan/yalm](https://github.com/andrewkchan/yalm)
- [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [vLLM](https://github.com/vllm-project/vllm)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Flash Attention from Scratch](https://lubits.ch/flash)
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
