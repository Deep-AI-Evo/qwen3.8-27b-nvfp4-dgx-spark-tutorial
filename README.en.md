# Qwen3.8-27B-NVFP4 on NVIDIA DGX Spark (GB10) — Deployment & Benchmarks

**[中文 README](README.zh.md)**

This repo documents the deployment and benchmarking of
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
(dense 27B vision-language model, NVFP4 quantization, native 262,144-token context)
on an **NVIDIA DGX Spark** (GB10, Grace-Blackwell, aarch64, 128GB unified memory).

## Documentation

- [Deployment Tutorial](docs/en/deployment-tutorial.md) — pure-pip setup (no root, no Docker),
  torch 2.13.0+cu130 + vLLM 0.27.1, incl. mirror download and every pitfall we hit
- [Test Report](docs/en/test-report.md) — functional tests, concurrency & long-context
  benchmarks, MTP speculative-decoding optimization, daily-use recommendations

## Benchmark Highlights

> vLLM 0.27.1 · torch 2.13.0+cu130 · 256k context · MTP ×3 speculative decoding on

**Single-stream decode**

| Scenario | Speed |
|---|---|
| Thinking mode | **22.7 tok/s** (11.8 without MTP, +92%) |
| Non-thinking | **16.7 tok/s** |
| At 200k context | **≈14.2 tok/s** |

**Concurrency 1–16**

| Concurrency | Per-stream decode | Aggregate throughput |
|---|---|---|
| 1 | 21.6 tok/s | 20.0 tok/s |
| 2 | 16.0 tok/s | 22.7 tok/s |
| 4 | 15.0 tok/s | 44.0 tok/s |
| 8 | ≈12.8 tok/s | 77.7 tok/s |
| 16 | ≈8.4 tok/s | 115.7 tok/s (peak 160) |

**Context-length scaling (single stream)**

| Context | Prefill | TTFT | Decode |
|---|---|---|---|
| 1k | ≈1800 tok/s | 0.57 s | ≈21 tok/s |
| 100k | ≈1230 tok/s (-32%) | 83 s | ≈16.6 tok/s |
| 200k | ≈840 tok/s (-53%) | 244 s | ≈14.2 tok/s |

Takeaway: decode degrades gently (48 of 64 layers are Gated DeltaNet linear attention, whose
decode cost is independent of context length); prefill slows super-linearly. KV cache holds
2.09M tokens → ~8 concurrent requests at the full 256k context.

## Daily-use Recommendations

- Interactive use: ≤4 concurrent streams (≥15 tok/s each); 8 is the interactive ceiling;
  use 16+ for offline batch to maximize aggregate throughput
- ≤100k context: use freely (~1.4 min ingest); ~200k: "ingest once, ask many turns"
- Prefix caching is not yet supported for this hybrid architecture — avoid re-ingesting
  the same long document in new sessions
- Thinking mode for reasoning/coding; disable it (`enable_thinking=false`) for casual chat

## 🤖 Hand This Repo to a Coding Agent (Saves Tokens)

Don't deploy by trial and error, and don't let an agent explore from scratch — this repo
encodes 5 real pitfalls (CPU-only aarch64 torch on PyPI, unreachable huggingface.co → mirror,
rootless Python headers, ninja missing from PATH, reasoning parser not enabled). Every failed
iteration an agent spends rediscovering them burns tokens. Paste this into any coding agent
(Kimi Code CLI, Claude Code, Cursor, Codex, …):

```text
My machine is an NVIDIA DGX Spark (GB10, aarch64). Read
https://github.com/Deep-AI-Evo/qwen3.8-27b-nvfp4-dgx-spark-tutorial
— docs/en/deployment-tutorial.md — and follow its exact versions and commands to deploy
Qwen3.8-27B-NVFP4 locally with vLLM. Then validate the deployment using the functional
tests described in docs/en/test-report.md and report the results.
```

The agent will hit the right versions and commands on the first pass.

## Repository Layout

- `docs/` — Chinese tutorial & report; `docs/en/` — English versions
- `tests/` — functional test scripts (chat/vision, stdlib only) and the benchmark matrix script
- `results/` — raw benchmark outputs

## Launch Command (once the environment is set up)

```bash
PATH=~/projects/qwen38-env/bin:$PATH \
~/projects/qwen38-env/bin/vllm serve ~/projects/models/Qwen3.8-27B-NVFP4 \
  --served-model-name unsloth/Qwen3.8-27B-NVFP4 \
  --reasoning-parser qwen3 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --port 8000
```
