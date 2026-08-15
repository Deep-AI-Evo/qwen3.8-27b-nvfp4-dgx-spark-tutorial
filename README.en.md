<div align="center">

# Qwen3.8-27B-NVFP4 × NVIDIA DGX Spark

**Deploy a 27B NVFP4 vision-language model from scratch on GB10 (Grace-Blackwell)**
**Pure pip · No root · 256K context · MTP speculative decoding · Full benchmarks**

[![Hugging Face](https://img.shields.io/badge/🤗%20Model-Qwen3.8--27B--NVFP4-yellow)](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
[![vLLM](https://img.shields.io/badge/vLLM-0.27.1-blue)](https://github.com/vllm-project/vllm)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0%2Bcu130-ee4c2c)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76b900)](https://developer.nvidia.com/cuda-toolkit)
[![Platform](https://img.shields.io/badge/Platform-DGX%20Spark%20GB10-green)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

### 🌐 **[切换到中文](README.md)**

</div>

---

## 📊 Benchmark Highlights

> Testbed: DGX Spark (GB10, 128GB unified memory) · vLLM 0.27.1 · torch 2.13.0+cu130 · 256K context · MTP ×3 enabled

**⚡ Single-stream decode**

| Scenario | Speed | Note |
|---|---|---|
| Thinking mode | **22.7 tok/s** | 11.8 without MTP — **+92%** |
| Non-thinking | **16.7 tok/s** | — |
| At 200K context | **≈14.2 tok/s** | gentle decay |

**🔀 Concurrency scaling**

| Concurrency | Per-stream decode | Aggregate throughput |
|---|---|---|
| 1 | 21.6 tok/s | 20.0 tok/s |
| 2 | 16.0 tok/s | 22.7 tok/s |
| 4 | 15.0 tok/s | 44.0 tok/s |
| 8 | ≈12.8 tok/s | 77.7 tok/s |
| 16 | ≈8.4 tok/s | **115.7 tok/s** (peak 160) |

**📏 Context-length scaling (single stream)**

| Context | Prefill speed | TTFT | Decode speed |
|---|---|---|---|
| 1K | ≈1800 tok/s | 0.57 s | ≈21 tok/s |
| 100K | ≈1230 tok/s (-32%) | 83 s | ≈16.6 tok/s |
| 200K | ≈840 tok/s (-53%) | 244 s | ≈14.2 tok/s |

> 💡 Why decode decays gently: 48 of the 64 layers are Gated DeltaNet linear attention, whose
> decode cost is independent of context length; only 16 full-attention layers grow with it.
> The KV cache holds 2.09M tokens → ~8 concurrent requests at the full 256K context.

---

## 🤖 Hand This Repo to a Coding Agent (Saves Tokens)

Don't copy commands by hand, and **don't let an agent explore from scratch** — this repo encodes
5 real pitfalls (CPU-only aarch64 torch on PyPI, unreachable huggingface.co → mirror, rootless
Python headers, ninja missing from PATH, reasoning parser not enabled). Every failed iteration
an agent spends rediscovering them burns tokens.

**How to use: copy the prompt below into any coding agent** (Kimi Code CLI, Claude Code,
Cursor, Codex, …):

```text
My machine is an NVIDIA DGX Spark (GB10, aarch64). Read
https://github.com/Deep-AI-Evo/qwen3.8-27b-nvfp4-dgx-spark-tutorial
— docs/en/deployment-tutorial.md — and follow its exact versions and commands to deploy
Qwen3.8-27B-NVFP4 locally with vLLM. Then validate the deployment using the functional
tests described in docs/en/test-report.md and report the results.
```

The agent hits the right versions and commands on the first pass — zero trial and error.

---

## 🚀 Launch Command (once the environment is set up)

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

Then call the OpenAI-compatible API at `http://127.0.0.1:8000`.

---

## 📚 Full Documentation

| Doc | Contents |
|---|---|
| 📖 [Deployment Tutorial](docs/en/deployment-tutorial.md) | Environment setup, mirror download, flag-by-flag launch guide, API examples, 5 real pitfalls & fixes |
| 📈 [Test Report](docs/en/test-report.md) | 7 functional tests, concurrency & long-context benchmarks, MTP comparison, daily-use recommendations |

## 💡 Daily-use Recommendations

- **Concurrency**: ≤4 streams for interactive use (≥15 tok/s each); 8 is the interactive ceiling; 16+ for offline batch
- **Long context**: ≤100K use freely (~1.4 min ingest); ~200K works best as "ingest once, ask many turns"
- **Caveat**: no prefix caching for this hybrid architecture yet — avoid re-ingesting the same long document in new sessions
- **Modes**: thinking on for reasoning/coding; off (`enable_thinking=false`) for chat/translation/summaries

## 🗂 Repository Layout

```
├── README.md / README.en.md      # 中文 / English
├── docs/                         # 中文教程与报告
│   └── en/                       # English tutorial & report
├── tests/                        # functional tests (chat/vision, stdlib-only) + bench matrix
└── results/                      # raw benchmark outputs
```

---

<div align="center">

## 👤 About the Author

**Evo AI (壹我AI)** · [@Deep-AI-Evo](https://github.com/Deep-AI-Evo)

Making frontier models accessible to everyone — AI tooling & open-source LLM applications.

Projects: [easyCodex](https://github.com/Deep-AI-Evo/eazycodex) (one-click Codex Desktop + DeepSeek) ·
[AI Office Workbuddy](https://github.com/Deep-AI-Evo/ai-office-workbuddy) ·
[Self-LLM Guide](https://github.com/Deep-AI-Evo/self-llm)

**If this saved you time, a ⭐ Star is appreciated**

**Made by Evo AI (壹我AI出品)**

</div>
