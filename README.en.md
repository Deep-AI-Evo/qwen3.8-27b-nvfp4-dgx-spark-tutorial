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

## 📑 Contents

- [What this repo solves](#intro)
- [Benchmark highlights (single node)](#single)
- [Dual-node results (2× DGX Spark)](#dual)
- [Quick start](#quickstart)
- [Hand this repo to a coding agent (saves tokens)](#agent)
- [Daily-use recommendations](#daily)
- [Cross-device comparison: DGX Spark / RTX PRO 5000 / PRO 6000](#compare) ⭐
- [Full documentation & layout](#docs)
- [Related repos & author](#links)

---

<a id="intro"></a>

## 🎯 What this repo solves

Qwen3.8-27B-NVFP4 is a newly released (Aug 2026) 27B vision-language model with a native
256K context. This repo answers three questions:

1. **How to deploy it** — pure pip on DGX Spark (aarch64/GB10), no root, no Docker,
   with every pitfall documented
2. **How fast it runs** — full decode/prefill/concurrency/long-context measurements
   (including a maxed-out 256K needle-in-a-haystack verification)
3. **Whether it's worth it** — does a second node help? How does it compare to an
   RTX PRO 5000 / PRO 6000?

> ⭐ **If you only care about "how does this machine stack up"**: jump straight to the
> [cross-device comparison](#compare) — DGX Spark vs RTX PRO 5000 vs RTX PRO 6000.
> But skimming the single-node and dual-node numbers first will make the gaps much clearer.

---

<a id="single"></a>

## 📊 Benchmark Highlights (single node)

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
| **256K (maxed, 261K tokens)** | ≈715 tok/s (-60%) | 366 s | ≈11.4 tok/s |

> 🎯 **256K context verified for real**: in a needle-in-a-haystack test (255,376 tokens,
> needle at 70% depth) the model recalled the hidden passcode on the first try.

> 💡 Why decode decays gently: 48 of the 64 layers are Gated DeltaNet linear attention, whose
> decode cost is independent of context length; only 16 full-attention layers grow with it.
> The KV cache holds 2.09M tokens → ~8 concurrent requests at the full 256K context.

---

<a id="dual"></a>

## 🖥🖥 Dual-node Results (2× DGX Spark TP=2)

> Full data & setup: [Dual-node Notes](docs/en/dual-node-benchmarks.md) (vLLM native mp mode + MTP×3)

| Metric | Single node | 2-node TP=2 | Verdict |
|---|---|---|---|
| Single-stream decode (thinking) | 22.7 tok/s | 20.6–22.0 tok/s | Tie to slightly worse |
| c=4 aggregate throughput | 44 tok/s | **65.4 tok/s (+49%)** | ✅ Clear dual-node win |
| Full-256K concurrency | 8 | **18** | ✅ KV pool doubled |
| Prefill 8K | ≈1900 tok/s | ≈1358 tok/s | Dual-node penalty |
| Prefill 100K | 1230 tok/s | **1325 tok/s** | ✅ Pulls ahead |

**In one line: a 27B model fits on one node — dual-node doesn't speed up a single
stream; what it buys is concurrency throughput and KV capacity.**
Use one node for daily interactive work; bring up the second for multi-agent fleets
or multi-session long-document workloads.

---

<a id="quickstart"></a>

## 🚀 Quick Start

Once the environment is set up (full steps in the [deployment tutorial](docs/en/deployment-tutorial.md)):

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

<a id="agent"></a>

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

<a id="daily"></a>

## 💡 Daily-use Recommendations

- **Concurrency**: ≤4 streams for interactive use (≥15 tok/s each); 8 is the interactive ceiling; 16+ for offline batch
- **Long context**: ≤100K use freely (~1.4 min ingest); ~200K works best as "ingest once, ask many turns"
- **Caveat**: no prefix caching for this hybrid architecture yet — avoid re-ingesting the same long document in new sessions
- **Modes**: thinking on for reasoning/coding; off (`enable_thinking=false`) for chat/translation/summaries

---

<a id="compare"></a>

## 🆚 Cross-device Comparison (DGX Spark / RTX PRO 5000 / RTX PRO 6000)

> Summary from the [3-device comparison doc](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark/blob/main/docs/Qwen3.8-27B-跨设备横向对比.md) (full tiers and methodology there).
> ⚠️ Methodology differs slightly across devices (vLLM version / MTP / OS) — order-of-magnitude guidance, not exact parity.

**Single-stream decode (tok/s)**

| Context | DGX Spark NVFP4 (this repo) | PRO 5000 FP8 (no MTP) | PRO 5000 Q6_K | **PRO 6000 FP8+MTP** | **PRO 6000 FP8 no MTP** | PRO 6000 NVFP4+MTP | PRO 6000 NVFP4 no MTP | PRO 6000 Q6_K |
|---|---|---|---|---|---|---|---|---|
| Short (~1-3K) | ~21 | 37.3 | 39.7 | 95.6 | 52.6 | 100.2 | 58.6 | 55.4 |
| ~200K | 14.2 | 26.4 | 39.9 | 19.1 | 39.6 | 18.2 | **43.7** | 35.7 |

**Single-stream prefill ~200K (tok/s) / TTFT (s)**

| DGX Spark NVFP4 (this repo) | PRO 5000 FP8 | PRO 5000 Q6_K | **PRO 6000 FP8+MTP** | PRO 6000 NVFP4+MTP | PRO 6000 Q6_K |
|---|---|---|---|---|---|
| 840 / 244s | 2,114 / 110s | 768 / 302s | 3,869 / 45.8s | **4,447 / 39.9s** | 1,676 / 105.9s |

**Concurrent decode aggregate throughput (tok/s)**

| Concurrency | DGX Spark NVFP4 (this repo) | PRO 5000 FP8 | **PRO 6000 FP8+MTP** | PRO 6000 NVFP4+MTP | PRO 6000 Q6_K (4 slots) |
|---|---|---|---|---|---|
| 1 | 20.0 | 37.3 | 78.8 | 95.0 | 49.6 |
| 2 | 22.7 | 64.7 | 137.0 | 151.3 | 92.8 |
| 4 | 44.0 | — | 299.3 | 346.5 | 158.5 |
| 8 | 77.7 | — | 556.2 | **654.1** | — |

**How to read this table**:

- **Raw speed**: the PRO 6000 leads prefill/TTFT/concurrency across the board (~4–8× the DGX
  Spark); FP8 without MTP is the strongest no-fuss baseline (52.6 t/s short, still 39.6 t/s at 200K)
- **Long-context decode champion**: the PRO 5000's NVFP4+MTP (n=1, marlin) at 57.8 t/s @148K
  is the best measured across all three devices; MTP's long-context payoff flips with backend/n
  value (the PRO 6000's n=2 setup collapses at 200K, where no-MTP gives 43.7 t/s) — measure your
  own config before trusting MTP at long context
- **The DGX Spark's value isn't raw speed**: 128GB unified memory running a full 256K context,
  a quiet low-power desktop form factor, and dual-node scalability (see [dual-node results](#dual))

---

<a id="docs"></a>

## 📚 Full Documentation & Layout

| Doc | Contents |
|---|---|
| 📖 [Deployment Tutorial](docs/en/deployment-tutorial.md) | Environment setup, mirror download, flag-by-flag launch guide, API examples, 5 real pitfalls & fixes |
| 📈 [Test Report](docs/en/test-report.md) | 7 functional tests, concurrency & long-context benchmarks, MTP comparison, daily-use recommendations |
| 🖥🖥 [Dual-node Notes](docs/en/dual-node-benchmarks.md) | 2× DGX Spark TP=2: concurrency/capacity gains, mp-vs-Ray pitfalls, prefill/decode comparison |

```
├── README.md / README.en.md      # 中文 / English
├── docs/                         # 中文教程与报告
│   └── en/                       # English tutorial & report
├── tests/                        # functional tests (chat/vision, stdlib-only) + bench matrix
└── results/                      # raw benchmark outputs
```

---

<a id="links"></a>

## 🔗 Related Repos

- [RTX PRO 6000 FP8 vs NVFP4 benchmark](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark) — includes the [3-device comparison](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark/blob/main/docs/Qwen3.8-27B-跨设备横向对比.md)
- [RTX PRO 5000 llama.cpp Q6_K vs vLLM FP8](https://github.com/Deep-AI-Evo/qwen3.8-27b-q6k-fp8-rtx-pro5000-serving-benchmark)
- [DeepSeek-V4-Flash dual-node field notes](https://github.com/Deep-AI-Evo/deepseek-v4-flash-2x-dgx-spark)
- [DGX Spark cluster monitor](https://github.com/Deep-AI-Evo/dgx-spark-cluster-monitor)

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
