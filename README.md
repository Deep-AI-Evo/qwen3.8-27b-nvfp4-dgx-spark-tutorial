# Qwen3.8-27B-NVFP4 on NVIDIA DGX Spark (GB10)

**语言 / Language: [中文](README.zh.md) | [English](README.en.md)**

在 NVIDIA DGX Spark（GB10, Grace-Blackwell, aarch64, 128GB 统一内存）上部署与实测
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
（27B 稠密视觉-语言模型，NVFP4 量化，262,144 tokens 原生上下文）的完整记录。

Deployment & benchmark record of unsloth/Qwen3.8-27B-NVFP4 (dense 27B vision-language model,
NVFP4, native 262k context) on the NVIDIA DGX Spark.

---

## 实测数据速览 / Benchmark Highlights

> vLLM 0.27.1 · torch 2.13.0+cu130 · `--max-model-len 262144` · MTP ×3 speculative decoding on

**单流解码 / Single-stream decode**

| 场景 Scenario | 速度 Speed |
|---|---|
| 思考模式（MTP ×3）/ Thinking mode | **22.7 tok/s**（无 MTP 11.8，+92%） |
| 非思考模式 / Non-thinking | **16.7 tok/s** |
| 200k 长上下文下 / at 200k context | **≈14.2 tok/s** |

**并发 1–4（日常场景）/ Concurrency 1–4 (daily use)**

| 并发 Concurrency | 单流解码 Per-stream | 聚合吞吐 Aggregate |
|---|---|---|
| 1 | 21.6 tok/s | 20.0 tok/s |
| 2 | 16.0 tok/s | 22.7 tok/s |
| 4 | 15.0 tok/s | 44.0 tok/s |
| 8 | ≈12.8 tok/s | 77.7 tok/s |
| 16 | ≈8.4 tok/s | 115.7 tok/s（峰值 peak 160） |

**上下文衰减 / Context-length scaling（单并发 single stream）**

| 上下文 Context | Prefill | 首 token TTFT | 解码 Decode |
|---|---|---|---|
| 1k | ≈1800 tok/s | 0.57 s | ≈21 tok/s |
| 100k | ≈1230 tok/s (-32%) | 83 s | ≈16.6 tok/s |
| 200k | ≈840 tok/s (-53%) | 244 s | ≈14.2 tok/s |

解码衰减平缓（64 层中 48 层为 Gated DeltaNet 线性注意力，解码成本与长度无关）；
prefill 超线性变慢（全注意力层的二次方项）。
Decode degrades gently (48/64 layers are linear attention with length-independent cost);
prefill slows super-linearly (quadratic full-attention term).

---

## 🤖 把本仓库交给代码智能体，一键复现部署（省 Token）

DGX Spark 用户**不必照着手动部署，也不建议让 Agent 从零摸索**——本教程沉淀了 5 个真实踩坑
（PyPI 的 aarch64 torch 是 CPU 版、HF 直连不通要走镜像、无 root 补 Python.h、PATH 缺 ninja、
reasoning parser 未开启），Agent 自行试错会消耗大量 token 和时间。把仓库交给它照做即可：

在任意代码智能体（Kimi Code CLI、Claude Code、Cursor、Codex 等）中输入：

```text
我的机器是 NVIDIA DGX Spark（GB10，aarch64）。请阅读仓库
https://github.com/Deep-AI-Evo/qwen3.8-27b-nvfp4-dgx-spark-tutorial
中 docs/Qwen3.8-27B-NVFP4-部署教程.md（英文：docs/en/deployment-tutorial.md），
严格按照其中的版本号与命令在本机部署 Qwen3.8-27B-NVFP4 并启动 vLLM 服务；
完成后参照 docs/ 下测试报告中的方法做功能验证并汇报结果。
```

### English: Hand this repo to a coding agent (saves tokens)

Don't deploy by trial and error — this repo encodes 5 real pitfalls (CPU-only aarch64 torch on PyPI,
HF mirror, rootless Python headers, ninja PATH, reasoning parser). An agent rediscovering them burns
tokens on every failed iteration. Paste this into any coding agent (Kimi Code CLI, Claude Code,
Cursor, Codex, …):

```text
My machine is an NVIDIA DGX Spark (GB10, aarch64). Read
https://github.com/Deep-AI-Evo/qwen3.8-27b-nvfp4-dgx-spark-tutorial
— docs/en/deployment-tutorial.md — and follow its exact versions and commands to deploy
Qwen3.8-27B-NVFP4 locally with vLLM. Then validate the deployment using the functional
tests described in docs/en/test-report.md and report the results.
```

---

## 完整文档 / Full Documentation

| 中文 | English |
|---|---|
| [部署教程](docs/Qwen3.8-27B-NVFP4-部署教程.md) | [Deployment Tutorial](docs/en/deployment-tutorial.md) |
| [测试报告](docs/Qwen3.8-27B-NVFP4-测试报告.md) | [Test Report](docs/en/test-report.md) |
| [中文 README](README.zh.md) | [English README](README.en.md) |

## 目录 / Repository Layout

- `docs/` —— 中文教程与报告 + `docs/en/` English versions
- `tests/` —— 功能测试脚本（对话/视觉，仅标准库）与压测矩阵脚本 / test scripts (stdlib-only)
- `results/` —— 原始压测输出 / raw benchmark outputs

## 一键启动命令 / Launch Command（环境就绪后 / once the env is set up）

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
