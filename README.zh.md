# Qwen3.8-27B-NVFP4 在 NVIDIA DGX Spark（GB10）上的部署与实测

**[English README](README.en.md)**

本仓库记录 [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
（27B 稠密视觉-语言模型，NVFP4 量化，262,144 tokens 原生上下文）在 **NVIDIA DGX Spark**
（GB10，Grace-Blackwell，aarch64，128GB 统一内存）上的完整部署教程与实测报告。

## 文档

- [部署教程](docs/Qwen3.8-27B-NVFP4-部署教程.md) —— 纯 pip 部署（无需 root/Docker），
  torch 2.13.0+cu130 + vLLM 0.27.1，含国内镜像下载与全部踩坑记录
- [测试报告](docs/Qwen3.8-27B-NVFP4-测试报告.md) —— 功能测试、并发/长上下文压测、
  MTP 推测解码优化、日常使用建议

## 实测数据速览

> vLLM 0.27.1 · torch 2.13.0+cu130 · 256k 上下文 · MTP ×3 推测解码已开启

**单流解码**

| 场景 | 速度 |
|---|---|
| 思考模式 | **22.7 tok/s**（无 MTP 11.8，+92%） |
| 非思考模式 | **16.7 tok/s** |
| 200k 长上下文下 | **≈14.2 tok/s** |

**并发 1–16**

| 并发 | 单流解码 | 聚合吞吐 |
|---|---|---|
| 1 | 21.6 tok/s | 20.0 tok/s |
| 2 | 16.0 tok/s | 22.7 tok/s |
| 4 | 15.0 tok/s | 44.0 tok/s |
| 8 | ≈12.8 tok/s | 77.7 tok/s |
| 16 | ≈8.4 tok/s | 115.7 tok/s（峰值 160） |

**上下文长度衰减（单并发）**

| 上下文 | Prefill | TTFT | 解码 |
|---|---|---|---|
| 1k | ≈1800 tok/s | 0.57 s | ≈21 tok/s |
| 100k | ≈1230 tok/s（-32%） | 83 s | ≈16.6 tok/s |
| 200k | ≈840 tok/s（-53%） | 244 s | ≈14.2 tok/s |

结论：解码衰减平缓（64 层中 48 层为 Gated DeltaNet 线性注意力，解码成本与上下文长度无关）；
prefill 超线性变慢。KV cache 209 万 tokens，满 256k 上下文约 8 路并发。

## 日常使用建议

- 交互场景并发 ≤4（单流 ≥15 tok/s），8 是交互上限，批量任务用 16+ 吃满聚合吞吐
- 100k 以内随意用（灌入约 1.4 分钟）；200k 级适合"一次灌入 + 多轮问答"
- 该架构暂不支持前缀缓存，避免在新会话反复重灌同一长文档
- 复杂任务开思考模式，闲聊关思考（`enable_thinking=false`）

## 🤖 把本仓库交给代码智能体，一键复现部署（省 Token）

不必手动部署，也不建议让 Agent 从零摸索——本教程沉淀了 5 个真实踩坑
（PyPI 的 aarch64 torch 是 CPU 版、HF 直连不通要走镜像、无 root 补 Python.h、
PATH 缺 ninja、reasoning parser 未开启），Agent 自行试错每一步都在烧 token。
在任意代码智能体（Kimi Code CLI、Claude Code、Cursor、Codex 等）中输入：

```text
我的机器是 NVIDIA DGX Spark（GB10，aarch64）。请阅读仓库
https://github.com/Deep-AI-Evo/qwen3.8-27b-nvfp4-dgx-spark-tutorial
中 docs/Qwen3.8-27B-NVFP4-部署教程.md，严格按照其中的版本号与命令在本机部署
Qwen3.8-27B-NVFP4 并启动 vLLM 服务；完成后参照 docs/Qwen3.8-27B-NVFP4-测试报告.md
中的方法做功能验证并汇报结果。
```

Agent 会按教程直接命中正确版本与命令，跳过全部试错环节。

## 目录

- `docs/` —— 教程与测试报告（`docs/en/` 为英文版）
- `tests/` —— 功能测试脚本（对话/视觉，仅标准库）与压测矩阵脚本
- `results/` —— 原始压测输出

## 一键启动命令（环境就绪后）

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
