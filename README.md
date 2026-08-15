<div align="center">

# Qwen3.8-27B-NVFP4 × NVIDIA DGX Spark

**在 GB10（Grace-Blackwell）上从零部署 27B NVFP4 视觉-语言大模型**
**纯 pip · 无需 root · 256K 上下文 · MTP 推测解码 · 完整实测数据**

[![Hugging Face](https://img.shields.io/badge/🤗%20Model-Qwen3.8--27B--NVFP4-yellow)](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
[![vLLM](https://img.shields.io/badge/vLLM-0.27.1-blue)](https://github.com/vllm-project/vllm)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0%2Bcu130-ee4c2c)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76b900)](https://developer.nvidia.com/cuda-toolkit)
[![Platform](https://img.shields.io/badge/Platform-DGX%20Spark%20GB10-green)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

### 🌐 **[Switch to English](README.en.md)**

</div>

---

## 📊 实测数据速览

> 测试环境：DGX Spark（GB10，128GB 统一内存）· vLLM 0.27.1 · torch 2.13.0+cu130 · 256K 上下文 · MTP ×3 已开启

**⚡ 单流解码速度**

| 场景 | 速度 | 备注 |
|---|---|---|
| 思考模式 | **22.7 tok/s** | 无 MTP 仅 11.8，提升 **+92%** |
| 非思考模式 | **16.7 tok/s** | — |
| 200K 长上下文下 | **≈14.2 tok/s** | 衰减平缓 |

**🔀 并发扩展性**

| 并发 | 单流解码 | 聚合吞吐 |
|---|---|---|
| 1 | 21.6 tok/s | 20.0 tok/s |
| 2 | 16.0 tok/s | 22.7 tok/s |
| 4 | 15.0 tok/s | 44.0 tok/s |
| 8 | ≈12.8 tok/s | 77.7 tok/s |
| 16 | ≈8.4 tok/s | **115.7 tok/s**（峰值 160） |

**📏 上下文长度衰减（单并发）**

| 上下文 | Prefill 速度 | 首 Token (TTFT) | 解码速度 |
|---|---|---|---|
| 1K | ≈1800 tok/s | 0.57 s | ≈21 tok/s |
| 100K | ≈1230 tok/s（-32%） | 83 s | ≈16.6 tok/s |
| 200K | ≈840 tok/s（-53%） | 244 s | ≈14.2 tok/s |

> 💡 解码衰减平缓的秘密：64 层中 48 层是 Gated DeltaNet 线性注意力，解码成本与上下文长度无关；
> 只有 16 层全注意力随长度增长。KV cache 共 209 万 tokens，满 256K 上下文支持约 8 路并发。

---

## 🤖 把本仓库交给代码智能体，一键复现部署（省 Token）

不必照着文档手动敲，也**不建议让 Agent 从零摸索**——本教程沉淀了 5 个真实踩坑
（PyPI 的 aarch64 torch 是 CPU 版、HF 直连不通要走镜像、无 root 补 Python.h、PATH 缺 ninja、
reasoning parser 未开启），Agent 自行试错每一步都在烧 Token。

**用法：复制下面这段话，发给任意代码智能体**（Kimi Code CLI、Claude Code、Cursor、Codex 等）：

```text
我的机器是 NVIDIA DGX Spark（GB10，aarch64）。请阅读仓库
https://github.com/Deep-AI-Evo/qwen3.8-27b-nvfp4-dgx-spark-tutorial
中 docs/Qwen3.8-27B-NVFP4-部署教程.md，严格按照其中的版本号与命令在本机部署
Qwen3.8-27B-NVFP4 并启动 vLLM 服务；完成后参照 docs/Qwen3.8-27B-NVFP4-测试报告.md
中的方法做功能验证并汇报结果。
```

Agent 会按教程直接命中正确版本与命令，跳过全部试错环节，一次跑通。

---

## 🚀 一键启动命令（环境就绪后）

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

服务就绪后访问 `http://127.0.0.1:8000`（OpenAI 兼容 API）。

---

## 📚 完整文档

| 文档 | 内容 |
|---|---|
| 📖 [部署教程](docs/Qwen3.8-27B-NVFP4-部署教程.md) | 环境准备、镜像下载、启动参数详解、API 调用示例、5 个踩坑实录 |
| 📈 [测试报告](docs/Qwen3.8-27B-NVFP4-测试报告.md) | 7 项功能测试、并发/长上下文压测、MTP 优化对比、日常使用建议 |

## 💡 日常使用建议

- **并发**：交互场景 ≤4 路（单流 ≥15 tok/s）；8 路是交互上限；批量任务用 16+ 吃满吞吐
- **长上下文**：100K 以内随意用（灌入约 1.4 分钟）；200K 级适合"一次灌入 + 多轮问答"
- **注意**：该混合架构暂不支持前缀缓存，避免在新会话反复重灌同一长文档
- **模式**：复杂推理/代码开思考模式；闲聊/翻译/摘要关思考（`enable_thinking=false`）更快

## 🗂 仓库结构

```
├── README.md / README.en.md      # 中文 / English
├── docs/                         # 中文教程与报告
│   └── en/                       # English tutorial & report
├── tests/                        # 功能测试脚本（对话/视觉，仅标准库）+ 压测矩阵
└── results/                      # 全部原始压测输出
```

---

<div align="center">

## 👤 关于作者

**壹我AI（Evo AI）** · [@Deep-AI-Evo](https://github.com/Deep-AI-Evo)

专注 AI 工具落地与开源大模型应用，让前沿模型人人可用。

出品：[easyCodex](https://github.com/Deep-AI-Evo/eazycodex)（一键安装 Codex 桌面版 + DeepSeek）·
[AI 办公提效实操课](https://github.com/Deep-AI-Evo/ai-office-workbuddy) ·
[Self-LLM 开源大模型食用指南](https://github.com/Deep-AI-Evo/self-llm)

**如果本教程帮你省了时间，欢迎点个 ⭐ Star**

**壹我AI出品**

</div>
