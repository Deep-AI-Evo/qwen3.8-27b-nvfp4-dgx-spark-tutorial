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

## 📑 目录

- [这个仓库解决什么问题](#intro)
- [实测数据速览（单机）](#single)
- [双机部署结论（2× DGX Spark）](#dual)
- [快速开始](#quickstart)
- [把本仓库交给代码智能体（省 Token）](#agent)
- [日常使用建议](#daily)
- [跨设备横向对比：DGX Spark / RTX PRO 5000 / PRO 6000](#compare) ⭐
- [完整文档与仓库结构](#docs)
- [相关仓库与作者](#links)

---

<a id="intro"></a>

## 🎯 这个仓库解决什么问题

Qwen3.8-27B-NVFP4 是 2026-08 新发布的 27B 视觉-语言模型（256K 原生上下文）。
本仓库回答三个问题：

1. **怎么装** —— DGX Spark（aarch64/GB10）上纯 pip 部署，无需 root、无需 Docker，含全部踩坑解法
2. **跑多快** —— 完整的 decode/prefill/并发/长上下文实测数据（含 256K 顶格大海捞针验证）
3. **值不值** —— 双机有没有意义？和 RTX PRO 5000 / PRO 6000 比是什么水平？

> ⭐ **如果你只关心"这机器什么水平"**：直接跳到文末的
> [跨设备横向对比](#compare)——DGX Spark vs RTX PRO 5000 vs RTX PRO 6000，
> 但建议先快速浏览单机和双机数据，对比时才看得懂差距来自哪里。

---

<a id="single"></a>

## 📊 实测数据速览（单机）

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
| **256K（顶格 261K tokens）** | ≈715 tok/s（-60%） | 366 s | ≈11.4 tok/s |

> 🎯 **256K 真实性已验证**：大海捞针测试在 255,376 tokens 文本的 70% 深度处藏入密码，模型一次答对。

> 💡 解码衰减平缓的秘密：64 层中 48 层是 Gated DeltaNet 线性注意力，解码成本与上下文长度无关；
> 只有 16 层全注意力随长度增长。KV cache 共 209 万 tokens，满 256K 上下文支持约 8 路并发。

---

<a id="dual"></a>

## 🖥🖥 双机部署结论（2× DGX Spark TP=2）

> 完整数据与部署方式：[双机部署实测](docs/Qwen3.8-27B-双机部署实测.md)（vLLM 原生 mp 多机模式 + MTP×3）

| 维度 | 单机 | 双机 TP=2 | 判断 |
|---|---|---|---|
| 单流解码（思考模式） | 22.7 tok/s | 20.6–22.0 tok/s | 持平略降 |
| c=4 聚合吞吐 | 44 tok/s | **65.4 tok/s（+49%）** | ✅ 双机明显优势 |
| 256K 满上下文并发 | 8 路 | **18 路** | ✅ KV 池翻倍 |
| prefill 8K | ≈1900 tok/s | ≈1358 tok/s | 短输入双机吃亏 |
| prefill 100K | 1230 tok/s | **1325 tok/s** | ✅ 长上下文反超 |

**一句话：27B 单机就装得下，双机不为单流提速——它买的是并发吞吐和 KV 容量。**
日常单人交互用单机；多 agent 并发、长文档多会话再上双机。

---

<a id="quickstart"></a>

## 🚀 快速开始

环境就绪后一条命令启动（完整步骤见 [部署教程](docs/Qwen3.8-27B-NVFP4-部署教程.md)）：

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

<a id="agent"></a>

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

<a id="daily"></a>

## 💡 日常使用建议

- **并发**：交互场景 ≤4 路（单流 ≥15 tok/s）；8 路是交互上限；批量任务用 16+ 吃满吞吐
- **长上下文**：100K 以内随意用（灌入约 1.4 分钟）；200K 级适合"一次灌入 + 多轮问答"
- **注意**：该混合架构暂不支持前缀缓存，避免在新会话反复重灌同一长文档
- **模式**：复杂推理/代码开思考模式；闲聊/翻译/摘要关思考（`enable_thinking=false`）更快

---

<a id="compare"></a>

## 🆚 跨设备横向对比（DGX Spark / RTX PRO 5000 / RTX PRO 6000）

> 摘要自 [三设备横向对比](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark/blob/main/docs/Qwen3.8-27B-跨设备横向对比.md)（完整档位与测试方法见该文档）。
> ⚠️ 口径差异：三台设备 vLLM 版本 / MTP 设置 / 操作系统不完全相同，数量级参考意义大于精确对比。

**单并发 decode（tok/s）**

| 上下文 | DGX Spark NVFP4（本仓库） | PRO 5000 FP8（无MTP → +MTP） | PRO 5000 NVFP4（无MTP → +MTP） | PRO 5000 Q6_K（无MTP → +MTP） | PRO 6000 FP8+MTP | PRO 6000 FP8 无MTP | PRO 6000 NVFP4+MTP | PRO 6000 NVFP4 无MTP | PRO 6000 Q6_K |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 短（~1-11K） | ~21 | 37.3 → 43.2 | 49.6 → **63.6** | 39.7 → 61.9 | 95.6 | 52.6 | 100.2 | 58.6 | 55.4 |
| ~148K | — | 26.4 → 15.3 ❌ | 42.1 → **57.8** ✅ | 39.9 → 40.0 | — | — | — | — | — |
| ~200K | 14.2 | 26.4（无MTP） | 42.1（无MTP） | 39.9 | 19.1 | 39.6 | 18.2 | **43.7** | 35.7 |

**单并发 prefill ~200K（tok/s）/ TTFT（s）**

| DGX Spark NVFP4（本仓库） | PRO 5000 FP8 | PRO 5000 Q6_K | **PRO 6000 FP8+MTP** | PRO 6000 NVFP4+MTP | PRO 6000 Q6_K |
|---|---|---|---|---|---|
| 840 / 244s | 2,114 / 110s | 768 / 302s | 3,869 / 45.8s | **4,447 / 39.9s** | 1,676 / 105.9s |

**并发 decode 聚合吞吐（tok/s）**

| 并发 | DGX Spark NVFP4（本仓库） | PRO 5000 FP8 | **PRO 6000 FP8+MTP** | PRO 6000 NVFP4+MTP | PRO 6000 Q6_K（4 槽） |
|---|---|---|---|---|---|
| 1 | 20.0 | 37.3 | 78.8 | 95.0 | 49.6 |
| 2 | 22.7 | 64.7 | 137.0 | 151.3 | 92.8 |
| 4 | 44.0 | — | 299.3 | 346.5 | 158.5 |
| 8 | 77.7 | — | 556.2 | **654.1** | — |

**怎么读这张表**：

- **要 raw 速度**：PRO 6000 在 prefill/TTFT/并发上全面领先（对 DGX Spark 约 4~8 倍），其中
  FP8 无 MTP 是最省心的强基线（52.6 t/s 起步，200K 仍有 39.6 t/s）
- **长上下文 decode 最高报告值**：PRO 5000 的 NVFP4+MTP（n=1, marlin, Windows）57.8 t/s @148K；
  但 PRO 6000（Linux）用**完全相同的软件栈**（vLLM 0.26 + marlin + n=1）复测仅 21.4 t/s——
  n 值/后端/vLLM 版本/调度参数逐一排除、接受率全程健康，差异指向 OS/驱动层。
  Linux 下可复现的最佳实践是关 MTP（PRO 6000 NVFP4 无 MTP 43.7 t/s @200K），用前实测自己的配置
- 📉 **MTP 长上下文衰减的根因与跨引擎验证**：见 [PRO 6000 横向对比文档的 MTP 专题](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark/blob/main/docs/Qwen3.8-27B-跨设备横向对比.md)——
  含 vLLM Issue #47602（单层草稿头接受率坍塌）与 PRO 6000 变量排除实验
  （接受率健康但每步开销失控，呈现不同签名）
- **DGX Spark 的价值不在绝对速度**：128GB 统一内存跑满 256K 上下文、安静低功耗的桌面形态、
  以及双机扩展的灵活性（见[双机结论](#dual)）

---

<a id="docs"></a>

## 📚 完整文档与仓库结构

| 文档 | 内容 |
|---|---|
| 📖 [部署教程](docs/Qwen3.8-27B-NVFP4-部署教程.md) | 环境准备、镜像下载、启动参数详解、API 调用示例、5 个踩坑实录 |
| 📈 [测试报告](docs/Qwen3.8-27B-NVFP4-测试报告.md) | 7 项功能测试、并发/长上下文压测、MTP 优化对比、日常使用建议 |
| 🖥🖥 [双机部署实测](docs/Qwen3.8-27B-双机部署实测.md) | 2× DGX Spark TP=2：并发/容量收益、mp 模式与 Ray 的坑、prefill/解码对比 |

```
├── README.md / README.en.md      # 中文 / English
├── docs/                         # 中文教程与报告
│   └── en/                       # English tutorial & report
├── tests/                        # 功能测试脚本（对话/视觉，仅标准库）+ 压测矩阵
└── results/                      # 全部原始压测输出
```

---

<a id="links"></a>

## 🔗 相关仓库

- [RTX PRO 6000 FP8 vs NVFP4 benchmark](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark) — 含 [DGX Spark / PRO 5000 / PRO 6000 三设备横向对比](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark/blob/main/docs/Qwen3.8-27B-跨设备横向对比.md)
- [RTX PRO 5000 llama.cpp Q6_K vs vLLM FP8](https://github.com/Deep-AI-Evo/qwen3.8-27b-q6k-fp8-rtx-pro5000-serving-benchmark)
- [DeepSeek-V4-Flash 双机部署实录](https://github.com/Deep-AI-Evo/deepseek-v4-flash-2x-dgx-spark)
- [DGX Spark 集群监控面板](https://github.com/Deep-AI-Evo/dgx-spark-cluster-monitor)

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
