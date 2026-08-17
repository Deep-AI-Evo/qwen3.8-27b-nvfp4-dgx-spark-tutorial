<div align="center">

# Qwen3.8-27B-NVFP4 × NVIDIA DGX Spark

**在 GB10（Grace-Blackwell）上从零部署 27B NVFP4 视觉-语言大模型**
**双引擎实战：vLLM 纯 pip · SGLang Docker + DSPARK · 256K 上下文 · 完整实测数据**

[![Hugging Face](https://img.shields.io/badge/🤗%20Model-Qwen3.8--27B--NVFP4-yellow)](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
[![vLLM](https://img.shields.io/badge/vLLM-0.27.1-blue)](https://github.com/vllm-project/vllm)
[![SGLang](https://img.shields.io/badge/SGLang-DSPARK-5B4FE0)](https://github.com/sgl-project/sglang)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0%2Bcu130-ee4c2c)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76b900)](https://developer.nvidia.com/cuda-toolkit)
[![Platform](https://img.shields.io/badge/Platform-DGX%20Spark%20GB10-green)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

### 🌐 **[Switch to English](README.en.md)**

</div>

---

## 📑 目录

- [这个仓库解决什么问题](#intro)
- [两条部署路径怎么选](#paths) ⭐
- [实测数据速览（单机）](#single)
- [双机部署结论（2× DGX Spark）](#dual)
- [快速开始](#quickstart)
- [Skill：一步一步照着执行的部署包](#skill) ⭐
- [把本仓库交给代码智能体（省 Token）](#agent)
- [日常使用建议](#daily)
- [跨设备横向对比：DGX Spark / RTX PRO 5000 / PRO 6000](#compare)
- [完整文档与仓库结构](#docs)
- [相关仓库与作者](#links)

---

<a id="intro"></a>

## 🎯 这个仓库解决什么问题

Qwen3.8-27B-NVFP4 是 2026-08 新发布的 27B 视觉-语言模型（256K 原生上下文）。
本仓库回答三个问题：

1. **怎么装** —— DGX Spark（aarch64/GB10）上两条路线任选：**vLLM 纯 pip**（无 root、无 Docker）
   或 **SGLang 官方 Docker + DSPARK 推测解码**（一条 docker run），含全部踩坑解法
2. **跑多快** —— 完整的 decode/prefill/并发/长上下文实测数据（含 256K 顶格大海捞针验证、
   DSPARK 在 256K 上下文下的实测衰减曲线）
3. **值不值** —— 双机有没有意义？和 RTX PRO 5000 / PRO 6000 比是什么水平？

> ⭐ **如果你只关心"这机器什么水平"**：直接跳到文末的
> [跨设备横向对比](#compare)——DGX Spark vs RTX PRO 5000 vs RTX PRO 6000，
> 但建议先快速浏览单机和双机数据，对比时才看得懂差距来自哪里。

---

<a id="paths"></a>

## 🔀 两条部署路径怎么选

| | 🅰️ vLLM 纯 pip（主教程） | 🅱️ SGLang Docker + DSPARK |
|---|---|---|
| 部署方式 | venv + pip，无 root 无容器 | 官方 digest 镜像，一条 `docker run` |
| 推测解码 | MTP（模型自带草稿头，×3） | DSPARK（外挂 draft 模型，DGX Spark 专属） |
| 配置难度 | 中（torch/vllm 版本要装对，教程已踩平） | 低（参数照抄即可） |
| 单流 decode（短上下文） | 思考 22.7 / 非思考 16.7 tok/s | 英文代码 **34–43 tok/s**，散文 17–20 tok/s |
| 速度特点 | 各场景均衡 | **内容敏感**：代码/英文吃香，中文散文收益低 |
| 长上下文 | 已验证 256K 顶格 + 大海捞针 | 256K 实测见下文衰减曲线 |
| 适合谁 | 要稳定均衡、要已验证的 256K 全档数据 | 主写代码/英文场景，想省心上 Docker |

**一句话**：主写代码选 🅱️（DSPARK 对代码内容加速明显）；要均衡通用、中文场景多选 🅰️。
两条路径的权重相同（NVFP4，~23GB），可以都装、按需切换。

> DSPARK 中文散文只有 ~17 tok/s 不是部署错了——draft 模型是英文语料训练的，
> 中文 n-gram 猜不准。代码内容（无论中英文注释）都能上 30+ tok/s。

---

<a id="single"></a>

## 📊 实测数据速览（单机）

> 测试环境：DGX Spark（GB10，128GB 统一内存）· 256K 上下文

### 🅰️ vLLM 0.27.1 + MTP ×3（torch 2.13.0+cu130）

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

### 🅱️ SGLang + DSPARK（官方 digest 镜像，mem-fraction 0.50 + chunked 8192 甜点配置）

**⚡ 单流解码速度（内容敏感）**

| 场景 | 速度 | DSPARK 接受情况 |
|---|---|---|
| 英文·写代码（贪心） | **34.7 tok/s** | 高 |
| 英文·二叉树遍历（峰值） | **42.8 tok/s** | 高，达到官方 34–38 宣称区间并超出 |
| 中文·写代码 | 31–32 tok/s | 高（代码内容不挑语言） |
| 中文·散文 | ~17 tok/s | 低（1.25–1.52），draft 英文语料训练所致 |
| 散文/summary（短上下文） | 17.7–20 tok/s | 中 |

**📏 上下文长度衰减（单并发，随机前缀无前缀缓存，散文内容）**

| 上下文 | Prompt tokens | TTFT | Prefill 速度 | Decode 速度 |
|---|---|---|---|---|
| 1K | 1,022 | 0.42 s | ≈2,400 tok/s | 17.9–20.0 tok/s |
| 32K | 32,022 | 19.3 s | 1,660 tok/s | 17.7 tok/s |
| 64K | 64,022 | 45.5 s | 1,408 tok/s | 15.4 tok/s |
| 100K | 100,022 | 85.4 s | 1,172 tok/s | 16.1 tok/s |
| 150K | 150,022 | 157 s（2.6 min） | 953 tok/s | 15.6 tok/s |
| 200K | 200,022 | 249 s（4.1 min） | 805 tok/s | 13.1 tok/s |
| **245K（≈顶格）** | 245,022 | **347 s（5.8 min）** | 705 tok/s（-71%） | **14.0 tok/s（仅 -30%）** |

> 🔍 **与 vLLM 同档几乎重合**（100K prefill 1,172 vs 1,230；顶格 705@245K vs 715@261K）——
> 散文内容下 DSPARK 加速有限，两引擎裸速趋同；SGLang 的速度优势集中在代码/英文短上下文。
> decode 从 1K 到 245K 只衰减 30%（20→14），**DSPARK 长上下文不坍塌**。
> 245K 一次灌入约 5.8 分钟，配合 RadixAttention 适合"一次灌入 + 多轮问答"（后续轮次 prefill 几乎免费）。

> ⚙️ **配置教训**：别把 mem-fraction 从 0.50 往 0.70 调——KV 池会挤压 prefill 工作区，
> TTFT 恶化 17 倍（0.23s→3.99s），32K prefill 从 8.5s 恶化到 14.8s。**0.50 + 8192 就是甜点。**

> 🔄 RadixAttention 前缀缓存生效时，重复前缀 prefill 达 18K–71K tok/s——
> 多轮对话实际体验远好于上表冷启动数字。

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

### 🅰️ vLLM 路径（完整步骤见 [部署教程](docs/Qwen3.8-27B-NVFP4-部署教程.md)）

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

### 🅱️ SGLang DSPARK 路径（完整步骤见 [部署实录](docs/Qwen3.8-27B-SGLang-DSpark部署实录.md)）

```bash
# 1. 拉 digest 锁定镜像（不要用 dev！DSPARK 会 shape 不兼容）
docker pull lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1

# 2. 下载模型（国内必做两个环境变量）
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
huggingface-cli download RadixArk/Qwen3.8-27B-NVFP4 --local-dir ~/models/qwen38-nvfp4

# 3. 一条命令启动（生产实测参数，mem-fraction 0.50 是甜点别调）
docker run -d --name qwen38-nvfp4 --gpus all --network host --ipc host \
  -e HF_ENDPOINT=https://hf-mirror.com -e HF_HUB_DISABLE_XET=1 \
  -v ~/models/qwen38-nvfp4:/models -v ~/models:/root/.cache/huggingface \
  lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1 \
  python3 -m sglang.launch_server --trust-remote-code --model-path /models \
    --tp-size 1 --served-model-name qwen3.8-27b \
    --mem-fraction-static 0.50 --attention-backend flashinfer \
    --chunked-prefill-size 8192 --disable-prefill-cuda-graph --cuda-graph-max-bs 4 \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark \
    --speculative-dspark-block-size 7 --speculative-draft-model-quantization unquant \
    --mamba-scheduler-strategy extra_buffer \
    --enable-torch-compile --torch-compile-max-bs 4 \
    --num-continuous-decode-steps 2 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --host 0.0.0.0 --port 30000
```

服务就绪后分别访问 `http://127.0.0.1:8000` / `http://127.0.0.1:30000`（OpenAI 兼容 API）。

---

<a id="skill"></a>

## 📦 Skill：一步一步照着执行的部署包

[`skill/`](skill/SKILL.md) 目录是一个可直接交给代码智能体（或自己动手）执行的部署 skill，
把 SGLang DSPARK 路径拆成 **检查 → 拉镜像 → 下模型 → 启动 → 验证 → 测速** 六步，
每步一个脚本，全部下载链接和达标参考内置：

```
skill/
├── SKILL.md                    # 主文档：六步流程 + 常见问题速查表
└── scripts/
    ├── 01_download_model.sh    # 模型下载（hf-mirror + 禁 xet，wget 兜底）
    ├── 02_start_server.sh      # docker run 启动 + 等待就绪
    ├── 03_verify.py            # 三项验证：对话 / thinking / OpenAI 格式
    └── 04_bench.py             # 测速：1K→245K 全档位 TTFT/prefill/decode
```

手动跟着做，或直接把 `skill/SKILL.md` 发给代码智能体让它逐步执行——
每一步都有**完成判定**，过不了就别往下走，配置时间从半天压到 30 分钟。

---

<a id="agent"></a>

## 🤖 把本仓库交给代码智能体，一键复现部署（省 Token）

不必照着文档手动敲，也**不建议让 Agent 从零摸索**——本教程沉淀了两条路径的全部真实踩坑
（PyPI 的 aarch64 torch 是 CPU 版、HF 直连不通要走镜像、无 root 补 Python.h、PATH 缺 ninja、
reasoning parser 未开启、SGLang dev 镜像 DSPARK shape 不兼容、mem-fraction 0.70 反优化……），
Agent 自行试错每一步都在烧 Token。

**🅱️ SGLang DSPARK 路径（推荐，最省心）：复制下面这段话发给任意代码智能体**
（Kimi Code CLI、Claude Code、Cursor、Codex 等）：

```text
我的机器是 NVIDIA DGX Spark（GB10，aarch64）。请阅读仓库
https://github.com/Deep-AI-Evo/qwen3.8-27b-nvfp4-dgx-spark-tutorial
中 skill/SKILL.md，严格按其中 Step 0-5 逐步执行（scripts/ 里的脚本可直接用），
在本机部署 Qwen3.8-27B-NVFP4 + SGLang DSPARK 服务；每步按"完成判定"确认后再继续，
最后运行验证与快速测速并汇报结果。
```

**🅰️ vLLM 路径：**

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

- **并发**：交互场景 ≤4 路（vLLM 单流 ≥15 tok/s）；8 路是交互上限；批量任务用 16+ 吃满吞吐。
  SGLang 侧 mem-fraction 0.50 实测 4–8 路并发、8 路聚合 77.7 tok/s 饱和
- **长上下文**：100K 以内随意用；200K 级适合"一次灌入 + 多轮问答"
- **前缀缓存**：vLLM 路径该混合架构暂不支持前缀缓存，避免反复重灌同一长文档；
  SGLang 路径有 RadixAttention，重复前缀 prefill 极快（18K–71K tok/s），多轮长文档更友好
- **模式**：复杂推理/代码开思考模式；闲聊/翻译/摘要关思考（`enable_thinking=false`）更快
- **场景选引擎**：写代码/英文为主用 SGLang DSPARK（34+ tok/s）；中文散文/通用对话用 vLLM（均衡）

---

<a id="compare"></a>

## 🆚 跨设备横向对比（DGX Spark / RTX PRO 5000 / RTX PRO 6000）

> 摘要自 [三设备横向对比](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark/blob/main/docs/Qwen3.8-27B-跨设备横向对比.md)（完整档位与测试方法见该文档）。
> ⚠️ 口径差异：三台设备 vLLM 版本 / MTP 设置 / 操作系统不完全相同，数量级参考意义大于精确对比。

**单并发 decode（tok/s）**

| 上下文 | DGX Spark NVFP4（本仓库） | **PRO 5000 SGLang NVFP4+DSPARK** | PRO 5000 FP8（无MTP → +MTP） | PRO 5000 NVFP4（无MTP → +MTP） | PRO 5000 Q6_K（无MTP → +MTP） | PRO 6000 FP8+MTP | PRO 6000 NVFP4+MTP | PRO 6000 NVFP4 无MTP | PRO 6000 Q6_K |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 短（~1-11K） | ~21 | **85.2** | 37.3 → 43.2 | 49.6 → 63.6 | 39.7 → 61.9 | 95.6 | 100.2 | 58.6 | 55.4 |
| ~148K | — | **71.4** 🏆 | 26.4 → 15.3 ❌ | 42.1 → 57.8 ✅ | 39.9 → 40.0 | — | — | — | — |
| ~200K | 14.2 | — | 26.4（无MTP） | 42.1（无MTP） | 39.9 | 19.1 | 18.2 | **43.7** | 35.7 |

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
- **长上下文 decode 之王（更新）**：PRO 5000 + SGLang + DSPARK 71.4 t/s @148K（引擎优化 > 硬件差距，
  72GB 卡反超 96GB 卡的 vLLM 配置）；MTP 长上下文收益因后端/n 值而反转（PRO 6000 n=2 在 200K 崩盘），
  用前实测自己的配置
- 📉 **MTP 长上下文衰减的根因与跨引擎验证**（vLLM Issue #47602：单层草稿头容量不足，接受率随上下文
  坍塌）：见 [PRO 6000 横向对比文档的 MTP 专题](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark/blob/main/docs/Qwen3.8-27B-跨设备横向对比.md)
- **DGX Spark 的价值不在绝对速度**：128GB 统一内存跑满 256K 上下文、安静低功耗的桌面形态、
  以及双机扩展的灵活性（见[双机结论](#dual)）

---

<a id="docs"></a>

## 📚 完整文档与仓库结构

| 文档 | 内容 |
|---|---|
| 📦 [skill/SKILL.md](skill/SKILL.md) | **SGLang DSPARK 六步部署 skill**：检查/拉镜像/下模型/启动/验证/测速，含全部下载链接与达标参考 |
| 📖 [部署教程](docs/Qwen3.8-27B-NVFP4-部署教程.md) | 🅰️ vLLM 路径：环境准备、镜像下载、启动参数详解、API 调用示例、5 个踩坑实录 |
| 📈 [测试报告](docs/Qwen3.8-27B-NVFP4-测试报告.md) | 🅰️ vLLM 路径：7 项功能测试、并发/长上下文压测、MTP 优化对比、日常使用建议 |
| 🐳 [SGLang DSPARK 部署实录](docs/Qwen3.8-27B-SGLang-DSpark部署实录.md) | 🅱️ SGLang 路径：digest 镜像选型、DSPARK draft 配置、mem-fraction 反优化教训、中文场景实测 |
| 🖥🖥 [双机部署实测](docs/Qwen3.8-27B-双机部署实测.md) | 2× DGX Spark TP=2：并发/容量收益、mp 模式与 Ray 的坑、prefill/解码对比 |

```
├── README.md / README.en.md      # 中文 / English
├── skill/                        # SGLang DSPARK 六步部署 skill（含脚本）
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
