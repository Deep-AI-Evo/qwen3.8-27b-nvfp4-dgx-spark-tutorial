---
name: qwen38-dgx-spark-sglang
description: 在 NVIDIA DGX Spark（GB10, aarch64）上部署 Qwen3.8-27B-NVFP4 + SGLang DSPARK 推测解码的完整 skill。当用户需要在 DGX Spark 上部署 Qwen3.8-27B、复现官方 34-38 tok/s 速度、配置 DSPARK 推测解码、或排查 SGLang 部署问题时使用。含全部下载链接、生产级启动参数、验证与 256K 长上下文测速脚本。
---

# Qwen3.8-27B × SGLang DSPARK × DGX Spark 部署 Skill

**目标**：30 分钟内从零到可用服务，复现官方 34–38 tok/s（英文代码场景）。

**预期成果**：
- OpenAI 兼容服务运行在 `http://<机器IP>:30000/v1`
- 模型：Qwen3.8-27B-NVFP4（256K 上下文）+ DSPARK 推测解码
- 英文代码 decode 34+ tok/s；中文散文 ~17 tok/s 属正常（draft 是英文语料训练）

---

## 执行前检查（Step 0）

逐项确认，不满足就先解决：

```bash
# 1. 机器是 DGX Spark（GB10, aarch64, 128GB 统一内存）
uname -m                                    # 期望: aarch64
nvidia-smi --query-gpu=name --format=csv,noheader   # 期望: NVIDIA GB10

# 2. Docker 可用且能访问 GPU
docker info >/dev/null && echo docker-ok
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi -L

# 3. 磁盘剩余 ≥ 80GB（模型 23GB + 镜像 ~15GB + draft + 缓存）
df -h ~ | tail -1
```

**已知硬性约束**：
- 必须用 **digest 锁定镜像**，`lmsysorg/sglang:dev` 跑 DSPARK 会报 `mat1 and mat2 shapes cannot be multiplied (7x5120 and 2560x248320)`——别试。
- 推测算法名必须写 `DSPARK`（写 `MTP` 报 `Unknown speculative algorithm name`，写 `EAGLE` 报 draft 格式不认）。
- 国内网络必须走 hf-mirror 并禁用 xet（两个环境变量，脚本已内置）。

---

## Step 1：拉取 Docker 镜像（~15GB）

```bash
docker pull lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1
```

镜像来源：NVIDIA 论坛官方人员给出的 DGX Spark 可用配置
（<https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257>）。

---

## Step 2：下载模型（~23GB）

运行脚本：

```bash
bash scripts/01_download_model.sh
```

脚本做的事（也可手动执行）：

```bash
export HF_ENDPOINT=https://hf-mirror.com     # 国内必做，直连 huggingface.co 会超时
export HF_HUB_DISABLE_XET=1                  # 必做，hf-mirror 上 xet 协议握手失败

mkdir -p ~/models/qwen38-nvfp4
huggingface-cli download RadixArk/Qwen3.8-27B-NVFP4 --local-dir ~/models/qwen38-nvfp4
```

**下载链接**：
- 主模型：<https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4>（镜像 <https://hf-mirror.com/RadixArk/Qwen3.8-27B-NVFP4>）
- DSPARK draft（无需手动下，首次启动容器自动拉取）：<https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark>
- 若 huggingface-cli 失败：到镜像站文件列表页用 `wget` 逐个下载到 `~/models/qwen38-nvfp4/`

**完成判定**：`~/models/qwen38-nvfp4/` 下有 `config.json`、`tokenizer.json` 和若干 `.safetensors`，总大小约 23GB。

---

## Step 3：启动服务

```bash
bash scripts/02_start_server.sh
```

脚本使用以下**生产实测参数**（从运行中容器抄回，每项都有用，别乱改）：

| 参数 | 值 | 为什么 |
|---|---|---|
| `--mem-fraction-static` | 0.50 | ⚠️ 提到 0.70 会让 TTFT 恶化 17 倍（0.23s→3.99s），KV 池与 prefill 工作区争内存 |
| `--chunked-prefill-size` | 8192 | 与 0.50 配套，16384 同样恶化 prefill |
| `--speculative-algorithm` | DSPARK | DGX Spark 专属推测解码 |
| `--speculative-draft-model-path` | RadixArk/Qwen3.8-27B-DSpark | markov_rank=256 专用 draft，必填 |
| `--speculative-draft-model-quantization` | unquant | draft 不做量化，必填 |
| `--speculative-dspark-block-size` | 7 | DSPARK block 大小 |
| `--enable-torch-compile` + `--torch-compile-max-bs 4` | — | decode 提速 |
| `--attention-backend` | flashinfer | GB10 上最优后端 |
| `--reasoning-parser` | qwen3 | thinking 内容走 `reasoning_content` 字段 |

**就绪判定**（首次约 5-10 分钟：拉 draft 模型 + torch compile）：

```bash
docker logs -f qwen38-nvfp4 2>&1 | grep "ready to roll"
```

---

## Step 4：验证

```bash
python3 scripts/03_verify.py
```

验证三项：基础对话、thinking 模式（reasoning_content 非空）、OpenAI 端点格式。
全部 PASS 即部署成功。

手动快速验证：

```bash
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

---

## Step 5：测速（可选，含 256K 长上下文）

```bash
# 快速档（约 1 分钟）
python3 scripts/04_bench.py 1000 32000

# 全档位 1K→245K（约 15-25 分钟，测 TTFT/prefill/decode 衰减曲线）
python3 scripts/04_bench.py
```

**达标参考**（DGX Spark 实测）：

| 场景 | 期望值 |
|---|---|
| 英文代码 decode（贪心） | ≥ 34 tok/s |
| 散文/summary decode @1K | 17–20 tok/s |
| prefill @1K | ≥ 2,000 tok/s |
| prefill @100K | ≈1,170 tok/s（TTFT ~85s） |
| prefill @245K 顶格 | ≈700 tok/s（TTFT ~5.8min） |
| decode @245K 顶格 | ≈14 tok/s（1K→245K 仅衰减 30%） |
| 256K 顶格可用 | 是（`max_position_embeddings=262144`，启动无需指定） |

---

## 常见问题速查

| 症状 | 原因与解法 |
|---|---|
| `mat1 and mat2 shapes cannot be multiplied (7x5120 and 2560x248320)` | 用了 dev 镜像 → 换 Step 1 的 digest 镜像 |
| `Unknown speculative algorithm name: MTP` | 算法名改 `DSPARK` |
| draft 模型 `model_type=qwen3` 不识别 | 你走了 EAGLE 通路；DSPARK 必须同时给 draft-model-path + quantization unquant |
| HF 下载超时 | `HF_ENDPOINT=https://hf-mirror.com` |
| hf-mirror 下载卡死/握手失败 | `HF_HUB_DISABLE_XET=1` 或 wget 直拉 |
| 调 mem-fraction 0.70 后 prefill 暴慢 | 预期行为，回 0.50+8192 |
| 客户端报 `reasoning_effort: high` 400 | chat template 小毛病，去掉该参数即可，不影响正常使用 |
| 中文只有 ~17 tok/s | DSPARK draft 是英文语料训练，中文散文接受率低是预期；代码类内容（无论中英文）仍能 30+ tok/s |

---

## 参考链接

- SGLang 官方 cookbook：<http://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B>
- NVIDIA 论坛部署讨论：<https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257>
- 主模型：<https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4>
- DSPARK draft：<https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark>
- 完整踩坑与实测数据：仓库 `docs/Qwen3.8-27B-SGLang-DSpark部署实录.md`
