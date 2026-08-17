# Qwen3.8-27B × SGLang DSPARK 部署实录（NVIDIA DGX Spark / GB10）

> 部署日期：2026-08-17　部署机器：NVIDIA DGX Spark（GB10，Grace-Blackwell，aarch64，128GB 统一内存）
> 目标：复现 SGLang 官方 cookbook / NVIDIA 论坛宣称的 **34–38 tok/s**
> 结果：**复现成功**——英文代码峰值 42.8 tok/s，贪心稳定 34.7 tok/s
>
> 参考：
> - SGLang cookbook：<http://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B>
> - NVIDIA 论坛讨论帖：<https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257>

---

## 0. 结论速览

| 场景 | 速度 | 说明 |
|---|---|---|
| 英文代码（贪心） | **34.7 tok/s** | 达到官方宣称区间 |
| 英文代码（二叉树遍历） | **42.8 tok/s** | 单次峰值，DSPARK 接受率高 |
| 中文散文 | 17 tok/s | DSPARK 接受率仅 1.25–1.52，**内容敏感** |
| 中文代码 | 31–32 tok/s | 代码内容不挑语言，接受率回升 |
| 长上下文 decode | ~10 tok/s | 与上下文长度无关，稳定 |
| 8 并发总吞吐 | 77.7 tok/s | mem-fraction 0.50（~60GB）下饱和 |

**一句话：DSPARK 在 DGX Spark 上真实有效，但加速比高度依赖内容——代码/英文吃香，中文散文收益有限（draft 模型是英文数据训练的）。**

---

## 1. 方案选型：为什么是 SGLang + DSPARK

本仓库主文档记录的是 **vLLM + MTP** 路径（纯 pip，无需 Docker）。
SGLang 路径的差异：

| | vLLM + MTP（主文档） | SGLang + DSPARK（本文） |
|---|---|---|
| 部署方式 | 纯 pip venv | **Docker（官方 digest 镜像）** |
| 推测解码 | MTP（模型自带头） | DSPARK（外挂 draft 模型，markov 秩 256） |
| 配置难度 | 中（要装对 torch/vllm 版本） | 低（一条 docker run） |
| 峰值速度 | 见主文档实测 | 英文代码 34–43 tok/s |

DSPARK 是 DGX Spark 专属的推测解码方案：用一个轻量 draft 模型（`RadixArk/Qwen3.8-27B-DSpark`，markov_rank=256）先猜后验，猜中的 token 白赚。

---

## 2. 部署步骤

### 2.1 拉镜像（务必用 digest 锁定，不要用 dev/latest）

```bash
docker pull lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1
```

> ⚠️ **不要用 `lmsysorg/sglang:dev`**。我们实测 dev 镜像跑 DSPARK 会在首次推理时炸：
> ```
> RuntimeError: mat1 and mat2 shapes cannot be multiplied (7x5120 and 2560x248320)
> ```
> 这是 dev 镜像里 DSPARK draft 与主模型的 shape 不兼容（还在另一处触发 torch 第三方扩展编译错误）。
> 换成上面这个 digest 锁定镜像后一次通过。**digest 来自 NVIDIA 论坛帖子里官方人员给的配置。**

### 2.2 下载主模型（NVFP4 权重，~23GB）

主模型用 **RadixArk/Qwen3.8-27B-NVFP4**（HF 上另有 unsloth 版，本文用 RadixArk 版与 cookbook 对齐）。

国内直连 huggingface.co 会超时，走镜像站；**xet 协议在镜像站上会失败，必须禁用**：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

mkdir -p ~/models/qwen38-nvfp4
# 方式一：huggingface-cli（带上面的环境变量）
huggingface-cli download RadixArk/Qwen3.8-27B-NVFP4 \
  --local-dir ~/models/qwen38-nvfp4

# 方式二（更稳）：wget 逐个文件直拉
# 在 https://hf-mirror.com/RadixArk/Qwen3.8-27B-NVFP4/tree/main 拿到文件列表后循环 wget
```

draft 模型（RadixArk/Qwen3.8-27B-DSpark）不用手动下，首次启动时容器会经 HF_ENDPOINT 自动拉取（约几百 MB）。

### 2.3 启动（生产实测命令，从运行中容器 inspect 抄回）

```bash
docker run -d --name qwen38-nvfp4 \
  --gpus all --network host --ipc host \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e HF_HUB_DISABLE_XET=1 \
  -v ~/models/qwen38-nvfp4:/models \
  -v ~/models:/root/.cache/huggingface \
  lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1 \
  python3 -m sglang.launch_server \
    --trust-remote-code \
    --model-path /models \
    --tp-size 1 \
    --served-model-name qwen3.8-27b \
    --mem-fraction-static 0.50 \
    --attention-backend flashinfer \
    --chunked-prefill-size 8192 \
    --disable-prefill-cuda-graph \
    --cuda-graph-max-bs 4 \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark \
    --speculative-dspark-block-size 7 \
    --speculative-draft-model-quantization unquant \
    --mamba-scheduler-strategy extra_buffer \
    --enable-torch-compile \
    --torch-compile-max-bs 4 \
    --num-continuous-decode-steps 2 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --host 0.0.0.0 --port 30000
```

启动日志看到 `The server is fired up and ready to roll!` 即就绪（首次要编译 torch + 拉 draft 模型，等几分钟）。

### 2.4 验证

```bash
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Write a Python binary tree traversal"}],"max_tokens":512}'
```

---

## 3. 踩坑记录（每个都真金白银烧过时间）

### 坑 1：`--speculative-algorithm` 填什么

- 填 `MTP` → 报 `Unknown speculative algorithm name: MTP`。
  SGLang 把 MTP 映射到 EAGLE/EAGLE3 通路，但 Qwen3.8 的 MTP 头格式它不直接认。
- 填 `EAGLE` → draft 模型 `model_type=qwen3` 不被识别（SGLang 要 eagle 格式）。
- **正解：`DSPARK`**，并且**必须同时给** `--speculative-draft-model-path`（markov_rank=256 的专用 draft）和 `--speculative-draft-model-quantization unquant`。缺一个都起不来。

### 坑 2：镜像版本（见 2.1）

dev 镜像 = DSPARK draft shape 不兼容 + torch 扩展编译失败。**生产只认 digest。**

### 坑 3：HF 下载二连坑

1. huggingface.co 国内超时 → `HF_ENDPOINT=https://hf-mirror.com`
2. hf-mirror 上 xet 协议握手失败 → `HF_HUB_DISABLE_XET=1`，或干脆 wget 直拉文件

### 坑 4：调参反优化——mem-fraction 不是越高越好

想当然地「内存还有富余，把 mem-fraction 从 0.50 提到 0.70、chunked-prefill 从 8192 提到 16384」，结果 prefill **全面恶化**：

| 配置 | TTFT（短 prompt） | 32K prefill |
|---|---|---|
| 0.50 + 8192（最优） | **0.23s** | **8.54s** |
| 0.70 + 16384（反优化） | 3.99s（17×） | 14.79s（1.7×） |

**根因**：mem-fraction-static 是预留给 KV cache 的静态份额，开太高会挤压 prefill 阶段的临时工作显存，chunk 越大单次临时需求越高，两者叠加直接卡死 prefill。**DGX Spark 上 0.50 + 8192 是甜点，别动。**

---

## 4. 实测数据

### 4.1 单流 decode（DSPARK 加速比 = 内容敏感）

| 场景 | tok/s | DSPARK 接受长度 |
|---|---|---|
| 英文·写代码（贪心） | 34.7 | 高 |
| 英文·二叉树遍历（峰值） | 42.8 | 高 |
| 中文·散文写作 | 17 | 1.25–1.52 |
| 中文·写代码 | 31–32 | 高 |

**规律：draft 模型是英文语料训练的。代码（无论中英文注释）和英文文本的 n-gram 模式它猜得准；中文散文猜不准，DSPARK 近似白开。** 中文场景想上 30 tok/s，目前全网没有 SGLang 格式的中文训练 DSPARK draft（DimInfer 有中英双语 draft，但只出 GGUF/llama.cpp 格式，用不了）。

### 4.2 长上下文

- decode：**稳定 ~10 tok/s，与上下文长度无关**（这是推测解码关闭/低接受时的裸速底线）
- 32K prefill：8.54s（chunked 8192 限制下的正常水平）
- **缓存命中的重复前缀 prefill：18K–71K tok/s**（RadixAttention 前缀缓存生效，多轮对话实际体验远好于冷启动数字）

### 4.3 并发

mem-fraction 0.50 ≈ 60GB 可用，实测 **4–8 并发稳定**，8 并发时总吞吐 77.7 tok/s 饱和。100GB 内存的机器上不要为了堆并发去调 mem-fraction（见坑 4）。

---

## 5. 运维备忘

- 服务挂在 30000 端口，OpenAI 兼容端点 `/v1/chat/completions`
- 我们是通过 **SSH 隧道**（网关机 107 → 109:30000，launchctl 常驻）把它接进 New-API 网关统一分发的；网关侧 channel 的 base_url **必须写 `http://127.0.0.1:30000` 而不是 `localhost`**——Go 会把 `localhost` 解析成 `[::1]`（IPv6），而 SSH 隧道只绑了 IPv4，会报 `no route to host`。这个坑在网关侧，与 SGLang 无关但极易撞上。
- `reasoning-parser qwen3` 已开，thinking 内容走 `reasoning_content` 字段；客户端如果发 `reasoning_effort: "high"` 会 400（chat template 小毛病，可选打补丁，不影响正常调用）。

---

## 6. 一页纸复现清单

1. `docker pull` 第 2.1 节的 **digest 镜像**（不要 dev）
2. 下 RadixArk/Qwen3.8-27B-NVFP4 到 `~/models/qwen38-nvfp4`（hf-mirror + 禁 xet）
3. 抄 2.3 的 `docker run`，等 `ready to roll`
4. 英文代码贪心测速 ≈ 34 tok/s 即复现成功；中文散文只有 ~17 tok/s 是**预期行为**，不是部署错了
