# Qwen3.8-27B-NVFP4 本地部署教程（NVIDIA DGX Spark / GB10）

> 部署日期：2026-08-15　部署机器：NVIDIA DGX Spark（GB10，Grace-Blackwell，aarch64）
> 模型：<https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4>

## 1. 模型与硬件说明

| 项目 | 说明 |
|---|---|
| 模型 | Qwen3.8-27B，稠密 27B，原生视觉-语言模型（图文/视频理解），混合架构（Gated DeltaNet + Gated Attention），64 层，隐藏维度 5120 |
| 量化 | NVFP4（FP4 权重 + FP8 激活，compressed-tensors `nvfp4-pack-quantized` 格式），需 Blackwell（sm_100+）GPU |
| 权重体积 | 约 23.4 GB（`model.safetensors` 22.6 GB + `model_mtp.safetensors` 0.85 GB MTP 推测解码头） |
| 上下文 | 原生 262,144 tokens，可通过 YaRN 扩展至 1M |
| 本机硬件 | NVIDIA GB10（计算能力 sm_121），128GB 统一内存（可用约 119GB），驱动 580.95.05 / CUDA 13.0 |

## 2. 环境准备

系统：Ubuntu 24.04 (noble) aarch64，Python 3.12.3，已装 gcc。全程无需 root。

### 2.1 创建虚拟环境

```bash
cd ~/projects
python3 -m venv qwen38-env
qwen38-env/bin/pip install --upgrade pip
```

### 2.2 安装 PyTorch（cu130，aarch64 CUDA wheel）

PyPI 上的 aarch64 torch 是 CPU 版，必须从 PyTorch 官方 cu130 索引安装：

```bash
qwen38-env/bin/pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

### 2.3 安装 vLLM

`qwen3_5` 架构（`Qwen3_5ForConditionalGeneration`）需要 vLLM ≥ 0.27：

```bash
qwen38-env/bin/pip install vllm==0.27.1 huggingface_hub hf_transfer
```

验证：

```bash
qwen38-env/bin/python -c "import torch, vllm; print(torch.__version__, vllm.__version__, torch.cuda.is_available())"
# 输出：2.13.0+cu130 0.27.1 True
```

### 2.4 准备 Python 头文件（无 root 权限时）

vLLM 的 Gated DeltaNet 线性注意力走 Triton，首次运行需用 gcc 编译 `cuda_utils.c`，依赖 `Python.h`。
若有 root：`sudo apt install python3.12-dev` 即可。无 root 的替代方案（本次采用）：

```bash
mkdir -p ~/projects/pydev && cd ~/projects/pydev
apt-get download python3.12-dev libpython3.12-dev
mkdir -p root && for d in *.deb; do dpkg -x "$d" root; done
```

启动服务时通过 `CPATH` 指到头文件目录（见 3.2）。

## 3. 下载模型

国内网络直连 huggingface.co 不通，使用镜像站 hf-mirror.com：

```bash
mkdir -p ~/projects/models/Qwen3.8-27B-NVFP4 && cd ~/projects/models/Qwen3.8-27B-NVFP4
BASE="https://hf-mirror.com/unsloth/Qwen3.8-27B-NVFP4/resolve/main"
for f in .gitattributes README.md chat_template.jinja config.json generation_config.json \
         model.safetensors.index.json preprocessor_config.json tokenizer.json \
         tokenizer_config.json video_preprocessor_config.json vocab.json; do
  curl -sfL -o "$f" "$BASE/$f"
done
curl -fL --retry 5 -C - -o model_mtp.safetensors "$BASE/model_mtp.safetensors"
curl -fL --retry 10 -C - -o model.safetensors "$BASE/model.safetensors"   # 22.6GB，约 55MB/s 需 7 分钟
```

等价地也可以设置 `HF_ENDPOINT=https://hf-mirror.com` 后用 `huggingface-cli download unsloth/Qwen3.8-27B-NVFP4`。

## 4. 启动推理服务

```bash
cd ~/projects
PATH=$HOME/projects/qwen38-env/bin:$PATH \
CPATH=$HOME/projects/pydev/root/usr/include:$HOME/projects/pydev/root/usr/include/python3.12:$HOME/projects/pydev/root/usr/include/aarch64-linux-gnu/python3.12 \
qwen38-env/bin/vllm serve $HOME/projects/models/Qwen3.8-27B-NVFP4 \
  --served-model-name unsloth/Qwen3.8-27B-NVFP4 \
  --reasoning-parser qwen3 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --port 8000
```

要点说明：

- `PATH` 必须包含 venv 的 `bin`，否则引擎子进程找不到 `ninja`（Triton/torch.compile 需要）。
- `CPATH` 仅在未安装系统 `python3.12-dev` 时需要。
- `--reasoning-parser qwen3`：把思考链拆到响应的 `reasoning` 字段；不加该参数时，思考内容会和 `</think>` 标记一起混在 `content` 里。
- `--max-model-len 262144`：启用完整 256k 原生上下文；本机 KV cache 约 209 万 tokens，满 256k 上下文下约 8 路并发（64k 配置下为 238 万 / 36 路，内存紧张时可降回 65536）。
- `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`：启用模型自带的 MTP 推测解码头（`model_mtp.safetensors`），单流解码实测从约 12 tok/s 提升到 17–23 tok/s（思考模式约 +92%），并发吞吐无回退。对延迟不敏感、只要极限并发吞吐的场景可去掉。
- 首次启动约 4 分钟（权重加载 + torch.compile + warmup），第二次起有 AOT 编译缓存，约 2 分钟。
- 健康检查：`curl http://127.0.0.1:8000/health` 返回 200 即就绪。

## 5. 调用方式（OpenAI 兼容 API）

### 5.1 文本对话（思考模式，默认开启）

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "unsloth/Qwen3.8-27B-NVFP4",
  "messages": [{"role": "user", "content": "strawberry 里有几个 r？"}],
  "max_tokens": 800,
  "temperature": 1.0, "top_p": 0.95, "top_k": 20
}'
```

思考链在 `choices[0].message.reasoning`，最终回复在 `content`。

### 5.2 关闭思考模式（更快）

请求体加：

```json
"chat_template_kwargs": {"enable_thinking": false},
"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5
```

### 5.3 图片理解

```json
{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
  {"type": "text", "text": "描述这张图片"}
]}
```

### 5.4 推荐采样参数（官方建议）

| 模式 | temperature | top_p | top_k | presence_penalty |
|---|---|---|---|---|
| 思考模式 | 1.0 | 0.95 | 20 | 0.0 |
| 非思考模式 | 0.7 | 0.80 | 20 | 1.5 |

## 6. 压测

```bash
qwen38-env/bin/vllm bench serve \
  --model unsloth/Qwen3.8-27B-NVFP4 \
  --tokenizer ~/projects/models/Qwen3.8-27B-NVFP4 \
  --base-url http://127.0.0.1:8000 \
  --dataset-name random \
  --random-input-len 512 --random-output-len 128 \
  --num-prompts 32 --max-concurrency 8 --ignore-eos
```

注意：`--tokenizer` 必须给本地路径；给模型 ID 会去 huggingface.co 拉分词器，在无法直连的网络下会一直卡住。

## 7. 常见问题（本次实际踩坑记录）

1. **`Python.h: No such file or directory`** —— 缺 `python3.12-dev`，见 2.4。
2. **`FileNotFoundError: 'ninja'`** —— 启动命令的 `PATH` 未包含 venv 的 `bin`。
3. **直连 huggingface.co 超时** —— 用 hf-mirror.com 镜像（下载）和本地分词器路径（bench）。
4. **思考链混在 content 里** —— 加 `--reasoning-parser qwen3`；本版 vLLM 响应里思考字段名是 `reasoning`。
5. **低版本 vLLM 报 `qwen3_5` 未注册** —— 需要 vLLM ≥ 0.27。

## 8. 目录结构

```
~/projects/
├── qwen38-env/                 # Python 虚拟环境（torch 2.13.0+cu130, vllm 0.27.1）
├── pydev/                      # 本地解包的 python3.12-dev 头文件
├── models/Qwen3.8-27B-NVFP4/   # 模型权重（23.4GB）
└── qwen38-test/                # 测试脚本与结果、服务日志
    ├── test_chat.py            # 对话功能测试（思考/非思考/代码/多轮/流式）
    ├── test_vision.py          # 视觉理解测试（自带标准库 PNG 生成器）
    └── vllm_server.log         # 服务日志
```
