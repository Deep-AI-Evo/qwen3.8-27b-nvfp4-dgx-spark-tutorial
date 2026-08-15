# Deploying Qwen3.8-27B-NVFP4 on NVIDIA DGX Spark (GB10)

> Deployment date: 2026-08-15　Machine: NVIDIA DGX Spark (GB10, Grace-Blackwell, aarch64)
> Model: <https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4>
> [中文版本](../Qwen3.8-27B-NVFP4-部署教程.md)

## 1. Model & Hardware

| Item | Details |
|---|---|
| Model | Qwen3.8-27B, dense 27B, native vision-language model (image/video understanding), hybrid architecture (Gated DeltaNet + Gated Attention), 64 layers, hidden size 5120 |
| Quantization | NVFP4 (FP4 weights + FP8 activations, compressed-tensors `nvfp4-pack-quantized`), requires Blackwell (sm_100+) |
| Weight size | ~23.4 GB (`model.safetensors` 22.6 GB + `model_mtp.safetensors` 0.85 GB MTP speculative-decoding head) |
| Context | 262,144 tokens natively, extensible to 1M via YaRN |
| This machine | NVIDIA GB10 (sm_121), 128GB unified memory (~119GB usable), driver 580.95.05 / CUDA 13.0 |

## 2. Environment Setup

OS: Ubuntu 24.04 (noble) aarch64, Python 3.12.3, gcc installed. **No root required anywhere.**

### 2.1 Create a virtual environment

```bash
cd ~/projects
python3 -m venv qwen38-env
qwen38-env/bin/pip install --upgrade pip
```

### 2.2 Install PyTorch (cu130, aarch64 CUDA wheels)

⚠️ The aarch64 torch wheels on PyPI are **CPU-only**. You must install from the official PyTorch cu130 index:

```bash
qwen38-env/bin/pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

### 2.3 Install vLLM

The `qwen3_5` architecture (`Qwen3_5ForConditionalGeneration`) requires vLLM ≥ 0.27:

```bash
qwen38-env/bin/pip install vllm==0.27.1 huggingface_hub hf_transfer
```

Verify:

```bash
qwen38-env/bin/python -c "import torch, vllm; print(torch.__version__, vllm.__version__, torch.cuda.is_available())"
# 2.13.0+cu130 0.27.1 True
```

### 2.4 Python headers without root

vLLM's Gated DeltaNet linear attention uses Triton, which compiles `cuda_utils.c` with gcc on first run and needs `Python.h`.
With root: `sudo apt install python3.12-dev`. Without root (the approach used here):

```bash
mkdir -p ~/projects/pydev && cd ~/projects/pydev
apt-get download python3.12-dev libpython3.12-dev
mkdir -p root && for d in *.deb; do dpkg -x "$d" root; done
```

Then point `CPATH` at the extracted headers when launching the server (see 3.2 / section 4).

## 3. Download the Model

If huggingface.co is unreachable (e.g. in China), use the hf-mirror.com mirror:

```bash
mkdir -p ~/projects/models/Qwen3.8-27B-NVFP4 && cd ~/projects/models/Qwen3.8-27B-NVFP4
BASE="https://hf-mirror.com/unsloth/Qwen3.8-27B-NVFP4/resolve/main"
for f in .gitattributes README.md chat_template.jinja config.json generation_config.json \
         model.safetensors.index.json preprocessor_config.json tokenizer.json \
         tokenizer_config.json video_preprocessor_config.json vocab.json; do
  curl -sfL -o "$f" "$BASE/$f"
done
curl -fL --retry 5 -C - -o model_mtp.safetensors "$BASE/model_mtp.safetensors"
curl -fL --retry 10 -C - -o model.safetensors "$BASE/model.safetensors"   # 22.6GB, ~7 min at 55MB/s
```

Equivalently: `HF_ENDPOINT=https://hf-mirror.com huggingface-cli download unsloth/Qwen3.8-27B-NVFP4`.

## 4. Launch the Inference Server

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

Flag notes:

- `PATH` must include the venv's `bin`, otherwise the engine subprocess can't find `ninja` (needed by Triton/torch.compile).
- `CPATH` is only needed when the system `python3.12-dev` package is not installed.
- `--reasoning-parser qwen3`: splits the thinking chain into the response's `reasoning` field; without it, reasoning text leaks into `content` together with a literal `</think>` marker.
- `--max-model-len 262144`: full native 256k context. KV cache on this machine: ~2.09M tokens → ~8 concurrent full-length requests (use 65536 if memory is tight: ~2.38M tokens / ~36 concurrent).
- `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`: enables the model's built-in MTP speculative head. Measured single-stream decode: ~12 → 17–23 tok/s (+92% in thinking mode), no throughput regression under concurrency.
- First launch takes ~4 minutes (weight load + torch.compile + warmup); subsequent launches ~2 minutes thanks to the AOT compile cache.
- Health check: `curl http://127.0.0.1:8000/health` → 200 means ready.

## 5. Usage (OpenAI-compatible API)

### 5.1 Chat (thinking mode, on by default)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "unsloth/Qwen3.8-27B-NVFP4",
  "messages": [{"role": "user", "content": "How many r in strawberry?"}],
  "max_tokens": 800,
  "temperature": 1.0, "top_p": 0.95, "top_k": 20
}'
```

Reasoning is in `choices[0].message.reasoning`, the final answer in `content`.

### 5.2 Disable thinking (faster)

```json
"chat_template_kwargs": {"enable_thinking": false},
"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5
```

### 5.3 Image understanding

```json
{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
  {"type": "text", "text": "Describe this image"}
]}
```

### 5.4 Recommended sampling parameters (official)

| Mode | temperature | top_p | top_k | presence_penalty |
|---|---|---|---|---|
| Thinking | 1.0 | 0.95 | 20 | 0.0 |
| Non-thinking | 0.7 | 0.80 | 20 | 1.5 |

## 6. Benchmarking

```bash
qwen38-env/bin/vllm bench serve \
  --model unsloth/Qwen3.8-27B-NVFP4 \
  --tokenizer ~/projects/models/Qwen3.8-27B-NVFP4 \
  --base-url http://127.0.0.1:8000 \
  --dataset-name random \
  --random-input-len 512 --random-output-len 128 \
  --num-prompts 32 --max-concurrency 8 --ignore-eos
```

⚠️ `--tokenizer` must be a **local path**; passing a model ID makes it try to reach huggingface.co and hang forever on networks that can't.

## 7. Troubleshooting (all actually hit during this deployment)

1. **`Python.h: No such file or directory`** — missing `python3.12-dev`; see 2.4.
2. **`FileNotFoundError: 'ninja'`** — the launch command's `PATH` didn't include the venv's `bin`.
3. **huggingface.co times out** — use the hf-mirror.com mirror (downloads) and a local tokenizer path (bench).
4. **Thinking text mixed into content with a leftover `</think>`** — add `--reasoning-parser qwen3`; in this vLLM version the reasoning field is named `reasoning`.
5. **Older vLLM reports `qwen3_5` unregistered** — you need vLLM ≥ 0.27.

## 8. Directory Layout

```
~/projects/
├── qwen38-env/                 # Python venv (torch 2.13.0+cu130, vllm 0.27.1)
├── pydev/                      # locally extracted python3.12-dev headers
├── models/Qwen3.8-27B-NVFP4/   # model weights (23.4GB)
└── qwen38-test/                # test scripts & results, server log
    ├── test_chat.py            # chat tests (thinking/non-thinking/code/multi-turn/streaming)
    ├── test_vision.py          # vision test (stdlib-only PNG generator included)
    └── vllm_server.log         # server log
```
