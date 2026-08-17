#!/usr/bin/env bash
# Step 3: 启动 Qwen3.8-27B-NVFP4 + SGLang DSPARK 服务（DGX Spark 生产实测参数）
# 镜像来源: NVIDIA 论坛官方配置 https://forums.developer.nvidia.com/t/380257
set -euo pipefail

IMAGE="lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1"
NAME="qwen38-nvfp4"
MODEL_DIR="${HOME}/models/qwen38-nvfp4"
PORT="${PORT:-30000}"

[ -d "${MODEL_DIR}" ] || { echo "!! 模型目录不存在: ${MODEL_DIR}，先跑 01_download_model.sh"; exit 1; }

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
    echo ">> 已存在容器 ${NAME}，先删除（数据在挂载目录，不丢）"
    docker rm -f "${NAME}" >/dev/null
fi

echo ">> 启动 ${NAME}（端口 ${PORT}，首次就绪需 5-10 分钟：拉 draft 模型 + torch compile）"
docker run -d --name "${NAME}" \
  --gpus all --network host --ipc host \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e HF_HUB_DISABLE_XET=1 \
  -v "${MODEL_DIR}:/models" \
  -v "${HOME}/models:/root/.cache/huggingface" \
  --restart unless-stopped \
  "${IMAGE}" \
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
    --host 0.0.0.0 --port "${PORT}"

echo ">> 等待就绪（Ctrl-C 退出等待不影响服务启动）..."
timeout 900 bash -c "until docker logs ${NAME} 2>&1 | grep -q 'ready to roll'; do sleep 10; done" \
    && echo ">> ✅ 服务就绪: http://127.0.0.1:${PORT}/v1" \
    || { echo "!! 等待超时，查看日志: docker logs -f ${NAME}"; exit 1; }
