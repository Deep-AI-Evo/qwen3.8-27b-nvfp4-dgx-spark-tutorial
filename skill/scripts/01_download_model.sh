#!/usr/bin/env bash
# Step 2: 下载 Qwen3.8-27B-NVFP4 主模型（~23GB）到 ~/models/qwen38-nvfp4
# 模型页: https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4
# 镜像页: https://hf-mirror.com/RadixArk/Qwen3.8-27B-NVFP4
set -euo pipefail

MODEL_REPO="RadixArk/Qwen3.8-27B-NVFP4"
DEST="${HOME}/models/qwen38-nvfp4"

# 国内网络两个必做项：走镜像站 + 禁用 xet（xet 在镜像站握手失败）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

mkdir -p "${DEST}"
echo ">> 下载 ${MODEL_REPO} -> ${DEST}"
echo ">> HF_ENDPOINT=${HF_ENDPOINT}  HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"

if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${MODEL_REPO}" --local-dir "${DEST}"
elif python3 -c "import huggingface_hub" 2>/dev/null; then
    python3 -m huggingface_hub.commands.huggingface_cli download "${MODEL_REPO}" --local-dir "${DEST}" || \
    python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("${MODEL_REPO}", local_dir="${DEST}")
EOF
else
    echo "!! 未找到 huggingface-cli，回退 wget 直拉文件列表"
    for f in $(curl -sL "https://hf-mirror.com/api/models/${MODEL_REPO}" | python3 -c "import sys,json; [print(s['rfilename']) for s in json.load(sys.stdin)['siblings']]"); do
        echo ">> wget ${f}"
        wget -c "https://hf-mirror.com/${MODEL_REPO}/resolve/main/${f}" -O "${DEST}/${f}"
    done
fi

echo ">> 完成。文件清单："
ls -lh "${DEST}" | head -20
du -sh "${DEST}"
echo ">> 期望总大小约 23GB，含 config.json / tokenizer.json / *.safetensors"
