#!/bin/bash
# 双机 Qwen3.8-27B 启动脚本。用法: qwen_2node_start.sh worker (在115) | head (本机)
export PATH=/home/dgx/projects/qwen38-env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin
export CPATH=/home/dgx/projects/pydev/root/usr/include:/home/dgx/projects/pydev/root/usr/include/python3.12:/home/dgx/projects/pydev/root/usr/include/aarch64-linux-gnu/python3.12
export NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1 GLOO_SOCKET_IFNAME=enp1s0f1np1 NCCL_IGNORE_CPU_AFFINITY=1

ARGS=(
  /home/dgx/projects/models/Qwen3.8-27B-NVFP4
  --served-model-name unsloth/Qwen3.8-27B-NVFP4
  --reasoning-parser qwen3
  --tensor-parallel-size 2 --nnodes 2
  --master-addr 18.18.11.1 --master-port 25000
  --max-model-len 262144 --gpu-memory-utilization 0.85
  --max-num-batched-tokens 16384 --max-num-seqs 8
  --kernel-config '{"enable_flashinfer_autotune":false}'
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
)

if [ "$1" = "worker" ]; then
  export VLLM_HOST_IP=18.18.11.2
  exec /home/dgx/projects/qwen38-env/bin/vllm serve "${ARGS[@]}" --node-rank 1 --headless
else
  export VLLM_HOST_IP=18.18.11.1
  exec /home/dgx/projects/qwen38-env/bin/vllm serve "${ARGS[@]}" --node-rank 0 --host 0.0.0.0 --port 8000
fi
