# Qwen3.8-27B-NVFP4 双机部署与实测（2× DGX Spark，TP=2）

> 日期：2026-08-15　环境：2× NVIDIA DGX Spark（GB10，aarch64），QSFP 直连 RoCE 200G
> 单机教程见 [Qwen3.8-27B-NVFP4-部署教程.md](Qwen3.8-27B-NVFP4-部署教程.md)

## 结论先行

27B 体量单机就装得下，双机 TP=2 的价值**不在单流速度，而在并发与容量**：

| 维度 | 单机 | 双机 TP=2 | 判断 |
|---|---|---|---|
| 单流解码（思考模式） | 22.7 tok/s | 20.6–22.0 tok/s | 持平略降（跨机同步开销） |
| c=4 聚合吞吐 | 44 tok/s | **65.4 tok/s（+49%）** | 双机明显优势 |
| 256K 上下文并发路数 | 8 | **18** | KV 池翻倍 |
| prefill 8K | ≈1900 tok/s | ≈1358 tok/s | 短输入双机吃亏 |
| prefill 100K | 1230 tok/s（83s） | **1325 tok/s（73s）** | 长上下文双机反超 |

**建议**：日常单/双路交互用单机；多 agent 并发、长文档多会话再上双机。

## 部署方式（与单机不同的关键点）

双机用 vLLM 原生 mp 多机模式（`--nnodes`），**不要用 Ray**（见踩坑 1）。
网络与集群准备（静态 IP、RoCE、NCCL 验证）见
[DeepSeek 双机实录仓库](https://github.com/Deep-AI-Evo/deepseek-v4-flash-2x-dgx-spark)。

启动脚本（两台相同环境：venv + 模型路径一致）：

```bash
# 关键参数：mp 模式 + 关闭 flashinfer autotune + MTP×3
vllm serve /path/to/Qwen3.8-27B-NVFP4 \
  --served-model-name unsloth/Qwen3.8-27B-NVFP4 \
  --reasoning-parser qwen3 \
  --tensor-parallel-size 2 --nnodes 2 \
  --master-addr 18.18.11.1 --master-port 25000 \
  --max-model-len 262144 --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 16384 --max-num-seqs 8 \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
# worker（第二台）：加 --node-rank 1 --headless，先启动
# head：加 --node-rank 0 --host 0.0.0.0 --port 8000，后启动
```

完整脚本（含 NCCL/编译环境变量）：本仓库 `tests/qwen_2node_start.sh`。

## 实测数据

**解码（自然文本，思考模式，MTP×3）**

| 并发 | 单流 | 聚合 |
|---|---|---|
| 1 | 20.6–22.0 tok/s | 22 tok/s |
| 4 | 18.3 tok/s | 65.4 tok/s |

**Prefill（单并发，输出 32 tokens）**

| 输入 | 耗时 | 速度 |
|---|---|---|
| 1K | 1.0 s | ≈1000 tok/s |
| 8K | 5.8 s | ≈1358 tok/s |
| 32K | 20.7 s | ≈1504 tok/s |
| 100K | 73.3 s | ≈1325 tok/s |

**稳定性**：压测全程 + 之后服务持续健康（ray 模式对比见下）。

## 踩坑记录

| # | 症状 | 根因 | 解法 |
|---|---|---|---|
| 1 | Ray 模式（`--distributed-executor-backend ray`）能启动，但一推理 worker 就无声死亡（无 Python 栈） | stock vLLM 0.27.1 对 qwen3_5 混合架构的多机 Ray 路径有 bug | 换 vLLM 原生 mp 多机模式（`--nnodes 2 --node-rank`，worker 先起） |
| 2 | mp 模式首次启动卡 40+ 分钟 | FlashInfer autotune 在该模式下全量重跑内核配置 | `--kernel-config '{"enable_flashinfer_autotune":false}'`，启动 4.5 分钟 |
| 3 | `--attention-backend flash_attn` 被拒 | 此 NVFP4 模型 KV 强制 FP8，SM121 上只有 FLASHINFER 后端支持 FP8 KV | 用默认后端（FLASHINFER），不要指定 flash_attn |
| 4 | head 起不来，报端口占用 | 之前 kill 的主进程残留子进程持有 TCPStore 端口 | `ss -tlnp` 找到残留 PID 清掉（`pkill -f` 注意别匹配到自己的包装命令） |

参考了 [mmike87/spark-vllm-docker 的 Qwen 双机分支](https://github.com/mmike87/spark-vllm-docker/tree/Qwen36-27B-FP8-MTP-Dual)
（flash_attn 后端、qwen3_next_mtp、fastsafetensors 等思路），其中 flash_attn 一条经实测不适用于本模型（坑 #3）。

## 运维

- 启动顺序：**worker（115）先，head 后**；停止时先停 head
- 与 DeepSeek 双机栈**不能同时运行**（内存不够），切换前停掉另一套
- 编译/调优缓存已在两台落盘，重启约 4–5 分钟
