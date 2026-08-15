# Qwen3.8-27B-NVFP4 on 2× DGX Spark (TP=2) — Deployment & Benchmarks

> Date: 2026-08-15　Setup: 2× NVIDIA DGX Spark (GB10, aarch64), QSFP direct-attach RoCE 200G
> Single-node guide: [deployment-tutorial.md](deployment-tutorial.md)

## TL;DR

A 27B model fits on one node, so dual-node TP=2 is **not about single-stream speed —
it's about concurrency and capacity**:

| Metric | Single node | 2-node TP=2 | Verdict |
|---|---|---|---|
| Single-stream decode (thinking) | 22.7 tok/s | 20.6–22.0 tok/s | Tie to slightly worse (fabric sync overhead) |
| c=4 aggregate throughput | 44 tok/s | **65.4 tok/s (+49%)** | Clear dual-node win |
| Concurrent full-256K requests | 8 | **18** | KV pool doubled |
| Prefill 8K | ≈1900 tok/s | ≈1358 tok/s | Dual-node penalty on short inputs |
| Prefill 100K | 1230 tok/s (83s) | **1325 tok/s (73s)** | Dual-node pulls ahead at long context |

**Recommendation**: single node for daily interactive use; dual node pays off for
multi-agent concurrency and multi-session long-document work.

## How it's deployed (diffs from single-node)

Use vLLM's native mp multi-node mode (`--nnodes`) — **not Ray** (see pitfall 1).
For fabric/cluster prep (static IPs, RoCE, NCCL verification) see the companion repo
[DeepSeek 2× DGX Spark field notes](https://github.com/Deep-AI-Evo/deepseek-v4-flash-2x-dgx-spark).

```bash
# key args: mp mode + flashinfer autotune off + MTP×3
vllm serve /path/to/Qwen3.8-27B-NVFP4 \
  --served-model-name unsloth/Qwen3.8-27B-NVFP4 \
  --reasoning-parser qwen3 \
  --tensor-parallel-size 2 --nnodes 2 \
  --master-addr 18.18.11.1 --master-port 25000 \
  --max-model-len 262144 --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 16384 --max-num-seqs 8 \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
# worker (2nd node): add --node-rank 1 --headless, start FIRST
# head: add --node-rank 0 --host 0.0.0.0 --port 8000, start SECOND
```

Full script with NCCL/compile env vars: `tests/qwen_2node_start.sh`.

## Benchmarks

**Decode (natural text, thinking mode, MTP×3)**

| Concurrency | Per-stream | Aggregate |
|---|---|---|
| 1 | 20.6–22.0 tok/s | 22 tok/s |
| 4 | 18.3 tok/s | 65.4 tok/s |

**Prefill (single stream, 32 output tokens)**

| Input | Time | Speed |
|---|---|---|
| 1K | 1.0 s | ≈1000 tok/s |
| 8K | 5.8 s | ≈1358 tok/s |
| 32K | 20.7 s | ≈1504 tok/s |
| 100K | 73.3 s | ≈1325 tok/s |

**Stability**: healthy through the full benchmark and beyond (contrast with Ray below).

## Pitfalls

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Ray mode starts but workers die silently on first inference (no Python stack) | stock vLLM 0.27.1's multi-node Ray path is buggy for the qwen3_5 hybrid arch | Use vLLM native mp multi-node (`--nnodes 2 --node-rank`), worker first |
| 2 | First mp-mode boot stalls 40+ min | FlashInfer autotune re-runs the full kernel sweep in this mode | `--kernel-config '{"enable_flashinfer_autotune":false}'` → 4.5 min boot |
| 3 | `--attention-backend flash_attn` rejected | This NVFP4 build forces FP8 KV; on SM121 only FLASHINFER supports FP8 KV | Keep the default (FLASHINFER) backend |
| 4 | Head fails: port busy | An earlier kill left a child holding the TCPStore port | `ss -tlnp` → kill the leftover PID (careful: `pkill -f` can match your own wrapper) |

Reference: [mmike87/spark-vllm-docker Qwen dual branch](https://github.com/mmike87/spark-vllm-docker/tree/Qwen36-27B-FP8-MTP-Dual)
(flash_attn backend, qwen3_next_mtp, fastsafetensors ideas) — flash_attn does not apply
to this model (pitfall 3).

## Operations

- Start order: **worker (2nd node) first, head second**; stop head first
- Cannot run alongside the dual-node DeepSeek stack (memory) — stop one before starting the other
- Compile/autotune caches are persisted on both nodes; restarts take ~4–5 minutes
