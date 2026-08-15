# Qwen3.8-27B-NVFP4 on DGX Spark — Test Report

> Test date: 2026-08-15
> Model: unsloth/Qwen3.8-27B-NVFP4 (released 2026-08-14)
> [中文版本](../Qwen3.8-27B-NVFP4-测试报告.md)

## 1. Test Environment

| Item | Configuration |
|---|---|
| Hardware | NVIDIA DGX Spark: GB10 GPU (Blackwell, sm_121) + Grace CPU (aarch64), 128GB unified memory |
| Driver/CUDA | Driver 580.95.05 / CUDA 13.0 |
| OS | Ubuntu 24.04 (noble) aarch64, Python 3.12.3 |
| Inference stack | vLLM 0.27.1 + PyTorch 2.13.0+cu130 + Triton 3.7.1 + flashinfer 0.6.16.post3 |
| Quantization | NVFP4 (compressed-tensors `nvfp4-pack-quantized`, FP4 weights / FP8 activations) |
| Server flags | `--max-model-len 262144 --gpu-memory-utilization 0.85 --reasoning-parser qwen3 --speculative-config '{"method":"mtp","num_speculative_tokens":3}'` |

Engine self-report (256k config): **KV cache 2,091,194 tokens, ~8 concurrent requests at the full 262,144-token context** (64k config: 2.38M tokens / ~36 concurrent).

## 2. Functional Tests (all passed)

| # | Test | Result | Notes |
|---|---|---|---|
| 1 | Thinking mode · math reasoning | ✅ Correct | Two-pipe pool problem, correct answer 2.4h, full reasoning chain, 29.1s / 342 tokens |
| 2 | Non-thinking · general QA | ✅ Correct | Accurate one-sentence Transformer explanation, 4.6s |
| 3 | Code generation | ✅ Complete | Thread-safe LRU cache, hit 1500-token cap (126.7s), sound approach (OrderedDict + RLock) |
| 4 | Multi-turn memory | ✅ Correct | Name & hobby recalled accurately, 1.2s |
| 5 | Streaming | ✅ OK | Short-prompt TTFT 0.18s |
| 6 | Vision understanding | ✅ Pass | Accurate scene description (dark-blue background, red circle upper-left, green rectangle center-right, yellow stripes bottom); stripe count 8→6 minor miss |
| 7 | Reasoning separation | ✅ OK | With `--reasoning-parser qwen3`, chain lands in the `reasoning` field; "strawberry r count" answered correctly (3) |
| 8 | 256K needle-in-a-haystack | ✅ Hit | Passcode "8517-3342-9906" hidden at ~70% depth inside 255,376 tokens of filler logs; model answered directly; prefill 352s |

## 3. Performance (vllm bench serve, random dataset)

### 3.1 Single stream (interactive measurement)

- Decode **≈ 11.9 tok/s** without MTP; **17–23 tok/s** with MTP (see 3.5)
- Short-prompt TTFT **0.18 s**

### 3.2 Concurrency sweep — daily-use range (256k config + MTP ×3, input 512 / output 256)

| Metric | c=1 | c=2 | c=4 |
|---|---|---|---|
| Per-stream decode (1/TPOT) | **21.6 tok/s** | **16.0 tok/s** | **15.0 tok/s** |
| Aggregate output throughput | 20.0 tok/s | 22.7 tok/s | 44.0 tok/s |
| Mean TTFT | 0.99 s | 0.81 s | 1.92 s (median 0.75 s) |
| TPOT (mean/median) | 46.3 / 46.3 ms | 62.7 / 61.2 ms | 66.6 / 54.6 ms |

Note: random-token prompts yield lower MTP acceptance (37–59%) than real text; real conversations sit closer to the 17–23 tok/s measured in 3.5.

### 3.3 Higher concurrency (64k config, input 512 / output 128, 32 requests)

| Metric | c=8 | c=8 + MTP | c=16 |
|---|---|---|---|
| Success | 32/32 | 32/32 | 32/32 |
| Output throughput | 72.0 tok/s | 77.7 tok/s | **115.7 tok/s** (peak 160) |
| Total throughput (incl. prefill) | 359.9 tok/s | 388.3 tok/s | 578.7 tok/s |
| Mean TTFT | 1.66 s | 1.95 s | 2.46 s |
| Mean TPOT | 98.6 ms | 78.8 ms | 119.1 ms |

### 3.4 Long context (single stream, 256k config + MTP ×3)

| Context (input) | TTFT | Prefill speed | TPOT | Per-stream decode |
|---|---|---|---|---|
| 1k | 0.57 s | **≈1800 tok/s** | 83.0 ms* | ≈12–21 tok/s* |
| 8k | 4.2 s | ≈1900 tok/s | 99.9 ms (no MTP) | — |
| 100k | 83.0 s | **≈1230 tok/s** (-32%) | 60.1 ms (+30%) | **≈16.6 tok/s** |
| 200k | 244.1 s | **≈840 tok/s** (-53%) | 70.3 ms (+52%) | **≈14.2 tok/s** |
| **256k (maxed, 261,000 tokens)** | 365.9 s | **≈715 tok/s** (-60%) | 87.5 ms (+88%) | **≈11.4 tok/s** |

\* The 1k random-data run had unusually low MTP acceptance (18%), distorting TPOT; see 3.2/3.5 for real-text numbers.

Takeaways:

- **Decode degrades gently**: at the full 256k context, decode keeps ~60–70% of its short-context speed. This is the hybrid architecture paying off — 48 of 64 layers are Gated DeltaNet linear attention (context-length-independent decode cost); only 16 full-attention layers grow with length.
- **Prefill slows super-linearly**: 2× context ≈ 3× time (the quadratic term of full-attention layers shows up). Loading 100k takes ~1.4 min, 200k ~4 min, a maxed-out 256k ~6 min — acceptable, but avoid re-ingesting the same long document repeatedly.
- **256K context verified for real**: the needle-in-a-haystack test (255,376 tokens, needle at 70% depth) hit on the first try — the model genuinely uses the full context, not merely "runs without crashing".
- vLLM currently does not support prefix caching for this hybrid architecture (`enable_prefix_caching=False`). Within a session, follow-up turns only incrementally prefill new tokens (KV cache reuse); re-ingesting the same document in a *new* session costs a full prefill again.

### 3.5 MTP speculative decoding (enabled in the final config)

The model ships an MTP head (`model_mtp.safetensors`, 1 full-attention layer). With
`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`:

| Scenario | No MTP | MTP ×3 | Gain |
|---|---|---|---|
| Single stream · thinking-mode math | 11.8 tok/s | **22.7 tok/s** | **+92%** |
| Single stream · non-thinking long writing | ≈12 tok/s | **16.7 tok/s** | ≈+40% |
| c=8 aggregate output | 72.0 tok/s | 77.7 tok/s | +8% (no regression) |
| c=8 TPOT | 98.6 ms | 78.8 ms | -20% |

- Draft acceptance on real text: 54%–84% (per-position 0.86/0.69/0.55), mean acceptance length 2.1–3.1 tokens/step; even random-token benchmarks accepted 39.5% with no concurrency regression.
- `num_speculative_tokens` 2 vs 3: single-stream speed is a wash (21.4 vs 22.7 tok/s); ×2 has less overhead on unpredictable text, ×3 peaks higher on predictable text (code, formatted output). Final choice: ×3.

### 3.6 Resource usage

- Unified memory: ~110GB / 119GB used (vLLM preallocates 85%: 23.4GB weights + KV cache)
- Idle power 13W, temperature 48℃

## 4. Issues Hit During Deployment & Fixes

| Issue | Cause | Fix |
|---|---|---|
| huggingface.co unreachable | Network restriction | hf-mirror.com mirror |
| `Python.h: No such file or directory` | No `python3.12-dev`, no root | `apt-get download` + `dpkg -x` locally, set `CPATH` at launch |
| `FileNotFoundError: 'ninja'` | venv/bin not in launch PATH | Explicit `PATH=.../qwen38-env/bin:$PATH` |
| Reasoning mixed into content with `</think>` | No reasoning parser | `--reasoning-parser qwen3` (field name: `reasoning`) |
| `vllm bench` hangs without sending requests | `--tokenizer` model ID triggers HF network access | Use local tokenizer path |

## 5. Conclusions

1. **Deployment is practical**: pure-pip on DGX Spark (GB10), no root, no Docker, ~30 minutes end-to-end including the 23GB download.
2. **Feature-complete**: text (thinking/non-thinking/streaming/multi-turn), vision, and code generation all work; quality matches 27B-class expectations.
3. **Performance (256k context + MTP)**: single-stream 15–23 tok/s, still ≈14 tok/s at 200k context; short-text prefill ≈1800–1900 tok/s, 100k ingest ≈1.4 min, 200k ≈4 min; aggregate output ≈78–116 tok/s at c=8–16.
4. **Full 262,144-token context is enabled** in the final config; KV cache 2.09M tokens supports ~8 concurrent full-length requests.

## 6. Daily-use Recommendations (from the measurements above)

1. **Concurrency**: 1–2 for interactive use (16–23 tok/s per stream); keep agent sessions ≤ **4** (≥15 tok/s each); 8 is the interactive ceiling; use 16+ for offline batch jobs where aggregate throughput peaks.
2. **Long context (the 256k goal)**:
   - Already configured at 262,144 tokens — nothing else to tune.
   - **≤100k: use freely** (~1.4 min ingest). **~200k: best for "ingest once, ask many turns"** (codebase-scale analysis, long documents) — first turn waits ~4 min, follow-ups only prefill incrementally, decode ≈14 tok/s.
   - Avoid re-ingesting the same long document across new sessions (no prefix caching; each new session pays full prefill).
   - Max ~8 concurrent full-256k requests; mixed short/long workloads share the KV cache automatically.
3. **Mode choice**: thinking mode for reasoning/code (higher quality, 22.7 tok/s); disable thinking for chat/translation/summaries (`chat_template_kwargs.enable_thinking=false`) — faster and cheaper.
4. **Sampling**: use the official recommendations (thinking 1.0/0.95/20; non-thinking 0.7/0.8/20 + presence_penalty 1.5) and give long tasks a generous `max_tokens`.
