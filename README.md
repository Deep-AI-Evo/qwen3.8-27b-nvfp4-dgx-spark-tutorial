# Qwen3.8-27B-NVFP4 在 NVIDIA DGX Spark（GB10）上的部署与实测

本仓库记录 [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
（27B 稠密视觉-语言模型，NVFP4 量化，256k 上下文）在 **NVIDIA DGX Spark**（GB10，
Grace-Blackwell，aarch64，128GB 统一内存）上的完整部署教程与实测报告。

## 文档

- [部署教程](docs/Qwen3.8-27B-NVFP4-部署教程.md) —— 纯 pip 部署（无需 root/Docker），
  torch 2.13.0+cu130 + vLLM 0.27.1，含国内镜像下载与全部踩坑记录
- [测试报告](docs/Qwen3.8-27B-NVFP4-测试报告.md) —— 功能测试、并发/长上下文压测数据、
  MTP 推测解码优化、日常使用建议

## 关键结论（TL;DR）

| 项目 | 实测结果 |
|---|---|
| 单流解码（MTP ×3 开启） | 15–23 tok/s（思考模式最高 22.7，较无 MTP +92%） |
| prefill | 短文本 ≈1800 tok/s；100k 灌入 ≈1.4 分钟；200k ≈4 分钟 |
| 长上下文解码衰减 | 很平缓：200k 上下文仍有 ≈14 tok/s（混合架构 48/64 层为线性注意力） |
| 并发吞吐 | 8 并发 ≈78 tok/s，16 并发 ≈116 tok/s（峰值 160） |
| 上下文 | 已启用完整 262,144 tokens，KV cache 209 万，满长上下文约 8 路并发 |
| 日常建议 | 交互 ≤4 并发；200k 级长文用于"一次灌入 + 多轮问答" |

## 目录

- `docs/` —— 教程与测试报告
- `tests/` —— 功能测试脚本（对话/视觉，仅标准库）与压测矩阵脚本
- `results/` —— 原始压测输出（vllm bench serve）与功能测试输出

## 一键启动命令（环境就绪后）

```bash
PATH=~/projects/qwen38-env/bin:$PATH \
~/projects/qwen38-env/bin/vllm serve ~/projects/models/Qwen3.8-27B-NVFP4 \
  --served-model-name unsloth/Qwen3.8-27B-NVFP4 \
  --reasoning-parser qwen3 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --port 8000
```
