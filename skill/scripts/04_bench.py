#!/usr/bin/env python3
"""Step 5（可选）: Qwen3.8-27B SGLang DSPARK 测速，含 256K 长上下文档位
逐档测量 TTFT / prefill 速度 / decode 速度，随机前缀避免 RadixAttention 缓存污染。

用法:
  python3 04_bench.py              # 全档位 1K→245K（约 15-25 分钟）
  python3 04_bench.py 1000 32000   # 只测指定档位（快速验证）

宿主机没有 transformers 时，自动接力到 qwen38-nvfp4 容器内运行。
"""
import json
import os
import random
import subprocess
import sys
import time

BASE = os.environ.get("BENCH_BASE", "http://127.0.0.1:30000")
MODEL = "qwen3.8-27b"
GEN_TOKENS = 256
TIERS = [1000, 32000, 64000, 100000, 150000, 200000, 245000]
CONTAINER = "qwen38-nvfp4"

# ---- 自动接力：宿主机缺 transformers 时，cp 进容器跑 -------------------------
try:
    from transformers import AutoTokenizer
    import requests  # noqa: F401
except ImportError:
    if os.environ.get("BENCH_IN_CONTAINER") != "1":
        r = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                           capture_output=True, text=True)
        if CONTAINER in r.stdout.split():
            print(f">> 宿主机缺 transformers，接力到容器 {CONTAINER} 内运行")
            self_path = os.path.abspath(__file__)
            subprocess.check_call(["docker", "cp", self_path, f"{CONTAINER}:/tmp/_bench.py"])
            env = os.environ.copy()
            env["BENCH_IN_CONTAINER"] = "1"
            rc = subprocess.call(
                ["docker", "exec", "-e", "BENCH_IN_CONTAINER=1",
                 CONTAINER, "python3", "/tmp/_bench.py"] + sys.argv[1:])
            sys.exit(rc)
    print("!! 需要 transformers + requests：pip install transformers requests")
    sys.exit(1)

import requests  # noqa: E402

tok = AutoTokenizer.from_pretrained(
    "/models" if os.environ.get("BENCH_IN_CONTAINER") == "1"
    else os.environ.get("MODEL_DIR", os.path.expanduser("~/models/qwen38-nvfp4")))

WORDS = ("the quick brown fox jumps over lazy dog science history market "
         "engine river mountain crystal fabric theorem signal harbor").split()


def make_prompt(n_tokens, salt):
    # O(n)：一次拼够文本、一次 encode 截断（逐句 encode 全文是 O(n²)，200K 档要跑十几分钟）
    rng = random.Random(salt)
    approx_chars = int(n_tokens * 8)  # 实测随机短词 ~6.1 字符/token，8 留足余量
    parts = [f"[bench-salt-{salt}]"]
    total = len(parts[0])
    while total < approx_chars:
        s = " ".join(rng.choice(WORDS) for _ in range(16)) + "."
        parts.append(s)
        total += len(s) + 1
    ids = tok.encode(" ".join(parts))
    return tok.decode(ids[:n_tokens]), min(len(ids), n_tokens)


def bench(n_tokens, salt):
    prompt, exact = make_prompt(n_tokens, salt)
    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt + "\n\nReply with a short summary of the above."}
        ],
        "max_tokens": GEN_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    ttft = None
    usage = {}
    with requests.post(f"{BASE}/v1/chat/completions", json=body, stream=True, timeout=3600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices and ttft is None:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    ttft = time.time() - t0
    total = time.time() - t0
    pt = usage.get("prompt_tokens", exact)
    ct = usage.get("completion_tokens", GEN_TOKENS)
    ttft = ttft or total
    decode_t = total - ttft
    return {
        "target_tokens": n_tokens,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "ttft_s": round(ttft, 2),
        "prefill_tps": round(pt / ttft, 1) if ttft > 0 else None,
        "decode_tps": round((ct - 1) / decode_t, 1) if decode_t > 0.01 else None,
        "total_s": round(total, 2),
    }


def main():
    tiers = [int(x) for x in sys.argv[1:]] or TIERS
    results = []
    for n in tiers:
        reps = 2 if n <= 1000 else 1
        for rep in range(reps):
            salt = f"{n}-{int(time.time())}-{rep}"
            print(f"--- {n} tokens (rep {rep}) ---", flush=True)
            try:
                r = bench(n, salt)
            except Exception as e:
                r = {"target_tokens": n, "error": str(e)[:200]}
            print(json.dumps(r, ensure_ascii=False), flush=True)
            results.append(r)
    out = f"bench_result_{int(time.time())}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("SAVED:", out)


if __name__ == "__main__":
    main()
