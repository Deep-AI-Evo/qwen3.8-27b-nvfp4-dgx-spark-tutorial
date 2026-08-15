#!/usr/bin/env python3
"""Qwen3.8-27B-NVFP4 基础对话功能测试（OpenAI 兼容接口，仅用标准库）。"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000/v1"
MODEL = "unsloth/Qwen3.8-27B-NVFP4"


def chat(messages, max_tokens=512, temperature=1.0, top_p=0.95, top_k=20,
         thinking=True, stream=False):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "stream": stream,
    }
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    if not stream:
        with urllib.request.urlopen(req, timeout=600) as r:
            dt = time.time() - t0
            return json.loads(r.read()), dt
    # 简单统计流式首 token 延迟
    first = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                if first is None:
                    first = time.time() - t0
    return {"ttft": first}, time.time() - t0


def show(title, resp, dt):
    print(f"\n===== {title} (耗时 {dt:.1f}s) =====")
    msg = resp["choices"][0]["message"]
    reasoning = msg.get("reasoning_content")
    if reasoning:
        print(f"[思考过程 {len(reasoning)} 字符] {reasoning[:200]}...")
    print(f"[回复] {msg.get('content','')[:600]}")
    print(f"[用量] {resp.get('usage')}")


def main():
    # 1. 思考模式 - 数学推理
    r, dt = chat([{"role": "user", "content":
                   "一个水池有甲乙两个进水管，甲单独注满需6小时，乙单独注满需4小时。两管同时开，多久注满？请给出推理过程。"}],
                 thinking=True)
    show("测试1: 思考模式-数学推理", r, dt)

    # 2. 非思考模式 - 常识问答
    r, dt = chat([{"role": "user", "content": "用一句话解释什么是 Transformer 架构。"}],
                 thinking=False, temperature=0.7, top_p=0.8, max_tokens=256)
    show("测试2: 非思考模式-常识问答", r, dt)

    # 3. 代码生成
    r, dt = chat([{"role": "user", "content":
                   "用 Python 写一个线程安全的 LRU 缓存，带容量上限，给出完整代码。"}],
                 thinking=True, max_tokens=1500)
    show("测试3: 代码生成", r, dt)

    # 4. 多轮对话
    msgs = [{"role": "user", "content": "我叫小明，我喜欢爬山。"}]
    r, _ = chat(msgs, thinking=False, max_tokens=128)
    msgs.append(r["choices"][0]["message"])
    msgs.append({"role": "user", "content": "我叫什么名字？爱好是什么？"})
    r, dt = chat(msgs, thinking=False, max_tokens=128)
    show("测试4: 多轮对话记忆", r, dt)

    # 5. 流式 + TTFT
    r, dt = chat([{"role": "user", "content": "写一首关于秋天的五言绝句。"}],
                 thinking=False, max_tokens=128, stream=True)
    print(f"\n===== 测试5: 流式输出 =====\nTTFT={r['ttft']:.2f}s, 总耗时={dt:.1f}s")

    print("\n全部对话测试完成")


if __name__ == "__main__":
    sys.exit(main())
