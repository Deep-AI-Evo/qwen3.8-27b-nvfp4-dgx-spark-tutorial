#!/usr/bin/env python3
"""Step 4: 验证 Qwen3.8-27B SGLang 服务（仅依赖标准库 + requests）
三项检查：基础对话 / thinking 模式 / OpenAI 端点格式
用法: python3 03_verify.py [base_url]   默认 http://127.0.0.1:30000
"""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30000"
MODEL = "qwen3.8-27b"


def chat(messages, max_tokens=128, thinking=False):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main():
    ok = True

    # 1. 基础对话
    try:
        d = chat([{"role": "user", "content": "用一句话介绍杭州"}], 128)
        msg = d["choices"][0]["message"]
        text = msg.get("content") or ""
        assert text.strip(), "空回复"
        print(f"✅ 基础对话 PASS: {text[:60]}")
    except Exception as e:
        ok = False
        print(f"❌ 基础对话 FAIL: {e}")

    # 2. thinking 模式（reasoning_content 应非空）
    try:
        d = chat([{"role": "user", "content": "9.11 和 9.9 哪个大？"}], 512, thinking=True)
        msg = d["choices"][0]["message"]
        reasoning = msg.get("reasoning_content") or ""
        assert reasoning.strip(), "reasoning_content 为空（--reasoning-parser qwen3 未生效？）"
        print(f"✅ thinking 模式 PASS: reasoning {len(reasoning)} chars")
    except Exception as e:
        ok = False
        print(f"❌ thinking 模式 FAIL: {e}")

    # 3. OpenAI 端点格式（usage 字段齐全）
    try:
        d = chat([{"role": "user", "content": "hi"}], 16)
        u = d.get("usage") or {}
        assert u.get("prompt_tokens") and u.get("completion_tokens") is not None
        print(f"✅ OpenAI 格式 PASS: usage={u}")
    except Exception as e:
        ok = False
        print(f"❌ OpenAI 格式 FAIL: {e}")

    print("\n" + ("🎉 全部通过，服务可用。" if ok else "⚠️ 有失败项，见上方日志。"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
