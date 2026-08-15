#!/usr/bin/env python3
"""Qwen3.8-27B-NVFP4 视觉理解测试：生成测试图并请求模型描述（仅用标准库）。"""
import base64
import io
import json
import struct
import time
import urllib.request
import zlib

BASE = "http://127.0.0.1:8000/v1"
MODEL = "unsloth/Qwen3.8-27B-NVFP4"


def make_png(path, w=512, h=384):
    """纯标准库生成一张测试图：蓝底、红色圆、绿色矩形、黄色条纹。"""
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    rows = b""
    for y in range(h):
        row = bytearray([0])  # filter type 0
        for x in range(w):
            r_, g_, b_ = 30, 60, 160  # 蓝底
            if (x - 130) ** 2 + (y - 120) ** 2 < 80 ** 2:
                r_, g_, b_ = 220, 40, 40   # 红圆
            if 300 < x < 470 and 200 < y < 330:
                r_, g_, b_ = 40, 180, 60   # 绿矩形
            if y > 340 and (x // 32) % 2 == 0:
                r_, g_, b_ = 230, 210, 40  # 黄条纹
            row += bytes((r_, g_, b_))
        rows += bytes(row)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)
    return path


def main():
    img = make_png("/home/dgx/projects/qwen38-test/test_image.png")
    b64 = base64.b64encode(open(img, "rb").read()).decode()
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text",
                 "text": "请详细描述这张图片的内容：背景颜色、有哪些图形、各自的颜色和大致位置。"},
            ],
        }],
        "max_tokens": 512,
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        dt = time.time() - t0
        resp = json.loads(r.read())
    print(f"===== 视觉理解测试 (耗时 {dt:.1f}s) =====")
    print(resp["choices"][0]["message"]["content"])
    print(f"[用量] {resp.get('usage')}")


if __name__ == "__main__":
    main()
