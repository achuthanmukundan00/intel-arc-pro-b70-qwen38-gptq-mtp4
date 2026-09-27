#!/usr/bin/env python3
"""p512/g128 short-context benchmark (stdlib only). Usage: [PORT=8000] ./bench.sh
Reports median/mean post-first-token decode + median TTFT. Warm up first (first
request triggers Triton JIT) — the script does one warmup automatically.
"""
import json, os, statistics, time, urllib.request
BASE = f"http://127.0.0.1:{os.environ.get('PORT', '8000')}"
MODEL = os.environ.get("MODEL", "qwen38")

def rpc(path, payload, timeout):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

seed = ("A retrieval augmented generation system retrieves relevant source passages, "
        "constructs a grounded context, sends the evidence to a language model, and "
        "generates a concise answer based on the supplied information. ")
text = seed
while True:
    toks = rpc("/tokenize", {"model": MODEL, "prompt": text}, 120)["tokens"]
    if len(toks) >= 512:
        break
    text += seed
text = rpc("/detokenize", {"model": MODEL, "tokens": toks[:512]}, 60)["prompt"]

def run():
    payload = {"model": MODEL, "prompt": text, "max_tokens": 128, "temperature": 0,
               "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = end = None
    usage = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                end = time.perf_counter()
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("choices") and obj["choices"][0].get("text") and first is None:
                first = time.perf_counter()
    ct = usage["completion_tokens"]
    return {"rate": (ct - 1) / (end - first), "ttft": first - start, "ct": ct}

print("warmup...", flush=True)
w = run()
print(f"warmup: {w['rate']:.2f} tok/s TTFT={w['ttft']:.4f}s", flush=True)
rates, ttfts = [], []
for i in range(1, 6):
    r = run()
    rates.append(r["rate"])
    ttfts.append(r["ttft"])
    print(f"run {i}: {r['rate']:.2f} tok/s TTFT={r['ttft']:.4f}s ct={r['ct']}", flush=True)
print(f"MEDIAN DECODE: {statistics.median(rates):.2f} tok/s")
print(f"MEAN DECODE:   {statistics.mean(rates):.2f} tok/s")
print(f"MEDIAN TTFT:   {statistics.median(ttfts):.4f} s")
