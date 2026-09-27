# b70-qwen38-vllm

One-command inference server: **Qwen3.8-27B GPTQ INT4 + MTP4** on a single **Intel Arc Pro B70 32GB** with vLLM XPU.

```bash
git clone <this-repo> && cd intel-arc-pro-b70-qwen38-gptq-mtp4 && ./serve.sh
```

That checks prerequisites, fetches the pinned model + patches (hash-verified),
starts the server, waits for health, and runs a smoke test. Endpoint:
`http://127.0.0.1:8000/v1` (OpenAI-compatible).

## What you get

| Setting | Value |
|---|---|
| Model | `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` @ `9d189a6` |
| Runtime | `vllm/vllm-openai-xpu@sha256:f01e24f6…` (vLLM `0.27.2rc1.dev77+gac7509e2b.xpu`) |
| Patches | `patch_mtp_nightly.py` + `patch_mtp_boundary.py` (SHA256-verified, fetched from upstream cookbook @ `1378950`) |
| Speculation | MTP4, BF16 draft (`B70_MTP_BF16_DRAFT=1`), XPU graphs on |
| KV cache | FP8, `gpu-memory-utilization 0.90` → ~162k-token capacity |
| Context | 161,000 tokens (`CTX=131072` also supported) |
| Concurrency | **1** (see limits) |
| Tool calling | `qwen3_xml` + `qwen3` reasoning parser, prefix caching on |

Measured on a B70 host (temp 0, post-first-token decode): **~63 tok/s @512 ctx,
~54 @14k, ~36 @130k**, TTFT 0.35s @512. Production thinking traffic (temp 1.0)
runs lower — MTP acceptance falls with entropy.

## Requirements

- Ubuntu 26.04 LTS (Debian-family likely), `xe` kernel driver for Battlemage
- Docker with `/dev/dri` access; card visible as `/dev/dri/renderD*`
- ~90GB free disk (18GB model + image + caches), 230W power cap recommended
- No other GPU workload on the B70

## Options

```bash
./serve.sh --ctx 131072 --port 8000        # smaller context if 161k doesn't fit
./serve.sh --dry-run                       # checks only, starts nothing
./serve.sh --stop                          # stop the server
./bench.sh                                 # p512/g128 short-context benchmark
./smoke.sh                                 # health + models + tiny generation
```

Env overrides: `MODEL_DIR`, `CTX`, `GPU_MEM`, `PORT`, `CONTAINER`.

## Hard limits (measured, not guessed)

- **Concurrency is 1 with MTP.** Two overlapping requests kill EngineCore:
  `causal_conv1d does not support spec-decode and non-spec tokens in the same
  invocation`. This is the hybrid-architecture cost of MTP — every fast config
  online is single-stream. Dropping MTP is the only path to parallel requests.
- 161k ctx leaves ~1.01x KV headroom. If startup reports less KV than that,
  drop to `--ctx 131072`.
- First request after start is slow (Triton JIT warmup) — always warm up
  before benchmarking.

## Troubleshooting

| Symptom | Check |
|---|---|
| `ze_fd_manager` / XCCL init failure | Add `--cap-add SYS_PTRACE --security-opt seccomp=unconfined --ipc=host` (see `serve.sh --compat-flags`) |
| Wrong GPU selected | Verify 32GB B70 via startup `Free memory on device (29.38/30.3 GiB)`; set `ZE_AFFINITY_MASK` |
| Slow first request | Triton JIT — repeat the same shape once |
| `EngineDeadError` + `causal_conv1d` | Concurrent requests with MTP — keep concurrency 1 |
| Post-Xe-fault check | `sudo dmesg -T \| grep -Ei 'engine reset\|fault\|coredump'` |

## Credits

Checkpoint, recipe, and both runtime patches: [SergiioB / intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook).
Measurement methodology cross-checked against
[MikeCaldera / intel-arc-pro-b70-qwen38-vllm](https://github.com/MikeCaldera/intel-arc-pro-b70-qwen38-vllm)
and [JP-devv / humble-b70-llm](https://github.com/JP-devv/humble-b70-llm).
