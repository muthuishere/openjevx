# 14 — Why long requests are slow on x86 (and what would fix it)

Date: 2026-10-03. Model 0.5.2 (`openjevx.w8.onnx`), ONNX Runtime 1.29.0 CPU, Python wheel (same MLAS kernels as the
server's `libonnxruntime`). Probe: `scripts/ort_cpu_probe.py`. Each number is the median of 3 runs after a warm-up, with
one item, 2 markers, seeded random tokens: "short" = 40 tokens, "long" = 512. Runners: GitHub Actions `ubuntu-latest`
(4 vCPU), native x86, no emulation (runs 37101332611 and 37100962544). Mac: M5 Pro, 1 thread.

## The question

Marketplace measured a 512-token request at about 3.9 s per core on Fargate x86, against about 0.5 s on M-series.
Which CPU features does ORT use, does the 8-bit graph get fast int8 kernels on x86, and does anything cheap help?

## Findings

1. **The graph is QDQ weight-only.** 120 per-channel int8 `DequantizeLinear` (axis 1) feed fp32 `MatMul`. With the server's
   settings (`ORT_ENABLE_ALL`) ORT fuses all 120 into `MatMulNBits` (8-bit, accuracy level 4 = int8 activations) on
   every CPU we tried. That one op is 86–88% of the time on x86 and 79% on the M5.
2. **On x86 there is no fast int8 path for it.** On AMD EPYC (AVX2 + FMA; no AVX-512, no VNNI), asking `MatMulNBits` for
   int8 compute is no faster than fp32 compute (long, 1 thread: 6.95 s vs 6.0 s for accuracy level 0). It is slower than
   ORT's plain fp32 `MatMul` over the same weights pre-dequantized (4.8 s). So the kernel behaves like
   dequantize-and-multiply every call, not like a VNNI/dot-product int8 GEMM.
3. **CPU features seen:** AMD EPYC 7763 and 9V74: `avx avx2 fma`. Intel Xeon Platinum 8573C: `avx avx2 fma avx512f
   avx512bw avx512_vnni`. Even with VNNI the default was 2.10 s at 2 threads (about 4 s per core), which matches
   Fargate's 3.9 s. The M5 Pro takes 1.44 s on 1 thread, so x86 AVX2 is about 4.8x slower per core.

| CPU (threads) | default (shipped) | QDQ fusion off (fp32 weights) | dynamic int8 graph |
|---|---|---|---|
| AMD EPYC 7763, AVX2 (1) | 531 / 6953 ms | 382 / 4788 ms | 214 / 2892 ms |
| AMD EPYC 7763, AVX2 (2) | 289 / 3600 ms | 213 / 2472 ms | 127 / 1527 ms |
| AMD EPYC 9V74, AVX2 (1) | 566 / 7445 ms | 406 / 5242 ms | 230 / 3159 ms |
| Intel 8573C, AVX-512 VNNI (2) | 173 / 2104 ms | 172 / 1780 ms | 90 / 1020 ms |
| Apple M5 Pro (1) | 109 / 1445 ms | 52 / 562 ms | 36 / 487 ms |

Each cell is short / long. Other settings changed nothing or made it worse:
- `ORT_ENABLE_EXTENDED`: the same as the default.
- `ORT_ENABLE_BASIC`: no fusion, so `DequantizeLinear` runs every call; slower on short requests.
- `session.qdq_matmulnbits_accuracy_level` 0 or 1: 2–3x slower on short requests, slightly faster on long.

## What would help, and why none of it is shipped here

- **QDQ fusion off** (`session.disable_quant_qdq=1`). ORT then folds the weights to fp32 at load time and uses its fp32
  GEMM. It is 1.4–1.45x faster on x86 and 2.6x on the M5. But the session holds about 2.7 GB instead of 1.3 GB, which risks
  running out of memory on a small Fargate task, and the logits move (max |diff| 20 against a max |logit| of 4024). The gate
  passed with the fused numerics, so this needs a gate run, and an opt-in setting rather than a default.
- **A dynamic-quantized graph** (QOperator `DynamicQuantizeMatMul`, int8 weights and activations; same size). It is 2.1–2.4x
  faster on x86 and 3x on the M5. But on AVX2 without VNNI its logits move by 1232 against a max of 4024, about 31%, which
  looks like the known int16 saturation of u8×s8 kernels on AVX2. On VNNI it is 231. A model built this way needs
  `reduce_range` (or VNNI-only use) and the full gate, so it is a model release, not a server change.
- **Threads:** this was the big cheap win. On a 1 vCPU Fargate task the server ran 2 threads and was CPU-throttled about
  4x. Server 0.5.5 reads the cgroup quota, then the ECS task's `Limits.CPU`.

## Next step, if the owner wants x86 speed

Build the dynamic-int8 model with `reduce_range=True` from the fine-tuned checkpoint's fp32 export, and run the gate on it
on x86 and on ARM. Ship it only if the gate passes (right AND confident). Re-run `scripts/ort_cpu_probe.py` on
`ubuntu-latest` to confirm the speed-up holds.
