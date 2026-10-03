# 15 — A dynamic-int8 build of model 0.5.2: passes the gate and is 1.7–3x faster on Apple Silicon; Linux servers need their own gate

Date: 2026-10-03. Follows llmresults/14-x86-cpu-latency.md. Tool: `finetuning/export/quantize_dynamic.py` (same weights, no retraining).
Probe: `scripts/ort_cpu_probe.py`, 1 thread, ONNX Runtime 1.29.0. Each number is the median of 3 runs, with seeded random
tokens: "short" = 40 tokens, "long" = 512. x86: GitHub `ubuntu-latest`, native (run 37132675745). Linux ARM: GitHub
`ubuntu-24.04-arm`, Neoverse N2 (Graviton-class; run 37136473465). Apple: M5 Pro, macOS.

## What it is

The shipped graph is int8 weights + `DequantizeLinear` + fp32 `MatMul`. ONNX Runtime fuses those into `MatMulNBits`,
which has no fast int8 path on x86 (llmresults/14-x86-cpu-latency.md). `quantize_dynamic.py` folds the exact weights and re-quantizes them
as a QOperator graph (`DynamicQuantizeMatMul`): per-channel int8 weights, activations quantized per request. The file
is the same size (598 MB) and the folder layout is the same (config.json temperatures unchanged, new sha256).

## Speed and fidelity

`diff_vs_fp32` is the largest logit difference from the exact fp32 weights on the same inputs; the largest logit is
about 4000. It is the CPU-independent measure of how much a build distorts the model.

| CPU | build | short ms | long ms | diff_vs_fp32 |
|---|---|---|---|---|
| Apple M5 Pro | shipped (MatMulNBits) | 116 | 1613 | 56 |
| Apple M5 Pro | **dynamic s8, reduce_range** | **38** | **520** | **54** |
| Apple M5 Pro | dynamic s8 | 48 | 526 | 363 |
| Apple M5 Pro | dynamic u8 | 94 | 1205 | 437 |
| AMD EPYC 7763 (AVX2) | shipped | 535 | 6975 | 20 |
| AMD EPYC 7763 (AVX2) | dynamic s8, reduce_range | 214 | 2911 | 424 |
| AMD EPYC 7763 (AVX2) | dynamic s8 | 215 | 2922 | 1211 |
| AMD EPYC 7763 (AVX2) | dynamic u8 | 263 | 3584 | 107 |
| Intel Xeon 6973P (AVX-512 VNNI, AMX) | shipped | 243 | 3121 | 57 |
| Intel Xeon 6973P (AVX-512 VNNI, AMX) | dynamic s8, reduce_range | 97 | 1087 | 415 |
| Intel Xeon 6973P (AVX-512 VNNI, AMX) | dynamic s8 | 86 | 1107 | 174 |
| Intel Xeon 6973P (AVX-512 VNNI, AMX) | dynamic u8 | 205 | 2805 | 327 |
| Neoverse N2, Linux (asimddp, i8mm, sve) | shipped | 345 | 4644 | 53 |
| Neoverse N2, Linux | dynamic s8, reduce_range | 185 | 2580 | 413 |
| Neoverse N2, Linux | dynamic s8 | 186 | 2581 | 111 |
| Neoverse N2, Linux | dynamic u8 | 184 | 2548 | 235 |

- **Apple Silicon (macOS):** s8 with reduce_range is 3.1x faster on short and long inputs, and as faithful to the weights
  as the shipped build (54 vs 56).
- **Every Linux server CPU we measured:** the same build is 1.8–2.9x faster, but it distorts the model 8x as much as the
  shipped build (413–424 vs 20–57). That holds on x86 AVX2, x86 AVX-512 VNNI and ARM Neoverse alike, so the M5's good
  numerics come from ORT's Apple kernels, not from ARM. No dynamic variant is both fast and as faithful as the shipped
  build on Linux. The closest are u8 on AVX2 (107, 1.9x faster) and s8 on Neoverse (111, 1.8x faster); both are still
  about twice the shipped build's distortion.

## Gate (Apple M5 Pro, macOS, the Go server on CPU)

`python3 ft.py gate` on the s8 reduce_range folder (`~/openjevx/data/work/runs/v0.5.2-dyn-s8rr/out/model`, sha256
`08e62bde09bb…`), against the shipped 0.5.2 report. Each cell is right & confident / confidently wrong:

| Set | shipped 0.5.2 | dynamic s8 reduce_range |
|---|---|---|
| Everyday basics (463) | 98.7% / 0.9% | 98.7% / **0.2%** |
| Rule basics (300) | 100% / 0.0% | 100% / 0.0% |
| Held-out rules (11,949) | 98.8% / 1.0% | 98.7% / 1.0% |
| Held-out work roles (9,159) | 97.8% / 1.0% | 97.6% / 1.0% |
| Logs gate (900) | 91.9% / 7.8% | 91.8% / 7.7% |
| BGL logs (214, the known 0.5.2 miss) | 64.5% / 35.0% | 64.5% / 35.5% |
| 13 fundamentals through jevx | 13/13 | 13/13 |

GATE PASS on the basics and the fundamentals, with no regression against shipped 0.5.2. The BGL miss is unchanged:
it was not fixed and not made worse.

Through the Go server on the M5 (4 threads; the machine was under other load, load average 40–100), p50 server_ms:

| Input | shipped 0.5.2 | dynamic s8 reduce_range |
|---|---|---|
| short (75 tokens) | 72 ms | 43 ms |
| typical (235 tokens) | 214 ms | 115 ms |
| long (587 tokens) | 565 ms | 294 ms |

This gate was run on macOS only. OrbStack's amd64 emulation does not expose AVX2, so it cannot stand in for x86. The gate
data is private, so it cannot run on the public repo's CI runners.

## Decision needed (owner)

1. **Ship it for macOS** (the `darwin-arm64` package). It is verified: it passed the gate, shows no regression, and is
   1.7–3x faster. That would mean two model folders per release, one for macOS and one for Linux. Or keep one model.
2. **For Linux servers (Fargate x86, Graviton):** before shipping any dynamic build there, run `ft.py gate` on that
   hardware. Options: an x86 box we own, or an ECS task in our AWS account (gate data stays private). Candidates: u8 for
   AVX2 and s8 for Neoverse. Until then the shipped 0.5.2 stays the Linux model. The cheap Linux win is already shipped:
   threads that follow the real CPU limit (server 0.5.5), 4x on a 1 vCPU Fargate task.

To rebuild: `python3 finetuning/export/quantize_dynamic.py <0.5.2 model folder> OUT --weights s8rr --version 0.5.3`.
