# ADR 0003 — Export, license, and what gets published

Status: accepted  
Date: 2026-09-28

## Context

The first ONNX export of the fine-tune was fp32 at 1.6GB. The ship target is one small download with no material quality loss. The server that users run is Go, not Python.

## Decision

1. Export ONNX on the GPU box with `scripts/export_onnx_gpu.py`. Refuse the export if the ORT session is not on CUDA.
2. Quantize 4-bit weight-only (`MatMulNBits`, block 32, symmetric) with `scripts/quantize_w4.py`; 461 MB. Superseded 2026-09-28: the first release used ORT dynamic int8 (598 MB), which was never benchmarked as a file. Measured on the Go server it scored 0.651 on typed-decisions test against 0.768 for fp32, and its answers changed with how many questions shared a request (activation scales are per batch). 4-bit weight-only keeps activations in float: 0.770 on the same subset where fp32 scored 0.774, stable across batching. Benchmark the exact shipped file, never only the PyTorch model.
3. Keep the fp16/fp32 safetensors as the trainable checkpoint. The 4-bit file is the downloadable graph.
4. Publish weights to Hugging Face and the 4-bit graph plus benchmark to the GitHub release. The release binary embeds that graph.
5. License is Apache-2.0 for our code and the weights. Laya and ModernBERT notices stay in `CREDITS`, `NOTICE`, and `licenses/`. Any company may use it commercially or not.
6. Next training does not start until ADR 0001 and ADR 0002 are still true. If the base model changes, write a new ADR first. Do not silently switch vendors.

## Consequences

A GPU config must name a CUDA-capable graph or fail. Auto mode may use CPU. The Go server is the product. Python remains only in `scripts/` for the training run.
