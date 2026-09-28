# ADR 0003 — Export, license, and what gets published

Status: accepted  
Date: 2026-09-28

## Context

The first ONNX export of the fine-tune was fp32 at 1.6GB. The ship target is one small download with no material quality loss. The server that users run is Go, not Python.

## Decision

1. Export ONNX on the GPU box with `scripts/export_onnx_gpu.py`. Refuse the export if the ORT session is not on CUDA.
2. Quantize with ORT dynamic int8, per-channel, `MatMul` only. That copy was 598,012,474 bytes. Int8 is the CPU artifact. It does not run on the CUDA provider. Do not pretend it does.
3. Keep the fp16/fp32 safetensors as the trainable checkpoint. The int8 file is the downloadable graph.
4. Publish weights to Hugging Face and the int8 graph plus benchmark to the GitHub release. The release binary embeds that graph.
5. License is Apache-2.0 for our code and the weights. Laya and ModernBERT notices stay in `CREDITS`, `NOTICE`, and `licenses/`. Any company may use it commercially or not.
6. Next training does not start until ADR 0001 and ADR 0002 are still true. If the base model changes, write a new ADR first. Do not silently switch vendors.

## Consequences

A GPU config must name a CUDA-capable graph or fail. Auto mode may use CPU. The Go server is the product. Python remains only in `scripts/` for the training run.
