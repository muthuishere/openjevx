# ADR 0011 — Serve on ONNX Runtime 1.29; a smaller model is the next speed step

Status: proposed
Date: 2026-10-01

## Context

Users said inference is slow. The target is under 100 ms per request, end to end, on a CPU server.
Measurements are in llmresults/12. On ONNX Runtime 1.22 the v0.5.0 model took 172 ms for a 36-token request and
302 ms for a typical 164-token one. Almost all of that was the graph re-dequantizing its 8-bit weights to fp32 on
every call. Tokenizer, decode and HTTP together are about 1.5 ms.

## Decision

1. Serve on **ONNX Runtime 1.29.0** (Go binding `onnxruntime_go` v1.36.0). It fuses the 8-bit weights into
   `MatMulNBits` with int8 activations. Result: 22 ms short, 85 ms typical, 540 ms long (912 tokens), with the same
   gate numbers. The model file does not change.
2. Keep ORT's default CPU session options. Measured thread counts, fp32 accuracy levels and spinning were equal or
   worse. There is no new setting.
3. Don't write custom kernels or switch runtimes. The hot op is already ORT's int8 GEMM.
4. The next speed step is a **smaller distilled 8-bit model** (ModernBERT-base size). It must pass the gate (right
   AND confident), and it needs a GPU run approved by the owner. A lower `max_len` is a product decision, because the
   gate has no input longer than 300 tokens and can't detect what truncation loses.

## Consequences

- Typical requests (the size of every gate set) are under 100 ms on the M5 Pro. Long inputs and many-question
  batches are not.
- The GPU training box keeps `onnxruntime-gpu==1.22.0` (ADR 0002). Only the serving runtime moves.
- ORT 1.29 has no Intel-Mac build. `task setup` fails loudly on Intel Macs.
- x86 numbers still need to be measured: run `go test ./cmd/openjevx -run TestBench` on the server.
