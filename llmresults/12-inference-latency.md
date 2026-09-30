# Report 12 — Inference latency: where the time goes, and what fixed it

Model: v0.5.0 folder (`openjevx.w8.onnx`, 598 MB, sha256 `109b2b1d…`). Machine: Apple M5 Pro, 18 cores, macOS,
CPU only. Benchmark: `cmd/openjevx/bench_test.go`, 20 rounds after 2 warm-up rounds:

    OPENJEVX_BENCH=$PWD/.local/model OPENJEVX_ORT=$PWD/.local/libonnxruntime.dylib go test ./cmd/openjevx -run TestBench -v -timeout 30m

Cases: `short` = 36 tokens, `typical` = 164 tokens (the longest gate set, logs, is p50 184 / max 288 tokens),
`long` = 912 tokens (close to `max_len` 1024). `x1` = one question; `x8` = eight questions about the same state.

## Where the time goes (before)

| stage | time |
|---|---|
| tokenizer + encode | 0.08–1.0 ms |
| graph execution | 99.6%+ of the request |
| decode (softmax, JSON map) | < 0.04 ms |
| HTTP + JSON (server vs in-process, typical) | ~0.5 ms |
| cold start: read + sha256 / session / first run | 284 / 326 / 246 ms |

The graph stores 8-bit weights as `DequantizeLinear -> MatMul` (120 pairs). ONNX Runtime 1.22 does not fuse that
pattern on CPU: it rebuilt ~300 M fp32 weights and ran fp32 MatMuls **on every request**. That is why a 36-token
request took 172 ms: most of it was dequantizing, not thinking.

## Fix 1: ONNX Runtime 1.22 -> 1.29 (no model change)

ORT 1.29 fuses `DequantizeLinear + MatMul` into `MatMulNBits` (8-bit weights stay 8-bit in memory) and by default
quantizes activations to int8 for it (`session.qdq_matmulnbits_accuracy_level` = 4), running on the ARM
SDOT/I8MM kernels here, AVX2/AVX-512-VNNI on x86. Profile after (36 tokens): MatMulNBits 47%, then small ops.

| case | tokens | before p50 | before p95 | after p50 | after p95 | after p99 | speed-up |
|---|---|---|---|---|---|---|---|
| short x1 | 36 | 172.2 | 175.9 | **22.4** | 23.0 | 23.0 | 7.7x |
| typical x1 | 164 | 302.2 | 308.0 | **84.8** | 85.8 | 85.8 | 3.6x |
| long x1 | 912 | 1224.3 | 1242.5 | 539.6 | 576.6 | 576.6 | 2.3x |
| typical x8 | 1319 | 1527.1 | 1551.0 | 666.4 | 689.5 | 689.5 | 2.3x |
| long x8 | 7303 | 9676.0 | 11171.4 | 4419.8 | 4480.8 | 4480.8 | 2.2x |

All times ms, end to end in process (encode + run + decode). Over HTTP the typical request is 85.4 ms p50 / 87.0 p95.
Cold start after: read + sha256 282 ms, session 678 ms (the fusion pass), first run 50 ms: about 1.0 s once per process.

Accuracy: full gate on ORT 1.29 against the same model gives the same numbers as the v0.5.0 report on ORT 1.22
(basics 96.3% / 94.3% right & confident, confidently wrong 3.0% / 3.7%; eval 98.7% / 97.5%; logs 91.7% (was 91.7%),
confidently wrong 7.9% (was 8.0%); jevx 13: 11/13). **v0.5.0 already failed three gate lines before this work**
(confidently wrong on both basics sets > 2%, jevx 13 < 12); the runtime change did not add or remove a failure.

## What did not help (measured)

| tried | typical 164 tok | long 912 tok | verdict |
|---|---|---|---|
| default (all cores, spinning on, level 4) | 81.9 | 540–690 | keep |
| intra-op threads 4 / 6 / 10 | 130 / 100 / 77 | – | default is as good; no knob needed |
| accuracy level 0 / 1 (fp32 compute) | 207 / 224 | 688 / 678 | slower, keep level 4 |
| intra-op spinning off | 93.7 | 590 | within noise (±15% at 912 tok) |

Session creation happens once, the session is reused, and the batch is padded to the longest item, not to
`max_len`, so the other session items (load once, dynamic padding) were already right. IO binding would save
copies of a few KB per request; not worth code.

## Why the long cases cannot reach 100 ms with this model

The encoder is ModernBERT-large sized (28 layers, hidden 1024, ~300 M encoder weights). One 912-token request is
about 2 × 300 M × 912 ≈ 0.55 TFLOP. MatMuls are ~93% of it (attention is ~7% at this length), and they already run
on ORT's int8 kernels. No runtime swap (OpenVINO, a hand-written Go/Rust/C path, SIMD kernels) changes the FLOP
count; at best it trims tens of percent. Hand-written kernels are not justified: the hot op is already a vendor int8
GEMM. Only three things move the long case:

1. **A smaller model** (distil to ModernBERT-base size: 22 layers, hidden 768, ~44% of the FLOPs, ~150 MB 8-bit).
   Needs a GPU training run and must pass the gate. Not started: renting a GPU needs the owner.
2. **A lower `max_len`** (for example 512). Every gate state is under 300 tokens, so the gate would pass while
   silently dropping the end of long inputs. That is a product decision, not a speed fix.
3. **Batching of many questions** re-encodes the shared state once per question (the question comes first in
   the input). A layout that shares the state would need retraining.

## Not measured

- x86 server CPUs. This Mac is ARM (no VNNI). `MatMulNBits` has AVX2 and AVX-512-VNNI int8 kernels in ORT, so the
  same win is expected, but the numbers above are M5 Pro numbers. The benchmark runs anywhere with `go test`.
- Intel Macs: ORT 1.29 ships no osx-x86_64 build; `task setup` now says "unsupported platform" there.
