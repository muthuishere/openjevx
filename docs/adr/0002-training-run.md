# ADR 0002 — How to train the next one

Status: accepted — launcher, image and watcher parts superseded by ADR 0009  
Date: 2026-09-28

## Context

The first good run was a single RTX 4090 on Vast.ai, not Kaggle and not the local Mac. The 2xT4 Laya notebook was adapted to one GPU. Interactive SSH setup wasted money. The job must be one immutable payload.

## Decision

Follow this order. Do not invent a new trainer.

1. Data. Public reproduction uses `LocalLLaMA/typed-decisions` train (1,200 cases, 6,000 decisions) and the untouched test split (400 cases, 2,000 decisions). A new domain replaces those two files but keeps the same row shape: `state`, `questions`, `gold` with full probability distributions.
2. Hold out calibration before training. Seed `20260922`, up to 400 items or 10%, whichever is smaller. Fit one temperature per type (`choice`, `score`, `noul`). Delete inherited `temperature_by_options` or the old buckets hide the new fit.
3. Code pin. Laya source is `9d955671415fc19f069b9cc998928075c1f255ec`. OpenJevX trainer is `finetuning/train/train_openjevx.py`. Launcher: see ADR 0009 (`finetuning/gpu/vast.py`, `finetuning/train/run_job.sh`).
4. Hardware. One verified RTX 4090, CUDA 12.4 image `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`, label `openjevx-...-DESTROY-AFTER`. Use the `vast-one-shot` skill. Do not assemble the box over SSH.
5. Runtime guard. `onnxruntime-gpu==1.22.0` only. Uninstall `onnxruntime` and `onnxruntime-gpu` first, then `pip install --force-reinstall --no-deps onnxruntime-gpu==1.22.0`. Version 1.30 needs CUDA 13 and silently falls back to CPU on this image. Abort unless the session provider is `CUDAExecutionProvider`.
6. Recipe, already in the script. 4 epochs, micro-batch 8, grad accumulation 8 (effective batch 64 on one GPU), encoder lr `2.5e-5`, head lr `1e-4`, AdamW, cosine, fp16 autocast, GradScaler, clip 1.0, `max_len` 1024, `head_max_len` 256. T4/4090 training uses fp16, not bf16.
   Superseded for v0.5.x: the live recipe is the `train` block of `finetuning/config.example.json` (1 epoch,
   micro-batch 16, grad accumulation 4, `max_len` 512, bf16 autocast on the 4090).
7. Benchmark on the GPU box before destroy. Pull artifacts, then destroy. A watcher that dies with the agent session leaves the GPU billing.

## Consequences

Override `BASE_MODEL`, `EPOCHS`, `MICRO_BATCH`, or `GRAD_ACCUM` with env vars. Do not edit the loss. If accuracy is under 0.70, do not publish. Personal chat and code transcripts are out of this run.
