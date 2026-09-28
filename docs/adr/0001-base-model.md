# ADR 0001 — Base model is Laya, not a Chinese chatbot

Status: accepted  
Date: 2026-09-28

## Context

OpenJevX must answer typed `choice`, `score`, and `noul` questions with probabilities, in one forward pass, and stay small enough to ship. A Qwen3-1.7B full fine-tune scored 0.562 on typed-decisions and was the wrong vendor. The product is a Jev-class decision model, not a chat model.

## Decision

Start from `convaiinnovations/laya` (commit family 0.3.21, Laya pin `9d955671415fc19f069b9cc998928075c1f255ec`).

- Encoder: `answerdotai/ModernBERT-large` (395M) plus Laya's decision head (421M total).
- Method: RLCD, Reinforcement Learning for Calibrated Decisions.
- Built by Nandakishor Mukkunnoth, ConvAI Innovations. Credits stay in `CREDITS`.
- Do not start from `laya-typed-decisions` when the goal is a new specialization. That checkpoint is already trained on the public benchmark. Use it only as a reference number (0.766).
- Do not train Qwen, and do not train on this Mac.

## Consequences

Next run copies `scripts/train_openjevx.py`. New domain data must already be `state + questions + gold.probabilities`. Hard labels alone are not an RLCD target.

Published result of the first run, untouched 400-case test, CUDA:

| | |
|---|---|
| Overall | 0.774 |
| Choice | 0.745 (n=600) |
| Score | 0.731 (n=800) |
| Noul | 0.860 (n=600) |
| p50 | 22.8 ms per five-question case on an RTX 4090 |

Release gate: do not publish below 0.70 overall.
