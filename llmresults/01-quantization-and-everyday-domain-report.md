# Report 1 — Quantization bug + everyday-domain accuracy

Ran 12,200+ decisions against the local OpenJevX server. Two separate problems, one fixable
without retraining, one that needs a new fine-tune.

## Test results (errors in brackets)

| Test set | Shipped int8 (v0.2.0) | fp32 (same weights) |
|---|---|---|
| Public test, 2,000 decisions | 65.1% (698) | 76.8% (464) |
| Public training data, 6,000 decisions | 71.9% (1,688) | not run |
| My everyday set, 5,200 decisions | 45.8% (2,821) | 46.8% (2,764) |

My everyday set: 1,700 cases across 20 common domains (incidents, refunds, invoices, fraud,
sales leads, deploys, security, and others). Every answer is computed by rule from facts in the
state; spot-checked the misses — labels are right.

## Problem 1: the int8 file cost ~12 points

ADR 0003 said int8 has "no material quality loss," but the 77.4% benchmark was measured on
PyTorch with a GPU, never on the shipped file. Re-quantized and scored each variant on 500
decisions through the Go server:

| Variant | Accuracy | Size | Time / 5-question case |
|---|---|---|---|
| Shipped int8 (dynamic) | 65.6% | 598 MB | 0.15 s |
| int8 except one MLP matrix per layer | 77.2% | 883 MB | 0.25 s |
| 4-bit weight-only | 77.0% | 461 MB | 0.42 s |
| fp32 | 77.4% | 1.7 GB | 0.40 s |

Root cause: dynamic int8 quantizes activations per-batch, so the model's answer to one question
depended on what else was in the same request. Verified directly: the same 6 yes/no questions,
asked one at a time vs. together in a batch, gave **0/6 matching answers** on int8 (e.g. "the
light is on" → 0.739 in a batch of 6, 0.291 asked alone). Weight-only quantization (4-bit, 8-bit)
does not have this bug — batch order changed nothing.

## Problem 2: the fine-tune only works on the four trained workflows

On everyday scenarios, even fp32 lands near guessing: 30% on score questions, 38% on choice
questions. Yes/no questions largely ignore the state:
- Stock at 0 units vs. 5,000 units → "reorder now" scored 0.117 vs. 0.102.
- Checkout at 95% errors vs. 0% → "urgent" scored 0.15 vs. 0.07.

The model learned "yes/no questions are usually no" and catches only 37% of true "yes" cases.
Training data explains this: 1,200 cases from 4 workflows, and on 41% of test decisions the
labellers didn't agree with each other — noise was rewarded more than reading the state.

## Proposed plan (as given at the time)

1. Ship the fix immediately (no GPU needed): switch the release to a non-batch-dependent
   quantization. → done in v0.3.0, see report 2.
2. Build better training data: public typed-decisions + rule-labelled everyday cases (grown to
   ~50 domains) + opencode/OpenRouter-generated cases + counterfactual pairs (same question,
   deciding fact flipped, answer flipped) to force the model to read the state.
3. Hold out whole domains from training to measure generalisation, not memory.
4. Train on a rented GPU (vast.ai) using the pinned recipe in ADR 0002.
5. Gate release on the exact shipped file: ≥77% public test, ≥85% held-out everyday domains,
   yes/no answers that flip when the deciding fact flips.
