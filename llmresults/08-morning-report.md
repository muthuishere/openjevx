# Morning report — overnight fine-tune (2026-09-29)

## Bottom line
The fine-tune worked and is ready to ship. It is **staged, not published**: the public
typed-decisions test stayed at 76.2%, short of the 80% bar we agreed, so under our rule I did
not publish while you were asleep. Everything else improved a lot and nothing regressed.
To publish: `bash .local/release/v0.4.0/publish.sh` (one command, ~30 min).

## Results (held-out; the exact 8-bit file the server ships; Go server on CPU)

| Test (never trained on) | v0.3.0 (shipped) | v0.4.0 (new) |
|---|---|---|
| Public typed-decisions test (2,000) | 76.1% | 76.2% |
| Software-role held-out test (9,159) | 63.7% | **98.1%** |
| Software-role families never seen in training (1,158) | 66.1% | **87.3%** |
| Your `eval_openjevx`, 1k sample (1,052) | 49.6% | **64.3%** |
| opencode + OpenRouter set (9,557) | 63.2% | **71.0%** |
| Rule-labelled everyday set (1,194) | 45.1% | **54.4%** |
| Your 1,000-question CSV | 46.9% | **54.5%** |

Through **jevx with default thresholds** (yes ≥ 0.8, no ≤ 0.2), obvious questions now get a
usable answer 4/5 times (was 1/5): out of stock → yes 0.85, checkout down → page on-call yes
0.85 (was 0.27), `git push --force origin main` destructive → yes 0.84, and "outage vs tea vs
newsletter" → outage 0.91. Spike checks: score 4/4, stable under option shuffling and batching.

## Why 80% on the public test isn't reachable by more training
Where the human labellers agreed, every model scores ~87–89%. Where they disagreed (41% of
the test) every model scores ~60%, because the "right" answer there is a split vote. Hitting
80% overall would need ~68% on questions humans themselves split on. The original
typed-decisions-only fine-tune topped out at 76.8% for the same reason. I stopped spending here.

## What ran
- **Data**: 443,394 training decisions = 183,543 rule-labelled software-role decisions
  (56 families, 7 roles incl. agent/jevx-style) + a stratified sample of `train_openjevx`
  + typed-decisions oversampled 4x. Two generator bugs found and fixed
  (`meeting_attendance`, `release_signoff`).
- **Leakage**: `train_openjevx.jsonl` contained 18,269 questions from `eval_openjevx`, 8,372
  from the opencode test set, 972 from your CSV and all 1,194 rule-set questions.
  22,576 leaked questions were removed before training, so every number above is honest.
- **Quality judges**: NLI DeBERTa and Selene-1-Mini (local) and free OpenRouter models were
  first scored on questions with known answers. None reached 90% (best: Selene yes/no 73%),
  so none was allowed to remove data — they'd have deleted good rows. Reports 06 and 07.
- **GPU**: vast.ai RTX 4090. Smoke $0.27, full run $1.53 (2.75 h budget, ~90% of one
  pass, 40 decisions/s). **Total $1.80**; both boxes destroyed; no data left on any box or
  bucket. (One 306 KB smoke shard reached the private S3 bucket before you rejected that step; I
  deleted it.)

## Staged for publishing (branch `release/v0.4.0`, local only, commit b864b12)
- Server defaults to `openjevx.w8.onnx`; packages for macOS/Linux/Windows built with it and
  checked from a clean folder; README section "Fine-tune it further"; versions → v0.4.0.
- Hugging Face folder: `model.safetensors`, `encoder/`, `tokenizer/`, `rl_agent_config.json`,
  `openjevx.onnx` (fp32), `openjevx.w8.onnx`, model card, `benchmark.json`. No data.
- GitHub release: packages, Docker tars (built at publish time), `openjevx.w8.onnx`,
  `openjevx-model-fp32.zip`, `benchmark.json`. No data.

## Try it now
- New model: port 21126 (`jevx ... --profile openjevx-new`). Shipped model: port 21118.

## Weak spots to fix next
- Held-out families still weakest: `tech_leads/hiring_signals` 71%, `managers/feedback_timeliness` 69%.
- Score questions on free-model data: 53.8%.
- Correction to earlier: my local server on 21118 served the retired int8 model for part of
  yesterday (fixed and re-measured), and a leftover test server on 21119 answered some of my
  package checks (fixed; the v0.4.0 package is verified correct on a clean port, and the published
  v0.3.0 macOS download was re-downloaded from GitHub and verified correct too).
