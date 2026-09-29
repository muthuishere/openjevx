# OpenJevX training data quality report

> **Free chat-model AND local specialist-model verdicts are ADVISORY ONLY. Nothing is ever deleted from data/it_worker_train.jsonl or data/train_openjevx.jsonl. A judge verdict only counts as evidence for (model,type) combos scoring >=0.90 accuracy on the trusted key (.local/eval/test.json, label_agreement[qid].argmax_agree==true rows only). No (model,type) combo -- free-chat or local specialist -- cleared that bar in this run, so NO source is recommended for review-as-suspect and NO row was removed or flagged by any judge.**

Generated 2026-09-29 00:41:32. Run started 00:32:36, stopped early by the owner at ~00:41 (~8-9 min in) before either judging pipeline produced trusted evidence.

## Judge calibration against the trusted key

Trusted key: `.local/eval/test.json`, restricted to rows where `label_agreement[qid].argmax_agree` is true. Trust threshold: 0.90.

This run's own free-chat-model calibration (openrouter/*:free + opencode/*-free, .local/eval/llm/gen.py-style pools) was started but stopped by the owner before any single model finished its calibration batches, so it produced zero usable (model,type) accuracy rows -- nothing from that pool is in the table below.

| judge | type | n | accuracy | trusted >=0.90 |
|---|---|---:|---:|:---:|
| selene | noul | 305 | 0.731 | no |
| selene | choice | 336 | 0.664 | no |
| nli | noul | 447 | 0.593 | no |
| selene | score | 433 | 0.494 | no |

**No (model, type) combo -- from either the free-chat-model pool (openrouter/*:free, opencode/*-free) or the local specialist pool (NLI, Selene) -- reached the 0.90 trust threshold.** Selene's best type (noul) topped out at 0.73; NLI at 0.59; Selene score at 0.49. The free-chat-model pool never finished a single model's calibration before the run was stopped, so it contributes zero rows.

## Decision rule

Using TRUSTED-ONLY judged questions per source: REVIEW (insufficient trusted evidence) if n_judged_trusted<20; REVIEW (high bad/unclear rate) if bad_rate>=0.25 or unclear_rate>=0.40; REVIEW (elevated bad rate) if 0.10<=bad_rate<0.25; else KEEP.

## Result: no rows removed, no source flagged

Because zero (model,type) combos are trusted, zero judged questions count as evidence (`total_judged_questions_trusted = 0`). Per the owner's rule that untrusted verdicts must never cause a removal or a review flag, **`review_sources.txt` is empty** and no family in `data/it_worker_train.jsonl` (49 families) or source in `data/train_openjevx.jsonl` (8 sources) was recommended for review. Nothing in either data file was touched, read-modified, or deleted.

## Totals

- Sources in scope: 57 (49 it-worker families + 8 train_openjevx.jsonl sources)
- Trusted judged questions: 0
- Sources KEPT: 0 · flagged REVIEW: 0 · insufficient trusted evidence (i.e. everything, no verdict either way): 57

## What ran

- **This script** (`scripts/data/judge_filter.py`): two-judge blind evaluation over free opencode/openrouter models, gated by a >=0.90 accuracy calibration pass on `.local/eval/test.json`. Launched detached at 00:32:36 with a 3h cap; calibration alone needs ~15 models x ~300 calibration questions each, and the owner stopped the run at ~8-9 minutes -- before the first model finished calibration, so it produced no usable data.
- **A separate, concurrently-run local specialist pipeline** (not authored by this script): NLI and Selene local models, calibrated on the same trusted key. Results in `.local/quality/calibration_local.json` / `calibration_raw.jsonl` / `local_judged.jsonl`. Both failed the 0.90 bar (table above), so per the owner's rule their per-row verdicts (`local_judged.jsonl`, 25,360 rows already produced) are advisory-only and were **not** used to compute any bad_rate/unclear_rate or recommend any source.

## Files

- `.local/quality/calibration_local.json`, `.local/quality/calibration_raw.jsonl` -- local specialist (NLI/Selene) calibration, completed
- `.local/quality/local_judged.jsonl` -- local specialist per-row verdicts (advisory only, not used for any decision since neither judge cleared the trust bar)
- `.local/quality/review_sources.txt` -- empty (nothing recommended)
- `.local/quality/report.json` -- this report as structured data

## Next step (not done here)

To get real evidence-backed recommendations, re-run `scripts/data/judge_filter.py` to completion (or use a smaller `--calib-per-type` / trimmed model pool to fit the time budget), and/or recalibrate or swap the local NLI/Selene judges for ones that can clear 0.90 accuracy per type on the trusted key. Until then, every family/source in both training files should be treated as unvetted by automated judges -- not as bad, just as unverified.
