# Local data-quality report (NLI + Selene-1-Mini, on-device)

_generated 2026-09-29 00:37:27_

## Stage 0 calibration (judge accuracy on the trusted key)

| judge | type | n | accuracy | trusted (>=0.90, n>=50) |
|---|---|---|---|---|
| nli | noul | 447 | 0.593 | no |
| selene | noul | 305 | 0.731 | no |
| selene | choice | 336 | 0.664 | no |
| selene | score | 433 | 0.494 | no |

Verdict rule (owner override): BAD only if two TRUSTED judges both disagree with gold on the same question. choice/score questions have only one possible judge (Selene; NLI never runs on them) so they cap at SUSPECT. Nothing is deleted -- bad_rows.jsonl and review_sources_local.txt are review lists only.

## Per-source results (sorted by bad_rate)

| source | n | bad_rate | suspect_rate | ambiguous_rate | decision |
|---|---|---|---|---|---|
| our-cases-it-worker/agent/claim_check | 2769 | 0.0 | 0.0 | 0.0 | KEEP |
| our-cases-it-worker/everyone/prioritize | 2172 | 0.0 | 0.0 | 0.0 | KEEP |
| our-cases-it-worker/agent/same_entity | 2769 | 0.0 | 0.0 | 0.0 | KEEP |
| our-cases-it-worker/agent/flaky_vs_real_ci | 1846 | 0.0 | 0.0 | 0.0 | KEEP |
| our-cases-it-worker/agent/build_verdict | 1846 | 0.0 | 0.0 | 0.0 | KEEP |
| our-cases-it-worker/agent/secrets_scan | 1846 | 0.0 | 0.0 | 0.0 | KEEP |
| our-cases-it-worker/agent/log_line_triage | 924 | 0.0 | 0.0 | 0.0 | KEEP |
| our-cases-it-worker/agent/injection_screen | 1846 | 0.0 | 0.0 | 0.0 | KEEP |
| our-cases-it-worker/agent/effort_routing | 923 | 0.0 | 0.0 | 0.0 | KEEP |
| our-cases-it-worker/agent/judge_turn | 2769 | 0.0 | 0.0 | 0.0 | KEEP |
| our-cases-it-worker/agent/ticket_routing | 1846 | 0.0 | 0.0 | 0.0 | KEEP |
| our-cases-it-worker/agent/bash_guard | 923 | 0.0 | 0.0 | 0.0 | KEEP |
| our-cases-it-worker/agent/tool_choice | 1846 | 0.0 | 0.0 | 0.0 | KEEP |
| our-cases-it-worker/agent/search_relevance | 923 | 0.0 | 0.0 | 0.0 | KEEP |
| our-cases-it-worker/everyone/phishing_email | 112 | 0.0 | 0.0 | 0.0 | KEEP |

## REVIEW sources (0) with example BAD questions

Row-level BAD questions (review, not deleted): 0 across 0 sources.

## Conclusion

Neither local specialist judge is reliable enough on this data to remove or
flag anything. Stage 0 calibration against the trusted key (1,216 questions:
~1,188 from `.local/eval/test.json` where human labellers agreed, plus 28
hand-verified "basic" rows from `.local/eval/questions_1000.csv`) put every
judge/type combination below the 0.90 trust bar (NLI noul 0.593; Selene noul
0.731, choice 0.664, score 0.494). Per the owner's rule, an untrusted
judge+type combo is recorded but never drives a verdict, so with zero
trusted combos every question resolves to OK by construction. Stage 1 (NLI
on `noul` questions) was run partially before being stopped once calibration
made the outcome obvious: 25,360 of ~123,530+ noul questions in source A
(plus the small `noul` slice of the source-B reservoir sample) were judged,
recording 118 raw `nli_conflict` flags -- informational only, since the NLI
judge is untrusted for `noul` and none of these can become BAD/SUSPECT.
Stage 2 (Selene) was stopped before it started running proper judging (only
the calibration pass used Selene). **No rows were removed or flagged for
review; `bad_rows.jsonl` and `review_sources_local.txt` are empty.**

## What a trustworthy judge would need

- **Bigger / better calibration set.** 1,216 questions (305-447 per type) is
  enough to *detect* an unreliable judge but leaves wide confidence
  intervals near the 0.90 bar; a few thousand per type would let "trusted"
  mean something.
- **A premise the NLI model was trained for.** `state_text()` flattens
  arbitrary JSON into `"key: value; key2: value2"` fragments -- not the kind
  of natural-language premise DeBERTa-v3-large-mnli-fever-anli-ling-wanli
  was fine-tuned on. A domain-specific templated renderer per `domain`
  (turning state into an actual sentence) would likely lift NLI noul
  accuracy well above 0.593.
- **A stronger or fine-tuned evaluator LLM.** Selene-1-Mini (8B, Q4_K_M) at
  temperature 0 landed at 0.49-0.73 depending on type; `score` questions
  (0.494, barely better than chance on a 4-way scale) suggest the rubric is
  either genuinely ambiguous or the model is too small/quantized for
  calibrated ordinal judgment. A larger evaluator (or one fine-tuned on
  OpenJevX's own score rubrics), few-shot examples in the prompt, or
  self-consistency voting over multiple samples (not just temperature-0
  greedy) would all plausibly help.
- **Independent, larger calibration keys per domain** rather than one pooled
  key -- accuracy on `it_worker` domains specifically wasn't measured
  (calibration only covers the `typed-decisions`/`basic` distributions), so
  even a "trusted" judge from this key would be an extrapolation onto A's
  data.

## Throughput / timing

- Stage 0 calibration (NLI + Selene, 1,216 questions, mixed noul/choice/score): ~14 min wall clock, ~1.45 questions/s combined (both judges called per applicable question).
- Stage 1 NLI (MPS, fp16, batch 16): peaked at ~235 q/s, averaged ~124-210 q/s once contention from a concurrent process on the same machine kicked in; 25,360 noul questions judged before the run was stopped on the owner's instruction.
- Stage 2 Selene (Ollama, temperature 0, format json): only run during calibration (~0.7-1 s/call observed there); the bulk choice/score judging pass was never started.
- Total wall clock for this run: ~23 minutes (venv/model setup + calibration + partial Stage 1).
- Disk free on `/System/Volumes/Data`: ~11 GiB at the start of this run, ~120 GiB at the end. The increase is not attributable to this pipeline (its own footprint is +2 venvs with torch ~1.8 GB, +NLI model ~0.8 GB, +Ollama Selene GGUF ~4.9 GB, one small reservoir sample of source B); a separate, unrelated agent shares this machine and was observed deleting large amounts of its own data during the run (see Caveats).

## Caveats

- Calibration key is small (~1,216 questions); accuracy estimates, especially per type, carry real sampling noise.
- NLI only ever judges noul questions; choice/score verdicts rely on Selene alone and can never reach BAD under the two-trusted-judge rule.
- Selene-1-Mini runs at temperature 0 via Ollama; JSON parse failures are retried once, then recorded as invalid (excluded from conflict math).
- **Shared-directory collision**: a separate, unrelated agent process also uses `.local/quality/` on this machine and was observed wiping its entire contents every ~5-6 minutes (its own `run.log`/`s3_delete_one.py` reappearing, everything else gone), which twice deleted this pipeline's venv, HF cache and log files mid-run. To avoid losing checkpointed work, all real working state (venv, HF cache, calibration/judged checkpoints) was moved to `.local/quality_local_work/` instead; only the four mandated deliverable filenames (`local_judged.jsonl`, `review_sources_local.txt`, `bad_rows.jsonl`, `calibration_local.json`) are copied into `.local/quality/` (via `python scripts/data/quality_local.py publish`), which is not touched otherwise. `.local/quality/judged.jsonl`, `report.json` and `drop_sources.txt` (the separate cloud-judge run's files) were never read or written by this pipeline.
- Stage 1 was stopped part-way through source A (25,360 of ~123,530 noul questions in A alone, plus B's small noul sample) once calibration showed no judge could be trusted; the remaining noul questions in A, and all of B's/A's choice/score questions, were never judged. `local_judged.jsonl` therefore reflects a partial, Stage-1-only pass.