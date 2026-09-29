# clauderesults

Everything from the OpenJevX model-quality investigation, in one place.

## Summary reports (my responses)

- `01-quantization-and-everyday-domain-report.md` — the int8 quantization bug, its fix (4-bit
  weight-only), and the finding that everyday-domain accuracy sits near chance regardless of
  quantization (a training-data problem, not a quantization one).
- `02-v0.3.0-release-and-jevx-report.md` — the v0.3.0 rebuild/release, the 8-bit comparison, the
  jevx CLI run with real repo context, and the confidence-calibration finding (jevx's default
  thresholds mark correct-but-weak answers "unsure").
- `03-status.md` — status snapshot: release state, opencode generator progress.
- `04-all-results-and-failures.md` — final consolidated numbers and where every failure row lives.
  (added once the last two background runs finish)

## Data files

- `questions_1000.csv` / `questions_1000_result.csv` — the owner's 1000-question set (state,
  question, criteria, expected) with the 8-bit model's actual answers and confidence.
- `cases_rules.jsonl` — my rule-labelled everyday-domain cases (deterministic gold, computed from
  facts in the state — not model-generated).
- `cases_llm.jsonl` — the opencode-generated, OpenRouter-blind-labelled test cases (~2,900 cases,
  ~9,500 questions kept where two free models agreed).
- `ALL_RESULTS_SUMMARY.csv` / `ALL_FAILURES.csv` — every eval run's accuracy and every wrong
  answer across all of them, one row each. (added once the last two background runs finish)

## Where the servers and scripts live

All of this ran against local OpenJevX servers under `.local/eval/` in the repo (not committed —
they're 400MB+ model files and scratch scripts). The shipped model is `.local/openjevx.w4.onnx`
(4-bit weight-only), released as v0.3.0.
