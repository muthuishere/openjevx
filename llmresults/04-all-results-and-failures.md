# Report 4 — All results and failures (baseline before the new fine-tune)

Model: shipped v0.3.0 (4-bit weight-only) unless noted. Every wrong answer is one row in
`ALL_FAILURES.csv` (run, id, domain, type, question, state, criteria, gold, predicted, confidence).
Totals are in `ALL_RESULTS_SUMMARY.csv`.

| Test set | Decisions | Accuracy | Errors | Note |
|---|---|---|---|---|
| Public typed-decisions test | 2,000 | 76.1% | 477 | shipped 4-bit |
| opencode + OpenRouter generated | 9,557 | 63.2% | 3,520 | shipped 4-bit; yes/no 69.9%, choice 63.1%, score 46.8% |
| Rule-labelled everyday set | 1,194 | 45.1% | 655 | shipped 4-bit |
| Owner's 1000-question CSV | 1,000 | 46.9% | 531 | 8-bit weight-only |
| Public typed-decisions train | 6,000 | 71.9% | 1,688 | old v0.2.0 int8, not re-run |

Correction: an earlier run of the public test and opencode set scored 65% because the server on
port 21118 was still running the retired int8 model. It was restarted on the 4-bit model and
both sets were re-run; the numbers above are from the corrected server. Released packages were
never affected.

Pattern: score questions are weakest everywhere; everyday software decisions sit near chance.
These are the numbers the new fine-tune has to beat.
