# Report 10 — v0.4.1 gate baseline (before training)

Everyday basics gate: `data/basics_gate.jsonl` (469 questions, 58 everyday rules, wording never used
in training). Condition basics gate: `data/conditions_basics_gate.jsonl` (300). "Usable" = right AND past
jevx's default thresholds (yes >= 0.8, no <= 0.2, choice/score >= 0.6). Scorer: `.local/gate041/gate_eval.py`.

| Model | Everyday basics: accuracy / usable / confidently wrong | Condition basics: accuracy / usable / confidently wrong |
|---|---|---|
| base (not fine-tuned) | 62.5% / 47.8% / 24.3% | 78.3% / 67.3% / 14.0% |
| v0.3 | 62.9% / 26.0% / 12.4% | 75.7% / 60.0% / 7.7% |
| v0.4 | 66.3% / 46.9% / 17.9% | 81.0% / 75.7% / 13.7% |

The 13 fundamentals through the jevx CLI (`.local/gate041/jevx13.sh`): v0.4 = 9/13 correct, 4 unsure, 0 wrong
(unsure: age 30 adult, checkout down -> page, ERROR db refused -> act, severity of site down).
