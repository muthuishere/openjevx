# Report 9 — small probe of the four target uses (30 hand-checked cases)

Script: `clauderesults/probe.py`; raw: `clauderesults/probe_results.json`. Base = convaiinnovations/laya
converted to ONNX (not fine-tuned); v0.3 and v0.4 = released models. All 8-bit/4-bit as shipped.

| Use | base | v0.3 | v0.4 |
|---|---|---|---|
| Tool calling: pick the right call among candidates (5) | 3/5 | 3/5 | **5/5** |
| Tool calling: verify one call, right vs one-wrong-argument (10) | 7/10 | 6/10 | 8/10 |
| Reranking: right passage ranked #1 (3 queries) | 2/3 | 2/3 | 2/3 |
| Diagnosing: root cause (5) | 5/5 | 5/5 | 5/5 |
| Programming: bug / vulnerability in a snippet (7) | 3/7 | 3/7 | 3/7 |

Findings
- Programming is the gap: every model says "no bug" to real bugs (off-by-one, `if (user = null)`,
  divide by empty list). v0.4 also missed SQL injection that base caught. It can't read code yet.
- Tool calling: v0.4 picks the right call every time, but a call with ONE wrong argument
  (tomorrow→today, order 8841→8814) is still accepted as correct by all models.
- Reranking: fine for prose; fails the code query (Python read-file snippet not ranked first).
- Diagnosing: all 5/5; base is more confident (avg 0.93) than v0.4 (0.87).
