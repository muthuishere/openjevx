# ADR 0004 — Training data for v0.5: what to gather

Status: proposed
Date: 2026-09-29

## Context

v0.4 fixed everyday software triage (software-role test 63.7% → 98.1%, families never seen in
training 66.1% → 87.3%), but a small probe of the real uses shows the gaps
(`clauderesults/09-usecase-probe.md`):

- **Conditions over state are shaky.** Age 18 → "adult" scores 0.51, stock 1 is called "out of
  stock", date comparisons are coin flips. Programming is a domain's rules written as conditions,
  so everything else rests on this.
- **Programming:** every model says "no bug" to real bugs (off-by-one, `if (user = null)`,
  divide by an empty list); v0.4 misses SQL injection.
- **Tool calling:** v0.4 picks the right call among candidates, but accepts a call with one wrong
  argument (`today` instead of `tomorrow`, order `8814` instead of `8841`).
- **Reranking:** fine for prose, fails for code.
- **Confidence:** correct answers stay near 0.75–0.88, below jevx's 0.8 bar, because training
  targets were 0.9/0.1. Certain answers must train toward 0.99.

The model is a classifier: it picks or verifies among options it is given. It does not write
code or arguments. Data must be shaped as questions with a known right answer.

## Decision

Build the v0.5 training set from seven kinds of data. We generate what can be computed by rule;
we gather what needs the real world.

### What we generate ourselves (no need to gather)

1. **Condition evaluation across ~40 domains** (`scripts/data/gen_conditions.py`): comparisons,
   and/or/not, membership, null, strings, counts, dates; boundary-heavy (17/18/19, 0/1/2,
   89/90/91); the same rule in English, rule text, and code (Python, JS, Go, SQL); "does this
   branch run", "what does this function return". Labels come from evaluating the condition.
2. **Everyday software-role decisions** (`scripts/data/gen_it_worker.py`, already in v0.4).
3. **Tool-call hard negatives:** from each correct call, change exactly one argument (value, type,
   unit, missing field, wrong tool).

### What we need you to gather

Each item: the source, how many, the shape, and what makes it good. Anything is useful even in
small amounts; real examples beat synthetic ones.

| # | Data | Target size | Where it can come from | Must have |
|---|---|---|---|---|
| A | **Real bugs with their fix** | 5k–50k pairs | Git history of your repos and good OSS repos: commits whose message says fix/bug; before = buggy, after = fixed | the buggy snippet, the fixed snippet, one line on what was wrong |
| B | **Code review comments** | 2k–20k | PR review threads (GitHub API) | the diff hunk, the comment, whether it was a real problem (changed after the comment) or a nit |
| C | **Real incidents** | 200–2k | Postmortems, incident tickets, on-call notes (yours, clients', public postmortems) | symptoms (alerts, logs, metrics), recent changes, the confirmed root cause, the fix |
| D | **Logs and stack traces with the answer** | 5k+ lines | Your services' logs, CI failures | the line or trace, and whether someone acted on it (or its category: config, dependency, code, infra) |
| E | **Tool-call traces** | 5k+ | Agent/LLM sessions (Claude Code, jevx logs that include content, OpenAPI-driven agents) | the user request, the available tools with argument schemas, the call that was made, and whether it was right (did it succeed / was it retried or corrected) |
| F | **Search / rerank judgments** | 1k queries, ~10 results each | Your docs search, code search, support-article search; clicks or "this answered it" | the query, the candidate results, which ones actually answered it (graded 0–3 if possible) |
| G | **Domain rules as written by people** | as many as possible | Policy docs, pricing pages, terms, runbooks, validation code in your repos | the rule text and, where it exists, the code that implements it |

| H | **Everyday basics, handmade** | 200–1,000 cases | Written by hand: the small rules people check every day | the rule, the facts, the right answer, and a near-miss twin that flips it |

**H in detail.** Small everyday rules the model must never get wrong: can I get a driving
licence (age >= 18), is the store open (time within opening hours), is the parcel late (today
after the promised date), is the battery low (below 20%), does the discount apply (cart >= Rs 500),
is it a fever (>= 38 C), does the bag fit (<= 23 kg), is the bill overdue, do two meetings clash.
Put the rule in the state and ask the model to apply it; write each case with its near-miss twin:

```json
{"state": {"rule": "You can apply for a driving licence at 18 or older.", "person": {"age": 18}},
 "question": "Can this person apply for a driving licence?", "type": "noul", "gold": "true"}
{"state": {"rule": "You can apply for a driving licence at 18 or older.", "person": {"age": 17}},
 "question": "Can this person apply for a driving licence?", "type": "noul", "gold": "false"}
```

Handmade cases are the most trusted part of the release gate; each can also be expanded by rule
into many variations around the threshold for training.

Useful format for all of them: one JSON object per line. The fields can be loose (we adapt them);
what matters is that the **right answer is known and comes from reality** (a fix that shipped, a
root cause that was confirmed, a call that succeeded), not from a model's guess.

### Quality rules (non-negotiable)

1. **No labels from a model.** Not hosted Jev, not an LLM judge. The judges we tried scored 49–73%
   on questions with known answers; they would teach mistakes.
2. **Keep a held-out slice untouched** for every source, split by repo, service or domain rather
   than by row, so tests measure generalisation.
3. **Leakage check before every training run** (`--exclude-keys`): no test question may appear
   in training. v0.4 found 22,576 leaked questions this way.
4. **Sharp targets only where the answer is certain** (0.99); disputed human votes stay soft.
5. **No secrets or personal data.** Scrub keys, tokens, emails and customer names from code,
   logs and tickets before they enter `data/`. Real-looking fake secrets are generated at
   runtime, never stored as literals.
6. **Data never goes to git, Hugging Face or the GitHub release.** It stays local; the model
   and scripts are what we publish.

### Where to put it

`data/incoming/<letter>-<source>-<date>.jsonl` (gitignored), plus a line in
`data/incoming/SOURCES.md`: where it came from, the licence or permission, and who collected it.

### Gates for v0.5 (on the shipped 8-bit file)

- Basics gate (`data/conditions_basics_gate.jsonl`): ≥ 99%, confident enough for jevx defaults.
- Programming (A/B-based held-out): clear gain over v0.4's 3/7 probe level.
- Tool calling: one-wrong-argument calls rejected; correct calls accepted with ≥ 0.8.
- Reranking: MRR@10 / NDCG@10 up on prose and code.
- Diagnosis (C/D-based held-out): root cause accuracy up.
- No regression on the v0.4 test sets (software-role, public typed-decisions, eval sample).

## Consequences

The generated parts (1–3) can be built and trained without waiting. The gathered parts (A–G)
decide how well v0.5 does on real work: bugs, reviews, incidents and tool traces from real
projects are worth more than any amount of synthetic data. Longer inputs (diffs, traces,
passages) need `MAX_LEN` 1024, which roughly doubles GPU time (still a few dollars per run).
