# ADR 0006 — Fine-tune OpenJevX on your own data: the guide, the toolkit, the skill

Status: accepted
Date: 2026-09-29

## Context

Every AI-native workflow is a chain of small decisions between things that are already
automated: watch the logs → *is this worth paging?* → page. Build finished → *does this diff
need a human?* → route. Ticket arrived → *which team owns it?* → assign. The endpoints and
functions exist; the missing piece is a tiny, fast, owned decision in between.

A general LLM can make that decision, but slowly, expensively, and without knowing the caller's
rules. RAG bolts the rules on at request time and pays for it on every call. A decision model
(jev, and OpenJevX as the open one) answers in milliseconds, but the shipped model does not know
*your* facts: that your week is four days, that a working day is three hours, that stock `1`
is "in stock", that repo `billing` is owned by team `pay`.

Today the repo can train such a model (`finetuning/train/train_openjevx.py`, `finetuning/dataprep/package_shards.py`,
`finetuning/train/run_job.sh`) and the site has a short "Fine-tune it further" page, but nothing
takes a developer from "here are my documents, policies and systems" to "here is a labelled
dataset I trust" to "here is my model running on port 21118". ADR 0004 says what data to gather
for *our* release; nothing says how *you* gather yours.

## Decision

Ship three things together, in this order, on the `finetune` page and in the repo:

1. **A long-form guide** at `site/src/pages/finetune/` written as a blog post for a developer
   who owns a production system: why a decision model, why not RAG or a prompt, then a numbered
   walk from documents to a served model. Every step names the script that does it.

2. **A toolkit** under `finetuning/yourdata/` (plain Python, stdlib plus what the trainer already
   needs). One file per step, each runnable on its own, each idempotent:

   | step | file | in | out |
   |---|---|---|---|
   | 1 collect | `collect.py` | folders of `.md`, `.txt`, `.pdf`, `.yaml`, OpenAPI specs, code | `data/yourdata/sources.jsonl` (one chunk per line, with origin) |
   | 2 extract rules | `extract_rules.py` | `sources.jsonl` | `data/yourdata/rules.yaml`: candidate rules (`subject`, `operator`, `threshold`, `unit`, `text`, `origin`) for the developer to confirm and edit |
   | 3 generate | `generate.py` | `rules.yaml` | `data/yourdata/train.jsonl`, `gate.jsonl`: boundary-heavy, many phrasings, near-miss twins, gold computed by evaluating the rule (never written by hand, never from a model) |
   | 4 label | `label.py` | records from your systems (tickets, alerts, tool calls, logs) as loose JSONL | the same rows in the trainer's shape, with the gold field mapped from the field that reality filled in (`resolved_as`, `succeeded`, `owner`) |
   | 5 test first | `probe.py` | `gate.jsonl` + a running OpenJevX (base or yours) | per-rule accuracy and confidence; this is the "test before you train" step and the release gate after |
   | 6 build | `build.py` | `train.jsonl` + the public shards | calls `finetuning/dataprep/package_shards.py --extra-train --exclude-keys` so no gate question leaks |
   | 7 train | reuse `finetuning/train/run_job.sh` | shard | fp32 weights + `openjevx.onnx` |
   | 8 ship | reuse `finetuning/export/quantize_w8.py` | onnx | `openjevx.w8.onnx`, set `"model"` in `openjevx.json` |

   `probe.py` runs at two points: before training, to show what the base model gets wrong on
   your rules, and after, on the same gate file, so the gain is measured on questions the model
   never saw.

3. **An agent skill** at `skills/openjevx-finetune/SKILL.md` (with `references/`) so any agent
   session in a user's repo can run the eight steps, draft `rules.yaml` from the user's docs,
   ask the user to confirm each rule, and never invent a label. The skill is the operational form
   of the guide; the guide is the readable form of the skill.

Rules that carry over from ADR 0004 and are restated here because they apply to the reader's data:

- Gold comes from a rule that is evaluated or from something reality confirmed. **No LLM labels.**
- Hold the gate file out by rule and by phrasing, not by row.
- Leak check before every run.
- Scrub secrets and personal data before anything enters `data/`. `data/yourdata/` is gitignored.
- Certain answers train toward 0.99; disputed ones stay soft.

## Options considered

### A. Where the guide lives

| | option | for | against |
|---|---|---|---|
| A1 | Rewrite `finetune/index.astro` as the long post | one URL, already in nav, no new layout | the current short "get the weights" content gets buried; page becomes long |
| A2 | Keep `finetune/` short, add `finetune/guide/` (or `blog/`) for the post | short page stays a landing; post gets room | two pages to keep consistent |
| A3 | Astro content collection + Markdown post under `site/src/content/blog/` | future posts are one `.md` each | new plumbing for one post now |

**Pick A2.** Keep the landing page, add one long post. Move to A3 when a second post appears.

### B. How rules get out of documents

| | option | for | against |
|---|---|---|---|
| B1 | Regex and heuristics (`at least N`, `N or older`, `below N%`, `within X hours`, `every K days`) | zero cost, deterministic, offline | misses prose that does not follow patterns |
| B2 | An LLM (Claude, via the skill) drafts the rules, the developer confirms each | catches prose, tables, implied rules | draft only; a wrong rule that slips through teaches a wrong model |
| B3 | Both: B1 in `extract_rules.py`, B2 in the skill, confirm step mandatory in either | best recall, human-owned truth | more surface |

**Pick B3.** The tool proposes, the developer disposes. The confirmation step is what makes
"no labels from a model" hold even when a model helped find the rule.

### C. What kinds of decisions to support in v1 of the toolkit

| | option | scope | note |
|---|---|---|---|
| C1 | Threshold rules only (numbers, dates, time windows, membership) | small, fully generated, matches `gen_basics.py` | covers "workdays are 4", "shift is 3 h", "stock 1 is in stock" |
| C2 | C1 + routing/ownership (`choice` from a fixed list: team, severity, queue) | adds the most common production decision | gold from a lookup table or from the field reality filled |
| C3 | C2 + verification of tool calls and log triage from historic records | the ADR 0004 A–F classes | needs real records; `label.py` maps them |

**Pick C2 for the guide's worked example, build `label.py` so C3 works when records exist.**

### D. Where training runs

| | option | cost | note |
|---|---|---|---|
| D1 | vast.ai one-shot, existing `finetuning/train/run_job.sh` | ~$2 per run (v0.4.0 was $1.80) | proven, documented, the only supported path (Kaggle was dropped in `8da45be`) |
| D2 | Any CUDA box the reader already has | $0 | same `train_job.py`, document the env knobs |
| D3 | Local CPU/MPS | $0 | hours for a small set; document as "possible, not recommended" |

**Pick D1 as the documented path, D2 as "if you have a GPU".** A reader's dataset is usually
thousands of rows, not 443k, so one run is minutes and cents.

### E. Skill placement

| | option | note |
|---|---|---|
| E1 | `skills/openjevx-finetune/` in this repo, installable by copying or symlink | visible, versioned with the scripts it drives |
| E2 | Only in the owner's private skills workspace | invisible to readers |
| E3 | Both, this repo canonical | one source of truth, private copy synced |

**Pick E3**, canonical here.

### F. Format of the labelled data

Keep the trainer's existing row shape (`state`, `questions {id: {type, instructions, criteria}}`,
`gold`), since `adapter.py` already turns it into targets and `package_shards.py` already dedupes
and leak-checks it. Do not add a second schema. `label.py` maps loose fields into it.

## Consequences

- The `finetune` page grows into a landing page plus one long post; the nav does not change.
- New folder `finetuning/yourdata/`, new gitignore line `data/yourdata/`, new folder `skills/`.
- The probe-before-train step gives readers a number on day one without renting anything.
- Nothing about the trainer, loss or export changes.
- Open: whether the `extract_rules.py` heuristics also read code (`if age >= 18`) in v1 or
  only prose. ADR 0004 item 1 already generates code-form conditions from rules, so prose-only
  extraction plus rule-driven code generation covers it.
