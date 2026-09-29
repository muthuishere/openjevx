# OpenJevX data + finetune session — what was done (2026-09-28)

Goal stated by the owner: gather every usable jev/typed-decision dataset, consolidate into ONE
English-only training set with no manufacturer license text embedded, then finetune
`muthuishere/openjevx` (a Laya / ModernBERT-large typed-decision model whose everyday-domain
decisions were judged nonsense per `clauderesults/01`) and ship ONNX. Serving build goes through
GitHub; everything needed to retrain lives in one place on Hugging Face. Training switched to
vast.ai (Kaggle packaging kept only as unused fallback).

## 1. What was gathered (raw, under `data/raw/`)

| source | what it is | size | status |
|---|---|---|---|
| `AlexWortega/openjev-data` | 2.35M-row NLI mixture behind openjev (text/agentic/faith/ifollow/longdoc/bullshit/distill parts, `premise/hypothesis/label` 0=contra,1=entail,2=neutral) | 238 MB | downloaded, verified 14 parts |
| `ZefanCai/Open-Jev` | typed-decision controls (extraction, mailroom, IR, citation), en/zh | 87 MB | downloaded (236 files) |
| `ZefanCai/Open-Jev-v1.1` | community-hard-mix-v2 (raw jsonl.gz trains/val/ood) | 90 MB | downloaded |
| `tasksource/tasksource-jev-typed-decisions` | 208,334 rows `state/kind(noul,choice,score)/question/options/target` compiled from dozens of task-source datasets | 1.45 GB | downloaded (12 train shards) |
| `SargeDev/jev-distill-corpus-v3` | distilled typed-decision corpus 430MB train | 490 MB | downloaded |
| `LocalLLaMA/typed-decisions` | THE base-recipe set of our model: per-workflow `state+questions+gold` (all/agent_trace/customer_service/invoice/security) | ~1.5 MB | downloaded incl. test split |
| `samatv256/jev-decisions-v1` | 22 GB RL-labeled decisions | 22 GB | background download ran/dropped; not merged yet |
| `muthuishere/openjevx` (model repo) | our shipped model: encoder 842.6MB safetensors + `openjevx.int8.onnx` 598MB + tokenizer + benchmark.json | — | inspected, not downloaded (weights stay on HF) |
| `openjev/openjev` (model repo) | weights only, no data | — | skipped, data lives in AlexWortega/openjev-data |

Ownership context looked up inside the repo: `scripts/train_openjevx.py` is the existing CUDA
finetuner (RLCD + calibrate + benchmark, requires CUDA); `scripts/export_onnx_gpu.py`,
`scripts/quantize_w8.py`, `scripts/finish_vast.py`, `scripts/publish_hf.py` complete the
GPU->ONNX->publish path, so all new work reuses these.

## 2. What was consolidated (under `data/`)

Builder: `data/build_master.py` (session log `clauderesults/build-notes.md` does not exist; runs
~166 s end-to-end). Outputs:

- `data/train_openjevx.jsonl` — **2,144,252 typed-decision records (2.7 GB)**, exact OpenJevX
  shape: `{"source","domain","state","questions":{qid:{type,instructions,criteria}},"gold"}`
- `data/eval_openjevx.jsonl` — **26,574 held-out records (34 MB)** (typed-decisions test split,
  tasksource test/validation)
- `data/jev_train_all.jsonl` — **5,455,001 flat aux rows (2.9 GB)** in NLI frame for auxiliary
  distillation (openjev-data + SargeDev 2.27M + OpenJev-v1.1 1.05M + …)

Per-source composition of train: tasksource-train 2,139,476; our-cases-llm 2,546;
our-cases-rules 400; typed-decisions train 2,030 across 5 workflows.

Native-gold win: `clauderesults/cases_rules.jsonl` (400 cases, 1,194 questions, deterministic
rule label) and `cases_llm.jsonl` (2,914 cases, 9,557 questions kept when two free OpenRouter
models agreed) were merged verbatim — these are the highest-value rows for the everyday-domain
failure fix.

Bugs fixed during merge (kept in builder script): file-handle/Counter name collision in flat(),
JSON-string-encoded gold/questions parsing, delegated dict/list state encoding, tasksource gold
handling (one-hot → chosen option, score → argmax list, noul 0.0/1.0 → false/true), option-label
mapping fix for question/options data, ASCII/English filter using `metadata.language` first with
an ASCII fallback heuristic.

## 3. What was NOT done yet

- HF publish of the packaged trainer+data (subagent run was cancelled; publish later)
- vast.ai training launch (owner directive switched away from Kaggle; scripts/kaggle/ package
  still staged locally as fallback)
- ONNX re-export of the finetuned model + publish to the model repo
- `samattjev-decisions-v1` 22GB fold-in (optional v2 of the merge)

## 4. Where things are when resumed

- Re-run merge any time: `python data/build_master.py` (reads only `data/raw/` +
  `clauderesults/`, writes the three outputs)
- Trainer inputs checked and locally validated: adapter produces 200 choice + 100 noul + 200
  score items from a 100-row smoke (0 skips), still at `scripts/kaggle/check_adapter.py`
- Known upstream trainer bugs parked upstream in `scripts/train_openjevx.py` are documented in
  the session notes: (1) build_item score branch defaults to 4 slots when criteria is a dict,
  (2) hardcoded LocalLLaMA load_dataset path, (3) benchmark compares score predictions against
  digit gold labels (structurally ~0 accuracy), (4) partial-accumulation loss cosmetic bug
- GitHub release v0.3.0 confirmed live: https://github.com/muthuishere/openjevx/releases/tag/v0.3.0
  Assets: `openjevx.w4.onnx` + sha256 + server binaries darwin-arm64/linux-amd64/windows-amd64 +
  docker tars, SHA256SUMS-server. Nothing new needed there until the finetune ships.
