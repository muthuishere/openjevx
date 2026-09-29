# OpenJevX: how we work

## Where things live
- **Code is in the repo. Data, settings and generated output are not.**
- Fine-tuning code: `finetuning/` → `dataprep/` · `datavalidate/` · `train/` · `export/` · `gate/` · `gpu/<provider>.py`.
  One runner, `finetuning/ft.py`; Taskfile in `finetuning/` (`task all`, `task smoke`, `task train`, `task gate -- model.onnx`).
- Settings: `~/.config/openjevx/config.json` (created from `finetuning/config.example.json`; override with
  `$OPENJEVX_FT_CONFIG`). Change the example when defaults change; never hard-code a setting in a script.
- Data: `~/openjevx/data/` (override with `$OPENJEVX_DATA`). All paths come from `finetuning/paths.py`:
  `raw/` · `incoming/` · `train/` · `eval/` · `gate/` · `work/{leak,shards,runs,gate,quality,samples}`.
  Never write data into the repo, and never add a new data path outside `paths.py`.
- Agent reports: `llmresults/` (ADR 0005). Decisions: `docs/adr/`.

## Model rules
- We ship **one model file: 8-bit ONNX, at most 750 MB** (`model.max_w8_mb`). The GPU job quantizes on the
  box and fails over the limit; the gate checks it again. We also keep the trainable checkpoint so the
  model can be fine-tuned again.
- Every run is one job, end to end: data ready → validate (leakage check) → smoke run → full run → gate.
  Freeze the data before renting a GPU. A model ships only when the gate passes, not when training ends.
- The gate is right AND confident on the everyday basics, never accuracy alone. Thresholds live in the config.
- Labels come from reality or from evaluating a rule, never from a model's guess. Test and gate questions
  never appear in training (the leakage check removes them).

## GPUs and storage
- GPU work runs on a rented provider, **not Kaggle**. Today it is Vast.ai (`gpu/vast.py`); a new provider is one
  `gpu/<name>.py` that prints `RUN_DIR=<dir>` and leaves `<dir>/out/openjevx.w8.onnx`, plus one line in the config.
- **The box does the whole job and cleans up after itself.** Inputs come from Cloudflare R2 through signed links;
  results (8-bit ONNX, checkpoint, eval report, log, status) go back to R2 the same way; then the box destroys
  itself by calling the destroy endpoint (`finetuning/destroy-worker/`, a Cloudflare Pages Function that holds the
  Vast key). A timer on the box destroys it at the deadline whatever happens; the local run destroys it too if it
  is still alive. No long-lived key ever goes on the box.
- R2 buckets: `openjevx-train` is **private** (shards, runs, checkpoints, gate reports); `openjevx` is public and
  holds only released models. `finetuning/r2.py` is the only R2 client.
- Label boxes `openjevx-...-DESTROY-AFTER` (the endpoint refuses anything else). The box clones a pushed commit,
  so commit `finetuning/` before training.

## Style
- Match the surrounding code; small, plain scripts; fail loudly with a clear message.
- Secrets only by env-var name or through `sec`; never in code, config or output.

<!-- ctx-optimize:begin -->
<ctx-optimize>
  <precondition>Run `command -v ctx-optimize` first. If it is NOT installed, IGNORE this entire
  block and answer by reading the code normally — the store is an optimization, not a requirement
  (install later with `npm install -g @muthuishere/ctx-optimize`, or download the binary). Everything
  below applies ONLY when the command exists.</precondition>
  <store>Pre-built knowledge store at `~/ctxoptimize/openjevx/` (config in `.ctxoptimize/` here).</store>
  <use>Use it INSTEAD of grep-and-read chains — PICK BY INTENT: find → `ctx-optimize query "<terms>"` ·
  inspect a symbol → `card <symbol>` · about to EDIT → `change-plan <symbol>` (callers+impact+tests, one
  call) · blast radius → `affected <symbol>` · connection → `path <a> <b>` ·
  list/filter (no jq): `nodes --kind K` / `edges --relation R` / `deps`.
  Output is parsed fact with exact file:line — cite it directly, do
  NOT re-verify in source; open a file only for a body the store didn't show. Exhaustive literal-string
  sweeps stay grep's job.</use>
  <deep-doc>The FULL usage card — verify discipline, store-vs-grep ladder, sources (databases/
  buckets/queues/APIs by env-var name), remote push/pull, `up` — is committed at
  `.ctxoptimize/instructions.md`. Read it before deeper store work.</deep-doc>
  <no-local-store>Fresh clone with nothing at `~/ctxoptimize/openjevx/`? Run `ctx-optimize up` —
  it pulls the team's prebuilt store when the config declares one, otherwise rebuilds in seconds.</no-local-store>
</ctx-optimize>
<!-- ctx-optimize:end -->
