---
title: "How to train your own Jev"
subtitle: "Your decisions in, your own 8-bit decision model out"
---

OpenJevX answers yes/no, pick-one and rating questions in milliseconds, on your own machine. The shipped
model knows everyday rules. It does not know yours: that your week is four days, that a discount starts
above Rs 500, that repo `billing` belongs to the payments team. This book takes you from your own
records to your own model, one step at a time. Every step has commands to copy and says what you should
see. Every number in it was measured on a real run; none is an estimate.

The path:

1. Prepare data from your documents and records
2. Validate it
3. Train on a rented GPU
4. Evaluate on held-out cases
5. Export the 8-bit ONNX model folder
6. Deploy it
7. The retrain loop

## What it costs and what you get

| What | Measured on the v0.5.0 run |
|---|---|
| GPU | one RTX 4090 on Vast.ai |
| Training data | 734,145 decisions after packaging |
| Time | one full pass in 2.4 h, about 88 items/s |
| Cost | about $1.10 in total |
| Test questions removed before training | 22,592 |
| 8-bit model (`openjevx.w8.onnx`) | 598,046,561 bytes; the limit is 750 MB |
| Model folder download | 467 MB |
| Trainable checkpoint | 777 MB |

You get a model folder (`openjevx.w8.onnx` + `config.json` + `tokenizer.json`) that runs on your
machine, plus a trainable checkpoint so you can train it again. Your decisions never go to a hosted model.

## 0. Install

You need Go, [Task](https://taskfile.dev), [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```sh
git clone {{SOURCE_URL}} openjevx
cd openjevx
go version && task --version && uv --version && python3 --version
```

For the GPU step (step 3) you also need:

- the `vastai` CLI and a Vast.ai API key with a few dollars of credit:
  `pip install vastai && vastai set api-key YOUR_KEY_HERE`;
- a Cloudflare account for R2 storage. The GPU box downloads your data from R2 and uploads the model back
  there. This creates the private bucket and the self-destroy endpoint, then checks the endpoint answers:

```sh
cd finetuning && task cloud-setup
```

Steps 1, 2 and 4 need no GPU and no cloud account.

Settings live outside the repo in `~/.config/openjevx/config.json` (created from
`finetuning/config.example.json` on first run). Data lives in `~/openjevx/data/` (set `$OPENJEVX_DATA` to
move it). Nothing you prepare is written into the repo.

## 1. Prepare data from your documents and records

A training row is **one decision**:

- `state`: the facts at the moment of deciding, exactly what your system will send the model later;
- `question`: the decision, phrased as you would ask a colleague;
- `answer`: the right decision, **from your rule or from what really happened**. Never a model's guess.

### From records to rows

Most rows come from records you already have. Take the record, keep the facts the rule needs, and write
the real outcome as the answer:

| Source | `state` |
|---|---|
| a log alert | `{"line": "OOMKilled payments-api", "env": "prod", "errors_last_10m": 42}` |
| a support ticket | `{"text": "Charged twice for order 8812", "customer_tier": "gold"}` |
| a database record | `{"promised_by": "2026-10-03", "delivered_on": "2026-10-04"}` |
| a pull request | `{"files_changed": 14, "touches": ["migrations/"], "summary": "adds a column"}` |
| a form | `{"leave_days_requested": 6, "leave_days_left": 4}` |

From a written policy or a document, turn each rule into many rows: both sides of every threshold
(499 → no, 500 → no, 501 → yes), different values, different wordings.

### The CSV

Write the rows in a spreadsheet and save it as CSV. Start from the example in the repo,
`finetuning/examples/decisions.csv` (39 rows: a 4-day week, working hours, a late parcel, a discount, team
routing, ticket urgency).

| column | required | what | example |
|---|---|---|---|
| `state` | yes | the facts, as a JSON object or plain text | `{"order_total_rs": 501}` |
| `question` | yes | one decision | `Does the order get the discount? Orders over Rs 500 get 10% off.` |
| `type` | no (`noul`) | `noul` = yes/no, `choice` = pick one, `score` = rating | `choice` |
| `options` | choice, score | keys separated by `\|`, optional meaning after `=`; score levels lowest first | `payments=billing repos\|web=frontend repos` |
| `answer` | yes | yes/no; one option key; a level name or index (0 = lowest) | `payments` |
| `split` | no | `train`, `test` or `gate`; empty = 90% train, 10% gate, fixed per row | `gate` |
| `source` | no | a tag for where the row came from | `shop` |

One row of each type:

```csv
state,question,type,options,answer
"{""order_total_rs"": 501}","Does the order get the discount? Orders over Rs 500 get 10% off.",noul,,yes
"{""repo"": ""billing""}",Which team owns this?,choice,payments=billing repos|platform=infra repos|web=frontend repos,payments
"{""customers_affected"": 40, ""workaround"": false}",How urgent is this ticket?,score,low|medium|high|critical,high
```

JSON inside a CSV cell: wrap the cell in double quotes and double the quotes inside. Any spreadsheet does
this when you save as CSV.

Rules that make a good set:

- near-miss pairs at every threshold;
- 20 or more rows per question, and no single answer above 80% of them;
- keep a few of the hardest rows as `split=gate`, so you can measure the result;
- never put the answer in the state (`"eligible": true`);
- never give the same state and question two different answers: a fact is missing.

### Import

```sh
python3 finetuning/dataprep/import_csv.py yours.csv --name mydata --add-to-config
```

You should see the good and bad row counts, then one line per question with its answers, then where the
files were written:

```text
yours.csv: 39 good rows, 0 bad
...
adapter: every row accepted
wrote    24 train rows -> ~/openjevx/data/train/mydata_train.jsonl
wrote    15 gate  rows -> ~/openjevx/data/gate/mydata_gate.jsonl
updated ~/.config/openjevx/config.json (backup: config.json.bak-…)
```

A bad row is reported with its line number and a fix, and nothing is written until every row is good
(unless `--skip-bad`). `--dry-run` checks without writing. `--add-to-config` adds your train file to the
training mix (repeated 3 times; change with `--repeat`), your gate file to the gate, and both to the
leakage check.

## 2. Validate

```sh
cd finetuning
task validate
```

This dry-parses every row through the trainer's adapter and runs the leakage check: any training question
that also appears in a test or gate file is removed, so the gate measures cases the model has never seen.
On the v0.5.0 data it removed 22,592 questions. Fix what it reports before going further; a data bug found
here costs nothing, the same bug found on a rented GPU costs money.

## 3. Train on a rented GPU

First measure the model you already have on your gate (step 4 explains the numbers). If it already scores
well on your rules, you may not need to train at all.

Then train:

```sh
cd finetuning
task all
```

It runs these stages and stops at the first failure: dataprep → validate → trainer check on CPU → smoke
run on the GPU → full run → gate. To run them one at a time: `task smoke`, then `task train`.

What happens on the GPU:

- The box is rented on Vast.ai (an RTX 4090, price cap in `providers.vast` in the config).
- It downloads the shard from your **private** R2 bucket through a signed link, trains, exports and
  quantizes to 8-bit, and uploads the model folder, checkpoint, eval report and log back to R2.
- Then it destroys itself. A timer on the box destroys it at the deadline whatever happens, and the local
  run destroys it too if it is still alive. No long-lived key goes on the box.
- Your laptop only waits on R2, so it can sleep; results wait in the bucket.

You should see `RUN_DIR=…` and, at the end, the model folder under `~/openjevx/data/work/runs/`.

The box clones a pushed commit, so commit and push any change under `finetuning/` before training. Your
data goes through R2, not git.

**Working from a fork?** The box clones your checkout's `origin` remote at the current commit
(`finetuning/gpu/vast.py`), over HTTPS and with no credentials, so `origin` must be a public repo you can
push to. Check and set it:

```sh
git remote get-url origin
git remote set-url origin https://your-git-host/you/openjevx.git
```

Another GPU provider is one file, `finetuning/gpu/<name>.py`, that prints `RUN_DIR=<dir>` and leaves the
model folder in `<dir>/out/model/`, plus one line in the config. Today only Vast.ai is implemented.

## 4. Evaluate on held-out cases

```sh
cd finetuning
task gate -- path/to/model-folder
```

This serves the model on a free local port and scores every gate file, one line each:

```text
gate/mydata_gate.jsonl   n=    15 accuracy  60.0% right&confident  40.0% confidently WRONG 13.3%
```

Three numbers per file:

- **accuracy**: the top answer was right;
- **right & confident**: right and sure enough to act on (yes ≥ 0.8, no ≤ 0.2, a choice or score ≥ 0.6).
  This is the number that matters: an unsure answer gets escalated, not acted on;
- **confidently wrong**: sure and wrong. The dangerous one; keep it near zero.

A model ships only when the gate passes, not when training ends. The default thresholds (in `gate` in the
config): the everyday basics at least 90% right & confident and at most 2% confidently wrong, and at least
12 of jevx's 13 fundamentals. The full report is saved to `~/openjevx/data/work/gate/<model>.json`.

For reference, the released v0.5.0 model on its clean gates (no state shared with training):

| Gate set | right & confident | confidently wrong |
|---|---|---|
| Everyday basics | 96.3% | 3.0% |
| Rule-checking basics | 94.3% | 3.7% |

Both confidently-wrong rates are above the 2% target. The owner accepted that for v0.5.0; the fix is
planned for v0.5.1.

## 5. Export the 8-bit ONNX model folder

The GPU job exports and quantizes on the box and fails if the 8-bit file is over 750 MB
(`model.max_w8_mb`); the gate checks it again. What comes back is one folder:

```text
model/
  openjevx.w8.onnx   the graph, 8-bit weight-only
  config.json        this model's calibration temperatures, lengths, token ids, sha256 of the graph
  tokenizer.json
```

The temperatures belong to this one model, so they live in its `config.json`, never in the server. To
build a folder by hand from an ONNX file and its eval report:

```sh
python3 finetuning/export/make_model_folder.py OUT_DIR model.onnx eval_report.json tokenizer.json --version 1.0.0
```

The server refuses a graph whose sha256 does not match `config.json`.

## 6. Deploy

Point the server at your folder in `openjevx.json`:

```json
{ "listen": "127.0.0.1:21118", "device": "auto", "model": "/path/to/my-model" }
```

`device` is `auto`, `cpu` or `gpu`. At startup the server logs the model path, version, temperatures and
sha256; `GET /health` reports `version` and `sha256`, so you can check which model is live.

With Docker, from the source folder (the image has no default credentials and refuses to start without them):

```sh
OPENJEVX_PASSWORD=<12+ characters> OPENJEVX_API_KEY=<16+ characters> docker compose up -d --build
```

Mount your model folder into the container and set the same `"model"` path in your own `openjevx.json`, mounted at
`/data/openjevx.json`. Clients send `Authorization: Bearer <key>`. For a cloud server, `docs/DEPLOY.md` has a
DigitalOcean 1-Click image, cloud-init for any VPS and an AWS AMI.

Use it from jevx with a **versioned** model name, because jevx caches answers by model name:

```sh
jevx profile add mymodel http://127.0.0.1:21118/v1/systemone --model mymodel-v1 \
  --header 'Authorization: Bearer $OPENJEVX_API_KEY'   # only for a server that has a key (Docker, any non-loopback listen)
jevx profile use mymodel
jevx cache clear
jevx ask --noul discount="Does the order get the discount? Orders over Rs 500 get 10% off." --in '{"order_total_rs": 501}'
```

You should see a yes with high confidence.

## 7. The retrain loop

Your rules change and reality surprises you. The loop:

1. **Collect** the decisions the model got wrong or was unsure about (answers below the confidence line),
   with the real outcome once it is known.
2. **Add** them as rows to your CSV, with near-miss pairs around the case it missed. Put a few in
   `split=gate`.
3. **Import** again (step 1). Re-running `--add-to-config` does not add duplicates.
4. **Gate the current model first** (step 4) on the new gate rows: that is your baseline.
5. **Train** (step 3). Each run saves a trainable checkpoint next to the model; `task all` does not yet
   start from it, so today every run trains from the base model on the whole mix.
6. **Ship only if the gate passes**, and compare with the baseline.
7. **Deploy** under a new model name (`mymodel-v2`), so no cached answer from the old model is reused.

## Troubleshooting

- **The GPU box is stuck loading.** Some hosts are slow to start. A box not running after 15 minutes is
  destroyed and the next machine is tried, up to 3 (`finetuning/gpu/vast.py`; change the wait with
  `VAST_START_MINUTES`).
- **Your network dropped.** The box does not depend on your laptop; results wait in R2. Run the step again
  to pick them up.
- **The trainer rejects a row.** The CPU check names the row and its source before any GPU is rented. Fix
  it in your CSV and import again.
- **The gate fails.** Look at which file failed. Your file: add near-miss rows and check the state holds
  every fact. The basics: lower `--repeat` so your data does not crowd them out.
