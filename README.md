# OpenJevX

Open, local decision model server for [jevx](https://github.com/muthuishere/jevx): yes/no, pick-one and rating
decisions in milliseconds, on your own machine.

- **Site:** [muthuishere.github.io/openjevx](https://muthuishere.github.io/openjevx/)
- **Source:** [github.com/muthuishere/openjevx](https://github.com/muthuishere/openjevx)
- **Releases:** [latest](https://github.com/muthuishere/openjevx/releases/latest) (macOS, Linux, Windows, Docker image tars)
- **Model:** [huggingface.co/muthuishere/openjevx](https://huggingface.co/muthuishere/openjevx)
- **Recipes:** [recipes/](recipes/README.md) (runnable examples) · [what you get](recipes/what-you-get.md) (measured size and latency)
- **Python:** [`python/openjevx.py`](python/openjevx.py), one file, same answers as the server (the container is still the recommended way to run it)
- **jevx CLI:** [github.com/muthuishere/jevx](https://github.com/muthuishere/jevx)

Default port: **21118**

http://127.0.0.1:21118/v1/systemone

Change it in `openjevx.json`:

```json
{ "listen": "127.0.0.1:21118", "device": "auto", "model": "model" }
```

`device` is `auto`, `cpu`, or `gpu`. `auto` uses the GPU when CUDA loads, otherwise CPU. `gpu` does not fall back.

## The model is a folder

The binary holds no model. A model is a folder, shipped in a release as `openjevx-model-<version>.tar.gz`:

```
model/
  openjevx.w8.onnx   the graph (8-bit weight-only)
  config.json        {"name":"openjevx","version":"0.5.0",
                      "temperature":{"choice":..,"score":..,"noul":..},
                      "max_len":1024,"head_max":256,
                      "special_ids":{"cls":50281,"sep":50282,"pad":50283,"mask":50284},
                      "quantization":"8-bit weight-only","base_model":"convaiinnovations/laya",
                      "sha256":"<of the onnx>"}
  tokenizer.json     optional; the built-in ModernBERT tokenizer otherwise
```

`temperature` is that model's own confidence calibration, so each model carries its own. `"model"` in
`openjevx.json` names a folder (or, for old models, a bare `.onnx` file, which gets the old built-in settings).
Unset, the server looks next to itself for `model/`, then `models/openjevx/`, then `openjevx.w8.onnx`. At startup
it logs the model path, version, temperatures and sha256 (and refuses a graph whose sha256 does not match
`config.json`); `GET /health` reports `version` and `sha256`. Build a folder with
`python finetuning/export/make_model_folder.py OUT_DIR model.onnx eval_report.json [tokenizer.json] --version X`.

## One step

```bash
npx git+https://github.com/muthuishere/openjevx.git
openjevx
```

## Manual

Download the file, then run it.

macOS:

```bash
curl -L -O https://github.com/muthuishere/openjevx/releases/download/v0.5.1/openjevx-darwin-arm64.tar
curl -L -O https://github.com/muthuishere/openjevx/releases/download/v0.5.1/openjevx-model-0.5.0.tar.gz
tar -xf openjevx-darwin-arm64.tar && tar -xzf openjevx-model-0.5.0.tar.gz && ./openjevx
```

Linux:

```bash
curl -L -O https://github.com/muthuishere/openjevx/releases/download/v0.5.1/openjevx-linux-amd64.tar
curl -L -O https://github.com/muthuishere/openjevx/releases/download/v0.5.1/openjevx-model-0.5.0.tar.gz
tar -xf openjevx-linux-amd64.tar && tar -xzf openjevx-model-0.5.0.tar.gz && ./openjevx
```

Windows: download https://github.com/muthuishere/openjevx/releases/download/v0.5.1/openjevx-windows-amd64.zip and
https://github.com/muthuishere/openjevx/releases/download/v0.5.1/openjevx-model-0.5.0.tar.gz, unpack both into the
same folder (`tar -xzf openjevx-model-0.5.0.tar.gz`) and run `openjevx.exe`.

## Docker

```bash
git clone https://github.com/muthuishere/openjevx.git && cd openjevx
OPENJEVX_PASSWORD=<12+ letters/digits> docker compose up -d --build
```

The image has no default dashboard password: it refuses to start without `OPENJEVX_PASSWORD` or a mounted `/app/openjevx.json`.

## jevx

Use OpenJevX from the [jevx CLI](https://github.com/muthuishere/jevx):

```bash
jevx profile add openjevx http://127.0.0.1:21118/v1/systemone --model openjevx
jevx profile use openjevx
```

Upgrading from an older model? jevx caches answers by model name, so run `jevx cache clear` after upgrading (or give the profile a versioned model name such as `--model openjevx-v0.5.0`).

## Recipes

18 runnable decisions (curl and jevx), each with the real answer and what to do next, plus a page of measured
facts. In the repo: [`recipes/`](recipes/README.md). In a running server: http://127.0.0.1:21118/recipes (dashboard
password). In the Docker image: `/usr/share/openjevx/recipes`.

Apache-2.0. Credits: `CREDITS`. Decisions behind the project: [`docs/adr/`](docs/adr/README.md).

## Dashboard

Open http://127.0.0.1:21118/ while the server runs. It shows request count, questions answered, input tokens, latency p50/p95/p99, errors, and recent requests.

Password default for the downloaded binary: `adminadmin`, change it in `openjevx.json` (`"password"`).

- `GET /stats` — JSON snapshot (same password)
- `GET /metrics` — Prometheus format (same password)
- `GET /health` — open, no password

## Deploy in your own cloud

DigitalOcean 1-Click, cloud-init for any VPS, Docker Compose, and an AWS AMI: [`docs/DEPLOY.md`](docs/DEPLOY.md).

## Run locally from source

Needs Go and [Task](https://taskfile.dev). `task run` fetches ONNX Runtime and the model folder into `.local/` (`.local/model/`), builds, and starts the server on http://127.0.0.1:21118/. `task build` only builds; `task test` runs the tests.

Release from this machine, no CI: `task package` builds the macOS, Linux and Windows packages plus `openjevx-model-<version>.tar.gz` from `MODEL_DIR` (default `.local/model`) (Go cross-compiles, [zig](https://ziglang.org) is the C compiler), `task docker` saves both Docker images as tars, and `task release VERSION=v0.4.0` uploads everything in `.local/dist` to that GitHub release.

## What v0.5.0 was trained on

734,795 decisions (408,505 rows). Labels come from evaluating rules or from real outcomes, never from another model's guesses.

| Area | Decisions | Share |
|---|---|---|
| Rule-checking across 38 business domains (retail pricing, recruiting, real estate, CI/CD, insurance claims, pharmacy stock, HR payroll, gaming, ...) | 200,139 | 27.2% |
| Software-work roles (developers, testers, tech leads, managers, operations, everyone, agent checks) | 183,543 | 25.0% |
| tasksource decision corpus | 146,567 | 19.9% |
| Public sets with real labels (CVE fixes from bigvul, defect detection, code search, ms_marco relevance, HDFS and BGL operator log alerts) | 104,990 | 14.3% |
| Log triage: application (Java/Python/Node/Go/nginx), database (Postgres/MySQL, real SQLSTATE codes), frontend (browser/Sentry) | 60,001 | 8.2% |
| Everyday basics (driving licence age, store hours, parcel late, discount thresholds, fever, bag weight, ...) | 30,405 | 4.1% |
| Public typed-decisions | 9,150 | 1.2% |

Run: one RTX 4090 on Vast.ai, 734,145 decisions after packaging, one full pass in 2.4 h at ~88 items/s, about $1.10 in total. Leakage-checked (22,592 test questions removed); every row checked against the trainer before renting.

## Fine-tune it further

The shipped model is the folder above; its graph is `openjevx.w8.onnx` (8-bit weight-only; activations stay float, so answers don't depend on what else is in the request).

**Fine-tune kit (v0.5.0, 777 MB):** [openjevx-finetune-v0.5.0.tar.gz](https://pub-8da821f06ff747cda688f8267ed2aa96.r2.dev/openjevx-finetune-v0.5.0.tar.gz) ([sha256](https://pub-8da821f06ff747cda688f8267ed2aa96.r2.dev/openjevx-finetune-v0.5.0.tar.gz.sha256)). It holds the trainable fine-tuned checkpoint (`openjevx-model/`: `model.safetensors`, `encoder/`, `tokenizer/`, `rl_agent_config.json`). No training data. The 8-bit ONNX is on the GitHub release and Hugging Face; the training scripts are in `finetuning/` in this repo. The same model files are on [Hugging Face](https://huggingface.co/muthuishere/openjevx).

1. **Data**: rows of `{state, questions:{id:{type: noul|choice|score, instructions, criteria}}, gold}`. `finetuning/dataprep/gen_it_worker.py` generates rule-labelled software-work decisions; add your own rows in the same shape.
2. **Shard**: `finetuning/dataprep/package_shards.py --out DIR --extra-train your.jsonl --exclude-keys leaked.json` adapts gold into targets, samples, and drops any question that also appears in your test sets.
3. **Train**: `cd finetuning && task all` runs data → leakage check → smoke run → full run → gate as one job. The GPU box
   pulls its data and runtime from a Cloudflare R2 bucket, uploads the 8-bit ONNX (≤ 750 MB) and the checkpoint, and
   destroys itself ([ADR 0009](docs/adr/0009-one-job-gpu-run-via-r2.md)). Settings live in
   `~/.config/openjevx/config.json`, data in `~/openjevx/data`. On your own CUDA box, run the steps in
   [`finetuning/ft.py`](finetuning/ft.py) and [`finetuning/train/run_job.sh`](finetuning/train/run_job.sh) directly.
4. **Ship**: the job already produces the model folder `<run>/out/model/` with that run's calibration temperatures
   (or build one with `finetuning/export/make_model_folder.py`), then set `"model"` in `openjevx.json` to it, or
   `MODEL_DIR=<folder> task package`.
