# OpenJevX

Open, local decision model server for [jevx](https://github.com/muthuishere/jevx): yes/no, pick-one and rating
decisions in milliseconds, on your own machine.

- **Site:** [muthuishere.github.io/openjevx](https://muthuishere.github.io/openjevx/)
- **Source:** [github.com/muthuishere/openjevx](https://github.com/muthuishere/openjevx)
- **Releases:** [latest](https://github.com/muthuishere/openjevx/releases/latest) (macOS Apple Silicon, Linux amd64 and arm64, Windows; build the Docker image with `docker compose`)
- **Model:** [huggingface.co/muthuishere/openjevx](https://huggingface.co/muthuishere/openjevx)
- **Recipes:** [recipes/](recipes/README.md) (runnable examples) · [what you get](recipes/what-you-get.md) (measured size and latency)
- **Python:** [`python/openjevx.py`](python/openjevx.py), one file, same answers as the server (the container is still the recommended way to run it)
- **jevx CLI:** [github.com/muthuishere/jevx](https://github.com/muthuishere/jevx)

Default port: **21160**

http://127.0.0.1:21160/v1/systemone

Change it in `openjevx.json`:

```json
{ "listen": "127.0.0.1:21160", "device": "auto", "model": "model" }
```

`device` is `auto`, `cpu`, or `gpu`. `auto` uses the first GPU provider that loads and matches the CPU on a probe
(CUDA, CoreML on macOS, DirectML on Windows), otherwise CPU. `gpu` does not fall back.

## Credentials

| openjevx.json | environment | guards | default |
|---|---|---|---|
| `api_key` | `OPENJEVX_API_KEY` | `POST /v1/systemone`, sent as `Authorization: Bearer <key>` (16+ characters) | **on** when `listen` is not loopback: unset, the server generates one into `openjevx.api-key` beside `openjevx.json`; **off** on `127.0.0.1`, `::1` and `localhost` unless set |
| `password` | `OPENJEVX_PASSWORD` | the dashboard `/`, `/stats`, `/metrics`, `/recipes` (HTTP Basic, any user name) | always on: unset, the server generates one into `openjevx.password` |
| `allow_no_api_key` | `OPENJEVX_ALLOW_NO_API_KEY=1` | turns the key off even when public, e.g. behind a proxy that checks it | off |
| `allow_no_password` | `OPENJEVX_ALLOW_NO_PASSWORD=1` | turns the dashboard password off | off |

The environment wins over `openjevx.json`. Generated files are created once with mode 0600 and reused: beside
`openjevx.json` (or beside the executable when there is none), else in `$OPENJEVX_DATA`, else the working folder,
whichever is writable, so a config mounted read-only still starts. A start that creates one shows the value once,
and only on a terminal; when stderr is a log (Docker, systemd, ECS, CloudWatch) it logs the file and a fingerprint
(`sha256 ...37dd`), never the secret, so read it from the file. Release archives carry no `openjevx.json`, so unpacking a new release over an old
one never replaces yours (the npx installer keeps it too and only adds keys it lacks).
`GET /health` stays open. The old published password `adminadmin` (in `openjevx.json` up to v0.5.6) is ignored and
replaced by a generated one. The 401 body is `{"error":"missing or wrong API key"}`, the same as the jev-cloud gate's.

```bash
curl -H "Authorization: Bearer $(cat openjevx.api-key)" http://<host>:21160/v1/systemone -d @body.json
```

`threads` (or `OPENJEVX_THREADS`, which wins) is the CPU threads one request uses. Unset, the server takes the first
of: the cgroup CPU quota (`/sys/fs/cgroup/cpu.max`, or v1 `cpu.cfs_quota_us / cpu.cfs_period_us`), rounded up; on
ECS / Fargate (`ECS_CONTAINER_METADATA_URI_V4` set) the task's `Limits.CPU` from the task metadata endpoint, else the
container's (1 s timeout); else `GOMAXPROCS`. It is never more than the machine's CPUs. Fargate limits CPU with shares
that neither the quota nor Go can see, so a 1 vCPU task reports 2 CPUs; running 2 threads there was 4x slower.
The server logs the choice at startup: `threads: intra-op 1 (from ecs), GOMAXPROCS 2, NumCPU 2` (the source is
`config`, `env`, `cgroup`, `ecs` or `GOMAXPROCS`). On Linux a second line names the CPU and the SIMD flags that set its speed, e.g.
`cpu: AMD EPYC 7763 64-Core Processor; has avx2; lacks avx512f avx512_vnni avx_vnni amx_int8`. On x86, AVX2-only hosts
are about 2x slower than AVX-512 VNNI ones (llmresults/14), and Fargate hands out both.

Every decision response carries a [`Server-Timing`](https://www.w3.org/TR/server-timing/) header in ms, for example
`encode;dur=0.3, wait;dur=0.0, run;dur=41.2, total;dur=42.0` (`wait` is time queued for the model session, `total`
runs from reading the request to the decoded answer), and the same total as `usage.server_ms` in the JSON body.

## The model is a folder

The binary holds no model. A model is a folder, shipped in a release as `openjevx-model-<version>.tar.gz`:

```
model/
  openjevx.w8.onnx   the graph (8-bit weight-only)
  config.json        {"name":"openjevx","version":"0.5.2",
                      "temperature":{"choice":..,"score":..,"noul":..},
                      "max_len":512,"head_max":256,
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

### The model from S3

`"model"` (or `-model` / `OPENJEVX_MODEL`) can also be an object-store URL: `s3://bucket/prefix/` holding the three
files, or one `s3://bucket/model.tar.gz` (or `.tar`) with them at its root or under `model/`. The server downloads it
with the AWS default credential chain (env, profile, EC2 instance role, EKS IRSA / Pod Identity, ECS task role; no
keys in the config), verifies it, and serves it from a local cache. It only calls `GetObject`/`HeadObject`, so the
role needs `s3:GetObject` on the objects and `s3:ListBucket` on the prefix (so a missing object is a 404). The bucket's
region is `AWS_REGION` if set, otherwise read from S3's redirect. gs:// and azblob:// are not supported yet.

| openjevx.json | environment | default | |
|---|---|---|---|
| `model_sha256` | `OPENJEVX_MODEL_SHA256` | none | pin: sha256 of the `.tar.gz`, or of the folder's `openjevx.w8.onnx` |
| `model_cache` | `OPENJEVX_MODEL_CACHE` | user cache dir`/openjevx/models`, else `$TMPDIR/openjevx-models` | where downloads live (e.g. `/tmp/jev-cache`) |
| `model_reload` | `OPENJEVX_MODEL_RELOAD` | off | check the ETag this often (`5m`; at least `10s`) |
| `model_fallback` | `OPENJEVX_MODEL_FALLBACK` | the `model/` lookup next to the executable | served while the bucket holds no model yet |
| `model_s3_path_style` | `AWS_S3_USE_PATH_STYLE=true` | false | bucket in the path, not the host name: MinIO, Ceph, R2 |

**S3-compatible stores** (MinIO, Ceph, Cloudflare R2): set the endpoint with `AWS_ENDPOINT_URL_S3` (or `AWS_ENDPOINT_URL`)
and turn on path-style (`AWS_S3_USE_PATH_STYLE=true` or `"model_s3_path_style": true`) unless the store has
wildcard bucket DNS. Example: `AWS_ENDPOINT_URL_S3=http://minio:9000 AWS_S3_USE_PATH_STYLE=true
OPENJEVX_MODEL=s3://models/current/ ./openjevx` (for R2, `AWS_REGION=auto`).

- **Verify:** `config.json`'s `sha256` must match the graph, and the pin (if set) must match. A mismatch fails loudly.
- **Start:** a cached model whose ETags still match starts without downloading. If the store is unreachable, denies
  access or holds a bad upload, a valid cache is served with a `WARNING`; with no valid cache the server refuses to start.
- **Empty bucket** (fresh deploy): the fallback folder is served, and the prefix is checked every `model_reload`
  (every minute if reload is off) until a model appears.
- **Reload:** on a new ETag the server downloads into a new cache folder, verifies it, opens a new session, swaps it in
  between requests, and keeps the previous folder for rollback. A bad upload is logged and the old model keeps serving.
- `GET /health` reports `version`, `sha256`, `source` (the s3 URL or local path), `fallback`, `loaded_at`, `dir`,
  `remote_version` (the ETags), `offline`, and `previous` (the model it replaced). Design: [ADR 0012](docs/adr/0012-model-from-object-store.md).

## One step

```bash
npx git+https://github.com/muthuishere/openjevx.git
openjevx
```

## Manual

Download the file, then run it.

macOS:

```bash
curl -L -O https://github.com/muthuishere/openjevx/releases/download/v0.5.11/openjevx-darwin-arm64.tar
curl -L -O https://github.com/muthuishere/openjevx/releases/download/v0.5.11/openjevx-model-0.5.2.tar.gz
tar -xf openjevx-darwin-arm64.tar && tar -xzf openjevx-model-0.5.2.tar.gz && ./openjevx
```

Linux (x86-64; glibc 2.28+):

```bash
curl -L -O https://github.com/muthuishere/openjevx/releases/download/v0.5.11/openjevx-linux-amd64.tar
curl -L -O https://github.com/muthuishere/openjevx/releases/download/v0.5.11/openjevx-model-0.5.2.tar.gz
tar -xf openjevx-linux-amd64.tar && tar -xzf openjevx-model-0.5.2.tar.gz && ./openjevx
```

Linux ARM (arm64, e.g. Graviton or Ampere): the same with
[`openjevx-linux-arm64.tar`](https://github.com/muthuishere/openjevx/releases/download/v0.5.11/openjevx-linux-arm64.tar) instead.

Windows: download https://github.com/muthuishere/openjevx/releases/download/v0.5.11/openjevx-windows-amd64.zip and
https://github.com/muthuishere/openjevx/releases/download/v0.5.11/openjevx-model-0.5.2.tar.gz, unpack both into the
same folder (`tar -xzf openjevx-model-0.5.2.tar.gz`) and run `openjevx.exe`.

## Docker

The prebuilt image, for linux/amd64 and linux/arm64 (each release tag, and `latest`):

```bash
docker run -d -p 127.0.0.1:21160:21160 \
  -e OPENJEVX_PASSWORD=<12+ characters> -e OPENJEVX_API_KEY=<16+ characters> \
  ghcr.io/deemwar-products/openjevx:v0.5.9
```

Or build it from source:

```bash
git clone https://github.com/muthuishere/openjevx.git && cd openjevx
OPENJEVX_PASSWORD=<12+ characters> OPENJEVX_API_KEY=<16+ characters> docker compose up -d --build
```

The image has no default credentials: it refuses to start without `OPENJEVX_PASSWORD` and `OPENJEVX_API_KEY` (or
your own `openjevx.json` mounted at `/data/openjevx.json`). It runs as uid 10001, not root; `/data` is its writable
working folder. Make a key with `openssl rand -hex 24`.

## jevx

Use OpenJevX from the [jevx CLI](https://github.com/muthuishere/jevx):

```bash
jevx profile add openjevx http://127.0.0.1:21160/v1/systemone --model openjevx
jevx profile use openjevx
```

With an API key (any server not on loopback), let jevx read it from the environment:

```bash
jevx profile add openjevx http://<host>:21160/v1/systemone --model openjevx --header 'Authorization: Bearer $OPENJEVX_API_KEY'
```

Upgrading from an older model? jevx caches answers by model name, so run `jevx cache clear` after upgrading (or give the profile a versioned model name such as `--model openjevx-v0.5.2`).

## Recipes

18 runnable decisions (curl and jevx), each with the real answer and what to do next, plus a page of measured
facts. In the repo: [`recipes/`](recipes/README.md). In a running server: http://127.0.0.1:21160/recipes (dashboard
password). In the Docker image: `/usr/share/openjevx/recipes`.

Apache-2.0. Credits: `CREDITS`. Decisions behind the project: [`docs/adr/`](docs/adr/README.md).

## Dashboard

Open http://127.0.0.1:21160/ while the server runs. It shows request count, questions answered, input tokens, latency p50/p95/p99, errors, and recent requests.

There is no default password. Set `"password"` in `openjevx.json` (or `OPENJEVX_PASSWORD`); unset, the server
generates one on its first start, shows it once on a terminal, and keeps it in `openjevx.password` (the npx install
keeps it in `~/.local/share/openjevx/`). Any user name works.

- `GET /stats` — JSON snapshot (same password)
- `GET /metrics` — Prometheus format (same password)
- `GET /health` — open, no password

## Deploy in your own cloud

DigitalOcean 1-Click, cloud-init for any VPS, Docker Compose, and an AWS AMI: [`docs/DEPLOY.md`](docs/DEPLOY.md).

## Run locally from source

The server turns ONNX Runtime's telemetry off (`ORT_DISABLE_TELEMETRY=1` unless you set it): nothing is sent to
Microsoft, and ORT 1.29 no longer runs `blkid`/`hostname` through `/bin/sh` at startup, which crashed it in distroless
images.

Needs Go and [Task](https://taskfile.dev). `task run` fetches ONNX Runtime and the model folder into `.local/` (`.local/model/`), builds, and starts the server on http://127.0.0.1:21160/. `task build` only builds; `task test` runs the tests.

Release from this machine, no CI: `task package` builds the macOS, Linux (amd64, arm64) and Windows packages plus `openjevx-model-<version>.tar.gz` from `MODEL_DIR` (default `.local/model`) (Go cross-compiles, [zig](https://ziglang.org) is the C compiler), `task docker` builds both Docker images locally (no image tars are released: users build the image with `docker compose`, ADR 0010), and `task release VERSION=v<version>` uploads the release assets by name (the four server archives, the model archive and `SHA256SUMS-server`) to that GitHub release.

## What v0.5.2 was trained on

791,889 decisions (465,583 rows). Labels come from evaluating rules or from real outcomes, never from another model's guesses.

| Area | Decisions | Share |
|---|---|---|
| Rule-checking across 38 business domains (retail pricing, recruiting, real estate, CI/CD, insurance claims, pharmacy stock, HR payroll, gaming, ...) | 200,155 | 25.3% |
| Software-work roles (developers, testers, tech leads, managers, operations, everyone, agent checks) | 183,543 | 23.2% |
| tasksource decision corpus | 146,567 | 18.5% |
| Public sets with real labels (CVE fixes from bigvul, defect detection, code search, ms_marco relevance, HDFS and BGL operator log alerts) | 104,990 | 13.3% |
| Rule-reading drills: the same everyday rules with many thresholds, decimals, dates over 2020-2035, meetings on the quarter hour, opening hours, missing/equal/contains checks | 75,321 | 9.5% |
| Log triage: application (Java/Python/Node/Go/nginx), database (Postgres/MySQL, real SQLSTATE codes), frontend (browser/Sentry) | 60,001 | 7.6% |
| Everyday basics (driving licence age, store hours, parcel late, discount thresholds, fever, bag weight, ...) | 12,162 | 1.5% |
| Public typed-decisions | 9,150 | 1.2% |

Run: one RTX 4090 on Vast.ai, 791,239 decisions after packaging, one full pass in 2.5 h at ~87 items/s, about $1.25. Leakage-checked; every row checked against the trainer before renting. Why the drills: [llmresults/13](llmresults/13-v0.5.2-gate-misses.md).

## Fine-tune it further

The shipped model is the folder above; its graph is `openjevx.w8.onnx` (8-bit weights). ONNX Runtime 1.29 runs it as
`MatMulNBits` and quantizes the activations to int8 on every call (ADR 0011), so an answer can move slightly with what
else is in the request: on 2026-10-03 a pick-one answer went from 0.9173 to 0.9214 when a long yes/no question was
added to the same request; the yes/no answers did not change.

**Fine-tune kit (v0.5.2, 777 MB):** [openjevx-finetune-v0.5.2.tar.gz](https://pub-8da821f06ff747cda688f8267ed2aa96.r2.dev/openjevx-finetune-v0.5.2.tar.gz) ([sha256](https://pub-8da821f06ff747cda688f8267ed2aa96.r2.dev/openjevx-finetune-v0.5.2.tar.gz.sha256)). It holds the trainable fine-tuned checkpoint (`openjevx-model/`: `model.safetensors`, `encoder/`, `tokenizer/`, `rl_agent_config.json`). No training data. The 8-bit ONNX is on the GitHub release and Hugging Face; the training scripts are in `finetuning/` in this repo. The same model files are on [Hugging Face](https://huggingface.co/muthuishere/openjevx).

1. **Data**: rows of `{state, questions:{id:{type: noul|choice|score, instructions, criteria}}, gold}`. `finetuning/dataprep/gen_it_worker.py` generates rule-labelled software-work decisions; add your own rows in the same shape.
2. **Shard**: `finetuning/dataprep/package_shards.py --out DIR --extra-train your.jsonl --exclude-keys leaked.json` adapts gold into targets, samples, and drops any question that also appears in your test sets.
3. **Train**: `cd finetuning && task all` runs data → leakage check → smoke run → full run → gate as one job. The GPU box
   pulls its data from a Cloudflare R2 bucket (its Python packages from PyPI, the base model from Hugging Face), uploads the 8-bit ONNX (≤ 750 MB) and the checkpoint, and
   destroys itself ([ADR 0009](docs/adr/0009-one-job-gpu-run-via-r2.md)). Settings live in
   `~/.config/openjevx/config.json`, data in `~/openjevx/data`. On your own CUDA box, run the steps in
   [`finetuning/ft.py`](finetuning/ft.py) and [`finetuning/train/run_job.sh`](finetuning/train/run_job.sh) directly.
4. **Ship**: the job already produces the model folder `<run>/out/model/` with that run's calibration temperatures
   (or build one with `finetuning/export/make_model_folder.py`), then set `"model"` in `openjevx.json` to it, or
   `MODEL_DIR=<folder> task package`.
