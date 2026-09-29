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
{ "listen": "127.0.0.1:21118", "device": "auto" }
```

`device` is `auto`, `cpu`, or `gpu`. `auto` uses the GPU when CUDA loads, otherwise CPU. `gpu` does not fall back.

## One step

```bash
npx git+https://github.com/muthuishere/openjevx.git
openjevx
```

## Manual

Download the file, then run it.

macOS:

```bash
curl -L -O https://github.com/muthuishere/openjevx/releases/download/v0.4.0/openjevx-darwin-arm64.tar
tar -xf openjevx-darwin-arm64.tar && ./openjevx
```

Linux:

```bash
curl -L -O https://github.com/muthuishere/openjevx/releases/download/v0.4.0/openjevx-linux-amd64.tar
tar -xf openjevx-linux-amd64.tar && ./openjevx
```

Windows: download https://github.com/muthuishere/openjevx/releases/download/v0.4.0/openjevx-windows-amd64.zip and run `openjevx.exe`.

## Docker

```bash
git clone https://github.com/muthuishere/openjevx.git && cd openjevx
docker compose up -d --build
```

No clone? Load a prebuilt image from the release (`amd64` or `arm64`):

```bash
curl -L -O https://github.com/muthuishere/openjevx/releases/download/v0.4.0/openjevx-docker-amd64.tar
docker load -i openjevx-docker-amd64.tar
docker run -d -p 127.0.0.1:21118:21118 ghcr.io/muthuishere/openjevx:latest-amd64
```

## jevx

Use OpenJevX from the [jevx CLI](https://github.com/muthuishere/jevx):

```bash
jevx profile add openjevx http://127.0.0.1:21118/v1/systemone --model openjevx
jevx profile use openjevx
```

## Recipes

18 runnable decisions (curl and jevx), each with the real answer and what to do next, plus a page of measured
facts. In the repo: [`recipes/`](recipes/README.md). In a running server: http://127.0.0.1:21118/recipes (dashboard
password). In the Docker image: `/usr/share/openjevx/recipes`.

Apache-2.0. Credits: `CREDITS`. Decisions behind the project: [`docs/adr/`](docs/adr/README.md).

## Dashboard

Open http://127.0.0.1:21118/ while the server runs. It shows request count, questions answered, input tokens, latency p50/p95/p99, errors, and recent requests.

Password default: `adminadmin`, change it in `openjevx.json` (`"password"`).

- `GET /stats` — JSON snapshot (same password)
- `GET /metrics` — Prometheus format (same password)
- `GET /health` — open, no password

## Run locally from source

Needs Go and [Task](https://taskfile.dev). `task run` fetches ONNX Runtime and the 8-bit model into `.local/`, builds, and starts the server on http://127.0.0.1:21118/. `task build` only builds; `task test` runs the tests.

Release from this machine, no CI: `task package` builds the macOS, Linux and Windows packages (Go cross-compiles, [zig](https://ziglang.org) is the C compiler), `task docker` saves both Docker images as tars, and `task release VERSION=v0.4.0` uploads everything in `.local/dist` to that GitHub release.

## Fine-tune it further

The shipped model is `openjevx.w8.onnx` (8-bit weight-only; activations stay float, so answers don't depend on what else is in the request).

**Fine-tune kit (v0.4.0, 2.2 GB):** [openjevx-finetune-v0.4.0.tar.gz](https://pub-8da821f06ff747cda688f8267ed2aa96.r2.dev/openjevx-finetune-v0.4.0.tar.gz) ([sha256](https://pub-8da821f06ff747cda688f8267ed2aa96.r2.dev/openjevx-finetune-v0.4.0.tar.gz.sha256)). It holds the fine-tuned checkpoint (`model.safetensors`, `encoder/`, `tokenizer/`, `rl_agent_config.json`), both ONNX files (8-bit and fp32), and the training scripts. No training data. The same model files are on [Hugging Face](https://huggingface.co/muthuishere/openjevx).

1. **Data**: rows of `{state, questions:{id:{type: noul|choice|score, instructions, criteria}}, gold}`. `finetuning/dataprep/gen_it_worker.py` generates rule-labelled software-work decisions; add your own rows in the same shape.
2. **Shard**: `finetuning/dataprep/package_shards.py --out DIR --extra-train your.jsonl --exclude-keys leaked.json` adapts gold into targets, samples, and drops any question that also appears in your test sets.
3. **Train**: `cd finetuning && task all` runs data → leakage check → smoke run → full run → gate as one job. The GPU box
   pulls its data and runtime from a Cloudflare R2 bucket, uploads the 8-bit ONNX (≤ 750 MB) and the checkpoint, and
   destroys itself ([ADR 0009](docs/adr/0009-one-job-gpu-run-via-r2.md)). Settings live in
   `~/.config/openjevx/config.json`, data in `~/openjevx/data`. On your own CUDA box, run the steps in
   [`finetuning/ft.py`](finetuning/ft.py) and [`finetuning/train/run_job.sh`](finetuning/train/run_job.sh) directly.
4. **Ship**: the job already produces `openjevx.w8.onnx` (or run `python finetuning/export/quantize_w8.py openjevx.onnx openjevx.w8.onnx`), then set `"model"` in `openjevx.json` or rebuild with `task package`.
