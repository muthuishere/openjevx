# OpenJevX

Open, local decision model server for [jevx](https://github.com/muthuishere/jevx). Source: [github.com/muthuishere/openjevx](https://github.com/muthuishere/openjevx) · Releases: [latest](https://github.com/muthuishere/openjevx/releases/latest)

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
curl -L -O https://github.com/muthuishere/openjevx/releases/download/v0.2.0/openjevx-darwin-arm64.tar
tar -xf openjevx-darwin-arm64.tar && ./openjevx
```

Linux:

```bash
curl -L -O https://github.com/muthuishere/openjevx/releases/download/v0.2.0/openjevx-linux-amd64.tar
tar -xf openjevx-linux-amd64.tar && ./openjevx
```

Windows: download https://github.com/muthuishere/openjevx/releases/download/v0.2.0/openjevx-windows-amd64.zip and run `openjevx.exe`.

## Docker

```bash
git clone https://github.com/muthuishere/openjevx.git && cd openjevx
docker compose up -d --build
```

No clone? Load a prebuilt image from the release (`amd64` or `arm64`):

```bash
curl -L -O https://github.com/muthuishere/openjevx/releases/download/v0.2.0/openjevx-docker-amd64.tar
docker load -i openjevx-docker-amd64.tar
docker run -d -p 127.0.0.1:21118:21118 ghcr.io/muthuishere/openjevx:latest-amd64
```

## jevx

Use OpenJevX from the [jevx CLI](https://github.com/muthuishere/jevx):

```bash
jevx profile add openjevx http://127.0.0.1:21118/v1/systemone --model openjevx
jevx profile use openjevx
```

Apache-2.0. Credits: `CREDITS`. Next training: `docs/adr/0002-training-run.md`.

## Dashboard

Open http://127.0.0.1:21118/ while the server runs. It shows request count, questions answered, input tokens, latency p50/p95/p99, errors, and recent requests.

Password default: `adminadmin`, change it in `openjevx.json` (`"password"`).

- `GET /stats` — JSON snapshot (same password)
- `GET /metrics` — Prometheus format (same password)
- `GET /health` — open, no password

## Run locally from source

Needs Go and [Task](https://taskfile.dev). `task run` fetches ONNX Runtime and the int8 model into `.local/`, builds, and starts the server on http://127.0.0.1:21118/. `task build` only builds; `task test` runs the tests.

Release from this machine, no CI: `task package` builds the macOS, Linux and Windows packages (Go cross-compiles, [zig](https://ziglang.org) is the C compiler), `task docker` saves both Docker images as tars, and `task release VERSION=v0.2.0` uploads everything in `.local/dist` to that GitHub release.
