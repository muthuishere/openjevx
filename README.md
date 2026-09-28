# OpenJevX

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
curl -L -O https://github.com/muthuishere/openjevx/releases/download/v0.1.0/openjevx-darwin-arm64.tar
tar -xf openjevx-darwin-arm64.tar && ./openjevx
```

Linux:

```bash
curl -L -O https://github.com/muthuishere/openjevx/releases/download/v0.1.0/openjevx-linux-amd64.tar
tar -xf openjevx-linux-amd64.tar && ./openjevx
```

Windows: download https://github.com/muthuishere/openjevx/releases/download/v0.1.0/openjevx-windows-amd64.zip and run `openjevx.exe`.

## jevx

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
