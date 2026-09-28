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
