# OpenJevX

Default port: **8000**

http://127.0.0.1:8000/v1/systemone

## One step

```bash
npx git+https://github.com/muthuishere/openjevx.git
```

That downloads the server and installs `openjevx`. Then run:

```bash
openjevx
```

## Manual

Download the file for your machine, unpack it, run the binary.

macOS — download this, then run:

```bash
curl -L -O https://github.com/muthuishere/openjevx/releases/download/v0.1.0/openjevx-darwin-arm64.tar
tar -xf openjevx-darwin-arm64.tar && ./openjevx
```

Linux — download this, then run:

```bash
curl -L -O https://github.com/muthuishere/openjevx/releases/download/v0.1.0/openjevx-linux-amd64.tar
tar -xf openjevx-linux-amd64.tar && ./openjevx
```

Windows — download this, then run `openjevx.exe`:

https://github.com/muthuishere/openjevx/releases/download/v0.1.0/openjevx-windows-amd64.zip

## jevx

```bash
jevx profile add openjevx http://127.0.0.1:8000/v1/systemone --model openjevx
jevx profile use openjevx
```

Config is `openjevx.json`. `"listen": "127.0.0.1:8000"`. `"device"` is `cpu` or `gpu`.

Apache-2.0. Credits: `CREDITS`. Next training: `docs/adr/0002-training-run.md`.
