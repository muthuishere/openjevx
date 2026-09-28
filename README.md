# OpenJevX

Jev-compatible decision model. No Python. Apache-2.0.

Default port: **8000**

Server: http://127.0.0.1:8000/v1/systemone

Config file `openjevx.json`:

```json
{ "listen": "127.0.0.1:8000", "device": "cpu" }
```

`device` is `cpu` or `gpu`. `gpu` does not fall back to CPU.

## Install

One command. It downloads the binary over HTTPS and installs `openjevx`. You do not unpack anything.

```bash
npx git+https://github.com/muthuishere/openjevx.git
```

Then:

```bash
openjevx
```

If `openjevx` is not found:

```bash
export PATH="$HOME/.local/bin:$PATH"
openjevx
```

## Install manually

macOS:

```bash
curl -L -o openjevx.tar https://github.com/muthuishere/openjevx/releases/download/v0.1.0/openjevx-darwin-arm64.tar
tar -xf openjevx.tar
./openjevx
```

Linux:

```bash
curl -L -o openjevx.tar https://github.com/muthuishere/openjevx/releases/download/v0.1.0/openjevx-linux-amd64.tar
tar -xf openjevx.tar
./openjevx
```

Windows: download https://github.com/muthuishere/openjevx/releases/download/v0.1.0/openjevx-windows-amd64.zip and run `openjevx.exe`.

## jevx

```bash
jevx profile add openjevx http://127.0.0.1:8000/v1/systemone --model openjevx
jevx profile use openjevx
```

## Credits

We stand on Laya by Nandakishor Mukkunnoth (ConvAI Innovations) and ModernBERT by Answer.AI and LightOn. See `CREDITS`.

Apache-2.0. Any company may use this commercially or not. See `LICENSE` and `licenses/`.

Next training: `docs/adr/0002-training-run.md`.

This project is not affiliated with TypeSafe. Jev is a trademark of its respective owner.
