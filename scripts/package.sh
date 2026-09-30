#!/bin/sh
# Build every OpenJevX release package on this machine (no CI).
# Go cross-compiles; zig is the C compiler for the cgo ONNX Runtime binding.
# A release is a small binary per platform plus ONE model folder archive (no model inside the binary):
#   .local/dist/openjevx-{darwin-arm64,linux-amd64}.tar, openjevx-windows-amd64.zip
#   .local/dist/openjevx-model-<version>.tar.gz   (model/: openjevx.w8.onnx, config.json, tokenizer.json)
# Unpack both into the same directory: the server finds model/ next to itself.
# MODEL_DIR (default .local/model) is the model folder to ship; build one with
# finetuning/export/make_model_folder.py, or take <run>/out/model from a training run.
set -eu
ORT=${ORT_VERSION:-1.29.0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/.local/dist
CACHE=$ROOT/.local/ort
MODEL_DIR=${MODEL_DIR:-$ROOT/.local/model}
cd "$ROOT"
mkdir -p "$OUT" "$CACHE"
for f in openjevx.w8.onnx config.json tokenizer.json; do
  [ -f "$MODEL_DIR/$f" ] || { echo "MODEL_DIR=$MODEL_DIR is not a model folder (missing $f)" >&2; exit 1; }
done
VERSION=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['version'])" "$MODEL_DIR/config.json")

fetch() { # $1 package name, $2 extension, $3 library path inside
  [ -f "$CACHE/$(basename "$3")" ] && return
  curl -fsSL -o "$CACHE/$1.$2" "https://github.com/microsoft/onnxruntime/releases/download/v$ORT/$1.$2"
  case "$2" in tgz) tar -xzf "$CACHE/$1.$2" -C "$CACHE" ;; zip) (cd "$CACHE" && unzip -qo "$1.$2") ;; esac
  cp -L "$CACHE/$1/$3" "$CACHE/"
}
fetch onnxruntime-osx-arm64-$ORT tgz lib/libonnxruntime.dylib
fetch onnxruntime-linux-x64-$ORT tgz lib/libonnxruntime.so
fetch onnxruntime-win-x64-$ORT zip lib/onnxruntime.dll

pack() { # $1 name, $2 GOOS, $3 CC target (empty = native), $4 exe, $5 lib, $6 launcher
  dir="$OUT/$1"; rm -rf "$dir"; mkdir -p "$dir"
  if [ -n "$3" ]; then cc="zig cc -target $3"; else cc=cc; fi
  GOOS=$2 GOARCH=${1##*-} CGO_ENABLED=1 CC="$cc" go build -trimpath -ldflags "-s -w" -o "$dir/$4" ./cmd/openjevx
  cp "$CACHE/$5" "$6" openjevx.json "$dir/"
  echo "built $1"
}
pack darwin-arm64 darwin "" openjevx libonnxruntime.dylib README
pack linux-amd64 linux x86_64-linux-gnu.2.28 openjevx libonnxruntime.so README
pack windows-amd64 windows x86_64-windows-gnu openjevx.exe onnxruntime.dll README.cmd

(cd "$OUT/darwin-arm64" && COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs --format ustar -cf ../openjevx-darwin-arm64.tar README openjevx openjevx.json libonnxruntime.dylib)
(cd "$OUT/linux-amd64" && COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs --format ustar -cf ../openjevx-linux-amd64.tar README openjevx openjevx.json libonnxruntime.so)
(cd "$OUT/windows-amd64" && rm -f ../openjevx-windows-amd64.zip && zip -q ../openjevx-windows-amd64.zip README.cmd openjevx.exe openjevx.json onnxruntime.dll)

# The model folder, always unpacked as model/.
rm -rf "$OUT/model" && mkdir -p "$OUT/model"
for f in openjevx.w8.onnx config.json tokenizer.json; do ln "$MODEL_DIR/$f" "$OUT/model/$f" 2>/dev/null || cp "$MODEL_DIR/$f" "$OUT/model/$f"; done
rm -f "$OUT"/openjevx-model-*.tar.gz
(cd "$OUT" && COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs --format ustar -czf "openjevx-model-$VERSION.tar.gz" model && rm -rf model)
echo "built openjevx-model-$VERSION.tar.gz"
(cd "$OUT" && shasum -a 256 openjevx-*.tar openjevx-*.zip openjevx-model-*.tar.gz > SHA256SUMS-server && cat SHA256SUMS-server)
