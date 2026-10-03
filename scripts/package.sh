#!/bin/sh
# Build every OpenJevX release package on this machine (no CI).
# Go cross-compiles; zig is the C compiler for the cgo ONNX Runtime binding.
# A release is a small binary per platform plus ONE model folder archive (no model inside the binary):
#   .local/dist/openjevx-{darwin-arm64,linux-amd64,linux-arm64}.tar, openjevx-windows-amd64.zip
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

fetch() { # $1 package name, $2 extension; unpacked as $CACHE/$1/
  [ -d "$CACHE/$1/lib" ] && return
  curl -fsSL -o "$CACHE/$1.$2" "https://github.com/microsoft/onnxruntime/releases/download/v$ORT/$1.$2"
  case "$2" in tgz) tar -xzf "$CACHE/$1.$2" -C "$CACHE" ;; zip) (cd "$CACHE" && unzip -qo "$1.$2") ;; esac
}
fetch onnxruntime-osx-arm64-$ORT tgz
fetch onnxruntime-linux-x64-$ORT tgz
fetch onnxruntime-linux-aarch64-$ORT tgz
fetch onnxruntime-win-x64-$ORT zip

pack() { # $1 name, $2 GOOS, $3 CC target (empty = native), $4 exe, $5 library inside $CACHE, $6 launcher
  dir="$OUT/$1"; rm -rf "$dir"; mkdir -p "$dir"
  if [ -n "$3" ]; then cc="zig cc -target $3"; else cc=cc; fi
  GOOS=$2 GOARCH=${1##*-} CGO_ENABLED=1 CC="$cc" go build -trimpath -ldflags "-s -w" -o "$dir/$4" ./cmd/openjevx
  cp -L "$CACHE/$5" "$6" openjevx.json LICENSE NOTICE CREDITS "$dir/" && cp -R licenses "$dir/"
  echo "built $1"
}
pack darwin-arm64 darwin "" openjevx onnxruntime-osx-arm64-$ORT/lib/libonnxruntime.dylib README
pack linux-amd64 linux x86_64-linux-gnu.2.28 openjevx onnxruntime-linux-x64-$ORT/lib/libonnxruntime.so README
pack linux-arm64 linux aarch64-linux-gnu.2.28 openjevx onnxruntime-linux-aarch64-$ORT/lib/libonnxruntime.so README
pack windows-amd64 windows x86_64-windows-gnu openjevx.exe onnxruntime-win-x64-$ORT/lib/onnxruntime.dll README.cmd

(cd "$OUT/darwin-arm64" && COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs --format ustar -cf ../openjevx-darwin-arm64.tar README openjevx openjevx.json libonnxruntime.dylib LICENSE NOTICE CREDITS licenses)
for a in amd64 arm64; do
  (cd "$OUT/linux-$a" && COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs --format ustar -cf ../openjevx-linux-$a.tar README openjevx openjevx.json libonnxruntime.so LICENSE NOTICE CREDITS licenses)
done
(cd "$OUT/windows-amd64" && rm -f ../openjevx-windows-amd64.zip && zip -qr ../openjevx-windows-amd64.zip README.cmd openjevx.exe openjevx.json onnxruntime.dll LICENSE NOTICE CREDITS licenses)

# The model folder, always unpacked as model/.
rm -rf "$OUT/model" && mkdir -p "$OUT/model"
for f in openjevx.w8.onnx config.json tokenizer.json; do ln "$MODEL_DIR/$f" "$OUT/model/$f" 2>/dev/null || cp "$MODEL_DIR/$f" "$OUT/model/$f"; done
# The weights are Apache-2.0 work derived from Laya and ModernBERT: their notices travel with them.
cp LICENSE NOTICE CREDITS "$OUT/model/" && cp -R licenses "$OUT/model/"
rm -f "$OUT"/openjevx-model-*.tar.gz
(cd "$OUT" && COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs --format ustar -czf "openjevx-model-$VERSION.tar.gz" model && rm -rf model)
echo "built openjevx-model-$VERSION.tar.gz"
(cd "$OUT" && shasum -a 256 openjevx-*.tar openjevx-*.zip openjevx-model-*.tar.gz > SHA256SUMS-server && cat SHA256SUMS-server)
