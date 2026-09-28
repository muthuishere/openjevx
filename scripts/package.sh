#!/bin/sh
# Build every OpenJevX release package on this machine (no CI).
# Go cross-compiles; zig is the C compiler for the cgo ONNX Runtime binding.
# Output: .local/dist/openjevx-{darwin-arm64,linux-amd64}.tar, openjevx-windows-amd64.zip
set -eu
ORT=${ORT_VERSION:-1.22.0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/.local/dist
CACHE=$ROOT/.local/ort
MODEL=$ROOT/.local/openjevx.int8.onnx
cd "$ROOT"
mkdir -p "$OUT" "$CACHE"
[ -f "$MODEL" ] || { [ -f artifacts/openjevx.int8.onnx ] && cp artifacts/openjevx.int8.onnx "$MODEL"; } || \
  curl -fsSL -o "$MODEL" https://huggingface.co/muthuishere/openjevx/resolve/main/openjevx.int8.onnx

fetch() { # $1 package name, $2 extension, $3 library path inside
  [ -f "$CACHE/$(basename "$3")" ] && return
  curl -fsSL -o "$CACHE/$1.$2" "https://github.com/microsoft/onnxruntime/releases/download/v$ORT/$1.$2"
  case "$2" in tgz) tar -xzf "$CACHE/$1.$2" -C "$CACHE" ;; zip) (cd "$CACHE" && unzip -qo "$1.$2") ;; esac
  cp -L "$CACHE/$1/$3" "$CACHE/"
}
fetch onnxruntime-osx-arm64-$ORT tgz lib/libonnxruntime.dylib
fetch onnxruntime-linux-x64-$ORT tgz lib/libonnxruntime.so
fetch onnxruntime-win-x64-$ORT zip lib/onnxruntime.dll

# Embed the real model for the build, restore the placeholder afterwards.
cp internal/assets/model.onnx "$OUT/.placeholder.onnx"
trap 'cp "$OUT/.placeholder.onnx" internal/assets/model.onnx' EXIT
cp "$MODEL" internal/assets/model.onnx

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
(cd "$OUT" && shasum -a 256 openjevx-*.tar openjevx-*.zip > SHA256SUMS-server && cat SHA256SUMS-server)
