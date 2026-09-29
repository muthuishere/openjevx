#!/usr/bin/env bash
# Build the GPU box runtime ONCE on this machine and put it in R2, so no GPU box ever builds it.
# Same image as the box (python:3.11-slim, linux/amd64 via Docker), same /opt/venv path, so the result is
# identical to what a box would build. Nothing large is stored locally: each tar is streamed from the
# container straight into an R2 multipart upload.
#   runtime/<key>/env.part.00   the venv (PyTorch + CUDA libs + packages), tar.gz
#   runtime/<key>/hf.tar        the base model's Hugging Face cache
#   runtime/<key>/env.count     "1"  (written last: its presence means the runtime is complete)
# Usage: finetuning/build_runtime.sh   (task runtime)   Needs Docker.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.local/eval/venv/bin/python"; [ -x "$PY" ] || PY=python3
IMAGE="${VAST_IMAGE:-python:3.11-slim}"
BASE_MODEL="${BASE_MODEL:-convaiinnovations/laya}"
REQ="$ROOT/finetuning/train/requirements-box.txt"
KEY=$("$PY" -c "import sys; sys.path.insert(0, '$ROOT/finetuning'); import r2; print(r2.runtime_key(open('$REQ').read(), '$IMAGE', '$BASE_MODEL'))")
RT="runtime/$KEY"
if "$PY" "$ROOT/finetuning/r2.py" ls "$RT/" | grep -q "env.count"; then echo "runtime $KEY already in R2"; exit 0; fi
echo "building runtime $KEY from $IMAGE (linux/amd64)"
run() { docker run --rm -i --platform linux/amd64 -e BASE_MODEL="$BASE_MODEL" -v "$REQ:/req.txt:ro" "$IMAGE" bash -c "$1"; }

echo "== venv -> r2://openjevx-train/$RT/env.part.00"
run 'set -e; { apt-get update -qq && apt-get install -y -qq --no-install-recommends git >/dev/null; } 1>&2
     python -m venv /opt/venv 1>&2
     /opt/venv/bin/pip install -q --no-cache-dir --upgrade pip 1>&2
     /opt/venv/bin/pip install -q --no-cache-dir -r /req.txt 1>&2
     /opt/venv/bin/pip uninstall -y onnxruntime onnxruntime-gpu 1>&2 || true
     /opt/venv/bin/pip install -q --no-cache-dir --force-reinstall --no-deps onnxruntime-gpu==1.22.0 1>&2
     du -sh /opt/venv 1>&2
     tar -C /opt -czf - venv' | "$PY" "$ROOT/finetuning/r2.py" put-stream "$RT/env.part.00"

echo "== base model -> r2://openjevx-train/$RT/hf.tar"
run 'set -e; pip install -q --no-cache-dir huggingface_hub 1>&2
     HF_HOME=/hf python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ[\"BASE_MODEL\"])" 1>&2
     du -sh /hf 1>&2
     tar -C /hf -cf - .' | "$PY" "$ROOT/finetuning/r2.py" put-stream "$RT/hf.tar"

echo 1 | "$PY" "$ROOT/finetuning/r2.py" put-stream "$RT/env.count"
"$PY" "$ROOT/finetuning/r2.py" ls "$RT/"
echo "runtime $KEY ready in R2"
