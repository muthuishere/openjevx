#!/usr/bin/env bash
# One-shot OpenJevX fine-tune on a rented CUDA box (vast.ai or any Linux GPU).
# Gets the private training shard (SSH copy or signed URLs), trains, calibrates,
# exports ONNX on CUDA, quantizes to 8-bit and ships ONLY the 8-bit file, which must be
# <= MAX_W8_MB (750). Markers: /root/JOB_COMPLETE or /root/JOB_FAILED.
# Artifacts: /root/out/openjevx-model (openjevx.w8.onnx + tokenizer/config). Log: /root/job.log.
# Usage: SHARD=full bash finetuning/train/run_job.sh   (data copied to /root/in over SSH, then /root/in/READY)
#    or: SHARD=full TRAIN_URL=... EVAL_URL=... RUN_URL=... bash finetuning/train/run_job.sh
set -euo pipefail
trap 'touch /root/JOB_FAILED' ERR

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SHARD="${SHARD:-full}"
IN=/root/in; OUT=/root/out; JOB=/root/jobsrc
mkdir -p "$IN" "$OUT" "$JOB"
if [ -n "${TRAIN_URL:-}" ]; then
  curl -fsSL -o "$IN/openjevx-${SHARD}-train.jsonl.gz" "$TRAIN_URL"
  curl -fsSL -o "$IN/openjevx-${SHARD}-eval.jsonl.gz" "${EVAL_URL:?set EVAL_URL}"
  curl -fsSL -o "$IN/openjevx-${SHARD}-run.json" "${RUN_URL:?set RUN_URL}"
else
  # No URLs: the data is copied onto this box over SSH, then $IN/READY is written.
  for i in $(seq 1 270); do [ -f "$IN/READY" ] && break; sleep 10; done
  [ -f "$IN/READY" ] || { echo "data never arrived"; false; }
fi
mv "$IN/openjevx-${SHARD}-run.json" "$IN/run.json"
if [ "$SHARD" = smoke ]; then
  mv "$IN/openjevx-smoke-train.jsonl.gz" "$IN/train_smoke.jsonl.gz"; mv "$IN/openjevx-smoke-eval.jsonl.gz" "$IN/eval_smoke.jsonl.gz"
else
  mv "$IN/openjevx-${SHARD}-train.jsonl.gz" "$IN/train.jsonl.gz"; mv "$IN/openjevx-${SHARD}-eval.jsonl.gz" "$IN/eval.jsonl.gz"
fi
cp "$ROOT"/finetuning/train/{train_job.py,eval_job.py,adapter.py,train_openjevx.py} "$ROOT"/finetuning/export/{export_onnx_gpu.py,quantize_w8.py} "$JOB/"

python -m pip install -q "laya @ git+https://github.com/NandhaKishorM/laya.git@9d955671415fc19f069b9cc998928075c1f255ec" \
  transformers==4.57.6 datasets sentencepiece protobuf safetensors accelerate huggingface_hub tokenizers onnx
python -m pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
python -m pip install -q --force-reinstall --no-deps onnxruntime-gpu==1.22.0
python - <<'PY'
import torch, onnxruntime as ort
assert torch.cuda.is_available(), "CUDA unavailable"
if hasattr(ort, "preload_dlls"): ort.preload_dlls()
assert "CUDAExecutionProvider" in ort.get_available_providers(), ort.get_available_providers()
print("preflight CUDA OK", torch.cuda.get_device_name(0), ort.__version__, flush=True)
PY

cd "$JOB"
INPUT_DIR="$IN" WORK_DIR="$OUT" PYTHONPATH="$JOB" python train_job.py
PYTHONPATH="$JOB" python export_onnx_gpu.py --model "$OUT/openjevx-model" --output "$OUT/fp32/openjevx.onnx"
python quantize_w8.py "$OUT/fp32/openjevx.onnx" "$OUT/openjevx-model/openjevx.w8.onnx"
MB=$(( $(stat -c %s "$OUT/openjevx-model/openjevx.w8.onnx") / 1048576 ))
echo "8-bit ONNX: ${MB} MB (limit ${MAX_W8_MB:-750})"
[ "$MB" -le "${MAX_W8_MB:-750}" ] || { echo "8-bit ONNX over the size limit"; false; }
rm -rf "$OUT/fp32"
cp /root/job.log "$OUT/" || true
touch /root/JOB_COMPLETE
