#!/usr/bin/env bash
# One-shot OpenJevX fine-tune on a rented CUDA box. The box does everything and cleans up after itself:
#   1. fetch its settings from JOB_ENV_URL (a signed R2 link; the file holds only signed links + the kill token)
#   2. download the shard from R2, train, calibrate, export ONNX on CUDA, quantize to 8-bit
#   3. fail if the 8-bit ONNX is over MAX_W8_MB (750); upload the 8-bit ONNX, the trainable checkpoint,
#      the eval report, the log and a status file to R2 through signed PUT links
#   4. destroy itself through the destroy endpoint (which holds the Vast key; this box never does)
# A timer destroys the box after DEADLINE_HOURS whatever happens. Markers: /root/JOB_COMPLETE or /root/JOB_FAILED.
# Usage: JOB_ENV_URL=<signed link> bash finetuning/train/run_job.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IN=/root/in; OUT=/root/out; JOB=/root/jobsrc; LOG=/root/job.log
mkdir -p "$IN" "$OUT" "$JOB"
curl -fsSL --retry 5 -o /root/job.env "${JOB_ENV_URL:?set JOB_ENV_URL}"
# shellcheck disable=SC1091
. /root/job.env
INSTANCE_ID="${CONTAINER_ID:-${VAST_CONTAINERLABEL#C.}}"

put() { curl -fsS --retry 5 -X PUT -T "$2" "$1" >/dev/null; }
destroy() {
  curl -fsS --retry 5 -X POST "$KILL_URL" -H "X-Kill-Token: $KILL_TOKEN" -H "Content-Type: application/json" \
    -d "{\"instance_id\": ${INSTANCE_ID:-0}}" || true
}
finish() {  # $1 = complete | failed | timeout
  printf '{"status":"%s","instance_id":"%s","finished":"%s","w8_mb":%s}\n' "$1" "$INSTANCE_ID" "$(date -u +%FT%TZ)" "${MB:-0}" > "$OUT/status.json"
  put "$PUT_LOG_URL" "$LOG" || true
  put "$PUT_STATUS_URL" "$OUT/status.json" || true
  destroy
}
trap 'touch /root/JOB_FAILED; finish failed' ERR
# Self-destroy timer, in its own session so it outlives this script.
setsid bash -c "sleep $(python3 -c "print(int(float('${DEADLINE_HOURS:-7}') * 3600))"); \
  [ -f /root/JOB_COMPLETE ] || { . /root/job.env; curl -fsS -X PUT -T $LOG \"\$PUT_LOG_URL\"; \
  curl -fsS -X POST \"\$KILL_URL\" -H \"X-Kill-Token: \$KILL_TOKEN\" -d '{\"instance_id\": ${INSTANCE_ID:-0}}'; }" \
  >/dev/null 2>&1 < /dev/null &

if [ "$SHARD" = smoke ]; then TR=train_smoke; EV=eval_smoke; else TR=train; EV=eval; fi
curl -fsSL --retry 5 -o "$IN/$TR.jsonl.gz" "$TRAIN_URL"
curl -fsSL --retry 5 -o "$IN/$EV.jsonl.gz" "$EVAL_URL"
curl -fsSL --retry 5 -o "$IN/run.json" "$RUN_URL"
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
python quantize_w8.py "$OUT/fp32/openjevx.onnx" "$OUT/openjevx.w8.onnx"
MB=$(( $(stat -c %s "$OUT/openjevx.w8.onnx") / 1048576 ))
echo "8-bit ONNX: ${MB} MB (limit ${MAX_W8_MB:-750})"
[ "$MB" -le "${MAX_W8_MB:-750}" ] || { echo "8-bit ONNX over the size limit"; false; }
rm -rf "$OUT/fp32"
# The trainable checkpoint (weights, tokenizer, config) so the model can be fine-tuned again.
tar -C "$OUT" -czf "$OUT/checkpoint.tar.gz" openjevx-model
put "$PUT_W8_URL" "$OUT/openjevx.w8.onnx"
put "$PUT_CKPT_URL" "$OUT/checkpoint.tar.gz"
put "$PUT_REPORT_URL" "$OUT/openjevx-model/eval_report.json"
touch /root/JOB_COMPLETE
trap - ERR
finish complete
