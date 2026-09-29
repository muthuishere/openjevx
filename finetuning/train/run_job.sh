#!/usr/bin/env bash
# One-shot OpenJevX fine-tune on a rented CUDA box. The box does everything and cleans up after itself:
#   0. runtime (Python wheels + base model) from our R2 bundle when it exists
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

fetch() { curl -fsSL --retry 8 --retry-all-errors --retry-delay 5 --connect-timeout 20 -C - -o "$2" "$1"; }
put() { curl -fsS --retry 8 --retry-all-errors --retry-delay 5 --connect-timeout 20 -X PUT -T "$2" "$1" >/dev/null; }
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

# Heartbeat: upload the log to R2 every 2 minutes, so progress is visible without SSH.
setsid bash -c ". /root/job.env; while [ ! -f /root/JOB_COMPLETE ] && [ ! -f /root/JOB_FAILED ]; do sleep 120; \
  curl -fsS -X PUT -T $LOG \"\$PUT_LOG_URL\" >/dev/null 2>&1; done" < /dev/null > /dev/null 2>&1 &

if [ "$SHARD" = smoke ]; then TR=train_smoke; EV=eval_smoke; else TR=train; EV=eval; fi
fetch "$TRAIN_URL" "$IN/$TR.jsonl.gz"
fetch "$EVAL_URL" "$IN/$EV.jsonl.gz"
fetch "$RUN_URL" "$IN/run.json"
cp "$ROOT"/finetuning/train/{train_job.py,eval_job.py,adapter.py,train_openjevx.py} "$ROOT"/finetuning/export/{export_onnx_gpu.py,quantize_w8.py} "$JOB/"

# Runtime from our R2 bucket: the box starts from a bare python:3.11-slim image and gets EVERYTHING else
# (PyTorch + CUDA libraries, all packages in a venv, and the base model) from R2 as a packed environment
# split into parts. If the environment for this key is not in R2 yet, this box builds it from PyPI /
# Hugging Face once and uploads it; every later box only downloads and unpacks.
export HF_HOME=/root/hf VENV=/opt/venv
command -v gcc >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends gcc g++ >/dev/null; }
command -v gcc >/dev/null || { echo "no C compiler, torch.compile will fail"; false; }
REQ="$ROOT/finetuning/train/requirements-box.txt"
got_env() {
  [ -n "${ENV_GET_URLS:-}" ] || return 1
  i=0; rm -f /root/env.part.*
  for u in $ENV_GET_URLS; do fetch "$u" "/root/env.part.$(printf %02d $i)" || return 1; i=$((i+1)); done
  fetch "$HF_GET_URL" /root/hf.tar || return 1
  cat /root/env.part.* | tar -xz -C /opt && mkdir -p "$HF_HOME" && tar -xf /root/hf.tar -C "$HF_HOME" \
    && rm -f /root/env.part.* /root/hf.tar
}
if got_env; then
  echo "runtime: from R2 ($(echo $ENV_GET_URLS | wc -w) parts)"
  export HF_HUB_OFFLINE=1
else
  [ -n "${ENV_GET_URLS:-}" ] && echo "runtime: R2 download failed, building from PyPI + Hugging Face"
  [ -z "${ENV_GET_URLS:-}" ] && echo "runtime: not in R2 yet, building from PyPI + Hugging Face and uploading it"
  rm -rf "$VENV"; python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r "$REQ"
  "$VENV/bin/pip" uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
  "$VENV/bin/pip" install -q --force-reinstall --no-deps onnxruntime-gpu==1.22.0
  "$VENV/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ.get("BASE_MODEL", "convaiinnovations/laya"))
PY
  if [ -n "${ENV_PUT_URLS:-}" ] && [ -z "${ENV_GET_URLS:-}" ]; then
    # 1000 MB parts: one signed PUT is limited to 5 GB.
    ( tar -C /opt -czf - venv | split -b 1000m - /root/env.part. \
      && tar -C "$HF_HOME" -cf /root/hf.tar . \
      && parts=$(ls /root/env.part.* | wc -l) && [ "$parts" -le "$(echo $ENV_PUT_URLS | wc -w)" ] \
      && set -- $ENV_PUT_URLS && for f in /root/env.part.*; do put "$1" "$f" || exit 1; shift; done \
      && put "$HF_PUT_URL" /root/hf.tar \
      && echo "$parts" > /root/env.count && put "$ENV_COUNT_PUT_URL" /root/env.count \
      && echo "runtime: uploaded to R2 ($parts parts)" ) || echo "runtime: upload to R2 failed (the next box builds again)"
    rm -f /root/env.part.* /root/hf.tar
  fi
fi
export PATH="$VENV/bin:$PATH"
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
