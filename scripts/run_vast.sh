#!/usr/bin/env bash
set -euo pipefail

trap 'touch /root/OPENJEVX_FAILED' ERR

cd /root/openjevx
export USE_TF=0
export USE_TORCH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m pip install -q -e /root/laya datasets transformers safetensors huggingface_hub pyarrow pandas scipy accelerate onnx
python -m pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
python -m pip install --force-reinstall --no-deps -q onnxruntime-gpu==1.22.0
python - <<'PY'
import torch
import onnxruntime as ort
assert torch.cuda.is_available(), "CUDA unavailable"
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()
assert "CUDAExecutionProvider" in ort.get_available_providers(), ort.get_available_providers()
print("preflight CUDA OK", torch.cuda.get_device_name(0), ort.__version__, flush=True)
PY

python scripts/train_openjevx.py 2>&1 | tee /root/openjevx-train.log
python scripts/export_onnx_gpu.py --model /root/openjevx-model --output /root/openjevx-model/openjevx.onnx \
  2>&1 | tee /root/openjevx-onnx.log
touch /root/OPENJEVX_COMPLETE
