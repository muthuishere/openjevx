#!/usr/bin/env python3
"""Make the shipped CPU graph: 4-bit weight-only (MatMulNBits, block 32, symmetric).

Activations stay float, so answers do not depend on what else is in the request
(dynamic int8 did, and lost ~12 points on typed-decisions test).
Needs: pip install onnx onnxruntime onnx_ir
Usage: quantize_w4.py artifacts/openjevx.onnx .local/openjevx.w4.onnx
"""
import sys

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

src, dst = sys.argv[1], sys.argv[2]
quantizer = MatMulNBitsQuantizer(onnx.load(src), algo_config=DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=4))
quantizer.process()
quantizer.model.save_model_to_file(dst, use_external_data_format=False)
