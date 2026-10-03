#!/usr/bin/env python3
"""Turn an 8-bit weight-only model folder into a dynamic-int8 one (llmresults/14).

The shipped graph is DequantizeLinear (int8, per output channel) + fp32 MatMul. ONNX Runtime fuses that into
MatMulNBits, which has no fast int8 path on x86. This writes the same weights as a QOperator graph
(DynamicQuantizeLinear + MatMulInteger, which ORT fuses into DynamicQuantizeMatMul): activations are quantized
per request, weights stay 8-bit, the file stays the same size. Two weight types:
  s8rr  int8 weights with reduce_range (7-bit), so AVX2 without VNNI cannot saturate its int16 sums
  u8    uint8 weights (U8U8 kernels; no saturation on AVX2)
Needs: pip install onnx onnxruntime numpy
Usage: quantize_dynamic.py MODEL_DIR OUT_DIR [--weights s8rr|u8] [--version X]
The folder gets a new config.json (same temperatures and ids; new sha256 and "quantization"). It must pass
the gate before it ships: `python3 ft.py gate OUT_DIR`.
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile

import numpy as np
import onnx
from onnx import numpy_helper

GRAPH = "openjevx.w8.onnx"


def dequantized(src, dst):
    """The graph with every constant DequantizeLinear folded into an fp32 initializer (exact weights)."""
    m = onnx.load(src)
    inits = {i.name: i for i in m.graph.initializer}
    keep, folded = [], {}
    for node in m.graph.node:
        if node.op_type == "DequantizeLinear" and all(x in inits for x in node.input if x):
            q = numpy_helper.to_array(inits[node.input[0]]).astype(np.float32)
            scale = numpy_helper.to_array(inits[node.input[1]]).astype(np.float32)
            zp = numpy_helper.to_array(inits[node.input[2]]).astype(np.float32) if len(node.input) > 2 and node.input[2] else 0
            axis = next((a.i for a in node.attribute if a.name == "axis"), 1)
            shape = [1] * q.ndim
            if np.ndim(scale):
                shape[axis] = -1
            w = (q - np.reshape(zp, shape)) * np.reshape(scale, shape)
            folded[node.output[0]] = numpy_helper.from_array(w.astype(np.float32), node.output[0])
        else:
            keep.append(node)
    used = {x for n in keep for x in n.input}
    del m.graph.node[:]
    m.graph.node.extend(keep)
    old = [i for i in m.graph.initializer if i.name in used]
    del m.graph.initializer[:]
    m.graph.initializer.extend(old + list(folded.values()))
    onnx.save(m, dst, save_as_external_data=True, location=os.path.basename(dst) + ".data")
    return dst


def dynamic(fp32, dst, weights):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(fp32, dst, per_channel=True, op_types_to_quantize=["MatMul"],
                     weight_type=QuantType.QUInt8 if weights == "u8" else QuantType.QInt8,
                     reduce_range=weights == "s8rr")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--weights", choices=["s8rr", "u8"], default="u8")
    ap.add_argument("--version", help="config.json version (default: the source's)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="qdyn-")
    try:
        fp32 = dequantized(os.path.join(a.model_dir, GRAPH), os.path.join(work, "fp32.onnx"))
        tmp = dynamic(fp32, os.path.join(work, "dyn.onnx"), a.weights)
        onnx.save(onnx.load(tmp), os.path.join(a.out_dir, GRAPH))  # one self-contained file (< 2 GB)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    graph = os.path.join(a.out_dir, GRAPH)
    cfg["sha256"] = hashlib.sha256(open(graph, "rb").read()).hexdigest()
    cfg["quantization"] = f"8-bit dynamic ({a.weights} weights per channel, int8 activations per request)"
    if a.version:
        cfg["version"] = a.version
    json.dump(cfg, open(os.path.join(a.out_dir, "config.json"), "w"), indent=2)
    tok = os.path.join(a.model_dir, "tokenizer.json")
    if os.path.exists(tok):
        shutil.copy(tok, a.out_dir)
    print(f"{graph}: {os.path.getsize(graph) / 1e6:.0f} MB, sha256 {cfg['sha256'][:12]}, {cfg['quantization']}")


if __name__ == "__main__":
    main()
