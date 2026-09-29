#!/usr/bin/env python3
"""Make an 8-bit weight-only CPU graph.

Every MatMul weight becomes int8 (symmetric, per output channel) followed by
DequantizeLinear, so the file is ~1/4 of fp32 while activations stay float:
answers do not depend on what else is in the request.
Needs: pip install onnx numpy
Usage: quantize_w8.py artifacts/openjevx.onnx .local/openjevx.w8.onnx
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
model = onnx.load(src)
graph = model.graph
inits = {i.name: i for i in graph.initializer}
done = {}
new_nodes = []
for node in graph.node:
    if node.op_type == "MatMul":
        for slot, name in enumerate(node.input):
            init = inits.get(name)
            if init is None or init.data_type != onnx.TensorProto.FLOAT or len(init.dims) != 2:
                continue
            if name not in done:
                w = numpy_helper.to_array(init)                     # [in, out]
                scale = np.abs(w).max(axis=0) / 127.0
                scale[scale == 0] = 1.0
                q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
                graph.initializer.remove(init)
                graph.initializer.extend([numpy_helper.from_array(q, name + "_q8"),
                                          numpy_helper.from_array(scale.astype(np.float32), name + "_scale")])
                new_nodes.append(helper.make_node("DequantizeLinear", [name + "_q8", name + "_scale"], [name + "_dq"],
                                                  name=name + "_DequantizeLinear", axis=1))
                done[name] = name + "_dq"
            node.input[slot] = done[name]
print("quantized", len(done), "weight matrices")
nodes = new_nodes + list(graph.node)
del graph.node[:]
graph.node.extend(nodes)
onnx.save(model, dst)
