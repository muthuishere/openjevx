#!/usr/bin/env python3
"""Where does CPU inference time go, and do cheap ORT settings or graph variants help?

    python scripts/ort_cpu_probe.py MODEL_DIR [--threads 1] [--rounds 5] [--out report.json]

Prints the CPU's SIMD flags, the ops left in the optimized graph (did the 8-bit weights fuse into
an int8 kernel?), the slowest ops from ORT's profiler, and short/long latency for:
  default            the server's settings (ORT_ENABLE_ALL)
  extended / basic   lower graph optimization levels
  no-qdq-fusion      session.disable_quant_qdq=1 (DequantizeLinear + fp32 MatMul every call)
  acc-level-N        session.qdq_matmulnbits_accuracy_level=N (how a fused MatMulNBits computes)
  dynamic-int8       the same weights as a QOperator graph (quantize_dynamic: int8 activations)
  fp32               the same weights dequantized once (4x the size; an upper bound, not a ship candidate)
Inputs are built like the server's: one item, two markers, qtype 0 (noul); token ids are seeded random.
max_logit_diff is against the default run on the same inputs (0 = identical numerics).
"""
import argparse
import collections
import gc
import json
import os
import platform
import statistics
import sys
import tempfile
import time

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort


def cpu_flags():
    want = ["avx", "avx2", "fma", "avx512f", "avx512bw", "avx512_vnni", "avx_vnni", "amx_int8", "asimddp", "i8mm", "sve", "sme"]
    flags = set()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith(("flags", "Features")):
                flags |= set(line.split(":", 1)[1].split())
    except OSError:
        pass
    model = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "") if os.path.exists("/proc/cpuinfo") else platform.processor()
    return {"model": model, "cores": os.cpu_count(), "flags": [f for f in want if f in flags]}


def inputs(length):
    ids = np.random.default_rng(length).integers(1000, 30000, size=(1, length), dtype=np.int64)
    ids[0, 0], ids[0, -1] = 0, 2
    return {
        "input_ids": ids,
        "attention_mask": np.ones((1, length), dtype=np.int64),
        "marker_pos": np.array([[1, 2]], dtype=np.int64),
        "marker_mask": np.ones((1, 2), dtype=bool),
        "qtype": np.zeros((1,), dtype=np.int64),
    }


def session(path, threads, level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL, configs=None, optimized=None, profile=None):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.graph_optimization_level = level
    for k, v in (configs or {}).items():
        o.add_session_config_entry(k, v)
    if optimized:
        o.optimized_model_filepath = optimized
    if profile:
        o.enable_profiling = True
        o.profile_file_prefix = profile
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return float("nan")


def timeit(s, rounds):
    out = {}
    for name, length in (("short", 40), ("long", 512)):
        feed = inputs(length)
        s.run(None, feed)  # warm-up
        ts = []
        for _ in range(rounds):
            t = time.perf_counter()
            s.run(None, feed)
            ts.append((time.perf_counter() - t) * 1000)
        out[name + "_ms"] = round(statistics.median(ts), 1)
    return out


def op_counts(path):
    m = onnx.load(path, load_external_data=False)
    c = collections.Counter(n.op_type for n in m.graph.node)
    keep = ["MatMulNBits", "DequantizeLinear", "MatMul", "MatMulInteger", "DynamicQuantizeMatMul", "DynamicQuantizeLinear", "FusedMatMul", "Attention", "MultiHeadAttention", "SkipLayerNormalization"]
    return {k: c[k] for k in keep if c[k]}


def top_ops(profile_file, n=8):
    ev = json.load(open(profile_file))
    tot = collections.Counter()
    for e in ev:
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
            tot[e["args"].get("op_name", "?")] += e.get("dur", 0)
    all_us = sum(tot.values()) or 1
    return [(op, round(us / 1000, 1), round(100 * us / all_us)) for op, us in tot.most_common(n)]


def dequantized(src, dst):
    """The graph with every constant DequantizeLinear folded into an fp32 initializer."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out")
    ap.add_argument("--skip-variants", action="store_true", help="only the session-option runs")
    a = ap.parse_args()
    graph = os.path.join(a.model_dir, "openjevx.w8.onnx")
    work = tempfile.mkdtemp(prefix="ortprobe-")
    report = {"ort": ort.__version__, "cpu": cpu_flags(), "threads": a.threads, "runs": {}}
    print(json.dumps(report["cpu"]), "ort", ort.__version__, "threads", a.threads, flush=True)

    reference = {}

    def record(name, path, **kw):
        opt = os.path.join(work, name + ".opt.onnx")
        prof = os.path.join(work, name)
        gc.collect()
        before = rss_mb()
        try:
            s = session(path, a.threads, optimized=opt, profile=prof, **kw)
        except Exception as e:  # a variant this build cannot run is a finding, not a crash
            report["runs"][name] = {"error": str(e)[:300]}
            print(name, "ERROR", str(e)[:200], flush=True)
            return
        r = timeit(s, a.rounds)
        r["session_rss_mb"] = round(rss_mb() - before)
        # How far this variant's logits are from the server's default on the same inputs.
        outs = [s.run(None, inputs(n)) for n in (40, 512)]
        if not reference:
            reference["outs"] = outs
            report["default_max_abs_logit"] = round(float(max(np.max(np.abs(o)) for got in outs for o in got)), 2)
        r["max_logit_diff"] = round(float(max(np.max(np.abs(o - ref)) for got, want in zip(outs, reference["outs"]) for o, ref in zip(got, want))), 4)
        r["optimized_ops"] = op_counts(opt)
        r["top_ops"] = top_ops(s.end_profiling())
        report["runs"][name] = r
        print(name, json.dumps(r), flush=True)
        del s

    L = ort.GraphOptimizationLevel
    record("default", graph)
    record("extended", graph, level=L.ORT_ENABLE_EXTENDED)
    record("basic", graph, level=L.ORT_ENABLE_BASIC)
    record("no-qdq-fusion", graph, configs={"session.disable_quant_qdq": "1"})
    for lvl in ("0", "1", "4"):
        record("acc-level-" + lvl, graph, configs={"session.qdq_matmulnbits_accuracy_level": lvl})
    if not a.skip_variants:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        fp32 = dequantized(graph, os.path.join(work, "fp32.onnx"))
        dyn = os.path.join(work, "dynamic-int8.onnx")
        quantize_dynamic(fp32, dyn, weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul"],
                         use_external_data_format=True)
        record("dynamic-int8", dyn)
        record("fp32", fp32)
        report["sizes_mb"] = {n: round(sum(os.path.getsize(os.path.join(work, f)) for f in os.listdir(work) if f.startswith(n) and not f.endswith(".opt.onnx")) / 1e6)
                              for n in ("dynamic-int8", "fp32")}
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
