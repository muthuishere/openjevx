"""Build an OpenJevX model folder: the graph, its config (calibration temperatures) and its tokenizer.

usage: make_model_folder.py OUT_DIR ONNX EVAL_REPORT_JSON [TOKENIZER_JSON] [--version X]
                            [--max-len 1024] [--head-max 256] [--base-model convaiinnovations/laya]

EVAL_REPORT_JSON is the eval_report.json the training job writes (or any JSON with "temperature":
[choice, score, noul], e.g. a laya rl_agent_config.json). TOKENIZER_JSON defaults to the tokenizer
embedded in the server (internal/bpe/tokenizer.json). Writes:
  OUT_DIR/openjevx.w8.onnx   OUT_DIR/config.json   OUT_DIR/tokenizer.json
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRAPH = "openjevx.w8.onnx"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def place(src, dst):
    """Hard-link when possible (same disk, no copy of a 600 MB graph), else copy."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("onnx")
    ap.add_argument("eval_report")
    ap.add_argument("tokenizer", nargs="?", default=str(ROOT / "internal/bpe/tokenizer.json"))
    ap.add_argument("--version", default="0.0.0")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max", type=int, default=256)
    ap.add_argument("--base-model", default="convaiinnovations/laya")
    a = ap.parse_args()

    temps = json.loads(Path(a.eval_report).read_text()).get("temperature")
    if not isinstance(temps, list) or len(temps) != 3 or not all(float(t) > 0 for t in temps):
        sys.exit(f"{a.eval_report}: need \"temperature\": [choice, score, noul], got {temps!r}")
    for p in (a.onnx, a.tokenizer):
        if not Path(p).is_file():
            sys.exit(f"missing {p}")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    place(Path(a.onnx), out / GRAPH)
    shutil.copyfile(a.tokenizer, out / "tokenizer.json")
    config = {
        "name": "openjevx",
        "version": a.version,
        # Order in eval_report.json is the qtype order: 0 choice, 1 score, 2 noul.
        "temperature": {"choice": float(temps[0]), "score": float(temps[1]), "noul": float(temps[2])},
        "max_len": a.max_len,
        "head_max": a.head_max,
        "special_ids": {"cls": 50281, "sep": 50282, "pad": 50283, "mask": 50284},
        "quantization": "8-bit weight-only",
        "base_model": a.base_model,
        "sha256": sha256(out / GRAPH),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"model folder {out}: version {a.version} temperatures {config['temperature']} sha256 {config['sha256'][:12]}")


if __name__ == "__main__":
    main()
