#!/usr/bin/env python3
"""Publish a validated OpenJevX checkpoint and ONNX graph to Hugging Face."""

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "muthuishere/openjevx"


def main():
    model_dir = Path(sys.argv[1]).resolve()
    report = json.loads((model_dir / "benchmark.json").read_text())
    if report["accuracy"] < 0.70:
        raise RuntimeError("refusing to publish a checkpoint below the 0.70 accuracy gate")
    card = f"""---
license: apache-2.0
library_name: laya
pipeline_tag: text-classification
tags:
- system-one
- typed-decisions
- rlcd
- jev-compatible
- onnx
base_model: convaiinnovations/laya
---

# OpenJevX

OpenJevX is an open-weight, non-autoregressive System One decision model specialized from
[Laya](https://huggingface.co/convaiinnovations/laya), which uses
`answerdotai/ModernBERT-large` plus a dynamic typed-decision head.

It accepts runtime-defined `choice`, `score`, and `noul` questions and returns calibrated
probabilities in one forward pass. It is compatible with the TypeSafe Jev `/v1/systemone`
request shape through the OpenJevX server.

## Benchmark

- Typed-decisions accuracy: **{report['accuracy']:.3f}**
- Choice: **{report['per_type']['choice']['accuracy']:.3f}**
- Score: **{report['per_type']['score']['accuracy']:.3f}**
- Noul: **{report['per_type']['noul']['accuracy']:.3f}**
- GPU: `{report['device']}`
- CUDA p50 latency per five-question case: **{report['latency_ms']['p50']:.1f} ms**

The benchmark uses the untouched 400-case, 2,000-decision test split from
`LocalLLaMA/typed-decisions`. Training uses only its 1,200-case train split.

## Attribution

OpenJevX is a derivative of Laya by ConvAI Innovations and ModernBERT by Answer.AI and
LightOn. Laya and ModernBERT are Apache-2.0 licensed.
"""
    (model_dir / "README.md").write_text(card)
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(repo_id=REPO_ID, repo_type="model", folder_path=model_dir,
                      commit_message="Publish OpenJevX RLCD checkpoint and ONNX export")
    api.update_repo_settings(REPO_ID, private=False)
    print(f"https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
