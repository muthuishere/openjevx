#!/usr/bin/env python3
"""Publish a validated OpenJevX checkpoint and ONNX graph to Hugging Face."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "muthuishere/openjevx"


def main():
    model_dir = Path(sys.argv[1]).resolve()
    report = json.loads((model_dir / "benchmark.json").read_text())
    if report["accuracy"] < 0.70:
        raise RuntimeError("refusing to publish a checkpoint below the 0.70 accuracy gate")
    checksums = []
    for path in sorted(model_dir.rglob("*")):
        if path.is_file() and path.name not in {"README.md", "SHA256SUMS"}:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            checksums.append(f"{digest.hexdigest()}  {path.relative_to(model_dir)}")
    (model_dir / "SHA256SUMS").write_text("\n".join(checksums) + "\n")

    card = f"""---
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
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
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="model",
        folder_path=model_dir,
        ignore_patterns=["openjevx.onnx", "checkpoint_latest/*", "checkpoint_latest/**"],
        commit_message="Publish OpenJevX RLCD checkpoint and ONNX export",
    )
    api.update_repo_settings(REPO_ID, private=False)
    release_notes = (
        f"OpenJevX v0.1.0: RLCD fine-tuned Laya decision model.\n\n"
        f"Typed-decisions accuracy: {report['accuracy']:.3f}. "
        f"CUDA p50: {report['latency_ms']['p50']:.1f} ms per five-question case.\n\n"
        "The ONNX graph requires the tokenizer and rl_agent_config.json from the Hugging Face model."
    )
    release_exists = subprocess.run(
        ["gh", "release", "view", "v0.1.0", "--repo", "muthuishere/openjevx"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    onnx_asset = model_dir / "openjevx.int8.onnx"
    if not onnx_asset.exists():
        onnx_asset = model_dir / "openjevx.onnx"
    assets = [
        str(onnx_asset),
        str(model_dir / "benchmark.json"),
        str(model_dir / "SHA256SUMS"),
        str(model_dir / "rl_agent_config.json"),
    ]
    if release_exists:
        subprocess.run(
            ["gh", "release", "upload", "v0.1.0", *assets, "--clobber", "--repo", "muthuishere/openjevx"],
            check=True,
        )
    else:
        subprocess.run(
            ["gh", "release", "create", "v0.1.0", *assets, "--repo", "muthuishere/openjevx",
             "--title", "OpenJevX v0.1.0", "--notes", release_notes],
            check=True,
        )
    print(f"https://huggingface.co/{REPO_ID}")
    print("https://github.com/muthuishere/openjevx/releases/tag/v0.1.0")


if __name__ == "__main__":
    main()
