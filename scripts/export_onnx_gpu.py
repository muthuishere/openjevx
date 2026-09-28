#!/usr/bin/env python3
"""Export OpenJevX to ONNX on CUDA and reject ONNX Runtime CPU fallback."""

import argparse
from pathlib import Path

import torch

from laya.agent import Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU export")
    agent = Agent(args.model, compile=False, device="cuda")
    if next(agent.model.parameters()).device.type != "cuda":
        raise RuntimeError("model is not on CUDA")
    device = torch.device("cuda")
    inputs = (
        torch.randint(0, 100, (1, 16), dtype=torch.long, device=device),
        torch.ones((1, 16), dtype=torch.long, device=device),
        torch.tensor([[1, 5]], dtype=torch.long, device=device),
        torch.tensor([[True, True]], dtype=torch.bool, device=device),
        torch.tensor([0], dtype=torch.long, device=device),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        agent.model,
        inputs,
        str(output),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "marker_pos": {0: "batch_size", 1: "num_markers"},
            "marker_mask": {0: "batch_size", 1: "num_markers"},
            "qtype": {0: "batch_size"},
            "logits": {0: "batch_size", 1: "num_markers"},
            "act_logits": {0: "batch_size"},
        },
    )
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    session = ort.InferenceSession(str(output), providers=["CUDAExecutionProvider"])
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"CUDA EP not active: {session.get_providers()}")
    print(f"exported {output} ({output.stat().st_size} bytes); providers={session.get_providers()}")


if __name__ == "__main__":
    main()
