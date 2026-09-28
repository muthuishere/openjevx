#!/usr/bin/env python3
"""Post-training: temperature calibration on held-out items, eval-set accuracy gate."""

import json
from pathlib import Path

import numpy as np
import torch

import train_openjevx


def calibrate_and_eval(model, tokenizer, cfg, calibration, eval_rows, eval_max, acc_gate, model_dir):
    import adapter
    model.eval()
    predictions = []
    for offset in range(0, len(calibration), 16):
        chunk = calibration[offset:offset + 16]
        batch = train_openjevx.collate(chunk, tokenizer.pad_token_id)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(
                batch["input_ids"].cuda(), batch["attention_mask"].cuda(),
                batch["marker_pos"].cuda(), batch["marker_mask"].cuda(),
                batch["qtype"].cuda(),
            )
        logits = logits.float().cpu().numpy()
        for index, item in enumerate(chunk):
            options = len(item["markers"])
            predictions.append((item["qtype"], logits[index, :options], item["target"]))
    temperatures = []
    for qtype in range(3):
        selected = [(logits, target) for item_type, logits, target in predictions if item_type == qtype]
        temperatures.append(train_openjevx.fit_temperature(selected))

    eval_items = []
    for row in eval_rows:
        for question_id, question in row["questions"].items():
            gold = row["gold"].get(question_id)
            if gold is None:
                continue
            item = train_openjevx.build_item(tokenizer, cfg, row["state"], question, gold)
            if item:
                eval_items.append(item)
            if len(eval_items) >= eval_max:
                break
        if len(eval_items) >= eval_max:
            break

    correct = 0
    total = 0
    for offset in range(0, len(eval_items), 16):
        chunk = eval_items[offset:offset + 16]
        batch = train_openjevx.collate(chunk, tokenizer.pad_token_id)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(
                batch["input_ids"].cuda(), batch["attention_mask"].cuda(),
                batch["marker_pos"].cuda(), batch["marker_mask"].cuda(),
                batch["qtype"].cuda(),
            )
        logits = logits.float().cpu().numpy()
        for index, item in enumerate(chunk):
            options = len(item["markers"])
            predicted = int(np.argmax(logits[index, :options]))
            correct += int(predicted == item["label"])
            total += 1
    accuracy = correct / max(1, total)

    cfg.update({"fine_tuned": True, "model_name": "openjevx", "temperature": temperatures})
    cfg.pop("temperature_by_options", None)
    train_openjevx.save_checkpoint(model, tokenizer, cfg, Path(model_dir))
    with (Path(model_dir) / "eval_report.json").open("w") as handle:
        json.dump({"eval_items": total, "accuracy": accuracy,
                   "temperature": temperatures, "acc_gate": acc_gate}, handle, indent=2)
    if accuracy < acc_gate:
        raise RuntimeError(f"accuracy gate failed: {accuracy:.3f} < {acc_gate}")
    return temperatures, accuracy
