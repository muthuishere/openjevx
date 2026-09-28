#!/usr/bin/env python3
"""Fine-tune Laya on typed-decisions with RLCD, calibrate, and benchmark on CUDA."""

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

import laya
from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_model, build_sequence, proper_reward, render_options


BASE_MODEL = os.environ.get("BASE_MODEL", "convaiinnovations/laya")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/root/openjevx-model"))
EPOCHS = int(os.environ.get("EPOCHS", "4"))
MICRO_BATCH = int(os.environ.get("MICRO_BATCH", "8"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
CALIB_MAX = 400


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to train or benchmark on CPU")
    print("CUDA:", torch.cuda.get_device_name(0), flush=True)


def build_item(tok, cfg, state, question, gold):
    qtype = question["type"]
    criteria = question.get("criteria", {})
    if qtype == "choice":
        target = [gold["probabilities"].get(key, 0.0) for key in criteria]
    elif qtype == "noul":
        target = [
            gold["probabilities"].get("false", 0.5),
            gold["probabilities"].get("true", 0.5),
        ]
    else:
        levels = len(criteria) if isinstance(criteria, list) else 4
        target = [gold["probabilities"].get(str(i), 0.0) for i in range(levels)]
    total = sum(target)
    target = [value / total for value in target] if total else [1 / len(target)] * len(target)
    sequence, markers = build_sequence(
        tok,
        state,
        {"t": qtype, "ins": question["instructions"], "crit": criteria},
        cfg["max_len"],
        cfg["head_max_len"],
    )
    if len(markers) != len(render_options({"t": qtype, "crit": criteria})):
        return None
    return {
        "ids": sequence,
        "markers": markers,
        "qtype": QTYPES[qtype],
        "target": target,
        "label": int(np.argmax(target)),
    }


def collate(items, pad_id):
    rows = len(items)
    length = max(len(item["ids"]) for item in items)
    options = max(len(item["markers"]) for item in items)
    ids = torch.full((rows, length), pad_id, dtype=torch.long)
    attention = torch.zeros((rows, length), dtype=torch.long)
    marker_pos = torch.zeros((rows, options), dtype=torch.long)
    marker_mask = torch.zeros((rows, options), dtype=torch.bool)
    target = torch.zeros((rows, options), dtype=torch.float32)
    for index, item in enumerate(items):
        item_length = len(item["ids"])
        item_options = len(item["markers"])
        ids[index, :item_length] = torch.tensor(item["ids"])
        attention[index, :item_length] = 1
        marker_pos[index, :item_options] = torch.tensor(item["markers"])
        marker_mask[index, :item_options] = True
        target[index, :len(item["target"])] = torch.tensor(item["target"])
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "target": target,
        "qtype": torch.tensor([item["qtype"] for item in items]),
    }


def fit_temperature(rows):
    if len(rows) < 10:
        return 1.0
    width = max(len(logits) for logits, _ in rows)
    logits = torch.full((len(rows), width), -1e4)
    targets = torch.zeros((len(rows), width))
    for index, (row_logits, row_target) in enumerate(rows):
        logits[index, :len(row_logits)] = torch.tensor(row_logits)
        targets[index, :len(row_target)] = torch.tensor(row_target)
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = -(targets * torch.log_softmax(logits / log_temperature.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(log_temperature.exp(), 0.1, 10.0).item())


def save_checkpoint(model, tokenizer, cfg, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {key: value.half().contiguous().cpu() for key, value in model.state_dict().items()}
    save_file(state, output_dir / "model.safetensors")
    model.encoder.config.save_pretrained(output_dir / "encoder")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    with (output_dir / "rl_agent_config.json").open("w") as handle:
        json.dump(cfg, handle, indent=2)


def benchmark(output_dir):
    require_cuda()
    dataset = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    agent = laya.Agent(str(output_dir), device="cuda", compile=False)
    if next(agent.model.parameters()).device.type != "cuda":
        raise RuntimeError("Laya benchmark model is not on CUDA")
    correct = {"choice": [0, 0], "score": [0, 0], "noul": [0, 0]}
    latencies = []
    for index, row in enumerate(dataset):
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = agent.predict(state, questions)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000)
        for question_id, question in questions.items():
            qtype = question["type"]
            answer = result["answers"][question_id]
            expected = str(gold[question_id]["label"]).lower()
            if qtype == "choice":
                predicted = str(answer["choice"]).lower()
            elif qtype == "noul":
                predicted = "true" if answer["noul"] >= 0.5 else "false"
            else:
                probabilities = answer["probabilities"]
                predicted = max(probabilities, key=probabilities.get)
            correct[qtype][0] += int(predicted == expected)
            correct[qtype][1] += 1
        if (index + 1) % 50 == 0:
            print(f"bench {index + 1}/{len(dataset)}", flush=True)
    total_correct = sum(value[0] for value in correct.values())
    total = sum(value[1] for value in correct.values())
    report = {
        "model": "openjevx",
        "base_model": BASE_MODEL,
        "accuracy": total_correct / total,
        "per_type": {key: {"accuracy": value[0] / value[1], "n": value[1]} for key, value in correct.items()},
        "latency_ms": {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "device": torch.cuda.get_device_name(0),
    }
    with (output_dir / "benchmark.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    if report["accuracy"] < 0.70:
        raise RuntimeError(f"accuracy gate failed: {report['accuracy']:.3f} < 0.70")


def main():
    require_cuda()
    model_dir = snapshot_download(BASE_MODEL)
    _fix_tokenizer_config(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as handle:
        cfg = json.load(handle)
    cfg.update({"gradient_checkpointing": True, "max_tokens_per_batch": 4096,
                "max_len": 1024, "head_max_len": 256})

    dataset = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
    items = []
    for row in dataset:
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        for question_id, question in questions.items():
            if question_id in gold:
                item = build_item(tokenizer, cfg, state, question, gold[question_id])
                if item:
                    items.append(item)
    print(f"prepared {len(items)} decision items", flush=True)

    order = list(range(len(items)))
    random.Random(20260922).shuffle(order)
    calibration_count = min(CALIB_MAX, len(items) // 10)
    calibration = [items[index] for index in sorted(order[:calibration_count])]
    training = [items[index] for index in sorted(order[calibration_count:])]

    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.cuda().train()

    encoder_parameters = [parameter for name, parameter in model.named_parameters() if "encoder." in name]
    head_parameters = [parameter for name, parameter in model.named_parameters() if "encoder." not in name]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": 2.5e-5},
        {"params": head_parameters, "lr": 1e-4},
    ], weight_decay=0.01)
    updates = math.ceil(len(training) / (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    started = time.time()

    for epoch in range(EPOCHS):
        random.Random(42 + epoch).shuffle(training)
        optimizer.zero_grad(set_to_none=True)
        sigma = 0.4 + (0.1 - 0.4) * epoch / max(1, EPOCHS - 1)
        epoch_loss = 0.0
        batches = 0
        for offset in range(0, len(training), MICRO_BATCH):
            batch = collate(training[offset:offset + MICRO_BATCH], tokenizer.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, action = model(
                    batch["input_ids"].cuda(), batch["attention_mask"].cuda(),
                    batch["marker_pos"].cuda(), batch["marker_mask"].cuda(),
                    batch["qtype"].cuda(),
                )
            logits = logits.float()
            mask = batch["marker_mask"].cuda()
            target = batch["target"].cuda()
            option_count = mask.sum(-1, keepdim=True).float()
            noise = torch.randn((4,) + logits.shape, device="cuda") * sigma * mask
            noise = (noise - noise.sum(-1, keepdim=True) / option_count) * mask
            sampled = logits.detach().unsqueeze(0) + noise
            distribution = torch.softmax(sampled.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                reward = proper_reward(distribution, target.unsqueeze(0), batch["qtype"].cuda(), mask,
                                       w_sph=0.75, w_rps=1.0)
                advantage = reward - reward.mean(0, keepdim=True)
                advantage = advantage / (advantage.std() + 1e-6)
            log_probability = -(((sampled - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            rl_loss = -(advantage * log_probability).mean()
            ce_loss = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (rl_loss + ce_loss) / GRAD_ACCUM + 0.0 * action.sum()
            scaler.scale(loss).backward()
            batches += 1
            if batches % GRAD_ACCUM == 0 or offset + MICRO_BATCH >= len(training):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item() * GRAD_ACCUM
            if batches % 100 == 0:
                print(f"epoch={epoch + 1}/{EPOCHS} batch={batches} loss={epoch_loss / batches:.4f}", flush=True)
        checkpoint_cfg = dict(cfg, fine_tuned=True, model_name="openjevx")
        save_checkpoint(model, tokenizer, checkpoint_cfg, OUTPUT_DIR / "checkpoint_latest")
        print(f"epoch {epoch + 1} complete in {time.time() - started:.1f}s", flush=True)

    model.eval()
    calibration_predictions = []
    with torch.no_grad():
        for offset in range(0, len(calibration), 16):
            chunk = calibration[offset:offset + 16]
            batch = collate(chunk, tokenizer.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(
                    batch["input_ids"].cuda(), batch["attention_mask"].cuda(),
                    batch["marker_pos"].cuda(), batch["marker_mask"].cuda(),
                    batch["qtype"].cuda(),
                )
            logits = logits.float().cpu().numpy()
            for index, item in enumerate(chunk):
                options = len(item["markers"])
                calibration_predictions.append((item["qtype"], logits[index, :options], item["target"]))
    temperatures = []
    for qtype in range(3):
        selected = [(logits, target) for item_type, logits, target in calibration_predictions if item_type == qtype]
        temperatures.append(fit_temperature(selected))
    cfg.update({"fine_tuned": True, "model_name": "openjevx", "temperature": temperatures})
    cfg.pop("temperature_by_options", None)
    save_checkpoint(model, tokenizer, cfg, OUTPUT_DIR)
    print("temperatures:", temperatures, flush=True)
    del model
    torch.cuda.empty_cache()
    benchmark(OUTPUT_DIR)
    (OUTPUT_DIR / "TRAINING_COMPLETE").touch()


if __name__ == "__main__":
    main()
