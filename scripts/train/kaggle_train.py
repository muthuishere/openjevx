#!/usr/bin/env python3
"""In-notebook driver: reuses scripts/train_openjevx.py functions on our shards.

fp16 autocast + GradScaler + fp32 master weights are already exactly what the
trainer does (T4/P100 have no real bf16). Gradient checkpointing is enabled by
the trainer via model.encoder.gradient_checkpointing_enable. Micro-batch is
sized for 16GB VRAM via run.json env (MICRO_BATCH), not hardcoded here.
"""

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

import adapter
import kaggle_eval
import train_openjevx

WORK = Path(os.environ.get("WORK_DIR", "/kaggle/working"))
INPUT = Path(os.environ.get("INPUT_DIR", "/kaggle/input"))
MODEL_DIR = WORK / "openjevx-model"


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; Kaggle notebook must enable GPU")
    print("CUDA:", torch.cuda.get_device_name(0), flush=True)


def find_input(name):
    matches = list(INPUT.rglob(name))
    if not matches:
        raise FileNotFoundError(name)
    return matches[0]


def items_from_rows(tokenizer, cfg, rows):
    items = []
    skipped = 0
    for row in rows:
        for question_id, question in row["questions"].items():
            gold = row["gold"].get(question_id)
            if gold is None:
                continue
            item = train_openjevx.build_item(tokenizer, cfg, row["state"], question, gold)
            if item:
                items.append(item)
            else:
                skipped += 1
    print(f"prepared {len(items)} decision items, build_item skipped {skipped}", flush=True)
    return items


def main():
    require_cuda()
    run_path = find_input("run.json")
    run_cfg = json.loads(Path(run_path).read_text())
    smoke = run_cfg.get("smoke", False)
    env = run_cfg.get("env", {})

    base_model = os.environ.get("BASE_MODEL", "convaiinnovations/laya")
    model_dir = snapshot_download(base_model)
    train_openjevx._fix_tokenizer_config(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as handle:
        cfg = json.load(handle)
    cfg.update({"gradient_checkpointing": True, "max_tokens_per_batch": 4096,
                "max_len": int(env.get("MAX_LEN", 1024)), "head_max_len": 256})

    train_rows = list(adapter.read_jsonl(find_input("train_smoke.jsonl.gz" if smoke else "train.jsonl.gz")))
    eval_rows = list(adapter.read_jsonl(find_input("eval_smoke.jsonl.gz" if smoke else "eval.jsonl.gz")))
    train_items = items_from_rows(tokenizer, cfg, train_rows)
    if len(train_items) < 100:
        raise RuntimeError(f"only {len(train_items)} trainable items; adapter/shard bug")

    epochs = int(env.get("EPOCHS", 1 if smoke else 4))
    micro = int(env.get("MICRO_BATCH", 2 if smoke else 8))
    accum = int(env.get("GRAD_ACCUM", 1 if smoke else 8))
    calib_max = int(env.get("CALIB_MAX", 40 if smoke else 400))
    eval_max = int(env.get("EVAL_MAX", 150 if smoke else 1500))
    acc_gate = float(env.get("ACC_GATE", 0.0 if smoke else 0.55))

    order = list(range(len(train_items)))
    random.Random(20260922).shuffle(order)
    calibration_count = min(calib_max, len(train_items) // 10)
    calibration = [train_items[index] for index in sorted(order[:calibration_count])]
    training = [train_items[index] for index in sorted(order[calibration_count:])]

    model = train_openjevx.build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    if env.get("GRAD_CKPT", "1") == "1":
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.cuda().train()

    encoder_parameters = [p for name, p in model.named_parameters() if "encoder." in name]
    head_parameters = [p for name, p in model.named_parameters() if "encoder." not in name]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": 2.5e-5},
        {"params": head_parameters, "lr": 1e-4},
    ], weight_decay=0.01)
    updates = math.ceil(len(training) / (micro * accum)) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=1e-6)
    # bf16 on Ampere/Ada/Hopper (no loss scaling needed); fp16 + GradScaler on T4/P100.
    amp_dtype = torch.bfloat16 if env.get("AMP", "fp16") == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    max_seconds = float(env.get("MAX_HOURS", 0)) * 3600
    started = time.time()

    for epoch in range(epochs):
        random.Random(42 + epoch).shuffle(training)
        optimizer.zero_grad(set_to_none=True)
        sigma = 0.4 + (0.1 - 0.4) * epoch / max(1, epochs - 1)
        epoch_loss = 0.0
        batches = 0
        for offset in range(0, len(training), micro):
            batch = train_openjevx.collate(training[offset:offset + micro], tokenizer.pad_token_id)
            with torch.autocast("cuda", dtype=amp_dtype):
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
                reward = train_openjevx.proper_reward(distribution, target.unsqueeze(0),
                                                      batch["qtype"].cuda(), mask, w_sph=0.75, w_rps=1.0)
                advantage = reward - reward.mean(0, keepdim=True)
                advantage = advantage / (advantage.std() + 1e-6)
            log_probability = -(((sampled - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            rl_loss = -(advantage * log_probability).mean()
            ce_loss = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (rl_loss + ce_loss) / accum + 0.0 * action.sum()
            scaler.scale(loss).backward()
            batches += 1
            if batches % accum == 0 or offset + micro >= len(training):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item() * accum
            if batches % 100 == 0:
                elapsed = time.time() - started
                print(f"epoch={epoch + 1}/{epochs} batch={batches} loss={epoch_loss / batches:.4f} "
                      f"items/s={batches * micro / elapsed:.1f}", flush=True)
            if max_seconds and time.time() - started > max_seconds:
                print(f"time budget reached after {batches} batches; stopping to calibrate", flush=True)
                break
        if max_seconds and time.time() - started > max_seconds:
            break
        print(f"epoch {epoch + 1} complete in {time.time() - started:.1f}s", flush=True)

    temperatures, accuracy = kaggle_eval.calibrate_and_eval(
        model, tokenizer, cfg, calibration, eval_rows, eval_max, acc_gate, MODEL_DIR)
    print("temperatures:", temperatures, "accuracy:", accuracy, flush=True)
    del model
    torch.cuda.empty_cache()
    (MODEL_DIR / "TRAINING_COMPLETE").touch()


if __name__ == "__main__":
    main()
