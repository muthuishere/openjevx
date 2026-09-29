#!/usr/bin/env python3
"""Build the staging directory for an openjevx training run.

Uses the gold adapter (finetuning/train/adapter.py) on data/train_openjevx.jsonl
and data/eval_openjevx.jsonl, writes gzipped shards (priority sources our-cases-*
and typed-decisions/* kept in full, tasksource-train hash-stratified to fit the
upload budget), plus the trainer, driver and runner code and run.json.
"""

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TRAIN_DIR = ROOT / "finetuning" / "train"
EXPORT_DIR = ROOT / "finetuning" / "export"

TRAIN_SMOKE_ROWS = 2000
EVAL_SMOKE_ROWS = 300
DEFAULT_BUDGET_GB = 1.8

COPY_FILES = ("train_openjevx.py", "export_onnx_gpu.py",
              "train_job.py", "eval_job.py", "adapter.py", "quantize_w8.py")


def question_key(state, instructions):
    """Same key as the leakage check: normalised state + normalised question text."""
    import hashlib
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except Exception:
            state = " ".join(state.split()).lower()
    text = state if isinstance(state, str) else json.dumps(state, sort_keys=True).lower()
    text += "||" + " ".join(str(instructions).split()).lower()
    return hashlib.blake2b(text.encode(), digest_size=12).hexdigest()


def filter_rows(rows, exclude_keys, drop_sources, counts):
    """Remove leaked test questions and reviewed-bad sources; drop rows left with no questions."""
    kept = []
    for row in rows:
        if row.get("source") in drop_sources:
            counts["dropped_source_rows"] += 1
            continue
        if exclude_keys:
            for qid in list(row["questions"]):
                if question_key(row["state"], row["questions"][qid].get("instructions", "")) in exclude_keys:
                    del row["questions"][qid]
                    row["gold"].pop(qid, None)
                    counts["leaked_questions_removed"] += 1
            if not row["questions"]:
                counts["rows_emptied"] += 1
                continue
        kept.append(row)
    return kept


def write_gzip_jsonl(path, rows):
    import gzip
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return Path(path).stat().st_size


def build(out_dir, smoke=False, budget_gb=DEFAULT_BUDGET_GB, extra_train=(), extra_eval=(),
          exclude_keys=frozenset(), drop_sources=frozenset()):
    import sys
    sys.path.insert(0, str(TRAIN_DIR))
    import adapter
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = DATA / "train_openjevx.jsonl"
    budget_bytes = int(budget_gb * 1e9)

    skips = Counter()
    if smoke:
        train_rows = []
        for row in adapter.read_jsonl(train_path):
            if len(train_rows) >= TRAIN_SMOKE_ROWS:
                break
            if not adapter.is_priority(row.get("source", "")):
                continue
            shaped = adapter.adapt_row(row)
            if shaped is None:
                skips[row["source"]] += 1
                continue
            train_rows.append(shaped)
        plan = {"mode": "smoke"}
    else:
        train_rows, plan = adapter.stratified_sample_rows(train_path, budget_bytes)
        skips["stratify_or_adapter_skips"] = plan.pop("skipped", 0)
    # Extra priority sets (e.g. data/it_worker_train.jsonl) are kept in full, never sampled.
    for extra in extra_train:
        for row in adapter.read_jsonl(extra):
            shaped = adapter.adapt_row(row)
            if shaped is None:
                skips[row.get("source", extra)] += 1
                continue
            train_rows.append(shaped)

    eval_rows = []
    for row in adapter.read_jsonl(DATA / "eval_openjevx.jsonl"):
        shaped = adapter.adapt_row(row)
        if shaped is None:
            skips[row["source"]] += 1
            continue
        eval_rows.append(shaped)
        if smoke and len(eval_rows) >= EVAL_SMOKE_ROWS:
            break

    removed = Counter()
    train_rows = filter_rows(train_rows, exclude_keys, drop_sources, removed)
    skips.update({"filter_" + key: value for key, value in removed.items()})

    for extra in extra_eval:
        for row in adapter.read_jsonl(extra):
            shaped = adapter.adapt_row(row)
            if shaped is not None:
                eval_rows.append(shaped)

    train_out = out_dir / ("train_smoke.jsonl.gz" if smoke else "train.jsonl.gz")
    eval_out = out_dir / ("eval_smoke.jsonl.gz" if smoke else "eval.jsonl.gz")
    train_bytes = write_gzip_jsonl(train_out, train_rows)
    eval_bytes = write_gzip_jsonl(eval_out, eval_rows)

    for name in COPY_FILES:
        source = TRAIN_DIR / name if (TRAIN_DIR / name).exists() else EXPORT_DIR / name
        shutil.copy2(source, out_dir / name)

    env = {
        "EPOCHS": "1" if smoke else os.environ.get("EPOCHS", "4"),
        "MICRO_BATCH": "2" if smoke else os.environ.get("MICRO_BATCH", "8"),
        "GRAD_ACCUM": "1" if smoke else os.environ.get("GRAD_ACCUM", "8"),
        "CALIB_MAX": "40" if smoke else os.environ.get("CALIB_MAX", "400"),
        "EVAL_MAX": "150" if smoke else os.environ.get("EVAL_MAX", "1500"),
        "ACC_GATE": os.environ.get("ACC_GATE", "0.0" if smoke else "0.55"),
    }
    env.update({k: os.environ[k] for k in ("MAX_LEN", "AMP", "GRAD_CKPT", "MAX_HOURS") if os.environ.get(k)})
    (out_dir / "run.json").write_text(json.dumps({"smoke": smoke, "env": env}))
    # The names push_data.py / run_job.sh expect.
    shard = "smoke" if smoke else "full"
    for src, suffix in ((train_out, "train.jsonl.gz"), (eval_out, "eval.jsonl.gz"), (out_dir / "run.json", "run.json")):
        shutil.copy2(src, out_dir / f"openjevx-{shard}-{suffix}")

    manifest = {
        "mode": plan.get("mode", "full"),
        "train_rows": len(train_rows), "train_gz_bytes": train_bytes,
        "eval_rows": len(eval_rows), "eval_gz_bytes": eval_bytes,
        "skips": dict(skips),
        "budget_gb": budget_gb,
        "payload_gz_bytes": train_bytes + eval_bytes,
        "plan": plan,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--budget-gb", type=float, default=DEFAULT_BUDGET_GB)
    parser.add_argument("--extra-train", action="append", default=[], help="JSONL kept in full")
    parser.add_argument("--extra-eval", action="append", default=[], help="JSONL added to eval")
    parser.add_argument("--exclude-keys", help="JSON list of question keys to remove (leaked test questions)")
    parser.add_argument("--drop-sources", help="text file, one source per line, removed from training")
    args = parser.parse_args()
    build(args.out, smoke=args.smoke, budget_gb=args.budget_gb, extra_train=args.extra_train, extra_eval=args.extra_eval,
          exclude_keys=frozenset(json.load(open(args.exclude_keys))) if args.exclude_keys else frozenset(),
          drop_sources=frozenset(l.strip() for l in open(args.drop_sources) if l.strip()) if args.drop_sources else frozenset())


if __name__ == "__main__":
    main()
