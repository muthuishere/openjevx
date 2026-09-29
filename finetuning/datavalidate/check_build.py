#!/usr/bin/env python3
"""Run every row of a packaged shard through the trainer's own build_item on CPU, before renting a GPU.

The adapter check proves rows are shaped right for us; this proves the trainer (laya's
build_sequence/render_options) accepts them. It caught string noul criteria that crashed a GPU box.

usage: check_build.py SHARD.jsonl.gz [...] [--sample N]
Run it with the trainer's packages, e.g.:
  uv run --with torch --with transformers==4.57.6 --with datasets --with safetensors --with huggingface_hub \
    --with "laya @ git+https://github.com/NandhaKishorM/laya.git@9d955671415fc19f069b9cc998928075c1f255ec" \
    python finetuning/datavalidate/check_build.py <shard>
"""
import argparse, collections, gzip, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import train_openjevx  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("shards", nargs="+")
ap.add_argument("--sample", type=int, default=0, help="rows per source (0 = all)")
a = ap.parse_args()
tok_dir = snapshot_download(train_openjevx.BASE_MODEL, allow_patterns=["tokenizer/*"])
tok = AutoTokenizer.from_pretrained(str(Path(tok_dir) / "tokenizer"))
cfg = {"max_len": 1024, "head_max_len": 256}
seen, ok, skipped, errors = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
examples = {}
rng = random.Random(1)
failed = False
for shard in a.shards:
    opener = gzip.open if shard.endswith(".gz") else open
    with opener(shard, "rt") as fh:
        for line in fh:
            row = json.loads(line)
            src = row.get("source", "?")
            seen[src] += 1
            if a.sample and seen[src] > a.sample:
                continue
            for qid, q in row["questions"].items():
                gold = row["gold"].get(qid)
                if gold is None:
                    continue
                try:
                    item = train_openjevx.build_item(tok, cfg, row["state"], q, gold)
                    (ok if item else skipped)[src] += 1
                except Exception as e:  # noqa: BLE001
                    errors[src] += 1
                    examples.setdefault(src, f"{type(e).__name__}: {e} | question={json.dumps(q)[:200]}")
for src in sorted(seen):
    print(f"{src:55s} built {ok[src]:7d}  skipped {skipped[src]:5d}  ERRORS {errors[src]}")
for src, ex in examples.items():
    print(f"  first error in {src}: {ex}")
if errors:
    sys.exit(f"{sum(errors.values())} rows crash the trainer; fix the data before renting a GPU")
print("every checked row builds")
