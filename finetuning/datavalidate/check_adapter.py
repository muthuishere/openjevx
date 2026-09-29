#!/usr/bin/env python3
"""Local CPU gate for the training pipeline: dry-parse N real rows through the adapter."""

import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import adapter

ROOT = Path(__file__).resolve().parents[2]


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seen = 0
    qtypes = Counter()
    examples = []
    rows = adapter.read_jsonl(paths.TRAIN / "train_openjevx.jsonl")
    for row in rows:
        shaped = adapter.adapt_row(row)
        if shaped is None:
            continue
        for qid, gold in shaped["gold"].items():
            q = shaped["questions"][qid]
            qtypes[q["type"]] += 1
            if len(examples) < 3:
                examples.append({"qid": qid, "qtype": q["type"], "gold": gold})
        seen += 1
        if seen >= limit:
            break
    if seen == 0:
        print(json.dumps({"error": "no rows adapted"}))
        return 1
    report = {"rows": seen, "qtypes": dict(qtypes), "examples": examples}
    print(json.dumps(report, indent=2)[:2000])
    assert sum(qtypes.values()) >= seen, "no items adapted at all"
    return 0


if __name__ == "__main__":
    sys.exit(main())
