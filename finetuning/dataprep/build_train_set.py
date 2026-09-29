#!/usr/bin/env python
"""Merge all raw jev decision datasets into ONE training JSONL.

Unified row schema (openjev decision format):
    {"premise": str, "hypothesis": str, "label": 0|1|2, "source": str, "image": ""}
0=contradiction, 1=entailment, 2=neutral. English-only (metadata lang or ASCII heuristic).

Multiple-choice rows are exploded one row per option: target option -> 1 (entailment),
others -> 2 (neutral), matching the openjev scoring scheme P(entailment) per option.
"""
import gzip, glob, json, os, sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
RAW = str(paths.RAW)
OUT = str(paths.TRAIN / "jev_train_all.jsonl")
OURS = {"contradiction": 0, "entailment": 1, "neutral": 2}

def is_english(meta_lang=None, probe=""):
    if meta_lang and str(meta_lang).strip():
        return str(meta_lang).strip().lower().startswith("en")
    return all(ord(c) < 128 or c.isalpha() for c in probe[:200])

def emit(w, premise, hypothesis, label, source, image="", lang=None):
    premise, hypothesis = str(premise).strip(), str(hypothesis).strip()
    if not premise or not hypothesis:
        return 0
    if not is_english(lang, premise + hypothesis):
        return 0
    w.write(json.dumps({"premise": premise, "hypothesis": hypothesis, "label": int(label),
                        "source": source, "image": image}, ensure_ascii=False) + "\n")
    return 1

def read_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

counts = Counter()

def write_nli_rows(rows, source):
    n = 0
    for row in rows:
        lbl = row.get("label", row.get("labels"))
        if isinstance(lbl, str) and lbl.strip().lower() in OURS:
            lbl = OURS[lbl.strip().lower()]
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        lang = meta.get("language") or row.get("language")
        n += emit(w_of_counts[0], row.get("premise", ""), row.get("hypothesis", ""), lbl, source,
                  row.get("image", "") or "", lang)
    return n

def mc_rows(question, options, target, source, lang=None):
    n = 0
    if not question or not options:
        return 0
    if isinstance(options, dict):
        options = [options[k] for k in sorted(options.keys(), key=lambda k: str(k))]
    for i, opt in enumerate(options):
        lab = 1 if (isinstance(target, int) and i == target) else 2
        n += emit(w_of_counts[0], question, opt, lab, source, "", lang)
    return n

with open(OUT, "w", encoding="utf-8") as w:
    w_of_counts = [w]

    # --- openjev-data: already core NLI schema
    for src in sorted(glob.glob(f"{RAW}/openjev-data/*.jsonl.gz")):
        name = os.path.basename(src)
        counts["openjev-data:" + name] += write_nli_rows(read_jsonl(src), "openjev-" + name.split(".")[0])
        print(f"  done openjev-data/{name}", flush=True)

    # --- Open-Jev v1.1 raw: question + options + target
    for src in sorted(glob.glob(f"{RAW}/open-jev-v1.1/raw/**/*.jsonl.gz", recursive=True)):
        name = os.path.basename(src).replace(".jsonl.gz", "")
        for row in read_jsonl(src):
            if "question" in row and "options" in row:
                meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                lang = meta.get("language")
                counts["openjev-v1.1:" + name] += mc_rows(
                    row.get("question"), row.get("options"), row.get("target"),
                    "openjev-v1.1", lang)
            elif "premise" in row and "hypothesis" in row:
                meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                counts["openjev-v1.1:" + name] += emit(
                    w, row.get("premise", ""), row.get("hypothesis", ""),
                    row.get("label", 2), "openjev-v1.1", "", meta.get("language"))
        print(f"  done open-jev-v1.1/{name}", flush=True)

    # --- raw jsonl (SargeDev distill corpus)
    for src in sorted(glob.glob(f"{RAW}/jev-distill-corpus-v3/*.jsonl")) + \
               sorted(glob.glob(f"{RAW}/open-jev/**/*.jsonl", recursive=True)):
        name = "SargeDev" if "distill" in src else "Open-Jev"
        for row in read_jsonl(src):
            if "premise" in row and "hypothesis" in row:
                counts[name] += emit(w, row.get("premise", ""), row.get("hypothesis", ""),
                                     row.get("label", row.get("labels", 2)), name)
            elif "question" in row and "options" in row:
                counts[name] += mc_rows(row.get("question"), row.get("options"),
                                        row.get("target", row.get("answer", row.get("label"))), name)
        print(f"  done jsonl {src}", flush=True)

    # --- parquet datasets (tasksource, jev-decisions-v1, Open-Jev splits)
    import pyarrow.parquet as pq
    for src in sorted(glob.glob(f"{RAW}/**/*.parquet", recursive=True)):
        skip_par = []
        if "open-jev/data" in src:  # Open-Jev parquet duplicates raw jsonl; skip to avoid double-count
            continue
        try:
            t = pq.read_table(src)
        except Exception as e:
            print(f"  skip {src}: {e}"); continue
        cols = t.column_names
        srcname = ("tasksource" if "tasksource" in src else
                   "jev-decisions-v1" if "jev-decisions" in src else os.path.basename(os.path.dirname(src)))
        if "premise" in cols and "hypothesis" in cols:
            counts[srcname] += write_nli_rows(t.to_pylist(), srcname)
        elif "question" in cols and "options" in cols:
            for row in t.to_pylist():
                tgt = row.get("target", row.get("answer", row.get("label")))
                counts[srcname] += mc_rows(row.get("question"), row.get("options"), tgt, srcname)
        else:
            print(f"  unknown schema {src}: {cols}"); continue
        print(f"  done {src} -> {srcname}", flush=True)

print("\nrows per source:", flush=True)
for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {k}: {v}")
print(f"TOTAL {sum(counts.values())} -> {OUT}", flush=True)
