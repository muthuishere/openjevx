#!/usr/bin/env python
"""Consolidate ALL gathered sources into ONE final training set for openjevx.

Outputs under <data>/train and <data>/eval (finetuning/paths.py):
  train_openjevx.jsonl - typed-decision records, exact OpenJevX format:
      {"source","domain","state","questions":{qid:{type,instructions,criteria}},"gold"}
  eval_openjevx.jsonl  - same shape, held-out rows (typed-decisions test split, tasksource holdout)
  jev_train_all.jsonl  - flat auxiliary merge of NLI/distill sources (premise/hypothesis/label)

English-only via language field or ASCII heuristic. No license text is emitted into the data.
"""
import glob, gzip, json, os, sys, time
from collections import Counter

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402
RAW = str(paths.RAW)
CLAUDE = os.path.join(os.path.dirname(os.path.dirname(HERE)), "llmresults")
OURS = {"contradiction": 0, "entailment": 1, "neutral": 2}

import pyarrow.parquet as pq


def read_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


ntr = nev = 0
typed_counts = Counter()   # source -> record count
flat_counts = Counter()    # source -> row count
flat_skips = Counter()     # source -> skipped row count


def ascii_ok(txt):
    return all(ord(c) < 128 for c in txt[:500])


def is_en(lang, probe):
    probe = str(probe or "")
    if lang is not None and str(lang).strip():
        if not str(lang).strip().lower().startswith("en"):
            return False
    return ascii_ok(probe)


def maybe_parse(v):
    if isinstance(v, str) and v[:1] in ("{", "["):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def norm_criteria(c):
    if isinstance(c, list):
        return {o: o for o in c}   # list -> {opt: opt}
    if isinstance(c, dict):
        return {str(k): str(v) for k, v in c.items()}
    return None


def write_typed(out, rec, source):
    """Write one typed-decision record; returns (records_written, questions_written)."""
    qs_in = maybe_parse(rec.get("questions") or {})
    gd_in = maybe_parse(rec.get("gold") or {})
    if not isinstance(qs_in, dict) or not isinstance(gd_in, dict):
        return 0, 0
    state = rec.get("state")
    if isinstance(state, (dict, list)):
        state = json.dumps(state, ensure_ascii=False)
    elif isinstance(state, str):
        state = state.strip()
    qs, gold = {}, {}
    for qid, q in qs_in.items():
        g = gd_in.get(qid)
        if isinstance(g, dict) and "label" in g:      # typed-decisions aggregations
            g = g.get("label")
        if not isinstance(q, dict) or g is None:
            continue
        k = str(q.get("type", "noul")).lower()
        qs[qid] = {"type": k if k in ("noul", "choice", "score") else "noul",
                   "instructions": str(q.get("instructions", "") or ""),
                   "criteria": norm_criteria(q.get("criteria"))}
        gold[qid] = g
    if not gold:
        return 0, 0
    probe = str(state or "") + " " + " ".join(q["instructions"] for q in qs.values()) \
        + " " + json.dumps((list(qs.values())[0].get("criteria") or {}), ensure_ascii=False)
    if not is_en(None, probe):
        return 0, 0
    out.write(json.dumps({"source": source,
                          "domain": rec.get("domain") or rec.get("workflow") or "general",
                          "state": state, "questions": qs, "gold": gold},
                         ensure_ascii=False) + "\n")
    return 1, len(gold)


def hard_target(kind, opts, tgt):
    """Map a tasksource/open-jev target (one-hot or soft list, or scalar) to a single gold label."""
    if isinstance(tgt, str):
        return tgt if tgt in opts else None
    if isinstance(tgt, (int, float)) and not isinstance(tgt, bool):
        if opts and 0 <= int(tgt) < len(opts):
            return opts[int(tgt)]
        if not opts and kind == "noul":
            return "true" if tgt >= 0.5 else "false"
        return None
    if not isinstance(tgt, list) or not tgt:
        return None
    best = max(range(len(tgt)), key=lambda i: float(tgt[i]))
    top = float(tgt[best])
    sorted_t = sorted((float(v) for v in tgt), reverse=True)
    if len(sorted_t) > 1 and sorted_t[1] >= top:
        return None            # ambiguous argmax tie -> skip
    if opts and len(tgt) == len(opts):
        return opts[best]
    if not opts and kind == "noul" and len(tgt) == 1:
        return "true" if top >= 0.5 else "false"
    return None


ftr = open(os.path.join(paths.TRAIN, "train_openjevx.jsonl"), "w", encoding="utf-8")
fev = open(os.path.join(paths.EVAL, "eval_openjevx.jsonl"), "w", encoding="utf-8")
fflat = open(os.path.join(paths.TRAIN, "jev_train_all.jsonl"), "w", encoding="utf-8")

try:
    # 1) LocalLLaMA typed-decisions (train split -> train, test split -> eval)
    for src in sorted(glob.glob(f"{RAW}/typed-decisions/*/train-*.parquet")):
        w = os.path.basename(os.path.dirname(src))
        sname = "typed-decisions/" + w
        for row in pq.read_table(src).to_pylist():
            r_, q_ = write_typed(ftr, {"domain": w, "state": row.get("state"),
                                       "questions": row.get("questions"), "gold": row.get("gold")},
                                 sname)
            typed_counts[sname] += r_
            ntr += r_

    for src in sorted(glob.glob(f"{RAW}/typed-decisions/*/test-*.parquet")):
        w = os.path.basename(os.path.dirname(src))
        sname = "typed-decisions/" + w
        for row in pq.read_table(src).to_pylist():
            r_, q_ = write_typed(fev, {"domain": w, "state": row.get("state"),
                                       "questions": row.get("questions"), "gold": row.get("gold")},
                                 sname)
            typed_counts[sname] += r_
            nev += r_
    print("after typed-decisions: train=%d eval=%d" % (ntr, nev), flush=True)

    # 2) our clauderesults gold cases (high quality -> train)
    tr_before = ntr
    for name, sname in (("cases_rules.jsonl", "our-cases-rules"), ("cases_llm.jsonl", "our-cases-llm")):
        p = os.path.join(CLAUDE, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as g:
                for line in g:
                    if not line.strip():
                        continue
                    r_, _ = write_typed(ftr, json.loads(line), sname)
                    typed_counts[sname] += r_
                    ntr += r_
    print(f"after our-cases: train={ntr} (+{ntr - tr_before}) eval={nev}", flush=True)

    # 3) tasksource-jev-typed-decisions (train -> train, validation/test -> eval)
    tr_before = ntr
    for src in sorted(glob.glob(f"{RAW}/tasksource-jev-typed-decisions/data/*.parquet")):
        in_train_default = os.path.basename(src).startswith("train")
        for row in pq.read_table(src).to_pylist():
            split = str(row.get("split") or "").lower()
            in_train = (split == "train") or (in_train_default and not split)
            out = ftr if in_train else fev
            sname = "tasksource-train" if in_train else "tasksource-holdout"
            kind = row.get("kind") or "choice"
            qid = row.get("question_id") or "decision"
            opts = row.get("options") or []
            gold_val = hard_target(kind, opts, row.get("target"))
            if gold_val is None:
                continue
            rec = {"domain": row.get("source") or "general", "state": row.get("state"),
                   "questions": {qid: {"type": kind, "instructions": row.get("question") or "",
                                       "criteria": {o: o for o in opts}}},
                   "gold": {qid: gold_val}}
            r_, _ = write_typed(out, rec, sname)
            typed_counts[sname] += r_
            if in_train:
                ntr += r_
            else:
                nev += r_
    print("after tasksource: train=%d (+%d) eval=%d" % (ntr, ntr - tr_before, nev), flush=True)

    # 4) flat auxiliary merge
    def flat(premise, hypothesis, label, source):
        premise, hypothesis = str(premise).strip(), str(hypothesis).strip()
        if not premise or not hypothesis:
            flat_skips[source] += 1
            return
        if not is_en(None, premise + " " + hypothesis):
            flat_skips[source] += 1
            return
        if label is None or not isinstance(label, (int, float)):
            flat_skips[source] += 1
            return
        lbl = int(label)
        if lbl not in (0, 1, 2):
            lbl = 0 if lbl < 0 else (1 if lbl == 1 else 2)
        flat_counts[source] += 1
        fflat.write(json.dumps({"premise": premise, "hypothesis": hypothesis, "label": lbl,
                                "source": source, "image": ""}, ensure_ascii=False) + "\n")

    # 4a) openjev-data (already NLI premise/hypothesis/label)
    for src in sorted(glob.glob(f"{RAW}/openjev-data/*.jsonl.gz")):
        S = "openjev-" + os.path.basename(src).split(".")[0]
        with gzip.open(src, "rt", encoding="utf-8", errors="replace") as g:
            for line in g:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    flat_skips[S] += 1
                    continue
                if isinstance(r.get("label"), int) and r["label"] in (0, 1, 2):
                    flat(r.get("premise", ""), r.get("hypothesis", ""), r["label"], S)
                else:
                    flat_skips[S] += 1
        print("flat done", os.path.basename(src), flush=True)

    # 4b) open-jev-v1.1 (question/options/target) + jev-distill-corpus-v3 (soft targets)
    for src in sorted(glob.glob(f"{RAW}/open-jev-v1.1/raw/**/*.jsonl.gz", recursive=True)) + \
               sorted(glob.glob(f"{RAW}/jev-distill-corpus-v3/*.jsonl")):
        S = "SargeDev" if "distill" in src else "OpenJev-v1.1"
        for r in read_jsonl(src):
            if "premise" in r and "hypothesis" in r:
                lbl = r.get("label", r.get("labels", 2))
                if isinstance(lbl, str) and lbl.strip().lower() in OURS:
                    lbl = OURS[lbl.strip().lower()]
                lang = (r.get("metadata") or {}).get("language")
                if lang is not None and not str(lang).strip().lower().startswith("en"):
                    flat_skips[S] += 1
                    continue
                flat(r.get("premise", ""), r.get("hypothesis", ""), lbl, S)
            elif "question" in r and "options" in r:
                lang = (r.get("metadata") or {}).get("language")
                if lang is not None and not str(lang).strip().lower().startswith("en"):
                    flat_skips[S] += 1
                    continue
                q = str(r.get("question") or "").strip()
                opts = [str(o) if not isinstance(o, dict) else (o.get("text") or str(o))
                        for o in (r.get("options") or [])]
                kind = r.get("kind") or "choice"
                gold_val = hard_target(kind, opts, r.get("target", r.get("answer", r.get("label"))))
                if not q or gold_val is None:
                    flat_skips[S] += 1
                    continue
                flat(q, str(gold_val), 1, S)          # target option -> label 1
                for o in opts:
                    if o != str(gold_val):
                        flat(q, o, 2, S)              # distractors -> label 2
        print("flat done", os.path.basename(src), flush=True)

finally:
    ftr.close()
    fev.close()
    fflat.close()

print("")
print("=== typed records: train=", ntr, "eval=", nev, flush=True)
print("=== typed records per source:", json.dumps(typed_counts.most_common()), flush=True)
print("=== flat rows per source:", json.dumps(flat_counts.most_common()), flush=True)
print("=== flat skips per source:", json.dumps(flat_skips.most_common()), flush=True)
print("=== runtime %.1fs" % (time.time() - T0), flush=True)
