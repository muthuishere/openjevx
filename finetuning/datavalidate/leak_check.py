#!/usr/bin/env python3
"""Find test questions that also appear in training data (same state + same question text).

usage: leak_check.py --train A.jsonl [--train B.jsonl ...] --eval X.jsonl [--eval Y.jsonl ...] --out leaked.json
Rows are {state, questions:{id:{instructions}}}. A test file may also be .json (list with string
state/questions) or .csv (columns state, instructions). The key matches package_shards.question_key.
"""
import argparse, csv, hashlib, json


def norm_state(s):
    if isinstance(s, str):
        try: s = json.loads(s)
        except Exception: return " ".join(s.split()).lower()
    return json.dumps(s, sort_keys=True).lower()


def key(state, instr):
    return hashlib.blake2b((norm_state(state) + "||" + " ".join(str(instr).split()).lower()).encode(),
                           digest_size=12).hexdigest()


def rows(path):
    with open(path) as f:
        for l in f:
            try: yield json.loads(l)
            except Exception: pass


def eval_questions(path):
    if path.endswith(".csv"):
        for r in csv.DictReader(open(path)): yield r["state"], r["instructions"]
    elif path.endswith(".json"):
        for r in json.load(open(path)):
            for q in json.loads(r["questions"]).values(): yield json.loads(r["state"]), q["instructions"]
    else:
        for r in rows(path):
            for q in r["questions"].values(): yield r["state"], q["instructions"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="append", required=True)
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    evals = {}
    for p in a.eval:
        for state, instr in eval_questions(p): evals.setdefault(key(state, instr), set()).add(p)
    print("eval question keys:", len(evals), flush=True)
    hits, leaked, total = {}, set(), 0
    for p in a.train:
        for r in rows(p):
            for q in r.get("questions", {}).values():
                total += 1
                k = key(r["state"], q.get("instructions", ""))
                if k in evals:
                    leaked.add(k)
                    for n in evals[k]: hits[(p, n)] = hits.get((p, n), 0) + 1
    json.dump(sorted(leaked), open(a.out, "w"))
    print("train questions scanned:", total)
    for (p, n), c in sorted(hits.items()): print(f"  {p} contains {c} questions from {n}")
    print("distinct leaked eval questions:", len(leaked), flush=True)


if __name__ == "__main__":
    main()
