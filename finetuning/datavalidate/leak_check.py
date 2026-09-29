#!/usr/bin/env python3
"""Find test questions that also appear in training data.

usage: leak_check.py --train A.jsonl [--train B.jsonl ...] --eval X.jsonl [--eval Y.jsonl ...]
                     [--gate G.jsonl ...] --out leaked.json [--fail-on-gate]

Three checks, each reported per test file:
  exact   same state + same question text. The key matches package_shards.question_key; --out gets
          this list (package_shards --exclude-keys removes those rows from training).
  state   same state + same question type, whatever the wording. A reworded question about a state
          the model trained on is still a leak. States are compared as their word/number tokens in
          order, so a dict and its plain-English rendering ("age: 17; ...") match.
  masked  the state check with every number masked: a warning only (same template, other values).
The state/masked results go to <out stem>.state.json. Training rows are never removed by state
(that would gut training); the test side must be clean instead. --gate files are release gates:
with --fail-on-gate a state-level overlap in any of them exits 1.

Rows are {state, questions:{id:{type, instructions}}}. A test file may also be .json (list with string
state/questions) or .csv (columns state, instructions; no type, so it matches any type).
"""
import argparse, csv, hashlib, json, re, sys
from collections import Counter
from pathlib import Path


def norm_state(s):
    if isinstance(s, str):
        try: s = json.loads(s)
        except Exception: return " ".join(s.split()).lower()
    return json.dumps(s, sort_keys=True).lower()


def key(state, instr):
    return hashlib.blake2b((norm_state(state) + "||" + " ".join(str(instr).split()).lower()).encode(),
                           digest_size=12).hexdigest()


TOK = re.compile(r"[a-z]+|-?\d+(?:\.\d+)?")


def _toks(text, masked):
    t = TOK.findall(text.lower().replace("_", " "))
    return [re.sub(r"\d+(?:\.\d+)?", "#", x) for x in t] if masked else t


def state_seqs(state, masked=False):
    """A state as its word/number tokens in order, with punctuation and JSON syntax dropped, so a
    dict and its plain-English rendering ("age: 17; ...") match. A dict gives two sequences (its own
    key order and sorted keys)."""
    if isinstance(state, str):
        try: state = json.loads(state)
        except Exception: pass
    texts = {state} if isinstance(state, str) else {json.dumps(state, ensure_ascii=False),
                                                     json.dumps(state, sort_keys=True, ensure_ascii=False)}
    return {" ".join(_toks(t, masked)) for t in texts}


def seq_key(seq, qtype):
    return hashlib.blake2b((seq + "||" + (qtype or "*")).encode(), digest_size=12).hexdigest()


def state_keys(state, qtype, masked=False):
    """Keys for 'same state, same question type' (None qtype: any type)."""
    return {seq_key(q, qtype) for q in state_seqs(state, masked)}


def in_keys(state, qtypes, keys):
    """True if the state asked with any of qtypes is in keys (a set built with state_keys)."""
    seqs = state_seqs(state)
    return any(seq_key(q, t) in keys for q in seqs for t in qtypes)


def rows(path):
    with open(path) as f:
        for l in f:
            try: yield json.loads(l)
            except Exception: pass


def eval_questions(path):
    """Yield (state, instructions, type) for every question in a test file."""
    path = str(path)
    if path.endswith(".csv"):
        for r in csv.DictReader(open(path)): yield r["state"], r["instructions"], None
    elif path.endswith(".json"):
        for r in json.load(open(path)):
            for q in json.loads(r["questions"]).values(): yield json.loads(r["state"]), q["instructions"], q.get("type")
    else:
        for r in rows(path):
            for q in r["questions"].values(): yield r["state"], q["instructions"], q.get("type")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="append", required=True)
    ap.add_argument("--eval", action="append", default=[])
    ap.add_argument("--gate", action="append", default=[], help="release-gate test file (state overlap is fatal with --fail-on-gate)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fail-on-gate", action="store_true")
    a = ap.parse_args()
    tests = [(p, "gate") for p in a.gate] + [(p, "eval") for p in a.eval]
    exact, st, mk = set(), set(), set()   # every test key
    total_q = Counter()
    for p, _ in tests:
        for state, instr, qtype in eval_questions(p):
            total_q[p] += 1
            exact.add(key(state, instr))
            st.update(state_keys(state, qtype))
            mk.update(state_keys(state, qtype, True))
    print("test questions:", sum(total_q.values()), "in", len(tests), "files", flush=True)
    found = {"exact": set(), "state": set(), "masked": set()}
    pair = {"exact": Counter(), "state": Counter(), "masked": Counter()}
    total = 0
    for p in a.train:
        for r in rows(p):
            qs = r.get("questions") or {}
            if not isinstance(qs, dict): continue
            seqs = None
            for q in qs.values():
                if not isinstance(q, dict): continue
                total += 1
                k = key(r["state"], q.get("instructions", ""))
                if k in exact and k not in found["exact"]:
                    found["exact"].add(k); pair["exact"][p] += 1
                if seqs is None: seqs = {"state": state_seqs(r["state"]), "masked": state_seqs(r["state"], True)}
                for name, d in (("state", st), ("masked", mk)):
                    for qt in (q.get("type"), None):   # None: csv test rows carry no type
                        for sq in seqs[name]:
                            sk = seq_key(sq, qt)
                            if sk in d and sk not in found[name]:
                                found[name].add(sk); pair[name][p] += 1
        print(f"  scanned {p}", flush=True)
    print("train questions scanned:", total)
    per_file = {n: Counter() for n in found}   # test questions hit, per file and check
    for p, _ in tests:
        for state, instr, qtype in eval_questions(p):
            per_file["exact"][p] += key(state, instr) in found["exact"]
            per_file["state"][p] += not found["state"].isdisjoint(state_keys(state, qtype))
            per_file["masked"][p] += not found["masked"].isdisjoint(state_keys(state, qtype, True))
    print(f"\n{'test file':60s} {'kind':5s} {'questions':>9s} {'exact':>7s} {'state':>7s} {'masked(warn)':>13s}")
    for p, kind in tests:
        print(f"{Path(p).name:60s} {kind:5s} {total_q[p]:9d} {per_file['exact'][p]:7d} "
              f"{per_file['state'][p]:7d} {per_file['masked'][p]:13d}")
    for name in found:
        print(f"train files contributing {name} overlaps (distinct keys):", dict(pair[name]))
    json.dump(sorted(found["exact"]), open(a.out, "w"))
    out2 = Path(a.out).with_suffix(".state.json")
    json.dump({"state": sorted(found["state"]), "masked": sorted(found["masked"]),
               "per_file": {n: dict(c) for n, c in per_file.items()}}, open(out2, "w"), indent=1)
    print("distinct leaked eval questions (exact):", len(found["exact"]), flush=True)
    bad_gates = [p for p, kind in tests if kind == "gate" and per_file["state"][p]]
    for p, kind in tests:
        if kind == "eval" and per_file["state"][p]:
            print(f"WARNING: eval file {p} shares {per_file['state'][p]} question states with training")
    if bad_gates:
        for p in bad_gates:
            print(f"ERROR: gate file {p} shares {per_file['state'][p]} question states with training")
        if a.fail_on_gate: sys.exit(1)


if __name__ == "__main__":
    main()
