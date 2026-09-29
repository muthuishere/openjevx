#!/usr/bin/env python3
"""Turn the usable public sets in raw/public-2026-09-29 into noul training rows.

Writes incoming/public/<name>_train.jsonl and <name>_eval.jsonl, plus gate-only rows to
gate/public_<name>.jsonl. Every gold comes from the dataset's own label (a CVE fix, a human
relevance mark, a block anomaly label, an author's docstring), never from a model.

usage: convert_public.py [--raw DIR]
Kept:  bigvul (CVE before/after pairs), mcanoglu defect detection, HDFS_v1 (grouped per block),
       code_search_net python (docstring vs own / other code), ms_marco v1.1 (is_selected),
       danluu postmortems (category, gate only).
Dropped: CyberNative DPO (LLM-generated), E-* agent traces (no success label), loghub other than
       HDFS (no labels), G policy docs (no labels), C-aifi (no labels), Qodo PR bench (only
       positives, no clean negatives), icco postmortems (no categorical answer to ask about).
"""
import argparse, collections, glob, gzip, hashlib, json, random, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
from datavalidate.leak_check import in_keys, state_keys  # noqa: E402

SEED = 20260929
EVAL_FRAC = 0.10
CODE_MAX = 1200
TEXT_MAX = 700


def stream(raw, prefix):
    files = glob.glob(str(raw / f"{prefix}-*.jsonl.gz"))
    if not files:
        sys.exit(f"missing raw file for {prefix} in {raw}")
    with gzip.open(files[0], "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def is_eval(key):
    digest = hashlib.blake2b(str(key).encode(), digest_size=8, key=b"openjevx-public").digest()
    return int.from_bytes(digest, "big") / 2 ** 64 < EVAL_FRAC


def cut(text, limit):
    text = str(text).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "\n..."


def row(name, rng, state, phrasings, gold):
    return {"source": f"our-cases-public/{name}", "domain": f"public/{name}", "state": state,
            "questions": {"q1": {"type": "noul", "instructions": rng.choice(phrasings),
                                 "criteria": {"true": "yes", "false": "no"}}},
            "gold": {"q1": "true" if gold else "false"}}


def balance(rows, rng):
    pos = [r for r in rows if r["gold"]["q1"] == "true"]
    neg = [r for r in rows if r["gold"]["q1"] == "false"]
    n = min(len(pos), len(neg))
    out = rng.sample(pos, n) + rng.sample(neg, n)
    rng.shuffle(out)
    return out


# ---------- sources ----------

VULN_Q = ["Does this function contain a security vulnerability?",
          "Is this code vulnerable to a security flaw?",
          "Would a security reviewer flag this function as exploitable?",
          "Does this code have a security bug that needs fixing?"]


def diff_window(before, after, limit):
    """Same window over both versions, starting just before the first changed line."""
    b, a = before.splitlines(), after.splitlines()
    first = next((i for i, (x, y) in enumerate(zip(b, a)) if x != y), min(len(b), len(a)))
    start = sum(len(x) + 1 for x in b[:max(0, first - 8)])
    return cut(before[start:], limit), cut(after[start:], limit)


def bigvul(raw, rng):
    train, test = [], []
    for r in stream(raw, "A-bstee615-bigvul"):
        w = r["raw"]
        if w.get("vul") != 1 or w["func_before"] == w["func_after"]:
            continue
        before, after = diff_window(w["func_before"], w["func_after"], CODE_MAX)
        if before == after:
            continue
        lang = w.get("lang") or "C"
        pair = [row("bigvul", rng, {"language": lang, "code": before}, VULN_Q, True),
                row("bigvul", rng, {"language": lang, "code": after}, VULN_Q, False)]
        (test if is_eval(w.get("project") or w.get("CVE ID")) else train).extend(pair)
    rng.shuffle(train), rng.shuffle(test)
    return train, test


DEFECT_Q = ["Does this code contain a defect?",
            "Is there a bug in this code?",
            "Would this code be flagged as defective in review?",
            "Does this snippet have a flaw that should be fixed?"]


def defect(raw, rng):
    train, test = [], []
    for r in stream(raw, "A-mcanoglu-defect-detection"):
        state = {"language": r["raw"].get("programming_language", ""), "code": cut(r["code"], CODE_MAX)}
        item = row("defect_detection", rng, state, DEFECT_Q, r["label"] == 1)
        (test if is_eval(r["code"]) else train).append(item)
    return balance(train, rng), balance(test, rng)


LOG_Q = ["Does this log sequence show an anomaly an engineer should look at?",
         "Is something wrong in this block's log lines?",
         "Should an on-call engineer investigate these HDFS log lines?",
         "Do these logs indicate an abnormal block?"]
BLOCK = re.compile(r"blk_-?\d+")


def hdfs(raw, rng):
    lines, labels = collections.defaultdict(list), collections.defaultdict(set)
    for r in stream(raw, "D-loghub-HDFS_v1"):
        for blk in set(BLOCK.findall(r["text"])):
            labels[blk].add(r["label"])
            if len(lines[blk]) < 15:
                lines[blk].append(r["text"])
    train, test = [], []
    for blk in sorted(lines):
        if len(labels[blk]) != 1:  # conflicting labels: cannot trust the block
            continue
        text = cut("\n".join(lines[blk]), 1400)
        item = row("hdfs_logs", rng, {"system": "HDFS", "block": blk, "log": text}, LOG_Q, 1 in labels[blk])
        (test if is_eval(blk) else train).append(item)
    return balance(train, rng), balance(test, rng)


CODE_Q = ["Does this code do what the description says?",
          "Does this function implement the described behaviour?",
          "Is this the code the description is talking about?",
          "Does the code match its description?",
          "Would this function satisfy the description?"]


def first_sentence(doc):
    doc = " ".join(doc.split())
    m = re.match(r"(.+?[.!?])(\s|$)", doc)
    return (m.group(1) if m else doc)[:300]


def strip_doc(code, doc):
    if doc and doc in code:
        code = code.replace(doc, "", 1)
        code = re.sub(r'(\"\"\"|\'\'\')\s*(\"\"\"|\'\'\')\n?', "", code, count=1)
    return code


def codesearch(raw, rng, cap=40000):
    items = []
    for r in stream(raw, "F-code-search-net-code_search_net-python"):
        w = r["raw"]
        desc = first_sentence(w["func_documentation_string"])
        if len(desc) < 15:
            continue
        code = strip_doc(w["func_code_string"], w["func_documentation_string"])
        items.append({"repo": w["repository_name"], "name": w["func_name"], "desc": desc,
                      "code": cut(code, CODE_MAX)})
    by_repo = collections.defaultdict(list)
    for index, it in enumerate(items):
        by_repo[it["repo"]].append(index)
    train, test = [], []
    for index, it in enumerate(items):
        positive = rng.random() < 0.5
        if positive:
            code = it["code"]
        else:
            same = [j for j in by_repo[it["repo"]] if j != index and items[j]["desc"] != it["desc"]]
            j = rng.choice(same) if same else rng.randrange(len(items))
            if items[j]["desc"] == it["desc"]:
                continue
            code = items[j]["code"]
        item = row("code_search", rng, {"description": it["desc"], "code": code}, CODE_Q, positive)
        (test if is_eval(it["repo"]) else train).append(item)
    train, test = balance(train, rng), balance(test, rng)
    return train[:cap], test[:max(1, cap // 9)]


MARCO_Q = ["Does this passage answer the question?",
           "Does the passage contain the answer to the query?",
           "Would this passage satisfy someone asking the question?",
           "Is this passage a good answer to the search query?"]


def marco(raw, rng, cap=20000):
    train, test = [], []
    for r in stream(raw, "F-microsoft-ms_marco-v1.1"):
        rel, texts = r["relevance"], r["passage_texts"]
        if 1 not in rel or len(rel) != len(texts):
            continue
        pos = [t for t, s in zip(texts, rel) if s == 1]
        neg = [t for t, s in zip(texts, rel) if s == 0 and t not in pos]
        if not neg:
            continue
        positive = rng.random() < 0.5
        text = rng.choice(pos if positive else neg)
        item = row("ms_marco", rng, {"question": r["query"], "passage": cut(text, TEXT_MAX)}, MARCO_Q, positive)
        (test if is_eval(r["query"]) else train).append(item)
    train, test = balance(train, rng), balance(test, rng)
    return train[:cap], test[:cap // 9]


def danluu_gate(raw, rng):
    labels = ["Config Errors", "Hardware/Power Failures", "Database", "Conflicts", "Time"]
    out = []
    for r in stream(raw, "C-danluu-post-mortems"):
        if r["category_label"] not in labels:
            continue
        out.append({"source": "our-cases-public/danluu_postmortems", "domain": "public/danluu_postmortems",
                    "state": {"company": r["company"], "incident": cut(r["text"], TEXT_MAX)},
                    "questions": {"q1": {"type": "choice",
                                         "instructions": "Which kind of failure caused this incident?",
                                         "criteria": {l: l for l in labels}}},
                    "gold": {"q1": r["category_label"]}})
    return out


SOURCES = {"bigvul": bigvul, "defect_detection": defect, "hdfs_logs": hdfs,
           "code_search": codesearch, "ms_marco": marco}
DROPPED = {
    "A-CyberNative-Code_Vulnerability_Security_DPO": "synthetic, LLM-generated pairs",
    "E-* agent traces": "answers come from a model, no success label",
    "D-loghub (Apache/BGL/Hadoop/Linux/Mac/OpenStack/Zookeeper)": "no labels",
    "G-huggingface-policy-docs": "no labels",
    "C-aifi-ai-failure-intelligence": "no negatives or labels",
    "B-Qodo-PR-Review-Bench": "only injected issues (positives), no clean negatives or diff context",
    "C-icco-postmortems": "free-text markdown, no single categorical answer",
}


def drop_trained(test, train):
    """Drop eval rows whose state (code, passage, log block) is also a train state: truncation and
    duplicate functions in the source sets put the same text on both sides of the split."""
    seen = {k for r in train for q in r["questions"].values() for k in state_keys(r["state"], q["type"])}
    return [r for r in test if not in_keys(r["state"], [q["type"] for q in r["questions"].values()], seen)]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")


def summary(name, train, test):
    for split, rows in (("train", train), ("eval", test)):
        t = sum(r["gold"]["q1"] == "true" for r in rows)
        big = max((len(json.dumps(r["state"])) for r in rows), default=0)
        print(f"  {split}: {len(rows)} rows  true={t} false={len(rows) - t}  max_state_chars={big}")
    for r in train[:2]:
        print("  sample:", json.dumps(r, ensure_ascii=False)[:400])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(paths.RAW / "public-2026-09-29"))
    raw = Path(ap.parse_args().raw)
    out = paths.INCOMING / "public"
    made = {}
    for name, fn in SOURCES.items():
        made[name] = fn(raw, random.Random(f"{SEED}-{name}"))
    all_train = [r for train, _ in made.values() for r in train]  # the same function can sit in two sets
    for name, (train, test) in made.items():
        kept = drop_trained(test, all_train)
        print(f"[{name}] dropped {len(test) - len(kept)} eval rows whose state is in a public train set")
        test = kept
        write(out / f"{name}_train.jsonl", train)
        write(out / f"{name}_eval.jsonl", test)
        print(f"[{name}] -> {out}")
        summary(name, train, test)
    gate = danluu_gate(raw, random.Random(SEED))
    write(paths.GATE / "public_danluu_postmortems.jsonl", gate)
    print(f"[danluu_postmortems] gate only: {len(gate)} rows",
          collections.Counter(r["gold"]["q1"] for r in gate))
    print("dropped:")
    for k, v in DROPPED.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
