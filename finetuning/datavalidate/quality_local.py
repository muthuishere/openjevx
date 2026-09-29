#!/usr/bin/env python3
"""
Local data-quality pipeline for OpenJevX training data (no MLX, no cloud calls).

Two specialist judges, both fully local:
  - NLI verifier: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli via
    transformers + PyTorch on MPS. Used for "noul" (true/false) questions:
    premise = state rendered as text, hypothesis = the instruction statement.
  - Evaluator LLM: Atla Selene-1-Mini via local Ollama (temperature 0, JSON
    format). Used for "choice"/"score" questions, plus every Stage-1
    nli_conflict (so a disputed noul question also gets a second local
    opinion).

Stage 0 (calibration, REQUIRED before anything can be flagged): each judge's
accuracy is measured per question TYPE against a trusted key built from
- .local/eval/test.json questions where label_agreement[qid].argmax_agree
  is true (human labellers agreed), and
- .local/eval/questions_1000.csv rows with domain == "basic" (hand-verified).
A judge is TRUSTED for a type only if accuracy >= 0.90 there and n >= 50.
Untrusted judge+type combos are recorded in the report but never drive a
verdict.

Stage 1 (NLI): every noul question in data/it_worker_train.jsonl (source A)
plus every sampled noul question from the reservoir sample of
data/train_openjevx.jsonl (source B, 300 questions/source). Flags
"nli_conflict" when NLI is confident (>=0.9 softmax prob) in the label
opposite of gold.

Stage 2 (Selene): choice/score questions -- a stratified sample of up to 150
per source in A and all sampled ones in B -- plus every Stage-1
nli_conflict question. Flags "judge_conflict" (answer != gold) and
"ambiguous" (self-reported clarity <= 2). Retries once on invalid/unparsable
JSON.

Verdict per question (owner override, 2026-09-29): a question can only be
BAD if TWO TRUSTED judges both ran on it and both disagreed with gold. If a
type has only one trusted judge (choice/score can only ever have Selene,
since NLI does not run on them), that type maxes out at SUSPECT. Nothing is
ever deleted -- bad_rows.jsonl is a REVIEW list, not a removal list.

Outputs (all under .local/quality/, this script's own files -- never touches
.local/quality/judged.jsonl / report.json / drop_sources.txt, which belong
to a separate cloud-judge run):
  - calibration_local.json       Stage-0 judge x type accuracy table
  - local_judged.jsonl           every judged question (checkpointed, resumable)
  - review_sources_local.txt     sources/families with bad_rate >= 0.15
  - bad_rows.jsonl               {source, state_hash, qid} for BAD questions (review list)
  - quality_local_state.json     run bookkeeping (elapsed time, stage progress)
clauderesults/07-local-quality-report.md is written by --report.

Usage (run each stage detached, e.g. via nohup/setsid, with a 3h wall-clock cap
shared across the whole run -- pass --deadline-ts <unix ts> to every stage so
they all respect the same budget):
  python quality_local.py calibrate
  python quality_local.py sample-b          # build the reservoir sample of B once
  python quality_local.py stage1
  python quality_local.py stage2
  python quality_local.py aggregate
  python quality_local.py report
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# NOTE: a separate agent shares .local/quality/ and periodically wipes its
# contents (observed every ~5-6 min: run.log / s3_delete_one.py reappear and
# everything else vanishes, mid-run). To avoid losing checkpointed work, all
# real working state lives in .local/quality_local_work/ instead, and only
# the mandated final deliverable filenames are copied into .local/quality/
# as the very last step (see cmd_publish / --publish).
QDIR = REPO / ".local" / "quality_local_work"
QDIR.mkdir(parents=True, exist_ok=True)
PUBLISH_DIR = REPO / ".local" / "quality"
EVAL_DIR = REPO / ".local" / "eval"
DATA_A = REPO / "data" / "it_worker_train.jsonl"
DATA_B = REPO / "data" / "train_openjevx.jsonl"

SAMPLE_B_PATH = QDIR / "sample_b.jsonl"
CALIB_PATH = QDIR / "calibration_local.json"
JUDGED_PATH = QDIR / "local_judged.jsonl"
REVIEW_SOURCES_PATH = QDIR / "review_sources_local.txt"
BAD_ROWS_PATH = QDIR / "bad_rows.jsonl"
STATE_PATH = QDIR / "quality_local_state.json"
AGG_PATH = QDIR / "aggregate_local.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
SELENE_MODEL = "hf.co/AtlaAI/Selene-1-Mini-Llama-3.1-8B-Q4_K_M-GGUF"

B_PER_SOURCE = 300
A_STAGE2_PER_FAMILY = 150
NLI_BATCH = 16
NLI_CONF = 0.9


def now():
    return time.time()


def deadline_hit(deadline_ts):
    return deadline_ts is not None and now() >= deadline_ts


def state_hash(state):
    if not isinstance(state, str):
        state = json.dumps(state, sort_keys=True, ensure_ascii=False)
    return hashlib.blake2b(state.encode("utf-8"), digest_size=10).hexdigest()


def load_json_maybe_str(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def flatten_state(state, prefix="", depth=0, out=None):
    if out is None:
        out = []
    if depth > 4 or len(out) > 40:
        return out
    if isinstance(state, dict):
        for k, v in state.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            flatten_state(v, key, depth + 1, out)
    elif isinstance(state, list):
        for i, v in enumerate(state[:10]):
            flatten_state(v, f"{prefix}[{i}]", depth + 1, out)
    else:
        out.append(f"{prefix}: {state}")
    return out


def state_text(state):
    parts = flatten_state(state)
    return "; ".join(parts)[:2000] or "(empty state)"


def criteria_options(criteria):
    """Normalise criteria (dict or list) into an ordered list of (key, desc)."""
    if isinstance(criteria, dict):
        return [(k, v) for k, v in criteria.items()]
    if isinstance(criteria, list):
        return [(str(i), v) for i, v in enumerate(criteria)]
    return []


def gold_label(gold, qtype, criteria):
    """Extract a plain gold label string from a gold field (dict-with-label,
    or a bare scalar as in data/*.jsonl)."""
    if isinstance(gold, dict):
        if "label" in gold:
            return str(gold["label"])
        return None
    text = str(gold).strip()
    if qtype == "score":
        opts = criteria_options(criteria)
        keys = [k for k, _ in opts]
        if text in keys:
            return text
        if text.lstrip("-").isdigit() and 0 <= int(text) < len(keys):
            return keys[int(text)]
        return text
    return text


# --------------------------------------------------------------------------
# Stage 0: calibration key
# --------------------------------------------------------------------------

def load_trusted_key():
    """Returns list of dicts: {source, state, qid, type, instructions, criteria, gold}"""
    items = []
    test_path = EVAL_DIR / "test.json"
    data = json.load(open(test_path))
    for row in data:
        la = load_json_maybe_str(row.get("label_agreement", "{}"))
        qs = load_json_maybe_str(row.get("questions", "{}"))
        gold = load_json_maybe_str(row.get("gold", "{}"))
        state = load_json_maybe_str(row.get("state", "{}"))
        for qid, agree in la.items():
            if not agree.get("argmax_agree"):
                continue
            q = qs.get(qid)
            g = gold.get(qid)
            if q is None or g is None:
                continue
            label = g.get("label") if isinstance(g, dict) else g
            if label is None:
                continue
            items.append({
                "source": "eval/test.json", "state": state, "qid": qid,
                "type": q.get("type"), "instructions": q.get("instructions"),
                "criteria": q.get("criteria"), "gold": str(label),
            })

    import csv
    csv_path = EVAL_DIR / "questions_1000.csv"
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("domain") != "basic":
                continue
            items.append({
                "source": "eval/questions_1000.csv", "state": json.loads(row["state"]),
                "qid": row["qid"], "type": row["type"],
                "instructions": row["instructions"], "criteria": json.loads(row["criteria"]),
                "gold": row["expected"],
            })
    return items


def nli_predict_one(nli, state, instructions):
    return nli.predict_batch([(state, instructions)])[0]


def selene_answer(qtype, state, instructions, criteria, retry=True):
    opts = criteria_options(criteria)
    opt_lines = "\n".join(f'- "{k}": {v}' for k, v in opts)
    prompt = (
        "You are a strict evaluator. Given the CONTEXT and QUESTION, choose the single "
        "best OPTION key and rate how unambiguous the question is (clarity: 5 = totally "
        "unambiguous given the context, 1 = very ambiguous/underspecified).\n\n"
        f"CONTEXT:\n{state_text(state)}\n\n"
        f"QUESTION ({qtype}): {instructions}\n\n"
        f"OPTIONS:\n{opt_lines}\n\n"
        'Respond with ONLY a JSON object: {"answer": "<one option key exactly as given>", '
        '"clarity": <integer 1-5>}'
    )
    body = {
        "model": SELENE_MODEL, "prompt": prompt, "stream": False,
        "format": "json", "options": {"temperature": 0},
    }
    for attempt in range(2 if retry else 1):
        try:
            req = urllib.request.Request(
                OLLAMA_URL, data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            raw = out.get("response", "")
            parsed = json.loads(raw)
            answer = str(parsed.get("answer", "")).strip()
            clarity = int(parsed.get("clarity", 3))
            valid_keys = {k for k, _ in opts}
            if answer in valid_keys:
                return {"answer": answer, "clarity": clarity, "valid": True, "raw_ok": True}
            # try matching by description text as a fallback
            for k, v in opts:
                if answer.lower() == str(v).lower():
                    return {"answer": k, "clarity": clarity, "valid": True, "raw_ok": True}
            if attempt == 0:
                continue
            return {"answer": answer, "clarity": clarity, "valid": False, "raw_ok": True}
        except Exception as e:
            if attempt == 0:
                continue
            return {"answer": None, "clarity": None, "valid": False, "raw_ok": False, "error": str(e)}
    return {"answer": None, "clarity": None, "valid": False, "raw_ok": False}


def cmd_calibrate(args):
    items = load_trusted_key()
    print(f"trusted key: {len(items)} questions", flush=True)
    from nli_local import NLIJudge
    nli = NLIJudge()

    calib_rows = []
    deadline_ts = args.deadline_ts
    for i, it in enumerate(items):
        if deadline_hit(deadline_ts):
            print("deadline hit during calibration, stopping early", flush=True)
            break
        row = {"source": it["source"], "qid": it["qid"], "type": it["type"], "gold": it["gold"]}
        if it["type"] == "noul":
            pred = nli_predict_one(nli, it["state"], it["instructions"])
            row["nli_pred"] = pred["label"]
            row["nli_conf"] = pred["conf"]
        if it["type"] in ("noul", "choice", "score"):
            sel = selene_answer(it["type"], it["state"], it["instructions"], it["criteria"])
            gold_norm = gold_label(it["gold"], it["type"], it["criteria"])
            row["selene_pred"] = sel["answer"]
            row["selene_valid"] = sel["valid"]
            row["gold_norm"] = gold_norm
        calib_rows.append(row)
        if i % 50 == 0:
            print(f"calibration {i+1}/{len(items)}", flush=True)

    with open(QDIR / "calibration_raw.jsonl", "w") as f:
        for r in calib_rows:
            f.write(json.dumps(r) + "\n")

    table = {}
    for judge in ("nli", "selene"):
        for qtype in ("noul", "choice", "score"):
            if judge == "nli" and qtype != "noul":
                continue
            n, correct = 0, 0
            for r in calib_rows:
                if r["type"] != qtype:
                    continue
                if judge == "nli":
                    if "nli_pred" not in r:
                        continue
                    n += 1
                    correct += int(r["nli_pred"] == r["gold"])
                else:
                    if "selene_pred" not in r or not r.get("selene_valid"):
                        continue
                    n += 1
                    correct += int(r["selene_pred"] == r.get("gold_norm"))
            acc = (correct / n) if n else None
            trusted = bool(n >= 50 and acc is not None and acc >= 0.90)
            table[f"{judge}:{qtype}"] = {"judge": judge, "type": qtype, "n": n, "accuracy": acc, "trusted": trusted}

    with open(CALIB_PATH, "w") as f:
        json.dump(table, f, indent=2)
    print(json.dumps(table, indent=2))
    nli.unload()


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def cmd_sample_b(args):
    """Single streaming pass: reservoir sample B_PER_SOURCE questions per source."""
    rng = random.Random(20260929)
    reservoirs = defaultdict(list)
    counts = defaultdict(int)
    n_rows = 0
    t0 = now()
    with open(DATA_B, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_rows += 1
            if n_rows % 200000 == 0:
                print(f"B streamed {n_rows} rows, {now()-t0:.0f}s", flush=True)
            if args.deadline_ts and deadline_hit(args.deadline_ts):
                print("deadline hit during sample-b, stopping early", flush=True)
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = row.get("source", "unknown")
            questions = row.get("questions", {})
            gold = row.get("gold", {})
            state = row.get("state")
            for qid, q in questions.items():
                g = gold.get(qid)
                if g is None:
                    continue
                counts[source] += 1
                k = counts[source]
                item = {"source": source, "domain": row.get("domain"), "state": state,
                        "qid": qid, "question": q, "gold": g}
                res = reservoirs[source]
                if len(res) < B_PER_SOURCE:
                    res.append(item)
                else:
                    j = rng.randint(1, k)
                    if j <= B_PER_SOURCE:
                        res[j - 1] = item
    total = sum(len(v) for v in reservoirs.values())
    with open(SAMPLE_B_PATH, "w") as f:
        for source, items in reservoirs.items():
            for it in items:
                f.write(json.dumps(it) + "\n")
    print(f"sample_b: {n_rows} rows scanned, {len(reservoirs)} sources, {total} questions sampled -> {SAMPLE_B_PATH}", flush=True)


# --------------------------------------------------------------------------
# Checkpointing helpers
# --------------------------------------------------------------------------

def load_done_keys(path):
    done = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["source"], r["state_hash"], r["qid"], r["stage"]))
    return done


def iter_a_rows():
    with open(DATA_A) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_b_sample():
    if not SAMPLE_B_PATH.exists():
        return
    with open(SAMPLE_B_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def a_questions(qtypes):
    for row in iter_a_rows():
        source = row.get("source")
        domain = row.get("domain")
        state = row.get("state")
        sh = state_hash(state)
        for qid, q in row.get("questions", {}).items():
            if q.get("type") not in qtypes:
                continue
            g = row.get("gold", {}).get(qid)
            if g is None:
                continue
            yield {"source": source, "domain": domain, "state": state, "state_hash": sh,
                   "qid": qid, "question": q, "gold": g}


def b_questions(qtypes):
    for it in iter_b_sample():
        if it["question"].get("type") not in qtypes:
            continue
        sh = state_hash(it["state"])
        yield {"source": it["source"], "domain": it.get("domain"), "state": it["state"],
               "state_hash": sh, "qid": it["qid"], "question": it["question"], "gold": it["gold"]}


# --------------------------------------------------------------------------
# Stage 1: NLI over noul
# --------------------------------------------------------------------------

def cmd_stage1(args):
    from nli_local import NLIJudge
    nli = NLIJudge()
    done = load_done_keys(JUDGED_PATH)
    out = open(JUDGED_PATH, "a")
    t0 = now()
    n_done = 0
    n_conflict = 0
    batch = []

    def flush(batch):
        nonlocal n_done, n_conflict
        if not batch:
            return
        pairs = [(b["state"], b["question"]["instructions"]) for b in batch]
        preds = nli.predict_batch(pairs)
        for item, pred in zip(batch, preds):
            gold_str = gold_label(item["gold"], "noul", item["question"].get("criteria"))
            gold_norm = "true" if gold_str.lower() in ("true", "yes", "1") else ("false" if gold_str.lower() in ("false", "no", "0") else gold_str)
            conflict = pred["label"] != gold_norm and pred["conf"] >= NLI_CONF
            rec = {
                "stage": "1", "source": item["source"], "domain": item.get("domain"),
                "state_hash": item["state_hash"], "qid": item["qid"], "type": "noul",
                "gold": gold_norm, "nli_pred": pred["label"], "nli_conf": pred["conf"],
                "nli_probs": pred["probs"], "nli_conflict": conflict,
            }
            out.write(json.dumps(rec) + "\n")
            n_done += 1
            if conflict:
                n_conflict += 1
        out.flush()

    for src_iter in (a_questions({"noul"}), b_questions({"noul"})):
        for item in src_iter:
            key = (item["source"], item["state_hash"], item["qid"], "1")
            if key in done:
                continue
            batch.append(item)
            if len(batch) >= NLI_BATCH:
                flush(batch)
                batch = []
                if n_done % (NLI_BATCH * 20) == 0:
                    rate = n_done / max(1e-6, now() - t0)
                    print(f"stage1: {n_done} judged, {n_conflict} conflicts, {rate:.1f} q/s", flush=True)
                if args.deadline_ts and deadline_hit(args.deadline_ts):
                    flush(batch)
                    batch = []
                    print("deadline hit during stage1, stopping early", flush=True)
                    out.close()
                    nli.unload()
                    return
        flush(batch)
        batch = []
    out.close()
    dt = now() - t0
    rate = n_done / max(1e-6, dt)
    print(f"stage1 done: {n_done} judged in {dt:.0f}s ({rate:.1f} q/s), {n_conflict} nli_conflicts", flush=True)
    nli.unload()


# --------------------------------------------------------------------------
# Stage 2: Selene over choice/score + nli_conflict
# --------------------------------------------------------------------------

def stratified_a_choice_score():
    by_source = defaultdict(list)
    for item in a_questions({"choice", "score"}):
        by_source[item["source"]].append(item)
    rng = random.Random(20260929)
    out = []
    for source, items in by_source.items():
        rng.shuffle(items)
        out.extend(items[:A_STAGE2_PER_FAMILY])
    return out


def get_nli_conflicts():
    conflicts = []
    if not JUDGED_PATH.exists():
        return conflicts
    a_by_key = {}
    for row in iter_a_rows():
        sh = state_hash(row.get("state"))
        for qid, q in row.get("questions", {}).items():
            a_by_key[(row.get("source"), sh, qid)] = {"source": row.get("source"), "domain": row.get("domain"),
                                                        "state": row.get("state"), "state_hash": sh, "qid": qid,
                                                        "question": q, "gold": row.get("gold", {}).get(qid)}
    b_by_key = {}
    for it in iter_b_sample():
        sh = state_hash(it["state"])
        b_by_key[(it["source"], sh, it["qid"])] = {"source": it["source"], "domain": it.get("domain"),
                                                     "state": it["state"], "state_hash": sh, "qid": it["qid"],
                                                     "question": it["question"], "gold": it["gold"]}
    with open(JUDGED_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get("stage") != "1" or not r.get("nli_conflict"):
                continue
            key = (r["source"], r["state_hash"], r["qid"])
            item = a_by_key.get(key) or b_by_key.get(key)
            if item:
                conflicts.append(item)
    return conflicts


def cmd_stage2(args):
    done = load_done_keys(JUDGED_PATH)
    items = []
    items.extend(stratified_a_choice_score())
    items.extend(list(b_questions({"choice", "score"})))
    items.extend(get_nli_conflicts())

    out = open(JUDGED_PATH, "a")
    t0 = now()
    n_done = 0
    n_conflict = 0
    n_ambig = 0
    seen = set()
    for item in items:
        key = (item["source"], item["state_hash"], item["qid"], "2")
        if key in done or key in seen:
            continue
        seen.add(key)
        if args.deadline_ts and deadline_hit(args.deadline_ts):
            print("deadline hit during stage2, stopping early", flush=True)
            break
        q = item["question"]
        qtype = q.get("type")
        criteria = q.get("criteria")
        sel = selene_answer(qtype, item["state"], q.get("instructions"), criteria)
        gold_norm = gold_label(item["gold"], qtype, criteria)
        conflict = sel["valid"] and sel["answer"] != gold_norm
        ambiguous = sel["clarity"] is not None and sel["clarity"] <= 2
        rec = {
            "stage": "2", "source": item["source"], "domain": item.get("domain"),
            "state_hash": item["state_hash"], "qid": item["qid"], "type": qtype,
            "gold": gold_norm, "selene_answer": sel["answer"], "selene_valid": sel["valid"],
            "selene_clarity": sel["clarity"], "judge_conflict": conflict, "ambiguous": ambiguous,
        }
        out.write(json.dumps(rec) + "\n")
        n_done += 1
        if conflict:
            n_conflict += 1
        if ambiguous:
            n_ambig += 1
        if n_done % 50 == 0:
            out.flush()
            rate = n_done / max(1e-6, now() - t0)
            print(f"stage2: {n_done}/{len(items)} judged, {n_conflict} conflicts, {n_ambig} ambiguous, {rate:.2f} q/s", flush=True)
    out.flush()
    out.close()
    dt = now() - t0
    rate = n_done / max(1e-6, dt)
    print(f"stage2 done: {n_done} judged in {dt:.0f}s ({rate:.2f} q/s), {n_conflict} judge_conflicts, {n_ambig} ambiguous", flush=True)
    # unload selene (keep_alive 0)
    try:
        body = {"model": SELENE_MODEL, "prompt": "", "keep_alive": 0}
        req = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode("utf-8"),
                                      headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=30).read()
        print("selene unloaded (keep_alive=0)", flush=True)
    except Exception as e:
        print(f"selene unload failed: {e}", flush=True)


# --------------------------------------------------------------------------
# Aggregate + verdicts (owner override: only trusted judges count)
# --------------------------------------------------------------------------

def cmd_aggregate(args):
    calib = json.load(open(CALIB_PATH)) if CALIB_PATH.exists() else {}
    trusted = {(v["judge"], v["type"]): v["trusted"] for v in calib.values()}

    stage1 = {}
    stage2 = {}
    if JUDGED_PATH.exists():
        with open(JUDGED_PATH) as f:
            for line in f:
                r = json.loads(line)
                key = (r["source"], r["state_hash"], r["qid"])
                if r["stage"] == "1":
                    stage1[key] = r
                else:
                    stage2[key] = r

    all_keys = set(stage1) | set(stage2)
    per_source = defaultdict(lambda: {"n": 0, "bad": 0, "suspect": 0, "ambiguous": 0})
    bad_rows = []
    verdicts = {}

    for key in all_keys:
        s1 = stage1.get(key)
        s2 = stage2.get(key)
        qtype = (s1 or s2)["type"]
        source = key[0]
        votes = []
        if s1 is not None:
            votes.append(("nli", "noul", bool(s1["nli_conflict"])))
        if s2 is not None:
            votes.append(("selene", qtype, bool(s2["judge_conflict"])))
        trusted_votes = [disagree for judge, t, disagree in votes if trusted.get((judge, t), False)]
        n_disagree_trusted = sum(1 for d in trusted_votes if d)
        if n_disagree_trusted >= 2:
            verdict = "BAD"
        elif n_disagree_trusted == 1:
            verdict = "SUSPECT"
        else:
            verdict = "OK"
        ambiguous = bool(s2 and s2.get("ambiguous"))

        per_source[source]["n"] += 1
        if verdict == "BAD":
            per_source[source]["bad"] += 1
            bad_rows.append({"source": source, "state_hash": key[1], "qid": key[2]})
        elif verdict == "SUSPECT":
            per_source[source]["suspect"] += 1
        if ambiguous:
            per_source[source]["ambiguous"] += 1
        verdicts[key] = verdict

    with open(BAD_ROWS_PATH, "w") as f:
        for r in bad_rows:
            f.write(json.dumps(r) + "\n")

    agg = []
    for source, c in per_source.items():
        n = c["n"]
        agg.append({
            "source": source, "n": n,
            "bad_rate": round(c["bad"] / n, 4) if n else 0.0,
            "suspect_rate": round(c["suspect"] / n, 4) if n else 0.0,
            "ambiguous_rate": round(c["ambiguous"] / n, 4) if n else 0.0,
            "decision": "REVIEW" if n and (c["bad"] / n) >= 0.15 else "KEEP",
        })
    agg.sort(key=lambda r: -r["bad_rate"])
    with open(AGG_PATH, "w") as f:
        json.dump(agg, f, indent=2)
    with open(REVIEW_SOURCES_PATH, "w") as f:
        for r in agg:
            if r["decision"] == "REVIEW":
                f.write(r["source"] + "\n")
    print(f"aggregate: {len(agg)} sources, {len(bad_rows)} BAD questions, "
          f"{sum(1 for r in agg if r['decision']=='REVIEW')} sources flagged for review", flush=True)


def cmd_report(args):
    calib = json.load(open(CALIB_PATH)) if CALIB_PATH.exists() else {}
    agg = json.load(open(AGG_PATH)) if AGG_PATH.exists() else []
    bad_rows = []
    if BAD_ROWS_PATH.exists():
        bad_rows = [json.loads(l) for l in open(BAD_ROWS_PATH)]
    lines = []
    lines.append("# Local data-quality report (NLI + Selene-1-Mini, on-device)\n")
    lines.append(f"_generated {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")
    lines.append("## Stage 0 calibration (judge accuracy on the trusted key)\n")
    lines.append("| judge | type | n | accuracy | trusted (>=0.90, n>=50) |")
    lines.append("|---|---|---|---|---|")
    for v in calib.values():
        acc = f"{v['accuracy']:.3f}" if v["accuracy"] is not None else "n/a"
        lines.append(f"| {v['judge']} | {v['type']} | {v['n']} | {acc} | {'yes' if v['trusted'] else 'no'} |")
    lines.append("")
    lines.append("Verdict rule (owner override): BAD only if two TRUSTED judges both "
                  "disagree with gold on the same question. choice/score questions have "
                  "only one possible judge (Selene; NLI never runs on them) so they cap at "
                  "SUSPECT. Nothing is deleted -- bad_rows.jsonl and review_sources_local.txt "
                  "are review lists only.\n")
    lines.append("## Per-source results (sorted by bad_rate)\n")
    lines.append("| source | n | bad_rate | suspect_rate | ambiguous_rate | decision |")
    lines.append("|---|---|---|---|---|---|")
    for r in agg:
        lines.append(f"| {r['source']} | {r['n']} | {r['bad_rate']} | {r['suspect_rate']} | {r['ambiguous_rate']} | {r['decision']} |")
    lines.append("")
    review = [r for r in agg if r["decision"] == "REVIEW"]
    lines.append(f"## REVIEW sources ({len(review)}) with example BAD questions\n")
    for r in review:
        lines.append(f"### {r['source']} (bad_rate={r['bad_rate']})")
        examples = [b for b in bad_rows if b["source"] == r["source"]][:3]
        for ex in examples:
            lines.append(f"- qid `{ex['qid']}`, state_hash `{ex['state_hash']}`")
        lines.append("")
    lines.append(f"Row-level BAD questions (review, not deleted): {len(bad_rows)} across "
                  f"{len(set(b['source'] for b in bad_rows))} sources.\n")
    state = json.load(open(STATE_PATH)) if STATE_PATH.exists() else {}
    lines.append("## Throughput / timing\n")
    for k, v in state.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Caveats\n")
    lines.append("- Calibration key is small (~1,216 questions); accuracy estimates, "
                  "especially per type, carry real sampling noise.")
    lines.append("- NLI only ever judges noul questions; choice/score verdicts rely on "
                  "Selene alone and can never reach BAD under the two-trusted-judge rule.")
    lines.append("- Selene-1-Mini runs at temperature 0 via Ollama; JSON parse failures "
                  "are retried once, then recorded as invalid (excluded from conflict math).")
    (REPO / "clauderesults").mkdir(exist_ok=True)
    (REPO / "clauderesults" / "07-local-quality-report.md").write_text("\n".join(lines))
    print("report written to clauderesults/07-local-quality-report.md", flush=True)


def cmd_publish(args):
    """Copy the mandated deliverable filenames into .local/quality/ (retrying
    since a concurrent process periodically empties that directory)."""
    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "local_judged.jsonl": JUDGED_PATH,
        "review_sources_local.txt": REVIEW_SOURCES_PATH,
        "bad_rows.jsonl": BAD_ROWS_PATH,
        "calibration_local.json": CALIB_PATH,
    }
    import shutil
    for name, src in files.items():
        if not src.exists():
            print(f"publish: skip {name}, source missing", flush=True)
            continue
        dst = PUBLISH_DIR / name
        for attempt in range(3):
            try:
                shutil.copy2(src, dst)
                if dst.exists() and dst.stat().st_size == src.stat().st_size:
                    print(f"publish: {name} -> {dst} ({dst.stat().st_size} bytes)", flush=True)
                    break
            except Exception as e:
                print(f"publish: attempt {attempt} for {name} failed: {e}", flush=True)
    print("publish done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calibrate", "sample-b", "stage1", "stage2", "aggregate", "report", "publish"])
    ap.add_argument("--deadline-ts", type=float, default=None)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    {
        "calibrate": cmd_calibrate,
        "sample-b": cmd_sample_b,
        "stage1": cmd_stage1,
        "stage2": cmd_stage2,
        "aggregate": cmd_aggregate,
        "report": cmd_report,
        "publish": cmd_publish,
    }[args.stage](args)


if __name__ == "__main__":
    main()
