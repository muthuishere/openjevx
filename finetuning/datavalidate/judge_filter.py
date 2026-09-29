#!/usr/bin/env python3
"""Blind-judge sampled OpenJevX training questions with two free LLM judges and
recommend which sources/families to flag for REVIEW.

Free chat-model verdicts are ADVISORY ONLY: nothing is ever deleted from the
input files, and a judge's verdict counts as evidence only for (model, question
type) combos that scored >=0.90 accuracy on a trusted key
(.local/eval/test.json, restricted to rows where label_agreement[qid].argmax_agree
is true). Untrusted combos never move a source's recommendation.

Reuses the model-calling / retry / cooldown / concurrency pattern from
.local/eval/llm/gen.py (opencode subprocess calls, weighted model picking with
per-model cooldown on failure). stdlib only.

Run with: .local/eval/venv/bin/python finetuning/datavalidate/judge_filter.py
Detached, resumable, hard wall-clock cap (default 3h, override with --max-hours).

Inputs:
  data/it_worker_train.jsonl   (A) rule-generated, source = our-cases-it-worker/<role>/<family>
  data/train_openjevx.jsonl    (B) 2.14M rows, streamed, many sources but few distinct
  .local/eval/test.json        trusted key used ONLY to calibrate judge accuracy per type

Outputs (all under .local/quality/ except the markdown report):
  calibration.json                 - per (model, type) accuracy on the trusted key
  sample_a.jsonl, sample_b.jsonl   - cached samples (so resume doesn't re-stream 2.7GB)
  judged.jsonl                     - every judged question, both judge answers (checkpoint)
  report.json                      - full structured report
  review_sources.txt               - one source per line, recommended for REVIEW (never deleted)
  clauderesults/06-data-quality-report.md
"""
import json, os, random, re, subprocess, sys, threading, time, importlib.util, argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402
DATA_A = str(paths.TRAIN / "it_worker_train.jsonl")
DATA_B = str(paths.TRAIN / "train_openjevx.jsonl")
QDIR = str(paths.WORK / "quality")
os.makedirs(QDIR, exist_ok=True)
SAMPLE_A_PATH = os.path.join(QDIR, "sample_a.jsonl")
SAMPLE_B_PATH = os.path.join(QDIR, "sample_b.jsonl")
JUDGED_PATH = os.path.join(QDIR, "judged.jsonl")
REPORT_JSON = os.path.join(QDIR, "report.json")
REVIEW_PATH = os.path.join(QDIR, "review_sources.txt")
CALIBRATION_PATH = os.path.join(QDIR, "calibration.json")
TRUSTED_KEY_PATH = str(paths.EVAL / "test.json")
REPORT_MD = os.path.join(ROOT, "clauderesults", "06-data-quality-report.md")
LOG_PATH = os.path.join(QDIR, "run.log")
TRUST_THRESHOLD = 0.90

ap = argparse.ArgumentParser()
ap.add_argument("--max-hours", type=float, default=float(os.environ.get("MAX_HOURS", "3")))
ap.add_argument("--par", type=int, default=int(os.environ.get("PAR", "8")))
ap.add_argument("--per-a", type=int, default=40)
ap.add_argument("--per-b", type=int, default=30)
ap.add_argument("--min-needed", type=int, default=20)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--calib-per-type", type=int, default=100)
args = ap.parse_args()

START = time.time()
DEADLINE = START + args.max_hours * 3600
# Reserve at most 40% of the wall-clock budget for calibration so judging always
# gets a real share; a model that times out mid-calibration just stays untrusted
# (safe default: its verdicts then never count as evidence).
CALIB_DEADLINE = START + args.max_hours * 3600 * 0.4

# ---- reuse adapter.gold_to_probabilities (stdlib only, no side deps) ----
spec = importlib.util.spec_from_file_location("openjevx_adapter", os.path.join(ROOT, "scripts", "train", "adapter.py"))
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

lock = threading.Lock()

def log(*a):
    with lock, open(LOG_PATH, "a") as f:
        f.write(time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a) + "\n")

# ---------------------------------------------------------------------------
# Model pools (free only; verified via `opencode models | grep -i free`)
# ---------------------------------------------------------------------------
OR_PREFERRED = "openrouter/nvidia/nemotron-3-super-120b-a12b:free"
OR_POOL = [
    OR_PREFERRED,
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/nvidia/nemotron-3.5-lightning:free",
    "openrouter/qwen/qwen3.8-27b:free",
    "openrouter/google/gemma-4-31b-it:free",
    "openrouter/dots-studio/dots-3-note-preview:free",
    "openrouter/poolside/laguna-s-2.1:free",
    "openrouter/thinkingmachines/inkling:free",
]
OC_PREFERRED = "opencode/mimo-v2.6-flash-free"
OC_SECOND = "opencode/longcat-2.5-preview-free"
OC_POOL = [
    OC_PREFERRED,
    OC_SECOND,
    "opencode/nemotron-3-ultra-free",
    "opencode/nemotron-3.5-lightning-free",
    "opencode/ling-3.0-flash-fin-free",
    "opencode/space-bunny-free",
    "opencode/muse-spark-1.3-contributor-free",
]
# Trimmed pools (below) keep calibration + judging wall-clock inside the 3h cap while
# still giving each judge slot real fallback options; the full free-model list is
# longer (`opencode models | grep -i free`) but every extra model calibrated costs
# ~10-15 minutes of the shared time budget.
ALL_MODELS = list(dict.fromkeys(OR_POOL + OC_POOL))
mstats = {m: {"ok": 0, "fail": 0, "cool_until": 0.0} for m in ALL_MODELS}


def call(model, prompt, timeout=180):
    try:
        r = subprocess.run(["opencode", "run", "-m", model, prompt], capture_output=True, text=True,
                            timeout=timeout, cwd=ROOT, stdin=subprocess.DEVNULL)
        return re.sub(r"\x1b\[[0-9;]*m", "", r.stdout)
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:
        log("call exc", model, repr(e))
        return ""


def pick(pool, preferred, exclude=None):
    now = time.time()
    with lock:
        cands = [m for m in pool if m != exclude] or list(pool)
        avail = [m for m in cands if mstats[m]["cool_until"] < now] or cands
        if preferred in avail and preferred != exclude and random.random() < 0.65:
            return preferred
        w = [(mstats[m]["ok"] + 2) / (mstats[m]["ok"] + mstats[m]["fail"] + 3) for m in avail]
        return random.choices(avail, w)[0]


def mark(model, ok):
    with lock:
        s = mstats[model]
        if ok:
            s["ok"] += 1
        else:
            s["fail"] += 1
            s["cool_until"] = time.time() + 90


def json_lines(text):
    objs = []
    for l in text.splitlines():
        l = l.strip().strip("`")
        if l.startswith("{"):
            try:
                objs.append(json.loads(l))
            except Exception:
                pass
    if not objs:
        m = re.search(r"\[.*\]", text, re.S)
        if m:
            try:
                v = json.loads(m.group(0))
                if isinstance(v, list):
                    objs = [x for x in v if isinstance(x, dict)]
            except Exception:
                pass
    return objs


# ---------------------------------------------------------------------------
# Sampling: A (in-memory, stratified by template/instructions text)
# ---------------------------------------------------------------------------
def normalize_gold(question, gold_raw):
    """Returns (normalized_label:str|None, question_with_canonical_criteria)."""
    q = dict(question)
    if isinstance(q.get("criteria"), dict):
        q["criteria"] = dict(q["criteria"])
    elif isinstance(q.get("criteria"), list):
        q["criteria"] = list(q["criteria"])
    try:
        result = adapter.gold_to_probabilities(gold_raw, q)
    except Exception:
        return None, q
    if not result or "probabilities" not in result:
        return None, q
    label = max(result["probabilities"].items(), key=lambda kv: kv[1])[0]
    return label, q


def sample_a(n_per_family):
    if os.path.exists(SAMPLE_A_PATH):
        items = [json.loads(l) for l in open(SAMPLE_A_PATH)]
        log("sample_a: loaded cached", len(items))
        return items
    by_source = defaultdict(list)  # source -> template(instructions) -> list of raw items
    unmappable_sources = Counter()
    total_sources_seen = set()
    with open(DATA_A) as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            src = row.get("source")
            dom = row.get("domain")
            state = row.get("state")
            total_sources_seen.add(src)
            for qid, q in row.get("questions", {}).items():
                gold_raw = row.get("gold", {}).get(qid)
                if gold_raw is None:
                    continue
                label, qn = normalize_gold(q, gold_raw)
                if label is None:
                    unmappable_sources[src] += 1
                    continue
                by_source[src].append({
                    "source": src, "domain": dom, "state": state, "qid": qid,
                    "question": qn, "gold": label, "template": qn.get("instructions", qid),
                })
    items = []
    counter = 0
    for src, pool in by_source.items():
        by_tmpl = defaultdict(list)
        for it in pool:
            by_tmpl[it["template"]].append(it)
        for lst in by_tmpl.values():
            random.shuffle(lst)
        tmpl_keys = list(by_tmpl.keys())
        picked = []
        i = 0
        while len(picked) < n_per_family and any(by_tmpl[k] for k in tmpl_keys):
            k = tmpl_keys[i % len(tmpl_keys)]
            if by_tmpl[k]:
                picked.append(by_tmpl[k].pop())
            i += 1
        for it in picked:
            counter += 1
            it["item_id"] = f"A{counter:06d}"
            items.append(it)
    with open(SAMPLE_A_PATH, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    log("sample_a: families", len(by_source), "of", len(total_sources_seen), "seen; sampled", len(items),
        "gold_unmappable_sources", dict(unmappable_sources))
    return items


# ---------------------------------------------------------------------------
# Sampling: B (streamed two-pass reservoir, since B is 2.7GB/2.14M rows but
# only a handful of distinct `source` values)
# ---------------------------------------------------------------------------
SRC_RE = re.compile(rb'"source":\s*"((?:[^"\\]|\\.)*)"')


def extract_source(raw_bytes):
    m = SRC_RE.search(raw_bytes)
    if not m:
        return None
    try:
        return json.loads(b'"' + m.group(1) + b'"')
    except Exception:
        return m.group(1).decode("utf-8", "replace")


def sample_b(n_per_source, rows_per_source=250):
    if os.path.exists(SAMPLE_B_PATH):
        items = [json.loads(l) for l in open(SAMPLE_B_PATH)]
        log("sample_b: loaded cached", len(items))
        return items
    t0 = time.time()
    reservoir = defaultdict(list)  # source -> [line_no,...]
    counts = Counter()
    with open(DATA_B, "rb") as f:
        for line_no, raw in enumerate(f):
            src = extract_source(raw)
            if src is None:
                continue
            counts[src] += 1
            res = reservoir[src]
            if len(res) < rows_per_source:
                res.append(line_no)
            else:
                j = random.randrange(counts[src])
                if j < rows_per_source:
                    res[j] = line_no
    selected = set()
    for res in reservoir.values():
        selected.update(res)
    log("sample_b: pass1 done", len(reservoir), "sources", len(selected), "rows selected",
        "%.1fs" % (time.time() - t0))
    rows_by_lineno = {}
    with open(DATA_B, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f):
            if line_no in selected:
                try:
                    rows_by_lineno[line_no] = json.loads(line)
                except Exception:
                    pass
    log("sample_b: pass2 done parsed", len(rows_by_lineno), "%.1fs" % (time.time() - t0))

    items = []
    counter = 0
    unmappable_sources = Counter()
    for src, line_nos in reservoir.items():
        pool = []
        for ln in line_nos:
            row = rows_by_lineno.get(ln)
            if not row:
                continue
            dom = row.get("domain")
            state = row.get("state")
            for qid, q in row.get("questions", {}).items():
                gold_raw = row.get("gold", {}).get(qid)
                if gold_raw is None:
                    continue
                label, qn = normalize_gold(q, gold_raw)
                if label is None:
                    unmappable_sources[src] += 1
                    continue
                pool.append({"source": src, "domain": dom, "state": state, "qid": qid,
                             "question": qn, "gold": label})
        random.shuffle(pool)
        picked = pool[:n_per_source]
        for it in picked:
            counter += 1
            it["item_id"] = f"B{counter:06d}"
            items.append(it)
    with open(SAMPLE_B_PATH, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    log("sample_b: sources", len(reservoir), "sampled", len(items),
        "gold_unmappable_sources", dict(unmappable_sources))
    return items


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------
def state_text(state, limit):
    s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    if len(s) > limit:
        s = s[:limit] + "...[truncated]"
    return s


def criteria_repr(q):
    t = q["type"]
    cr = q.get("criteria")
    if t == "score":
        return {str(i): d for i, d in enumerate(cr)}
    return cr


JUDGE_PROMPT = """You are a careful, skeptical judge for a decision-model training dataset. For EACH item below, read "state" and answer "question" using ONLY evidence present in the state. Never use outside/general knowledge to fill a gap the state doesn't give you.

Answer format by type:
- "noul": answer is the string "true" or "false".
- "choice": answer is exactly one of the criteria keys, verbatim.
- "score": answer is a level index as a string ("0","1",...) matching position in the criteria list (0-based, listed low-to-high).

Also set "unclear": true if a competent person could reasonably disagree given the state, or the state is missing the deciding fact; otherwise false.

Output ONLY JSON Lines, one line per item, nothing else: {{"id":"<item id>","answer":"<answer>","unclear":true|false}}
No prose, no markdown, no code fences, do not use tools.

ITEMS:
{items}
"""


def legal(question, ans):
    if ans is None:
        return False
    a = str(ans).strip().strip('"')
    t = question["type"]
    cr = question.get("criteria")
    if t == "noul":
        return a.lower() in ("true", "false")
    if t == "choice":
        return a in cr
    if t == "score":
        return a.isdigit() and int(a) < len(cr)
    return False


def norm_ans(question, ans):
    a = str(ans).strip().strip('"')
    if question["type"] == "noul":
        return a.lower()
    return a


def build_batch_prompt(items, state_limit):
    parts = []
    for it in items:
        q = it["question"]
        parts.append(json.dumps({
            "id": it["item_id"], "type": q["type"], "state": state_text(it["state"], state_limit),
            "question": q["instructions"], "criteria": criteria_repr(q),
        }, ensure_ascii=False))
    return JUDGE_PROMPT.format(items="\n".join(parts))


def judge_batch(items, pool, preferred, state_limit=1500):
    """Runs one judge over a batch of items, retrying/rotating models on failure.
    Returns {item_id: {"answer":..., "unclear":..., "model":...}}"""
    by_id = {it["item_id"]: it for it in items}
    remaining = list(items)
    out = {}
    model = None
    for attempt in range(3):
        if not remaining or time.time() > DEADLINE:
            break
        model = pick(pool, preferred, exclude=model)
        prompt = build_batch_prompt(remaining, state_limit)
        text = call(model, prompt)
        got = {}
        for o in json_lines(text):
            iid = o.get("id")
            if iid not in by_id:
                continue
            ans = o.get("answer")
            if legal(by_id[iid]["question"], ans):
                got[iid] = {"answer": norm_ans(by_id[iid]["question"], ans),
                            "unclear": bool(o.get("unclear", False)), "model": model}
        ok = len(got) >= max(1, len(remaining) // 2)
        mark(model, ok)
        out.update(got)
        remaining = [it for it in remaining if it["item_id"] not in out]
        if not remaining:
            break
    if remaining:
        log("judge_batch: giving up on", len(remaining), "items, pool", pool[:1])
    return out


# ---------------------------------------------------------------------------
# Calibration: per (model, question type) accuracy on a trusted key, BEFORE
# any judge verdict is allowed to count as evidence. Free chat models are
# advisory only; only combos scoring >=TRUST_THRESHOLD here matter later.
# ---------------------------------------------------------------------------
def load_calibration_items(per_type, seed=20260929):
    rng = random.Random(seed)
    rows = json.load(open(TRUSTED_KEY_PATH))
    by_type = defaultdict(list)
    for r in rows:
        try:
            qs = json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
            gold = json.loads(r["gold"]) if isinstance(r["gold"], str) else r["gold"]
            la = json.loads(r["label_agreement"]) if isinstance(r["label_agreement"], str) else r["label_agreement"]
            state = r["state"]
        except Exception:
            continue
        for qid, q in qs.items():
            agree = la.get(qid, {})
            if not agree.get("argmax_agree"):
                continue
            g = gold.get(qid, {})
            label = g.get("label") if isinstance(g, dict) else g
            if label is None:
                continue
            by_type[q["type"]].append({
                "item_id": f"CAL_{r['id']}_{qid}", "source": "__calibration__", "domain": r.get("workflow"),
                "state": state, "qid": qid, "question": q, "gold": str(label),
            })
    items = []
    for t, lst in by_type.items():
        rng.shuffle(lst)
        items.extend(lst[:per_type])
    log("calibration items loaded", {t: min(len(l), per_type) for t, l in by_type.items()})
    return items


def calibrate_model(model, items_by_type):
    """Single-model accuracy per type, no rotation (we want THIS model's own
    accuracy). Two attempts per batch to absorb transient failures only."""
    result = {}
    for t, items in items_by_type.items():
        correct = 0
        answered = 0
        for i in range(0, len(items), args.batch):
            if time.time() > CALIB_DEADLINE:
                log("calibrate_model: deadline hit, stopping early", model, t, "at", i, "/", len(items))
                break
            chunk = items[i:i + args.batch]
            by_id = {it["item_id"]: it for it in chunk}
            got = {}
            for attempt in range(2):
                prompt = build_batch_prompt(chunk, state_limit=1500)
                text = call(model, prompt)
                for o in json_lines(text):
                    iid = o.get("id")
                    if iid not in by_id or iid in got:
                        continue
                    ans = o.get("answer")
                    if legal(by_id[iid]["question"], ans):
                        got[iid] = norm_ans(by_id[iid]["question"], ans)
                if len(got) >= len(chunk):
                    break
            mark(model, len(got) >= max(1, len(chunk) // 2))
            for it in chunk:
                a = got.get(it["item_id"])
                if a is None:
                    continue
                answered += 1
                if a == it["gold"]:
                    correct += 1
        n = len(items)
        result[t] = {"n": n, "answered": answered, "correct": correct,
                     "accuracy": round(correct / n, 4) if n else 0.0}
    return result


def run_calibration():
    if os.path.exists(CALIBRATION_PATH):
        cal = json.load(open(CALIBRATION_PATH))
        log("calibration: loaded cached for", list(cal.get("models", {}).keys()))
        return cal
    items = load_calibration_items(args.calib_per_type)
    by_type = defaultdict(list)
    for it in items:
        by_type[it["question"]["type"]].append(it)
    models = list(dict.fromkeys(OR_POOL + OC_POOL))
    results = {}
    with ThreadPoolExecutor(args.par) as ex:
        futs = {ex.submit(calibrate_model, m, by_type): m for m in models}
        for fut, m in futs.items():
            try:
                results[m] = fut.result(timeout=max(60, CALIB_DEADLINE - time.time() + 400))
            except Exception as e:
                log("calibration failed for", m, repr(e))
                results[m] = {t: {"n": len(v), "answered": 0, "correct": 0, "accuracy": 0.0} for t, v in by_type.items()}
    cal = {"threshold": TRUST_THRESHOLD, "per_type_n": {t: len(v) for t, v in by_type.items()}, "models": results}
    json.dump(cal, open(CALIBRATION_PATH, "w"), indent=1)
    log("calibration done", {m: {t: v["accuracy"] for t, v in r.items()} for m, r in results.items()})
    return cal


def trusted_combos_from(cal):
    trusted = set()
    for m, per_type in cal.get("models", {}).items():
        for t, stats in per_type.items():
            if stats.get("accuracy", 0.0) >= TRUST_THRESHOLD:
                trusted.add((m, t))
    return trusted


def already_judged_ids():
    ids = set()
    if os.path.exists(JUDGED_PATH):
        for l in open(JUDGED_PATH):
            try:
                ids.add(json.loads(l)["item_id"])
            except Exception:
                pass
    return ids


def process_source_group(src, items):
    done = already_judged_ids()
    todo = [it for it in items if it["item_id"] not in done]
    if not todo:
        return []
    j1 = judge_batch(todo, OR_POOL, OR_PREFERRED)
    j2 = judge_batch(todo, OC_POOL, OC_PREFERRED)
    recs = []
    for it in todo:
        a1 = j1.get(it["item_id"])
        a2 = j2.get(it["item_id"])
        rec = {
            "item_id": it["item_id"], "source": it["source"], "domain": it.get("domain"),
            "qid": it["qid"], "type": it["question"]["type"], "instructions": it["question"]["instructions"],
            "state_excerpt": state_text(it["state"], 500), "gold": it["gold"],
            "judge1_model": a1["model"] if a1 else None,
            "judge1_answer": a1["answer"] if a1 else None,
            "judge1_unclear": a1["unclear"] if a1 else None,
            "judge2_model": a2["model"] if a2 else None,
            "judge2_answer": a2["answer"] if a2 else None,
            "judge2_unclear": a2["unclear"] if a2 else None,
        }
        both_valid = a1 is not None and a2 is not None
        rec["both_valid"] = both_valid
        rec["bad"] = bool(both_valid and a1["answer"] != it["gold"] and a2["answer"] != it["gold"])
        rec["unclear_both"] = bool(a1 and a2 and a1["unclear"] and a2["unclear"])
        recs.append(rec)
    with lock:
        with open(JUDGED_PATH, "a") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return recs


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(hit_deadline, insufficient_note, cal, trusted):
    rows = [json.loads(l) for l in open(JUDGED_PATH)] if os.path.exists(JUDGED_PATH) else []
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    sources_report = []
    for src, recs in by_source.items():
        valid = [r for r in recs if r["both_valid"]]
        # Owner rule: a judge verdict counts as evidence only if that judge's
        # ACTUAL model is trusted (>=0.90 accuracy) for that question's type.
        trusted_valid = [r for r in valid
                          if (r["judge1_model"], r["type"]) in trusted
                          and (r["judge2_model"], r["type"]) in trusted]
        n_all = len(valid)
        n = len(trusted_valid)
        bad = sum(1 for r in trusted_valid if r["bad"])
        unclear_both = sum(1 for r in trusted_valid if r["unclear_both"])
        j1_agree = sum(1 for r in trusted_valid if r["judge1_answer"] == r["gold"])
        j2_agree = sum(1 for r in trusted_valid if r["judge2_answer"] == r["gold"])
        bad_rate = bad / n if n else 0.0
        unclear_rate = unclear_both / n if n else 0.0
        if n < args.min_needed:
            decision = "REVIEW (insufficient trusted evidence)"
        elif bad_rate >= 0.25 or unclear_rate >= 0.40:
            decision = "REVIEW (high bad/unclear rate)"
        elif 0.10 <= bad_rate < 0.25:
            decision = "REVIEW (elevated bad rate)"
        else:
            decision = "KEEP"
        examples = [r for r in trusted_valid if r["bad"]][:3]
        sources_report.append({
            "source": src, "n_judged_trusted": n, "n_judged_all": n_all, "n_sampled": len(recs),
            "bad_rate": round(bad_rate, 4), "unclear_rate": round(unclear_rate, 4),
            "judge1_agreement": round(j1_agree / n, 4) if n else None,
            "judge2_agreement": round(j2_agree / n, 4) if n else None,
            "decision": decision, "insufficient_trusted_evidence": n < args.min_needed,
            "examples_bad": [{"state": e["state_excerpt"][:300], "gold": e["gold"],
                               "judge1_answer": e["judge1_answer"], "judge2_answer": e["judge2_answer"],
                               "question": e["instructions"]} for e in examples],
        })
    sources_report.sort(key=lambda s: -s["bad_rate"])

    totals = {
        "total_sources": len(sources_report),
        "total_judged_questions_all": sum(s["n_judged_all"] for s in sources_report),
        "total_judged_questions_trusted": sum(s["n_judged_trusted"] for s in sources_report),
        "review_high_bad": sum(1 for s in sources_report if "high bad" in s["decision"]),
        "review_elevated": sum(1 for s in sources_report if "elevated" in s["decision"]),
        "review_insufficient": sum(1 for s in sources_report if "insufficient" in s["decision"]),
        "keep": sum(1 for s in sources_report if s["decision"] == "KEEP"),
        "insufficient_trusted_sources": [s["source"] for s in sources_report if s["insufficient_trusted_evidence"]],
    }
    model_stats = {m: {"ok": v["ok"], "fail": v["fail"]} for m, v in mstats.items()}
    elapsed = time.time() - START
    cal_table = []
    for m, per_type in cal.get("models", {}).items():
        for t, stats in per_type.items():
            cal_table.append({"model": m, "type": t, "n": stats["n"], "answered": stats["answered"],
                               "correct": stats["correct"], "accuracy": stats["accuracy"],
                               "trusted": (m, t) in trusted})
    cal_table.sort(key=lambda r: (-r["accuracy"]))

    report = {
        "advisory_notice": "Free chat-model verdicts are ADVISORY ONLY. Nothing is ever deleted from "
                            "data/it_worker_train.jsonl or data/train_openjevx.jsonl. A judge verdict counts as "
                            "evidence only for (model, question type) combos scoring >=%.2f accuracy on the "
                            "trusted key (.local/eval/test.json, argmax_agree rows only). review_sources.txt "
                            "lists sources recommended for human review, not for automatic removal." % TRUST_THRESHOLD,
        "decision_rule": "Using TRUSTED-ONLY judged questions per source: REVIEW (insufficient trusted evidence) "
                          "if n_judged_trusted<%d; REVIEW (high bad/unclear rate) if bad_rate>=0.25 or "
                          "unclear_rate>=0.40; REVIEW (elevated bad rate) if 0.10<=bad_rate<0.25; else KEEP." % args.min_needed,
        "judges": {"judge1_pool": "openrouter/*:free", "judge1_preferred": OR_PREFERRED,
                   "judge2_pool": "opencode/*-free", "judge2_preferred": OC_PREFERRED},
        "calibration": {"trust_threshold": TRUST_THRESHOLD, "trusted_key": ".local/eval/test.json",
                         "trusted_key_filter": "label_agreement[qid].argmax_agree == true",
                         "per_type_calibration_n": cal.get("per_type_n"), "table": cal_table},
        "sources": sources_report, "totals": totals, "model_stats": model_stats,
        "wall_clock_seconds": round(elapsed, 1), "hit_deadline": hit_deadline,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(START)),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": insufficient_note,
    }
    json.dump(report, open(REPORT_JSON, "w"), indent=1)

    with open(REVIEW_PATH, "w") as f:
        for s in sources_report:
            if s["decision"] != "KEEP":
                f.write(s["source"] + "\n")

    md = []
    md.append("# OpenJevX training data quality report\n\n")
    md.append("> **" + report["advisory_notice"] + "**\n\n")
    md.append(f"Generated {report['finished_at']}. Wall clock: {elapsed/60:.1f} min. "
              f"Hit 3h cap: {hit_deadline}.\n\n")
    md.append("## Judge calibration against the trusted key\n\n")
    md.append(f"Trusted key: `.local/eval/test.json`, restricted to rows where "
              f"`label_agreement[qid].argmax_agree` is true. Per-type calibration n: "
              f"{report['calibration']['per_type_calibration_n']}. Trust threshold: {TRUST_THRESHOLD}.\n\n")
    md.append("| model | type | n | answered | correct | accuracy | trusted >=0.90 |\n")
    md.append("|---|---|---:|---:|---:|---:|:---:|\n")
    for r in cal_table:
        md.append(f"| {r['model']} | {r['type']} | {r['n']} | {r['answered']} | {r['correct']} | "
                  f"{r['accuracy']:.3f} | {'YES' if r['trusted'] else 'no'} |\n")
    md.append("\n## Decision rule\n\n")
    md.append(f"{report['decision_rule']}\n\n")
    md.append("## Judges\n\n")
    md.append(f"- Judge 1 pool: openrouter/*:free (preferred `{OR_PREFERRED}`)\n"
              f"- Judge 2 pool: opencode/*-free (preferred `{OC_PREFERRED}`)\n\n")
    md.append("## Totals\n\n")
    md.append(f"- Sources judged: {totals['total_sources']}\n"
              f"- Questions judged (all): {totals['total_judged_questions_all']}\n"
              f"- Questions judged (trusted-combo only, this is what decisions are based on): "
              f"{totals['total_judged_questions_trusted']}\n"
              f"- REVIEW (high bad/unclear rate): {totals['review_high_bad']} · "
              f"REVIEW (elevated bad rate): {totals['review_elevated']} · "
              f"REVIEW (insufficient trusted evidence): {totals['review_insufficient']} · "
              f"KEEP: {totals['keep']}\n\n")
    if totals["insufficient_trusted_sources"]:
        md.append(f"- Sources with fewer than {args.min_needed} TRUSTED judged questions: "
                  + ", ".join(totals["insufficient_trusted_sources"]) + "\n\n")
    md.append("## Per-source table (sorted by bad_rate desc, trusted-only evidence)\n\n")
    md.append("| source | n_trusted | n_all_judged | bad_rate | unclear_rate | judge1 agree | judge2 agree | recommendation |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---|\n")
    for s in sources_report:
        md.append(f"| {s['source']} | {s['n_judged_trusted']} | {s['n_judged_all']} | {s['bad_rate']:.3f} | "
                  f"{s['unclear_rate']:.3f} | {s['judge1_agreement']} | {s['judge2_agreement']} | {s['decision']} |\n")
    md.append("\n## Examples of bad questions per source (trusted-combo judges only)\n")
    for s in sources_report:
        if not s["examples_bad"]:
            continue
        md.append(f"\n### {s['source']} ({s['decision']}, bad_rate={s['bad_rate']:.3f})\n")
        for e in s["examples_bad"]:
            md.append(f"- Q: {e['question']}\n  state: `{e['state']}`\n  gold=`{e['gold']}` "
                      f"judge1=`{e['judge1_answer']}` judge2=`{e['judge2_answer']}`\n")
    md.append("\n## Model call stats (all calls made, including untrusted models)\n\n")
    md.append("| model | ok | fail |\n|---|---:|---:|\n")
    for m, v in sorted(model_stats.items(), key=lambda kv: -(kv[1]["ok"] + kv[1]["fail"])):
        if v["ok"] or v["fail"]:
            md.append(f"| {m} | {v['ok']} | {v['fail']} |\n")
    if insufficient_note:
        md.append("\n## Notes\n\n" + insufficient_note + "\n")
    with open(REPORT_MD, "w") as f:
        f.writelines(md)
    return report


def main():
    log("=== start ===", "max_hours", args.max_hours, "par", args.par)
    cal = run_calibration()
    trusted = trusted_combos_from(cal)
    log("trusted (model,type) combos", sorted(trusted))
    items_a = sample_a(args.per_a)
    items_b = sample_b(args.per_b)
    all_items = items_a + items_b
    groups = defaultdict(list)
    for it in all_items:
        groups[it["source"]].append(it)
    log("groups", len(groups), "total sampled items", len(all_items))

    hit_deadline = False
    with ThreadPoolExecutor(args.par) as ex:
        futs = {ex.submit(process_source_group, src, items): src for src, items in groups.items()}
        for fut in futs:
            src = futs[fut]
            if time.time() > DEADLINE:
                hit_deadline = True
                continue
            try:
                fut.result(timeout=max(1, DEADLINE - time.time()) if time.time() < DEADLINE else 1)
            except Exception as e:
                log("group failed", src, repr(e))
                hit_deadline = hit_deadline or time.time() > DEADLINE

    note_parts = []
    for src, items in groups.items():
        target = args.per_a if items and items[0]["item_id"].startswith("A") else args.per_b
        if len(items) < target:
            note_parts.append(f"{src}: only {len(items)}/{target} sampleable (gold unmappable or too few rows)")
    note = "; ".join(note_parts) if note_parts else ""
    report = build_report(hit_deadline, note, cal, trusted)
    log("=== done ===", "elapsed_s", round(time.time() - START, 1), "hit_deadline", hit_deadline)
    print(json.dumps(report["totals"], indent=1))


if __name__ == "__main__":
    main()
