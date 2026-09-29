#!/usr/bin/env python3
"""Gold adapter: shape our typed-decisions gold into what build_item expects.

Rules (build_item is scripts/train_openjevx.py:37):
- choice: probabilities keyed by the exact criteria dict keys, 0.9 on gold, 0.1 spread.
- noul:   probabilities keyed "false"/"true", 0.9 on gold.
- score:  build_item hardcodes levels=4 for dict criteria (train_openjevx.py:48), so
          dict criteria are converted to a LIST of option keys and gold becomes
          {"0".."N-1"} spread over the index positions.
Unmappable gold (label absent from criteria, out-of-range index) drops the question;
a record with zero usable questions is skipped by adapt_row.
"""

import gzip
import hashlib
import json

PRIORITY_PREFIXES = ("our-cases-", "typed-decisions/")


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def is_priority(source):
    return source.startswith(PRIORITY_PREFIXES)


# Target mass on the gold option. Rule-labelled data (our-cases-*: the answer is computed, so it
# is certain) trains toward near-certainty; human-labelled sources keep 0.9 because some labels are
# noisy. Soft gold that already carries probabilities (split human votes) is kept as-is.
CERTAIN_PEAK = 0.99
DEFAULT_PEAK = 0.9


def spread(target_key, keys, peak=DEFAULT_PEAK):
    probability = {}
    for key in keys:
        probability[key] = peak if key == target_key else round((1 - peak) / max(1, len(keys) - 1), 6)
    return probability


def gold_to_probabilities(gold, question, peak=DEFAULT_PEAK):
    if isinstance(gold, dict):
        label = gold.get("label", gold)
        if "probabilities" in gold and isinstance(label, str):
            return gold
        return gold_to_probabilities(label, question, peak)
    criteria = question.get("criteria")
    if isinstance(criteria, list):
        criteria = {str(index): str(value) for index, value in enumerate(criteria)}
    elif isinstance(criteria, dict):
        criteria = {str(key): str(value) for key, value in criteria.items()}
    else:
        criteria = {}
    qtype = question.get("type")
    text = str(gold).strip()
    if qtype == "noul":
        lowered = text.lower()
        if lowered in ("true", "yes", "1"):
            return {"label": "true", "probabilities": spread("true", ("false", "true"), peak)}
        if lowered in ("false", "no", "0"):
            return {"label": "false", "probabilities": spread("false", ("false", "true"), peak)}
        return None
    if len(criteria) < 2:
        return None
    keys = list(criteria)
    if qtype == "score" and isinstance(question.get("criteria"), dict):
        question["criteria"] = keys
    target_key = text if text in criteria else None
    if target_key is None and text.lstrip("-").isdigit() and 0 <= int(text) < len(keys):
        target_key = keys[int(text)]
    if target_key is None:
        return None
    if qtype == "score":
        target_key = str(keys.index(target_key))
        keys = [str(index) for index in range(len(keys))]
    return {"label": text, "probabilities": spread(target_key, keys, peak)}


def adapt_row(row):
    peak = CERTAIN_PEAK if str(row.get("source", "")).startswith("our-cases-") else DEFAULT_PEAK
    kept = {}
    for question_id, question in row.get("questions", {}).items():
        gold = row.get("gold", {}).get(question_id)
        if gold is None:
            continue
        shaped = gold_to_probabilities(gold, question, peak)
        if shaped is not None:
            kept[question_id] = shaped
    if not kept:
        return None
    return {"source": row.get("source"), "domain": row.get("domain"),
            "state": row.get("state"), "questions": row["questions"], "gold": kept}


def hash_slot(text, key=b"openjevx-stratify"):
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8, key=key).digest()
    return int.from_bytes(digest, "big") / 2 ** 64


def tasksource_rate(train_path, budget_bytes):
    priority_bytes = 0
    domain_bytes = {}
    with open(train_path, "rb") as handle:
        for raw in handle:
            start = raw.find(b'"source": "') + 11
            source = raw[start:raw.index(b'"', start)].decode("utf-8", "replace")
            if is_priority(source):
                priority_bytes += len(raw)
                continue
            marker = raw.find(b'"domain": "') + 11
            domain = raw[marker:raw.index(b'"', marker)].decode("utf-8", "replace")
            domain_bytes[domain] = domain_bytes.get(domain, 0) + len(raw)
    available = budget_bytes - priority_bytes
    total = sum(domain_bytes.values())
    return min(1.0, max(0.0, available / total)), priority_bytes, total


def stratified_sample_rows(train_path, budget_bytes, seed_key=b"openjevx-stratify"):
    """Every our-cases-* and typed-decisions/* row in full; per-domain hash sampling
    (deterministic, single pass, no re-versioning risk) for the tasksource bulk."""
    rate, priority_bytes, bulk_bytes = tasksource_rate(train_path, budget_bytes)
    rows = []
    skipped = 0
    for row in read_jsonl(train_path):
        if not is_priority(row.get("source", "")):
            if hash_slot(json.dumps({"s": row.get("state"), "d": row.get("domain")})[:300], seed_key) > rate:
                skipped += 1
                continue
        shaped = adapt_row(row)
        if shaped is None:
            skipped += 1
            continue
        rows.append(shaped)
    plan = {"rate": rate, "priority_bytes": priority_bytes, "bulk_bytes": bulk_bytes,
            "skipped": skipped, "rows": len(rows)}
    return rows, plan
