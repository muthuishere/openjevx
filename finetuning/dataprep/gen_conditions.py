#!/usr/bin/env python3
"""Deterministic CONDITION-EVALUATION training-data generator for OpenJevX.

Core insight (owner): programming is representing a domain's rules as
conditions over state. This generator teaches the model to evaluate
conditions over state CORRECTLY and CONFIDENTLY -- across ~40 business
domains, in English and in code (Python / JavaScript / Go / SQL) -- with
heavy sampling right at numeric/date boundaries (age 17/18/19, stock 0/1/2,
cpu 89/90/91, deadline yesterday/today/tomorrow, ...).

Row contract (same as finetuning/dataprep/gen_it_worker.py):
    {"source": "our-cases-conditions/<domain>/<kind>",
     "domain": "conditions/<domain>/<kind>",
     "state": <dict OR plain-English string>,
     "questions": {"<qid>": {"type": "noul"|"choice"|"score",
                              "instructions": "...", "criteria": ...}},
     "gold": {"<qid>": "<label>"}}

Every gold label is produced by ACTUALLY EVALUATING the condition in Python
against the concrete field values placed in `state` (see `evaluate_atom`).
Code renderings (Python/JS/Go/SQL) are strings for the model to read; they
are never executed or parsed -- the single source of truth for `gold` is
always the Python evaluation of the same underlying atom(s).

Render forms per case (balanced across a pool, ~3-5 questions sampled per
row, mirroring gen_it_worker.py's `build_row`):
    a. English statement about the entity, with the policy given in state.
    b. Explicit rule text ("adult means age >= 18").
    c. The condition as code (`if` in Python/JS/Go, or a SQL WHERE clause) --
       "Does this evaluate to true for this state?"
    d. A small 2-3 branch function ("what does this return for this input?")
       -- choice over the possible return labels.
    e. Which tier/band applies (score, lowest-first) -- e.g. shipping tiers,
       discount bands, alert severity.

Splits:
    - 5 whole domains held out of train, eval-only (HELD_OUT_DOMAINS).
    - The last phrasing template in every instruction pool is eval-only (T()).
    - ~20% of eval numeric/date samples are drawn from a wider range /
      different digit-length than train ever sees (see `sample_num`).

Also writes data/conditions_basics_gate.jsonl: ~300 hand-auditable, single
fact + single condition cases (age~18, stock 0/1/2/500, cpu 89/90/91,
explicit-date deadlines, status equals, list membership, null checks) as a
release gate. Every gate gold is also computed by evaluation, never by hand.

Usage:
    python3 finetuning/dataprep/gen_conditions.py
"""

import json
import random
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LOCAL = ROOT / ".local"

TRAIN_PATH = DATA / "conditions_train.jsonl"
EVAL_PATH = DATA / "conditions_eval.jsonl"
GATE_PATH = DATA / "conditions_basics_gate.jsonl"
SAMPLES_PATH = LOCAL / "conditions_samples.txt"

TRAIN_SEED = 20260929
EVAL_SEED = 47110002
GATE_SEED = 90210001

TRAIN_TARGET = 50000
EVAL_TARGET = 3000
GATE_TARGET = 300


def A(rng, seq):
    return rng.choice(seq)


def T(rng, split, templates, reserve=1, **kw):
    """Pick an instruction template. The last `reserve` templates are
    eval-only; train never draws them (mirrors gen_it_worker.py's T())."""
    reserve = min(reserve, len(templates) - 1) if len(templates) > 1 else 0
    pool = templates if split == "eval" else (templates[:-reserve] if reserve else templates)
    return A(rng, pool).format(**kw)


def NOUL(qkey, instr, true_text, false_text, gold_bool):
    return (qkey, "noul", instr, {"false": false_text, "true": true_text},
            "true" if gold_bool else "false")


def CHOICE(rng, qkey, instr, options, gold_key):
    keys = list(options.keys())
    rng.shuffle(keys)
    shuffled = {k: options[k] for k in keys}
    assert gold_key in shuffled, (gold_key, shuffled)
    return (qkey, "choice", instr, shuffled, gold_key)


def SCORE(qkey, instr, levels, gold_index):
    assert 0 <= gold_index < len(levels)
    return (qkey, "score", instr, list(levels), str(gold_index))


# ===========================================================================
# generic condition engine: field kinds, operators, evaluation, rendering
# ===========================================================================

TODAY_POOL = ["2026-09-29", "2026-01-15", "2026-12-01", "2026-06-10"]

NUM_WORD = {"lt": "is less than", "lte": "is at most", "gt": "is greater than",
            "gte": "is at least", "eq": "equals", "ne": "does not equal"}
NUM_PYOP = {"lt": "<", "lte": "<=", "gt": ">", "gte": ">=", "eq": "==", "ne": "!="}
NUM_JSOP = NUM_PYOP
NUM_GOOP = NUM_PYOP
NUM_SQLOP = {"lt": "<", "lte": "<=", "gt": ">", "gte": ">=", "eq": "=", "ne": "<>"}


def fmt_num(v):
    return str(v)


def eval_atom(atom, state):
    """Single source of truth for gold: evaluate one atom against `state`."""
    kind = atom["kind"]
    op = atom["op"]
    val = state[atom["field"]]
    if kind == "num":
        thr = atom["value"]
        if op == "between":
            lo, hi, lo_incl, hi_incl = thr
            left = val >= lo if lo_incl else val > lo
            right = val <= hi if hi_incl else val < hi
            return left and right
        if op == "lt":
            return val < thr
        if op == "lte":
            return val <= thr
        if op == "gt":
            return val > thr
        if op == "gte":
            return val >= thr
        if op == "eq":
            return val == thr
        if op == "ne":
            return val != thr
    if kind == "enum":
        thr = atom["value"]
        if op == "eq":
            return val == thr
        if op == "ne":
            return val != thr
        if op == "in":
            return val in thr
        if op == "not_in":
            return val not in thr
    if kind == "bool":
        return val is True if op == "is_true" else val is False
    if kind == "list":
        if op == "contains":
            return atom["value"] in val
        if op == "not_contains":
            return atom["value"] not in val
        if op.startswith("count_"):
            n = len(val)
            thr = atom["value"]
            sub = op[len("count_"):]
            return {"lt": n < thr, "lte": n <= thr, "gt": n > thr,
                    "gte": n >= thr, "eq": n == thr, "ne": n != thr}[sub]
    if kind == "nullable":
        return val is None if op == "is_null" else val is not None
    if kind == "date":
        f, t = date.fromisoformat(val), date.fromisoformat(state["today"])
        if op == "before":
            return f < t
        if op == "after":
            return f > t
        if op == "within_days":
            return 0 <= (f - t).days <= atom["value"]
        if op == "older_than":
            return (t - f).days > atom["value"]
    if kind == "text":
        kw = atom["value"]
        if op == "contains":
            return kw.lower() in val.lower()
        if op == "startswith":
            return val.lower().startswith(kw.lower())
        if op == "eq":
            return val == kw
        if op == "ne":
            return val != kw
    raise ValueError((kind, op))


def eval_node(node, state):
    tag = node[0]
    if tag == "atom":
        return eval_atom(node[1], state)
    if tag == "not":
        return not eval_node(node[1], state)
    if tag == "and":
        return eval_node(node[1], state) and eval_node(node[2], state)
    if tag == "or":
        return eval_node(node[1], state) or eval_node(node[2], state)
    raise ValueError(tag)


# ---- code-language support per atom (kind, op) -> tuple of usable langs -----

def atom_langs(atom):
    kind, op = atom["kind"], atom["op"]
    if kind == "num":
        return ("py", "js", "go", "sql")
    if kind == "enum":
        return ("py", "js", "go", "sql")
    if kind == "bool":
        return ("py", "js", "go", "sql")
    if kind == "list":
        if op in ("contains", "not_contains"):
            return ("py", "js")
        return ("py", "js", "go")  # count_*
    if kind == "nullable":
        return ("py", "js", "go", "sql")
    if kind == "date":
        if op in ("before", "after"):
            return ("py", "js", "go", "sql")
        return ("py",)  # within_days / older_than needs real date arithmetic
    if kind == "text":
        return ("py", "js", "go", "sql")
    return ("py",)


def node_langs(node):
    tag = node[0]
    if tag == "atom":
        return set(atom_langs(node[1]))
    if tag == "not":
        return node_langs(node[1])
    return node_langs(node[1]) & node_langs(node[2])


# ---- English fragment for a single atom (used by forms a/b) ---------------

def atom_english(atom):
    kind, op, field = atom["kind"], atom["op"], atom["field"]
    unit = atom.get("unit", "")
    if kind == "num":
        if op == "between":
            lo, hi, lo_incl, hi_incl = atom["value"]
            incl = "inclusive" if lo_incl and hi_incl else "exclusive"
            return f"{field} is between {lo} and {hi}{unit} ({incl})"
        return f"{field} {NUM_WORD[op]} {fmt_num(atom['value'])}{unit}"
    if kind == "enum":
        v = atom["value"]
        if op == "eq":
            return f'{field} is "{v}"'
        if op == "ne":
            return f'{field} is not "{v}"'
        if op == "in":
            return f"{field} is one of {list(v)}"
        return f"{field} is none of {list(v)}"
    if kind == "bool":
        return f"{field} is {'true' if op == 'is_true' else 'false'}"
    if kind == "list":
        if op == "contains":
            return f'{field} contains "{atom["value"]}"'
        if op == "not_contains":
            return f'{field} does not contain "{atom["value"]}"'
        sub = op[len("count_"):]
        return f"the number of items in {field} {NUM_WORD[sub]} {atom['value']}"
    if kind == "nullable":
        return f"{field} is {'missing' if op == 'is_null' else 'present'}"
    if kind == "date":
        if op == "before":
            return f"{field} is before today"
        if op == "after":
            return f"{field} is after today"
        if op == "within_days":
            return f"{field} is within {atom['value']} days from today"
        return f"{field} is more than {atom['value']} days ago"
    if kind == "text":
        v = atom["value"]
        if op == "contains":
            return f'{field} contains "{v}" (case-insensitive)'
        if op == "startswith":
            return f'{field} starts with "{v}" (case-insensitive)'
        if op == "eq":
            return f'{field} is exactly "{v}"'
        return f'{field} is not "{v}"'
    raise ValueError(kind)


def node_english(node):
    tag = node[0]
    if tag == "atom":
        return atom_english(node[1])
    if tag == "not":
        return f"NOT ({node_english(node[1])})"
    joiner = "AND" if tag == "and" else "OR"
    return f"({node_english(node[1])}) {joiner} ({node_english(node[2])})"


# ---- code fragment for a single atom in a given language -------------------

def _q(v):
    return v if isinstance(v, (int, float)) else f'"{v}"'


def atom_code(atom, lang):
    kind, op, field = atom["kind"], atom["op"], atom["field"]
    if kind == "num":
        if op == "between":
            lo, hi, lo_incl, hi_incl = atom["value"]
            if lang == "py":
                return f"{lo} {'<=' if lo_incl else '<'} {field} {'<=' if hi_incl else '<'} {hi}"
            if lang == "sql":
                return f"{field} BETWEEN {lo} AND {hi}" if lo_incl and hi_incl else \
                    f"({field} {'>=' if lo_incl else '>'} {lo} AND {field} {'<=' if hi_incl else '<'} {hi})"
            op1 = ">=" if lo_incl else ">"
            op2 = "<=" if hi_incl else "<"
            return f"({field} {op1} {lo} && {field} {op2} {hi})" if lang != "go" else \
                f"({field} {op1} {lo} && {field} {op2} {hi})"
        opmap = {"py": NUM_PYOP, "js": NUM_JSOP, "go": NUM_GOOP, "sql": NUM_SQLOP}[lang]
        return f"{field} {opmap[op]} {fmt_num(atom['value'])}"
    if kind == "enum":
        v = atom["value"]
        if op in ("eq", "ne"):
            eqop = {"py": "==", "js": "===", "go": "==", "sql": "="}[lang]
            neop = {"py": "!=", "js": "!==", "go": "!=", "sql": "<>"}[lang]
            quote = f"'{v}'" if lang == "sql" else f'"{v}"'
            return f"{field} {eqop if op == 'eq' else neop} {quote}"
        values = list(v)
        if lang == "py":
            return f"{field} {'in' if op == 'in' else 'not in'} {tuple(values)}"
        if lang == "sql":
            csv = ", ".join(f"'{x}'" for x in values)
            return f"{field} {'IN' if op == 'in' else 'NOT IN'} ({csv})"
        chain = " || ".join(f'{field} === "{x}"' for x in values) if lang == "js" else \
            " || ".join(f'{field} == "{x}"' for x in values)
        return chain if op == "in" else f"!({chain})"
    if kind == "bool":
        truthy = op == "is_true"
        if lang == "sql":
            return f"{field} = {'TRUE' if truthy else 'FALSE'}"
        return field if truthy else (f"not {field}" if lang == "py" else f"!{field}")
    if kind == "list":
        v = atom.get("value")
        if op in ("contains", "not_contains"):
            if lang == "py":
                expr = f'"{v}" in {field}'
            else:
                expr = f'{field}.includes("{v}")'
            return expr if op == "contains" else (f"not ({expr})" if lang == "py" else f"!({expr})")
        sub = op[len("count_"):]
        opmap = {"py": NUM_PYOP, "js": NUM_JSOP, "go": NUM_GOOP}[lang]
        lenexpr = {"py": f"len({field})", "js": f"{field}.length", "go": f"len({field})"}[lang]
        return f"{lenexpr} {opmap[sub]} {v}"
    if kind == "nullable":
        if op == "is_null":
            return {"py": f"{field} is None", "js": f"{field} == null",
                    "go": f"{field} == nil", "sql": f"{field} IS NULL"}[lang]
        return {"py": f"{field} is not None", "js": f"{field} != null",
                "go": f"{field} != nil", "sql": f"{field} IS NOT NULL"}[lang]
    if kind == "date":
        today = "today"  # rendered as a variable/column holding the given today
        if op == "before":
            return {"py": f'{field} < "{{today}}"', "js": f'{field} < "{{today}}"',
                    "go": f'{field} < "{{today}}"', "sql": f"{field} < '{{today}}'"}[lang]
        if op == "after":
            return {"py": f'{field} > "{{today}}"', "js": f'{field} > "{{today}}"',
                    "go": f'{field} > "{{today}}"', "sql": f"{field} > '{{today}}'"}[lang]
        n = atom["value"]
        if op == "within_days":
            return (f"0 <= (date.fromisoformat({field}) - date.fromisoformat(\"{{today}}\")).days <= {n}")
        return (f"(date.fromisoformat(\"{{today}}\") - date.fromisoformat({field})).days > {n}")
    if kind == "text":
        v = atom["value"]
        if op == "contains":
            return {"py": f'"{v.lower()}" in {field}.lower()',
                    "js": f'{field}.toLowerCase().includes("{v.lower()}")',
                    "go": f'strings.Contains(strings.ToLower({field}), "{v.lower()}")',
                    "sql": f"LOWER({field}) LIKE '%{v.lower()}%'"}[lang]
        if op == "startswith":
            return {"py": f'{field}.lower().startswith("{v.lower()}")',
                    "js": f'{field}.toLowerCase().startsWith("{v.lower()}")',
                    "go": f'strings.HasPrefix(strings.ToLower({field}), "{v.lower()}")',
                    "sql": f"LOWER({field}) LIKE '{v.lower()}%'"}[lang]
        eqop = {"py": "==", "js": "===", "go": "==", "sql": "="}[lang]
        neop = {"py": "!=", "js": "!==", "go": "!=", "sql": "<>"}[lang]
        quote = f"'{v}'" if lang == "sql" else f'"{v}"'
        return f"{field} {eqop if op == 'eq' else neop} {quote}"
    raise ValueError(kind)


def node_code(node, lang):
    tag = node[0]
    if tag == "atom":
        return atom_code(node[1], lang)
    if tag == "not":
        inner = node_code(node[1], lang)
        return {"py": f"not ({inner})", "js": f"!({inner})", "go": f"!({inner})",
                "sql": f"NOT ({inner})"}[lang]
    inner1, inner2 = node_code(node[1], lang), node_code(node[2], lang)
    joiner = {"py": {"and": "and", "or": "or"}, "js": {"and": "&&", "or": "||"},
              "go": {"and": "&&", "or": "||"}, "sql": {"and": "AND", "or": "OR"}}[lang][tag]
    return f"({inner1}) {joiner} ({inner2})"


def render_code_block(node, lang, today=None):
    body = node_code(node, lang)
    if today is not None:
        body = body.replace("{today}", today)
    if lang == "py":
        return f"if {body}:\n    ...", "Python"
    if lang == "js":
        return f"if ({body}) {{\n  ...\n}}", "JavaScript"
    if lang == "go":
        return f"if {body} {{\n\t...\n}}", "Go"
    return f"SELECT * FROM rows WHERE {body};", "SQL"


# ===========================================================================
# sampling helpers (boundary-heavy; correctness always comes from eval_atom)
# ===========================================================================

def sample_num(rng, lo, hi, thresholds, decimals, split):
    """Boundary-heavy numeric sampler. ~40% of draws land at
    threshold-step/threshold/threshold+step for a randomly chosen threshold;
    the rest vary magnitude/digit-length across the full range. In eval,
    ~20% of the "rest" draws are pushed into a wider range never seen in
    train, to test out-of-distribution magnitudes/digit-lengths honestly."""
    step = (1 / (10 ** decimals)) if decimals else 1
    if thresholds and rng.random() < 0.40:
        t = rng.choice(thresholds)
        v = t + rng.choice([-1, 0, 1]) * step
    else:
        lo2, hi2 = lo, hi
        if split == "eval" and rng.random() < 0.20:
            span = hi - lo
            hi2 = hi + max(span, 1)
        v = rng.uniform(lo2, hi2) if decimals else rng.randint(int(lo2), int(hi2))
    if decimals:
        v = round(v, decimals)
    else:
        v = int(round(v))
    return max(min(v, hi if (split != "eval") else hi * 3 + 1), lo - abs(lo) - 1 if lo < 0 else min(lo, v))


def sample_num_simple(rng, lo, hi, decimals):
    v = rng.uniform(lo, hi) if decimals else rng.randint(int(lo), int(hi))
    return round(v, decimals) if decimals else int(round(v))


def sample_day_offset(rng, anchor, split):
    """Boundary-heavy day-offset sampler for date fields (anchor = 0 for
    before/after-today rules, or N for within/older-than-N-days rules)."""
    if rng.random() < 0.40:
        off = anchor + rng.choice([-1, 0, 1])
    else:
        span = 400 if split != "eval" else 900
        off = rng.randint(-span, span)
    return off


def iso(base_str, offset_days):
    return (date.fromisoformat(base_str) + timedelta(days=offset_days)).isoformat()


# ===========================================================================
# domain catalog
# ===========================================================================

def NUM(key, unit="", lo=0, hi=100, decimals=0, thresholds=None):
    return dict(key=key, kind="num", unit=unit, lo=lo, hi=hi, decimals=decimals,
                thresholds=thresholds or [])


def ENUM(key, values):
    return dict(key=key, kind="enum", values=list(values))


def BOOL(key):
    return dict(key=key, kind="bool")


def LIST(key, pool, lenlo=0, lenhi=4):
    return dict(key=key, kind="list", pool=list(pool), lenlo=lenlo, lenhi=lenhi)


def NULLABLE(key, pool):
    return dict(key=key, kind="nullable", pool=list(pool))


def DATEF(key):
    return dict(key=key, kind="date")


def TEXTF(key, with_kw, without_kw):
    return dict(key=key, kind="text", with_kw=list(with_kw), without_kw=list(without_kw))


def BAND(key, unit, lo, hi, decimals, thresholds, labels, descs):
    assert len(labels) == len(thresholds) + 1 == len(descs)
    return dict(key=key, unit=unit, lo=lo, hi=hi, decimals=decimals,
                thresholds=thresholds, labels=labels, descs=descs)


def RULE(field, op, value, subject, true_desc, false_desc, rule_def, policy, unit=""):
    return dict(field=field, op=op, value=value, subject=subject, true_desc=true_desc,
                false_desc=false_desc, rule_def=rule_def, policy=policy, unit=unit)


def DOMAIN(id_, entity, band, rule1, rule2, fields, filler):
    all_fields = {f["key"]: f for f in fields}
    return dict(id=id_, entity=entity, band=band, rule1=rule1, rule2=rule2,
                fields=all_fields, filler=filler)


DOMAINS = [
    DOMAIN("ecommerce_orders", "order",
           BAND("order_total_usd", " USD", 0, 500, 0, [25, 50, 150],
                ["standard_shipping", "reduced_fee_shipping", "free_shipping", "priority_free_shipping"],
                ["Standard shipping fee applies.", "Reduced shipping fee applies.",
                 "Free shipping applies.", "Free priority shipping applies."]),
           RULE("order_total_usd", "gte", 50, "the order", "qualifies for free shipping",
                "does not qualify for free shipping",
                "free shipping applies when the order total is at least $50",
                "Orders totaling $50 or more ship free.", unit=" USD"),
           RULE("payment_status", "eq", "paid", "the order", "is ready to fulfill",
                "is not ready to fulfill", "an order is ready to fulfill once payment_status is \"paid\"",
                "Orders are only fulfilled once payment is confirmed as paid."),
           [NUM("order_total_usd", " USD", 0, 500), ENUM("payment_status", ["pending", "paid", "failed", "refunded"])],
           [NUM("items_count", " items", 1, 20), ENUM("warehouse_region", ["east", "west", "central"])]),
    DOMAIN("retail_pricing", "product",
           BAND("discount_pct", "%", 0, 80, 0, [10, 25, 50],
                ["no_discount", "minor_discount", "major_discount", "clearance"],
                ["No discount tier.", "Minor discount tier.", "Major discount tier.", "Clearance pricing."]),
           RULE("discount_pct", "gte", 25, "the product", "is on major discount",
                "is not on major discount",
                "a major discount is any discount of 25% or more",
                "Products discounted 25% or more are flagged as a major discount."),
           RULE("is_clearance", "is_true", None, "the product", "is clearance stock",
                "is regular stock", "clearance stock means the is_clearance flag is set",
                "Clearance items are marked with the is_clearance flag."),
           [NUM("discount_pct", "%", 0, 80), BOOL("is_clearance")],
           [NUM("price_usd", " USD", 1, 500), ENUM("category", ["apparel", "electronics", "home"])]),
    DOMAIN("banking_accounts", "account",
           BAND("balance_usd", " USD", -500, 50000, 0, [0, 100, 10000],
                ["overdrawn", "low_balance", "healthy", "high_value"],
                ["Overdrawn.", "Low balance.", "Healthy balance.", "High-value account."]),
           RULE("balance_usd", "lt", 0, "the account", "is overdrawn",
                "is not overdrawn", "an account is overdrawn when its balance is below $0",
                "Any account with a balance below zero is overdrawn.", unit=" USD"),
           RULE("kyc_verified", "is_true", None, "the account", "has passed KYC verification",
                "has not passed KYC verification",
                "an account has passed KYC once kyc_verified is true",
                "New accounts must pass KYC verification before large transfers are allowed."),
           [NUM("balance_usd", " USD", -500, 50000), BOOL("kyc_verified")],
           [NUM("overdraft_limit_usd", " USD", 0, 1000), ENUM("account_type", ["checking", "savings"])]),
    DOMAIN("banking_loans", "loan application",
           BAND("credit_score", "", 300, 850, 0, [580, 670, 740],
                ["poor", "fair", "good", "excellent"],
                ["Poor credit band.", "Fair credit band.", "Good credit band.", "Excellent credit band."]),
           RULE("credit_score", "gte", 670, "the applicant", "qualifies for standard rates",
                "does not qualify for standard rates",
                "standard rates require a credit score of at least 670",
                "Applicants need a credit score of 670 or higher for standard rates."),
           RULE("has_collateral", "is_true", None, "the loan", "is secured by collateral",
                "is unsecured", "a secured loan is one where has_collateral is true",
                "A loan is considered secured only when collateral is pledged."),
           [NUM("credit_score", "", 300, 850), BOOL("has_collateral")],
           [NUM("principal_usd", " USD", 1000, 500000), ENUM("loan_type", ["personal", "auto", "mortgage"])]),
    DOMAIN("payments_fraud", "transaction",
           BAND("fraud_risk_score", "", 0, 100, 0, [30, 60, 85],
                ["low_risk", "medium_risk", "high_risk", "critical_risk"],
                ["Low fraud risk.", "Medium fraud risk.", "High fraud risk.", "Critical fraud risk."]),
           RULE("fraud_risk_score", "gte", 60, "the transaction", "should be held for review",
                "can proceed automatically",
                "a transaction is held for review once its fraud risk score is 60 or higher",
                "Transactions scoring 60 or higher on fraud risk are held for manual review."),
           RULE("is_first_time_merchant", "is_true", None, "the transaction", "is with a first-time merchant",
                "is with a known merchant",
                "is_first_time_merchant marks a transaction going to a merchant never paid before",
                "First-time merchants get extra scrutiny on new transactions."),
           [NUM("fraud_risk_score", "", 0, 100), BOOL("is_first_time_merchant")],
           [NUM("amount_usd", " USD", 1, 20000), ENUM("channel", ["card", "ach", "wallet"])]),
    DOMAIN("insurance_claims", "claim",
           BAND("claim_amount_usd", " USD", 0, 100000, 0, [1000, 10000, 50000],
                ["minor_claim", "standard_claim", "major_claim", "catastrophic_claim"],
                ["Minor claim.", "Standard claim.", "Major claim.", "Catastrophic claim."]),
           RULE("claim_amount_usd", "gt", 50000, "the claim", "requires senior adjuster sign-off",
                "does not require senior adjuster sign-off",
                "claims over $50,000 require senior adjuster sign-off",
                "Any claim exceeding $50,000 must be signed off by a senior adjuster.", unit=" USD"),
           RULE("claim_status", "eq", "approved", "the claim", "has been approved",
                "has not been approved", "a claim is approved once claim_status equals \"approved\"",
                "Payouts are issued only once the claim status is approved."),
           [NUM("claim_amount_usd", " USD", 0, 100000), ENUM("claim_status", ["submitted", "under_review", "approved", "denied"])],
           [DATEF("incident_date"), BOOL("has_police_report")]),
    DOMAIN("insurance_policies", "policy",
           BAND("premium_usd", " USD", 0, 5000, 0, [500, 1500, 3000],
                ["basic_tier", "standard_tier", "premium_tier", "elite_tier"],
                ["Basic tier.", "Standard tier.", "Premium tier.", "Elite tier."]),
           RULE("premium_usd", "gte", 3000, "the policy", "is in the elite tier",
                "is not in the elite tier", "the elite tier requires a premium of $3,000 or more",
                "Policies with a premium of $3,000 or more are elite tier.", unit=" USD"),
           RULE("renewal_date", "within_days", 30, "the policy", "is due for renewal soon",
                "is not due for renewal soon",
                "a policy is due for renewal soon if its renewal date is within 30 days",
                "Policyholders are notified once their renewal date is within 30 days."),
           [NUM("premium_usd", " USD", 0, 5000), DATEF("renewal_date")],
           [ENUM("policy_type", ["auto", "home", "life"]), BOOL("autopay_enabled")]),
    DOMAIN("healthcare_appointments", "appointment",
           BAND("wait_time_minutes", " min", 0, 240, 0, [15, 30, 60],
                ["on_time", "minor_delay", "significant_delay", "severe_delay"],
                ["On time.", "Minor delay.", "Significant delay.", "Severe delay."]),
           RULE("wait_time_minutes", "gt", 60, "the appointment", "has a severe delay",
                "does not have a severe delay",
                "a severe delay is a wait time greater than 60 minutes",
                "Wait times over 60 minutes are logged as severe delays.", unit=" min"),
           RULE("insurance_verified", "is_true", None, "the appointment", "has insurance verified",
                "does not have insurance verified",
                "insurance is verified once insurance_verified is true",
                "Appointments require insurance verification before check-in."),
           [NUM("wait_time_minutes", " min", 0, 240), BOOL("insurance_verified")],
           [DATEF("appointment_date"), ENUM("department", ["cardiology", "pediatrics", "radiology"])]),
    DOMAIN("healthcare_eligibility", "patient",
           BAND("age_years", " years", 0, 100, 0, [18, 26, 65],
                ["minor", "young_adult", "adult", "senior"],
                ["Minor.", "Young adult.", "Adult.", "Senior."]),
           RULE("age_years", "gte", 18, "the patient", "is an adult",
                "is a minor", "adult means age >= 18",
                "Patients aged 18 or older are treated as adults for consent purposes.", unit=" years"),
           RULE("insurance_id", "is_not_null", None, "the patient", "has insurance on file",
                "is uninsured", "a patient is uninsured when insurance_id is missing",
                "Patients with no insurance_id on file are billed as self-pay."),
           [NUM("age_years", " years", 0, 100), NULLABLE("insurance_id", ["INS-1001", "INS-2044", "INS-3399"])],
           [ENUM("plan_tier", ["bronze", "silver", "gold"]), BOOL("has_referral")]),
    DOMAIN("pharmacy_stock", "medication",
           BAND("stock_units", " units", 0, 1000, 0, [0, 10, 200],
                ["out_of_stock", "low_stock", "adequate_stock", "overstocked"],
                ["Out of stock.", "Low stock.", "Adequate stock.", "Overstocked."]),
           RULE("stock_units", "gt", 0, "the medication", "is in stock",
                "is out of stock", "in stock means stock_units is greater than 0",
                "A medication is in stock only while stock_units is above zero.", unit=" units"),
           RULE("expiry_date", "before", None, "the medication", "has expired",
                "has not expired", "expired means the expiry_date is before today",
                "Expired stock (expiry_date before today) must be pulled from the shelf."),
           [NUM("stock_units", " units", 0, 1000), DATEF("expiry_date")],
           [ENUM("drug_class", ["antibiotic", "analgesic", "other"]), BOOL("requires_prescription")]),
    DOMAIN("hr_employees", "employee",
           BAND("tenure_months", " months", 0, 300, 0, [3, 12, 60],
                ["probation", "junior", "established", "veteran"],
                ["On probation.", "Junior tenure.", "Established tenure.", "Veteran tenure."]),
           RULE("tenure_months", "lt", 3, "the employee", "is on probation",
                "is past probation", "probation lasts while tenure_months is under 3",
                "New employees are on probation for their first 3 months.", unit=" months"),
           RULE("is_manager", "is_true", None, "the employee", "is a manager",
                "is not a manager", "is_manager marks people-management responsibility",
                "Only employees flagged is_manager can approve leave requests."),
           [NUM("tenure_months", " months", 0, 300), BOOL("is_manager")],
           [ENUM("department", ["eng", "sales", "hr", "finance"]), NUM("salary_usd", " USD", 30000, 250000)]),
    DOMAIN("hr_leave", "leave request",
           BAND("days_requested", " days", 1, 30, 0, [3, 10, 20],
                ["short_leave", "medium_leave", "long_leave", "extended_leave"],
                ["Short leave.", "Medium leave.", "Long leave.", "Extended leave."]),
           RULE("days_requested", "gt", 10, "the leave request", "needs HR approval",
                "does not need HR approval",
                "requests over 10 days require HR approval, not just a manager",
                "Leave requests longer than 10 days must be approved by HR, not just the manager.",
                unit=" days"),
           RULE("leave_type", "eq", "sick", "the leave request", "is sick leave",
                "is not sick leave", "sick leave is any request with leave_type equal to \"sick\"",
                "Sick leave does not count against the annual vacation allowance."),
           [NUM("days_requested", " days", 1, 30), ENUM("leave_type", ["vacation", "sick", "unpaid", "bereavement"])],
           [DATEF("start_date"), BOOL("manager_approved")]),
    DOMAIN("hr_payroll", "paycheck",
           BAND("gross_pay_usd", " USD", 500, 20000, 0, [2000, 5000, 10000],
                ["entry_band", "mid_band", "senior_band", "executive_band"],
                ["Entry pay band.", "Mid pay band.", "Senior pay band.", "Executive pay band."]),
           RULE("gross_pay_usd", "gte", 10000, "the paycheck", "is in the executive pay band",
                "is not in the executive pay band",
                "the executive band starts at $10,000 gross pay",
                "Gross pay of $10,000 or more is the executive pay band.", unit=" USD"),
           RULE("overtime_eligible", "is_true", None, "the employee", "is overtime-eligible",
                "is not overtime-eligible",
                "overtime_eligible marks hourly, non-exempt employees",
                "Only overtime_eligible employees are paid time-and-a-half for overtime."),
           [NUM("gross_pay_usd", " USD", 500, 20000), BOOL("overtime_eligible")],
           [ENUM("pay_frequency", ["weekly", "biweekly", "monthly"]), NUM("tax_withheld_usd", " USD", 0, 5000)]),
    DOMAIN("recruiting", "candidate",
           BAND("years_experience", " years", 0, 30, 0, [2, 5, 10],
                ["entry_level", "mid_level", "senior_level", "principal_level"],
                ["Entry level.", "Mid level.", "Senior level.", "Principal level."]),
           RULE("years_experience", "gte", 5, "the candidate", "meets the senior bar",
                "does not meet the senior bar",
                "the senior bar requires 5 or more years of experience",
                "Candidates need 5+ years of experience to be considered senior.", unit=" years"),
           RULE("stage", "eq", "hired", "the candidate", "has been hired",
                "has not been hired", "hired means stage equals \"hired\"",
                "Only candidates whose stage is \"hired\" receive an offer letter."),
           [NUM("years_experience", " years", 0, 30), ENUM("stage", ["applied", "interview", "offer", "hired", "rejected"])],
           [NUM("expected_salary_usd", " USD", 40000, 300000), LIST("skills", ["python", "go", "sql", "react", "aws"])]),
    DOMAIN("education_students", "student",
           BAND("gpa", "", 0.0, 4.0, 2, [2.0, 3.0, 3.5],
                ["at_risk", "satisfactory", "good_standing", "honors"],
                ["At academic risk.", "Satisfactory standing.", "Good standing.", "Honors standing."]),
           RULE("gpa", "lt", 2.0, "the student", "is on academic probation",
                "is not on academic probation",
                "academic probation applies when gpa is below 2.0",
                "Students with a GPA below 2.0 are placed on academic probation."),
           RULE("financial_aid_id", "is_null", None, "the student", "has no financial aid on file",
                "has financial aid on file",
                "no financial aid on file means financial_aid_id is missing",
                "Students with no financial_aid_id are billed the full tuition rate."),
           [NUM("gpa", "", 0.0, 4.0, 2), NULLABLE("financial_aid_id", ["AID-101", "AID-202", "AID-303"])],
           [ENUM("major", ["cs", "biology", "history"]), NUM("credits_completed", "", 0, 160)]),
    DOMAIN("education_enrollment", "enrollment",
           BAND("attendance_pct", "%", 0, 100, 0, [60, 80, 95],
                ["chronic_absence", "at_risk_attendance", "good_attendance", "excellent_attendance"],
                ["Chronic absence.", "At-risk attendance.", "Good attendance.", "Excellent attendance."]),
           RULE("attendance_pct", "lt", 60, "the enrollment", "is flagged for chronic absence",
                "is not flagged for chronic absence",
                "chronic absence is attendance below 60%",
                "Attendance below 60% triggers a chronic-absence flag.", unit="%"),
           RULE("tuition_paid", "is_true", None, "the enrollment", "has tuition paid",
                "has unpaid tuition", "tuition_paid tracks whether the balance is settled",
                "Course access is suspended while tuition_paid is false."),
           [NUM("attendance_pct", "%", 0, 100), BOOL("tuition_paid")],
           [ENUM("term", ["fall", "spring", "summer"]), NUM("credit_hours", "", 1, 21)]),
    DOMAIN("travel_flights", "flight booking",
           BAND("days_until_departure", " days", 0, 365, 0, [1, 7, 30],
                ["imminent", "departing_soon", "upcoming", "far_out"],
                ["Departing imminently.", "Departing soon.", "Upcoming.", "Far out."]),
           RULE("days_until_departure", "lte", 1, "the booking", "departs within a day",
                "does not depart within a day",
                "\"departs within a day\" means days_until_departure is 1 or fewer",
                "Bookings departing within a day trigger a check-in reminder.", unit=" days"),
           RULE("checkin_deadline", "within_days", 1, "the booking", "has an imminent check-in deadline",
                "does not have an imminent check-in deadline",
                "an imminent check-in deadline is one within 1 day of today",
                "Passengers are alerted when the check-in deadline is within 1 day."),
           [NUM("days_until_departure", " days", 0, 365), DATEF("checkin_deadline")],
           [ENUM("cabin_class", ["economy", "premium", "business"]), BOOL("has_checked_bag")]),
    DOMAIN("hotels", "reservation",
           BAND("nights_booked", " nights", 1, 30, 0, [3, 7, 14],
                ["short_stay", "standard_stay", "extended_stay", "long_term_stay"],
                ["Short stay.", "Standard stay.", "Extended stay.", "Long-term stay."]),
           RULE("nights_booked", "gte", 14, "the reservation", "qualifies for a long-stay rate",
                "does not qualify for a long-stay rate",
                "the long-stay rate applies at 14 nights or more",
                "Stays of 14 nights or more get the long-stay discount rate.", unit=" nights"),
           RULE("cancellation_deadline", "before", None, "the reservation", "is past its free-cancellation window",
                "is still within its free-cancellation window",
                "past the cancellation window means cancellation_deadline is before today",
                "Cancelling after the cancellation_deadline forfeits the first night's cost."),
           [NUM("nights_booked", " nights", 1, 30), DATEF("cancellation_deadline")],
           [ENUM("room_type", ["standard", "suite", "deluxe"]), BOOL("breakfast_included")]),
    DOMAIN("logistics_shipping", "shipment",
           BAND("transit_days", " days", 0, 30, 0, [2, 5, 10],
                ["express", "standard", "economy", "delayed"],
                ["Express transit.", "Standard transit.", "Economy transit.", "Delayed transit."]),
           RULE("transit_days", "gt", 10, "the shipment", "is delayed",
                "is not delayed", "a shipment is delayed once transit_days exceeds 10",
                "Shipments taking more than 10 transit days are flagged delayed.", unit=" days"),
           RULE("handling_flags", "contains", "fragile", "the shipment", "is marked fragile",
                "is not marked fragile", "fragile handling means \"fragile\" is in handling_flags",
                "Fragile shipments require extra packing and a signature on delivery."),
           [NUM("transit_days", " days", 0, 30), LIST("handling_flags", ["fragile", "hazmat", "oversized", "signature_required"])],
           [ENUM("carrier", ["fedex", "ups", "dhl"]), NUM("weight_kg", " kg", 0, 100)]),
    DOMAIN("warehousing", "pallet",
           BAND("capacity_used_pct", "%", 0, 150, 0, [70, 90, 100],
                ["low_utilization", "normal_utilization", "near_capacity", "overcapacity"],
                ["Low utilization.", "Normal utilization.", "Near capacity.", "Overcapacity."]),
           RULE("capacity_used_pct", "gt", 100, "the warehouse zone", "is overcapacity",
                "is not overcapacity", "overcapacity means capacity_used_pct is above 100%",
                "Any zone above 100% capacity used must stop accepting new pallets.", unit="%"),
           RULE("hazmat_flag", "is_true", None, "the pallet", "is hazmat",
                "is not hazmat", "hazmat_flag marks pallets requiring hazardous-materials handling",
                "Hazmat pallets must be stored in the designated hazmat zone only."),
           [NUM("capacity_used_pct", "%", 0, 150), BOOL("hazmat_flag")],
           [ENUM("zone", ["A", "B", "C"]), NUM("item_count", " items", 1, 500)]),
    DOMAIN("manufacturing_qa", "production batch",
           BAND("defect_rate_pct", "%", 0.0, 20.0, 2, [0.5, 2.0, 5.0],
                ["excellent", "acceptable", "marginal", "reject"],
                ["Excellent quality.", "Acceptable quality.", "Marginal quality.", "Reject batch."]),
           RULE("defect_rate_pct", "gt", 5.0, "the batch", "must be rejected",
                "does not have to be rejected",
                "a batch is rejected once its defect rate exceeds 5%",
                "Batches with a defect rate over 5% are automatically rejected.", unit="%"),
           RULE("passed_inspection", "is_true", None, "the batch", "passed final inspection",
                "failed final inspection",
                "passed_inspection is the final QA sign-off flag",
                "Only batches with passed_inspection true may ship."),
           [NUM("defect_rate_pct", "%", 0.0, 20.0, 2), BOOL("passed_inspection")],
           [ENUM("production_line", ["L1", "L2", "L3"]), NUM("units_produced", " units", 100, 20000)]),
    DOMAIN("energy_utilities", "utility bill",
           BAND("usage_kwh", " kWh", 0, 5000, 0, [300, 800, 2000],
                ["low_usage", "typical_usage", "high_usage", "excessive_usage"],
                ["Low usage.", "Typical usage.", "High usage.", "Excessive usage."]),
           RULE("usage_kwh", "gt", 2000, "the bill", "is flagged excessive usage",
                "is not flagged excessive usage",
                "excessive usage is above 2000 kWh in the billing period",
                "Usage over 2000 kWh triggers an excessive-usage notice.", unit=" kWh"),
           RULE("autopay_enabled", "is_true", None, "the account", "is on autopay",
                "is not on autopay", "autopay_enabled marks automatic bill payment",
                "Accounts on autopay never incur a late fee."),
           [NUM("usage_kwh", " kWh", 0, 5000), BOOL("autopay_enabled")],
           [ENUM("rate_plan", ["fixed", "variable"]), NUM("amount_due_usd", " USD", 10, 2000)]),
    DOMAIN("telecom_plans", "subscriber",
           BAND("data_used_gb", " GB", 0, 100, 0, [5, 20, 50],
                ["light_user", "moderate_user", "heavy_user", "over_limit"],
                ["Light user.", "Moderate user.", "Heavy user.", "Over the data limit."]),
           RULE("data_used_gb", "gt", 50, "the subscriber", "is over the data limit",
                "is not over the data limit",
                "over the limit means data_used_gb exceeds 50",
                "Usage above 50 GB triggers overage charges.", unit=" GB"),
           RULE("contract_end_date", "within_days", 30, "the subscriber", "has a contract ending soon",
                "does not have a contract ending soon",
                "ending soon means contract_end_date is within 30 days of today",
                "Subscribers get a renewal offer once contract_end_date is within 30 days."),
           [NUM("data_used_gb", " GB", 0, 100), DATEF("contract_end_date")],
           [ENUM("plan_tier", ["basic", "standard", "unlimited"]), BOOL("roaming_enabled")]),
    DOMAIN("real_estate_listings", "listing",
           BAND("days_on_market", " days", 0, 365, 0, [14, 45, 90],
                ["fresh_listing", "active_listing", "stale_listing", "long_stale_listing"],
                ["Fresh listing.", "Active listing.", "Stale listing.", "Long-stale listing."]),
           RULE("days_on_market", "gt", 90, "the listing", "is long-stale",
                "is not long-stale", "long-stale means more than 90 days on market",
                "Listings on the market more than 90 days are flagged long-stale.", unit=" days"),
           RULE("price_reduced", "is_true", None, "the listing", "has had a price reduction",
                "has not had a price reduction",
                "price_reduced marks any listing whose price has been lowered since listing",
                "Price-reduced listings are highlighted in the weekly buyer digest."),
           [NUM("days_on_market", " days", 0, 365), BOOL("price_reduced")],
           [ENUM("property_type", ["condo", "house", "townhome"]), NUM("list_price_usd", " USD", 50000, 3000000)]),
    DOMAIN("real_estate_leases", "lease",
           BAND("days_until_expiry", " days", 0, 730, 0, [30, 90, 180],
                ["expiring_now", "expiring_soon", "mid_term", "long_term"],
                ["Expiring now.", "Expiring soon.", "Mid-term.", "Long-term."]),
           RULE("days_until_expiry", "lte", 30, "the lease", "is expiring soon",
                "is not expiring soon", "expiring soon means 30 or fewer days until expiry",
                "Tenants are notified once their lease has 30 or fewer days remaining.", unit=" days"),
           RULE("security_deposit_ref", "is_not_null", None, "the lease", "has a security deposit on file",
                "has no security deposit on file",
                "a deposit on file means security_deposit_ref is not missing",
                "Move-in cannot be finalized without a security_deposit_ref on file."),
           [NUM("days_until_expiry", " days", 0, 730), NULLABLE("security_deposit_ref", ["DEP-1", "DEP-2", "DEP-3"])],
           [ENUM("lease_type", ["month_to_month", "fixed_term"]), NUM("monthly_rent_usd", " USD", 500, 10000)]),
    DOMAIN("saas_subscriptions", "subscription",
           BAND("usage_pct_of_limit", "%", 0, 150, 0, [70, 90, 100],
                ["low_usage", "normal_usage", "near_limit", "over_limit"],
                ["Low usage.", "Normal usage.", "Near limit.", "Over limit."]),
           RULE("usage_pct_of_limit", "gt", 100, "the subscription", "is over its usage limit",
                "is not over its usage limit",
                "over limit means usage_pct_of_limit exceeds 100%",
                "Accounts over 100% of their usage limit are throttled.", unit="%"),
           RULE("plan_status", "eq", "past_due", "the subscription", "is past due",
                "is not past due", "past due means plan_status equals \"past_due\"",
                "Past-due subscriptions lose access to premium features."),
           [NUM("usage_pct_of_limit", "%", 0, 150), ENUM("plan_status", ["trial", "active", "past_due", "cancelled"])],
           [ENUM("tier", ["free", "pro", "enterprise"]), BOOL("auto_renew")]),
    DOMAIN("saas_billing", "invoice",
           BAND("days_overdue", " days", 0, 120, 0, [0, 7, 30],
                ["current", "recently_due", "overdue", "severely_overdue"],
                ["Current.", "Recently due.", "Overdue.", "Severely overdue."]),
           RULE("days_overdue", "gt", 30, "the invoice", "is severely overdue",
                "is not severely overdue",
                "severely overdue means more than 30 days overdue",
                "Invoices more than 30 days overdue are sent to collections.", unit=" days"),
           RULE("payment_method_on_file", "is_true", None, "the account", "has a payment method on file",
                "has no payment method on file",
                "payment_method_on_file tracks whether a card or bank account is saved",
                "Auto-billing requires a payment_method_on_file."),
           [NUM("days_overdue", " days", 0, 120), BOOL("payment_method_on_file")],
           [NUM("amount_due_usd", " USD", 5, 5000), ENUM("currency", ["USD", "EUR"])]),
    DOMAIN("identity_access", "user account",
           BAND("failed_login_attempts", "", 0, 20, 0, [3, 5, 10],
                ["normal", "elevated", "high_risk", "locked_out"],
                ["Normal.", "Elevated risk.", "High risk.", "Locked out."]),
           RULE("failed_login_attempts", "gte", 10, "the account", "is locked out",
                "is not locked out",
                "an account is locked out at 10 or more failed login attempts",
                "Accounts are locked after 10 consecutive failed login attempts."),
           RULE("mfa_enabled", "is_true", None, "the account", "has MFA enabled",
                "does not have MFA enabled", "mfa_enabled marks two-factor authentication being active",
                "Admin-role accounts must have mfa_enabled to sign in."),
           [NUM("failed_login_attempts", "", 0, 20), BOOL("mfa_enabled")],
           [ENUM("role", ["viewer", "editor", "admin"]), DATEF("last_login_date")]),
    DOMAIN("devops_sre", "host",
           BAND("cpu_pct", "%", 0, 100, 0, [70, 90, 95],
                ["normal", "elevated", "warning", "critical"],
                ["Normal load.", "Elevated load.", "Warning level.", "Critical level."]),
           RULE("cpu_pct", "gt", 90, "the host", "is at a critical CPU level",
                "is not at a critical CPU level",
                "critical means CPU usage above 90%",
                "Hosts above 90% CPU trigger a page.", unit="%"),
           RULE("alert_acknowledged", "is_true", None, "the alert", "has been acknowledged",
                "has not been acknowledged",
                "alert_acknowledged tracks whether on-call has acked the page",
                "Unacknowledged alerts re-page after 5 minutes."),
           [NUM("cpu_pct", "%", 0, 100), BOOL("alert_acknowledged")],
           [ENUM("region", ["us-east", "us-west", "eu"]), NUM("disk_pct", "%", 0, 100)]),
    DOMAIN("cicd_builds", "build",
           BAND("test_coverage_pct", "%", 0, 100, 0, [60, 80, 90],
                ["poor_coverage", "acceptable_coverage", "good_coverage", "excellent_coverage"],
                ["Poor coverage.", "Acceptable coverage.", "Good coverage.", "Excellent coverage."]),
           RULE("test_coverage_pct", "lt", 60, "the build", "fails the coverage gate",
                "passes the coverage gate", "the coverage gate requires at least 60% coverage",
                "Builds under 60% coverage fail the coverage gate.", unit="%"),
           RULE("status", "eq", "failed", "the build", "has failed",
                "has not failed", "failed means status equals \"failed\"",
                "A failed build blocks the merge until it is fixed."),
           [NUM("test_coverage_pct", "%", 0, 100), ENUM("status", ["queued", "running", "passed", "failed"])],
           [ENUM("branch", ["main", "develop", "feature"]), NUM("duration_seconds", "", 5, 3600)]),
    DOMAIN("git_prs", "pull request",
           BAND("lines_changed", " lines", 0, 5000, 0, [50, 300, 1000],
                ["trivial", "small", "large", "massive"],
                ["Trivial change.", "Small change.", "Large change.", "Massive change."]),
           RULE("lines_changed", "gt", 1000, "the pull request", "is a massive change",
                "is not a massive change",
                "a massive PR changes more than 1000 lines",
                "PRs over 1000 changed lines require two senior reviewers.", unit=" lines"),
           RULE("ci_passing", "is_true", None, "the pull request", "has CI passing",
                "does not have CI passing", "ci_passing reflects the latest CI run's result",
                "PRs cannot be merged unless ci_passing is true."),
           [NUM("lines_changed", " lines", 0, 5000), BOOL("ci_passing")],
           [ENUM("base_branch", ["main", "develop"]), BOOL("has_conflicts")]),
    DOMAIN("databases", "database instance",
           BAND("replication_lag_seconds", " s", 0, 600, 0, [5, 30, 120],
                ["healthy", "minor_lag", "significant_lag", "critical_lag"],
                ["Healthy replication.", "Minor lag.", "Significant lag.", "Critical lag."]),
           RULE("replication_lag_seconds", "gt", 120, "the replica", "has critical replication lag",
                "does not have critical replication lag",
                "critical lag is replication lag above 120 seconds",
                "Replicas with lag over 120 seconds are pulled from the read pool.", unit=" s"),
           RULE("backups_enabled", "is_true", None, "the instance", "has backups enabled",
                "does not have backups enabled",
                "backups_enabled marks whether automated backups are configured",
                "Instances without backups_enabled fail the production readiness check."),
           [NUM("replication_lag_seconds", " s", 0, 600), BOOL("backups_enabled")],
           [ENUM("engine", ["postgres", "mysql", "mongodb"]), NUM("storage_used_gb", " GB", 1, 5000)]),
    DOMAIN("networking", "connection",
           BAND("packet_loss_pct", "%", 0.0, 100.0, 1, [1.0, 5.0, 20.0],
                ["healthy", "degraded", "poor", "down"],
                ["Healthy link.", "Degraded link.", "Poor link.", "Link down."]),
           RULE("packet_loss_pct", "gte", 20.0, "the connection", "is effectively down",
                "is not effectively down",
                "effectively down means packet loss of 20% or more",
                "Links with 20%+ packet loss are treated as down for routing.", unit="%"),
           RULE("vpn_active", "is_true", None, "the connection", "is routed over VPN",
                "is not routed over VPN", "vpn_active marks traffic tunneled through the corporate VPN",
                "Only vpn_active connections can reach internal admin endpoints."),
           [NUM("packet_loss_pct", "%", 0.0, 100.0, 1), BOOL("vpn_active")],
           [ENUM("protocol", ["tcp", "udp", "http"]), NUM("latency_ms", " ms", 1, 2000)]),
    DOMAIN("iot_sensors", "sensor",
           BAND("battery_pct", "%", 0, 100, 0, [10, 25, 50],
                ["critical_battery", "low_battery", "ok_battery", "full_battery"],
                ["Critical battery.", "Low battery.", "OK battery.", "Full battery."]),
           RULE("battery_pct", "lt", 10, "the sensor", "has a critical battery level",
                "does not have a critical battery level",
                "critical battery means below 10%",
                "Sensors under 10% battery get an urgent replacement ticket.", unit="%"),
           RULE("last_calibration_date", "is_null", None, "the sensor", "has never been calibrated",
                "has been calibrated at least once",
                "never calibrated means last_calibration_date is missing",
                "Uncalibrated sensor readings are excluded from reports."),
           [NUM("battery_pct", "%", 0, 100), NULLABLE("last_calibration_date", ["2026-01-10", "2026-05-02", "2026-08-20"])],
           [ENUM("sensor_type", ["temperature", "humidity", "motion"]), NUM("reading_value", "", -40, 150)]),
    DOMAIN("weather_alerts", "weather station",
           BAND("wind_speed_kmh", " km/h", 0, 250, 0, [40, 65, 120],
                ["calm", "advisory", "warning", "extreme"],
                ["Calm.", "Advisory level.", "Warning level.", "Extreme level."]),
           RULE("wind_speed_kmh", "gte", 120, "the station", "should issue an extreme wind alert",
                "should not issue an extreme wind alert",
                "an extreme alert triggers at 120 km/h or more",
                "Extreme wind alerts fire automatically at wind speeds of 120 km/h or above.",
                unit=" km/h"),
           RULE("flood_risk", "is_true", None, "the station", "is under flood risk",
                "is not under flood risk", "flood_risk is set by the regional hydrology feed",
                "Flood-risk stations must issue a joint wind+flood bulletin."),
           [NUM("wind_speed_kmh", " km/h", 0, 250), BOOL("flood_risk")],
           [ENUM("region", ["coastal", "inland", "mountain"]), NUM("rainfall_mm", " mm", 0, 400)]),
    DOMAIN("agriculture", "field record",
           BAND("soil_moisture_pct", "%", 0, 100, 0, [20, 40, 70],
                ["drought", "dry", "adequate", "saturated"],
                ["Drought conditions.", "Dry conditions.", "Adequate moisture.", "Saturated soil."]),
           RULE("soil_moisture_pct", "lt", 20, "the field", "is in drought condition",
                "is not in drought condition",
                "drought condition is soil moisture below 20%",
                "Fields under 20% soil moisture trigger an irrigation alert.", unit="%"),
           RULE("pest_detected", "is_true", None, "the field", "has a pest detection",
                "has no pest detection", "pest_detected comes from the field scout report",
                "Fields with pest_detected true are scheduled for treatment within 48 hours."),
           [NUM("soil_moisture_pct", "%", 0, 100), BOOL("pest_detected")],
           [ENUM("crop_type", ["wheat", "corn", "soy"]), NUM("acreage", "", 1, 5000)]),
    DOMAIN("restaurants_delivery", "order ticket",
           BAND("prep_time_minutes", " min", 0, 90, 0, [10, 20, 40],
                ["fast", "normal", "slow", "very_slow"],
                ["Fast prep.", "Normal prep.", "Slow prep.", "Very slow prep."]),
           RULE("prep_time_minutes", "gt", 40, "the order", "has a very slow prep time",
                "does not have a very slow prep time",
                "very slow means prep time over 40 minutes",
                "Orders taking over 40 minutes to prep trigger a kitchen alert.", unit=" min"),
           RULE("allergens", "contains", "peanut", "the order", "contains a peanut allergen",
                "does not contain a peanut allergen",
                "a peanut allergen is present when \"peanut\" is in allergens",
                "Orders with a peanut allergen get a separate prep station."),
           [NUM("prep_time_minutes", " min", 0, 90), LIST("allergens", ["peanut", "dairy", "gluten", "shellfish"])],
           [ENUM("order_type", ["dine_in", "takeout", "delivery"]), NUM("total_usd", " USD", 5, 200)]),
    DOMAIN("ride_hailing", "trip",
           BAND("eta_minutes", " min", 0, 60, 0, [3, 8, 15],
                ["imminent_pickup", "short_wait", "long_wait", "very_long_wait"],
                ["Imminent pickup.", "Short wait.", "Long wait.", "Very long wait."]),
           RULE("eta_minutes", "gt", 15, "the trip", "has a very long wait",
                "does not have a very long wait",
                "a very long wait is an ETA over 15 minutes",
                "ETAs over 15 minutes trigger a rider apology credit.", unit=" min"),
           RULE("surge_pricing_active", "is_true", None, "the trip", "is under surge pricing",
                "is not under surge pricing",
                "surge_pricing_active reflects real-time demand multipliers",
                "Riders see a surge notice whenever surge_pricing_active is true."),
           [NUM("eta_minutes", " min", 0, 60), BOOL("surge_pricing_active")],
           [ENUM("vehicle_type", ["economy", "premium", "xl"]), NUM("fare_usd", " USD", 3, 150)]),
    DOMAIN("gaming", "player account",
           BAND("player_level", "", 1, 100, 0, [10, 30, 60],
                ["novice", "intermediate", "advanced", "elite"],
                ["Novice.", "Intermediate.", "Advanced.", "Elite."]),
           RULE("player_level", "gte", 60, "the player", "is elite tier",
                "is not elite tier", "elite tier starts at player level 60",
                "Elite-tier players unlock the ranked ladder.", unit=""),
           RULE("is_banned", "is_true", None, "the player", "is banned",
                "is not banned", "is_banned reflects an active moderation ban",
                "Banned players cannot join matchmaking regardless of level."),
           [NUM("player_level", "", 1, 100), BOOL("is_banned")],
           [ENUM("region", ["na", "eu", "apac"]), NUM("in_game_currency", "", 0, 100000)]),
    DOMAIN("social_moderation", "post",
           BAND("report_count", "", 0, 50, 0, [1, 5, 15],
                ["no_reports", "low_reports", "moderate_reports", "high_reports"],
                ["No reports.", "Low reports.", "Moderate reports.", "High reports."]),
           RULE("report_count", "gte", 15, "the post", "must go to human review",
                "does not need human review",
                "15 or more reports forces human review",
                "Posts reaching 15 or more reports are routed to human moderators.", unit=""),
           RULE("content_text", "contains", "banned_term", "the post", "contains a banned term",
                "does not contain a banned term",
                "a banned term match is detected when content_text contains it",
                "Posts containing a banned term are auto-hidden pending review."),
           [NUM("report_count", "", 0, 50), TEXTF("content_text",
                ["this post has a banned_term in it", "why is banned_term allowed here"],
                ["great sunset photo today", "check out my new setup", "happy to help with that question"])],
           [ENUM("platform", ["forum", "comments", "dm"]), BOOL("is_verified_author")]),
    DOMAIN("marketing_campaigns", "campaign",
           BAND("ctr_pct", "%", 0.0, 20.0, 2, [0.5, 2.0, 5.0],
                ["poor_ctr", "average_ctr", "good_ctr", "excellent_ctr"],
                ["Poor CTR.", "Average CTR.", "Good CTR.", "Excellent CTR."]),
           RULE("ctr_pct", "lt", 0.5, "the campaign", "has a poor click-through rate",
                "does not have a poor click-through rate",
                "a poor CTR is below 0.5%",
                "Campaigns with CTR under 0.5% are paused for review.", unit="%"),
           RULE("end_date", "before", None, "the campaign", "has already ended",
                "is still running", "ended means end_date is before today",
                "Ended campaigns (end_date before today) stop serving impressions."),
           [NUM("ctr_pct", "%", 0.0, 20.0, 2), DATEF("end_date")],
           [ENUM("channel", ["email", "social", "search"]), NUM("budget_usd", " USD", 100, 100000)]),
    DOMAIN("crm_leads", "lead",
           BAND("lead_score", "", 0, 100, 0, [25, 50, 80],
                ["cold_lead", "warm_lead", "hot_lead", "sales_ready_lead"],
                ["Cold lead.", "Warm lead.", "Hot lead.", "Sales-ready lead."]),
           RULE("lead_score", "gte", 80, "the lead", "is sales-ready",
                "is not sales-ready", "sales-ready means a lead score of 80 or higher",
                "Leads scoring 80 or higher are handed straight to a sales rep.", unit=""),
           RULE("assigned_rep", "is_null", None, "the lead", "is unassigned",
                "is assigned to a rep", "unassigned means assigned_rep is missing",
                "Unassigned leads older than a day are escalated to the sales manager."),
           [NUM("lead_score", "", 0, 100), NULLABLE("assigned_rep", ["rep_amy", "rep_ben", "rep_cho"])],
           [ENUM("source", ["referral", "web", "event"]), NUM("company_size", "", 1, 50000)]),
    DOMAIN("legal_contracts", "contract",
           BAND("days_until_renewal", " days", 0, 730, 0, [30, 90, 180],
                ["imminent_renewal", "upcoming_renewal", "mid_term", "long_term"],
                ["Imminent renewal.", "Upcoming renewal.", "Mid-term.", "Long-term."]),
           RULE("days_until_renewal", "lte", 30, "the contract", "renews imminently",
                "does not renew imminently",
                "imminent renewal means 30 or fewer days until renewal",
                "Legal reviews contracts renewing in 30 days or fewer.", unit=" days"),
           RULE("auto_renew_clause", "is_true", None, "the contract", "auto-renews",
                "does not auto-renew",
                "auto_renew_clause marks contracts that renew without action",
                "Contracts with an auto_renew_clause need an opt-out notice to cancel."),
           [NUM("days_until_renewal", " days", 0, 730), BOOL("auto_renew_clause")],
           [ENUM("contract_type", ["nda", "msa", "sow"]), NUM("value_usd", " USD", 1000, 5000000)]),
]

DOMAIN_BY_ID = {d["id"]: d for d in DOMAINS}

# 5 whole domains held out of train, eval-only (a fair, non-load-bearing spread).
HELD_OUT_DOMAINS = {"warehousing", "telecom_plans", "social_moderation", "agriculture", "legal_contracts"}

TRAIN_DOMAIN_IDS = [d["id"] for d in DOMAINS if d["id"] not in HELD_OUT_DOMAINS]
ALL_DOMAIN_IDS = [d["id"] for d in DOMAINS]

# ---------------------------------------------------------------------------
# state -> plain English (generic flattener, ~20% of cases, mirrors gen_it_worker.py)
# ---------------------------------------------------------------------------

def _flatten(d, prefix=""):
    lines = []
    if isinstance(d, dict):
        for k, v in d.items():
            label = f"{prefix}{str(k).replace('_', ' ')}"
            if isinstance(v, (dict, list)):
                lines.extend(_flatten(v, prefix=f"{label} "))
            else:
                lines.append(f"{label} is {v}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            lines.append(f"{prefix}item {i + 1} is {v}")
    else:
        lines.append(f"{prefix.strip()} {d}")
    return lines


def state_to_english(state):
    return "Situation: " + ". ".join(_flatten(state)) + "."


def maybe_english(rng, state):
    if rng.random() < 0.20:
        return state_to_english(state)
    return state


# ===========================================================================
# instruction phrasing pools (last template reserved for eval, per T())
# ===========================================================================

NOUL_STATEMENT_INSTR = [
    "Given the policy in the state, is the following true? {stmt}.",
    "Based on the stated policy, does this hold? {stmt}.",
    "Is this statement true given the state above? {stmt}.",
]
NOUL_RULEDEF_INSTR = [
    "Given the rule \"{ruledef}\", is the following true for this {entity}? {stmt}.",
    "Applying the rule that {ruledef}, does this hold? {stmt}.",
    "The rule is: {ruledef}. Given that, is this true? {stmt}.",
]
CODE_INSTR = [
    "Does this {lang} condition evaluate to true for this state?\n\n{code}",
    "For this state, does the following {lang} branch run?\n\n{code}",
    "Given the state above, is this {lang} condition true?\n\n{code}",
]
SQL_INSTR = [
    "Given this state as a single row, is the row returned by this query?\n\n{code}",
    "Would this SQL query return this row, given the state above?\n\n{code}",
]
FUNC_INSTR = [
    "What does this function return for this input?\n\n{code}",
    "Given the state above as the function's input, what does it return?\n\n{code}",
]
TIER_SCORE_INSTR = [
    "Which tier applies to this {entity}, from lowest to highest?",
    "Given the state above, where does this {entity} fall on this scale?",
]
TIER_CHOICE_INSTR = [
    "Which of these categories applies to this {entity}?",
    "Given the state above, which category best fits this {entity}?",
]

FUNC_TEMPLATES = {
    "py": "def classify(state):\n    if {c1}:\n        return \"{l1}\"\n    elif {c2}:\n        return \"{l2}\"\n    else:\n        return \"{l3}\"",
    "js": "function classify(state) {{\n  if ({c1}) return \"{l1}\";\n  else if ({c2}) return \"{l2}\";\n  else return \"{l3}\";\n}}",
    "go": "func classify(state State) string {{\n\tif {c1} {{\n\t\treturn \"{l1}\"\n\t}} else if {c2} {{\n\t\treturn \"{l2}\"\n\t}}\n\treturn \"{l3}\"\n}}",
}


# ===========================================================================
# per-case sampling & pool building
# ===========================================================================

def sample_field(rng, field, split):
    kind = field["kind"]
    if kind == "num":
        return sample_num(rng, field["lo"], field["hi"], field["thresholds"], field["decimals"], split)
    if kind == "enum":
        return A(rng, field["values"])
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "list":
        k = rng.randint(field["lenlo"], min(field["lenhi"], len(field["pool"])))
        return rng.sample(field["pool"], k)
    if kind == "nullable":
        if rng.random() < 0.4:
            return None
        return A(rng, field["pool"])
    if kind == "date":
        return None  # filled in later relative to a rule/today
    if kind == "text":
        return A(rng, field["with_kw"] + field["without_kw"])
    raise ValueError(kind)


def sample_rule_field_value(rng, field_spec, rule, today, split):
    """Sample the concrete value for a rule's own field, boundary-heavy."""
    kind = field_spec["kind"]
    if kind == "num":
        thr = rule["value"]
        thresholds = [thr] if not isinstance(thr, (tuple, list)) else [thr[0], thr[1]]
        return sample_num(rng, field_spec["lo"], field_spec["hi"], thresholds, field_spec["decimals"], split)
    if kind == "date":
        op = rule["op"]
        if op in ("before", "after"):
            off = sample_day_offset(rng, 0, split)
        else:
            off = sample_day_offset(rng, rule["value"], split)
        return iso(today, off)
    if kind == "enum":
        # ~55% chance of landing on the rule's target value(s) for balance
        v = rule["value"]
        target = v if isinstance(v, str) else A(rng, list(v))
        if rng.random() < 0.55:
            return target
        return A(rng, field_spec["values"])
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "list":
        target = rule["value"]
        base = [x for x in field_spec["pool"] if x != target]
        k = rng.randint(0, min(field_spec["lenhi"], len(base)))
        items = rng.sample(base, k)
        if rng.random() < 0.5:
            items.append(target)
            rng.shuffle(items)
        return items
    if kind == "nullable":
        if rng.random() < 0.5:
            return None
        return A(rng, field_spec["pool"])
    if kind == "text":
        return A(rng, field_spec["with_kw"] if rng.random() < 0.5 else field_spec["without_kw"])
    raise ValueError(kind)


def build_atom_from_rule(rule, field_spec, state, unit=""):
    kind = field_spec["kind"]
    op = rule["op"]
    value = rule["value"]
    if kind == "date" and op in ("within_days", "older_than"):
        value = rule["value"]
    return dict(field=rule["field"], kind=kind, op=op, value=value, unit=unit or rule.get("unit", ""))


def band_index(value, thresholds):
    idx = 0
    for t in thresholds:
        if value >= t:
            idx += 1
        else:
            break
    return idx


def build_case(domain, rng, split):
    today = A(rng, TODAY_POOL)
    state = {"today": today}
    field_defs = domain["fields"]
    band = domain["band"]
    rule1, rule2 = domain["rule1"], domain["rule2"]

    # band field (may coincide with rule1's field)
    band_field_spec = field_defs.get(band["key"], dict(kind="num", lo=band["lo"], hi=band["hi"], decimals=band["decimals"]))
    if band["key"] == rule1["field"]:
        band_value = sample_rule_field_value(rng, band_field_spec, rule1, today, split)
    else:
        band_value = sample_num(rng, band["lo"], band["hi"], band["thresholds"], band["decimals"], split)
    state[band["key"]] = band_value

    for rule in (rule1, rule2):
        fkey = rule["field"]
        if fkey in state:
            continue
        fspec = field_defs[fkey]
        state[fkey] = sample_rule_field_value(rng, fspec, rule, today, split)

    for fspec in domain["filler"]:
        if fspec["key"] in state:
            continue
        v = sample_field(rng, fspec, split)
        if v is None and fspec["kind"] == "date":
            v = iso(today, sample_day_offset(rng, 0, split))
        state[fspec["key"]] = v

    atom1 = build_atom_from_rule(rule1, field_defs[rule1["field"]], state)
    atom2 = build_atom_from_rule(rule2, field_defs[rule2["field"]], state)
    gold1 = eval_atom(atom1, state)
    gold2 = eval_atom(atom2, state)

    pool = []
    entity = domain["entity"]

    # ---- form a: English statement + policy given in state -------------
    state["policy"] = rule1["policy"]
    stmt = f"{rule1['subject']} {rule1['true_desc']}"
    pool.append(NOUL(f"a_{rule1['field']}", T(rng, split, NOUL_STATEMENT_INSTR, stmt=stmt),
                      f"Yes, {rule1['subject']} {rule1['true_desc']}.",
                      f"No, {rule1['subject']} {rule1['false_desc']}.", gold1))

    # ---- form b: explicit rule text, composed with rule2 via and/or -----
    combine = A(rng, ["and", "or", "single"])
    if combine == "single":
        node_b = ("atom", atom2)
        gold_b = gold2
        ruledef_b = rule2["rule_def"]
        stmt_b = f"{rule2['subject']} {rule2['true_desc']}"
    else:
        node_b = (combine, ("atom", atom1), ("atom", atom2))
        gold_b = eval_node(node_b, state)
        ruledef_b = f"{rule1['rule_def']}; separately, {rule2['rule_def']}"
        conn = "and" if combine == "and" else "or"
        stmt_b = f"({rule1['subject']} {rule1['true_desc']}) {conn} ({rule2['subject']} {rule2['true_desc']})"
    pool.append(NOUL("b_ruledef", T(rng, split, NOUL_RULEDEF_INSTR, ruledef=ruledef_b, entity=entity, stmt=stmt_b),
                      "Yes, that holds.", "No, that does not hold.", gold_b))

    # ---- form c: code (py/js/go/sql), depth 1-3 with and/or/not ----------
    depth_choice = A(rng, ["atom1", "atom2", "and", "or", "not_and", "not_or"])
    if depth_choice == "atom1":
        node_c = ("atom", atom1)
    elif depth_choice == "atom2":
        node_c = ("atom", atom2)
    elif depth_choice == "and":
        node_c = ("and", ("atom", atom1), ("atom", atom2))
    elif depth_choice == "or":
        node_c = ("or", ("atom", atom1), ("atom", atom2))
    elif depth_choice == "not_and":
        node_c = ("not", ("and", ("atom", atom1), ("atom", atom2)))
    else:
        node_c = ("not", ("or", ("atom", atom1), ("atom", atom2)))
    gold_c = eval_node(node_c, state)
    langs = list(node_langs(node_c))
    if langs:
        lang = A(rng, langs)
        code, lang_label = render_code_block(node_c, lang, today=today if lang != "sql" or True else None)
        if lang == "sql":
            instr = T(rng, split, SQL_INSTR, code=code)
        else:
            instr = T(rng, split, CODE_INSTR, lang=lang_label, code=code)
        pool.append(NOUL("c_code", instr, "Yes, it evaluates to true.", "No, it evaluates to false.", gold_c))

    # ---- form d: small function, choice over labels ----------------------
    func_lang = A(rng, ["py", "js", "go"])
    if func_lang in atom_langs(atom1) and func_lang in atom_langs(atom2):
        c1 = atom_code(atom1, func_lang).replace("{today}", today)
        c2 = atom_code(atom2, func_lang).replace("{today}", today)
        l1, l2, l3 = f"{rule1['field']}_match", f"{rule2['field']}_match", "no_match"
        code_d = FUNC_TEMPLATES[func_lang].format(c1=c1, c2=c2, l1=l1, l2=l2, l3=l3)
        if gold1:
            gold_d = l1
        elif gold2:
            gold_d = l2
        else:
            gold_d = l3
        pool.append(CHOICE(rng, "d_func", T(rng, split, FUNC_INSTR, code=code_d),
                            {l1: f"Returns \"{l1}\".", l2: f"Returns \"{l2}\".", l3: f"Returns \"{l3}\"."},
                            gold_d))

    # ---- form e: tier band, score (lowest-first) or choice ---------------
    idx = band_index(band_value, band["thresholds"])
    if rng.random() < 0.5:
        pool.append(SCORE("e_tier", T(rng, split, TIER_SCORE_INSTR, entity=entity), band["descs"], idx))
    else:
        options = {label: desc for label, desc in zip(band["labels"], band["descs"])}
        pool.append(CHOICE(rng, "e_tier_choice", T(rng, split, TIER_CHOICE_INSTR, entity=entity),
                            options, band["labels"][idx]))

    # simple fact-check noul on rule1's atom directly (keeps operator-level
    # signal dense even when forms above compose rule1 with rule2)
    pool.append(NOUL(f"fact_{rule1['field']}", f"Is it true that {atom_english(atom1)}?",
                      "Yes, that's true.", "No, that's false.", gold1))
    pool.append(NOUL(f"fact_{rule2['field']}", f"Is it true that {atom_english(atom2)}?",
                      "Yes, that's true.", "No, that's false.", gold2))

    out_state = maybe_english(rng, state)
    return out_state, pool, {"ops": [atom1["op"], atom2["op"], node_c[0]],
                              "forms": ["a", "b", "c", "d", "e"] if len(pool) >= 5 else ["a", "b", "e"]}


def build_row(domain_id, rng, split):
    domain = DOMAIN_BY_ID[domain_id]
    for _attempt in range(5):
        state, pool, meta = build_case(domain, rng, split)
        if len(pool) < 3:
            continue
        if isinstance(state, dict):
            state = dict(state)
            ref_prefix = "".join(ch for ch in domain_id.upper() if ch.isalnum())[:5] or "REF"
            state["ref"] = f"{ref_prefix}-{rng.randint(100000, 999999)}"
        k = rng.randint(3, min(5, len(pool)))
        chosen = rng.sample(pool, k)
        rng.shuffle(chosen)
        questions, gold = {}, {}
        for i, (qkey, qtype, instr, criteria, g) in enumerate(chosen):
            qid = f"{qkey}_{i}"
            questions[qid] = {"type": qtype, "instructions": instr, "criteria": criteria}
            gold[qid] = g
        row = {"source": f"our-cases-conditions/{domain_id}/rules",
               "domain": f"conditions/{domain_id}/rules",
               "state": state, "questions": questions, "gold": gold}
        return row, meta
    raise RuntimeError(f"could not build row for {domain_id}")


def dedup_key(row):
    return json.dumps([row["state"], row["questions"]], sort_keys=True, default=str)


def generate(path, target, seed, domain_ids, split):
    rng = random.Random(seed)
    schedule = list(domain_ids)
    seen = set()
    n_cases, n_questions = 0, 0
    counts_domain = Counter()
    counts_type = Counter()
    counts_op = Counter()
    gold_dist = defaultdict(Counter)
    with open(path, "w", encoding="utf-8") as fh:
        i, misses = 0, 0
        while n_cases < target:
            domain_id = schedule[i % len(schedule)]
            i += 1
            row, meta = build_row(domain_id, rng, split)
            dk = dedup_key(row)
            if dk in seen:
                misses += 1
                if misses > 80000:
                    raise RuntimeError("too many dedup collisions")
                continue
            seen.add(dk)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_cases += 1
            counts_domain[domain_id] += 1
            for op in meta["ops"]:
                counts_op[op] += 1
            for qid, q in row["questions"].items():
                n_questions += 1
                counts_type[q["type"]] += 1
                gold_dist[q["type"]][row["gold"][qid]] += 1
    return {"cases": n_cases, "questions": n_questions, "per_domain": counts_domain,
            "per_type": counts_type, "per_op": counts_op, "gold_dist": gold_dist}


# ===========================================================================
# basics gate: ~300 hand-auditable, single-fact + single-condition cases
# ===========================================================================

def gate_case(rng, kind, qid_seed):
    today = A(rng, TODAY_POOL)
    if kind == "age_adult":
        age = A(rng, [5, 12, 16, 17, 18, 19, 21, 30, 45, 70]) if rng.random() < 0.8 else rng.randint(0, 100)
        state = {"customer": {"age_years": age}}
        gold = age >= 18
        q = ("is_adult", "noul", "Is the customer an adult? An adult is someone aged 18 or older.",
             {"false": "No, a minor.", "true": "Yes, an adult."}, "true" if gold else "false")
    elif kind == "stock_level":
        units = A(rng, [0, 1, 2, 500]) if rng.random() < 0.85 else rng.randint(0, 1000)
        state = {"product": {"stock_units": units}}
        gold = units > 0
        q = ("in_stock", "noul", "Is the product in stock? In stock means stock_units is greater than 0.",
             {"false": "No, out of stock.", "true": "Yes, in stock."}, "true" if gold else "false")
    elif kind == "cpu_critical":
        cpu = A(rng, [20, 85, 89, 90, 91, 95, 99]) if rng.random() < 0.85 else rng.randint(0, 100)
        state = {"host": {"cpu_pct": cpu}}
        gold = cpu > 90
        q = ("cpu_critical", "noul",
             "Is the host's CPU above the alert threshold? The alert threshold is 90%.",
             {"false": "No, at or below the threshold.", "true": "Yes, above the threshold."},
             "true" if gold else "false")
    elif kind == "deadline_passed":
        offset = A(rng, [-1, 0, 1, -10, 10]) if rng.random() < 0.8 else rng.randint(-30, 30)
        deadline = iso(today, offset)
        state = {"today": today, "task": {"deadline": deadline}}
        gold = deadline < today
        q = ("deadline_passed", "noul",
             f"Given today is {today}, has the deadline {deadline} already passed?",
             {"false": "No, not passed yet.", "true": "Yes, already passed."}, "true" if gold else "false")
    elif kind == "status_equals":
        status = A(rng, ["pending", "approved", "rejected", "cancelled"])
        target = A(rng, ["pending", "approved", "rejected", "cancelled"])
        state = {"request": {"status": status}}
        gold = status == target
        q = ("status_equals", "noul", f'Does the request\'s status equal "{target}"?',
             {"false": "No.", "true": "Yes."}, "true" if gold else "false")
    elif kind == "list_membership":
        pool_items = ["red", "green", "blue", "yellow", "black"]
        k = rng.randint(0, 4)
        items = rng.sample(pool_items, k)
        target = A(rng, pool_items)
        state = {"cart": {"tags": items}}
        gold = target in items
        q = ("tag_present", "noul", f'Does the cart\'s tags list contain "{target}"?',
             {"false": "No.", "true": "Yes."}, "true" if gold else "false")
    elif kind == "null_check":
        has_value = rng.random() < 0.5
        val = None if not has_value else A(rng, ["REF-100", "REF-200", "REF-300"])
        state = {"record": {"assigned_ref": val}}
        gold = val is None
        q = ("is_missing", "noul", "Is the record's assigned_ref field missing (null)?",
             {"false": "No, it's set.", "true": "Yes, it's missing."}, "true" if gold else "false")
    else:
        raise ValueError(kind)
    qid, qtype, instr, criteria, g = q
    return {"source": f"our-cases-conditions/basics_gate/{kind}",
            "domain": f"conditions/basics_gate/{kind}",
            "state": state,
            "questions": {qid: {"type": qtype, "instructions": instr, "criteria": criteria}},
            "gold": {qid: g}}


GATE_KINDS = ["age_adult", "stock_level", "cpu_critical", "deadline_passed",
              "status_equals", "list_membership", "null_check"]


def generate_gate(path, target, seed):
    rng = random.Random(seed)
    seen = set()
    counts = Counter()
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        i, misses = 0, 0
        while n < target:
            kind = GATE_KINDS[i % len(GATE_KINDS)]
            i += 1
            row = gate_case(rng, kind, i)
            dk = dedup_key(row)
            if dk in seen:
                misses += 1
                if misses > 20000:
                    raise RuntimeError("too many gate dedup collisions")
                continue
            seen.add(dk)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[kind] += 1
            n += 1
    return {"cases": n, "per_kind": counts}


# ===========================================================================
# driver
# ===========================================================================

def main():
    print(f"Domains registered: {len(DOMAINS)}  (held out from train: {len(HELD_OUT_DOMAINS)})")
    for d in sorted(HELD_OUT_DOMAINS):
        print(f"  held out: {d}")

    DATA.mkdir(exist_ok=True)

    print("\nGenerating train...")
    train_stats = generate(TRAIN_PATH, TRAIN_TARGET, TRAIN_SEED, TRAIN_DOMAIN_IDS, "train")
    print("Generating eval...")
    eval_stats = generate(EVAL_PATH, EVAL_TARGET, EVAL_SEED, ALL_DOMAIN_IDS, "eval")
    print("Generating basics gate...")
    gate_stats = generate_gate(GATE_PATH, GATE_TARGET, GATE_SEED)

    for name, stats, path in (("TRAIN", train_stats, TRAIN_PATH), ("EVAL", eval_stats, EVAL_PATH)):
        print(f"\n=== {name} ({path}) ===")
        print(f"cases={stats['cases']} questions={stats['questions']}")
        print("per domain:")
        for dom, c in sorted(stats["per_domain"].items()):
            print(f"  {dom}: {c}")
        print("per type:", dict(stats["per_type"]))
        print("per op (from meta, approximate operator coverage):", dict(stats["per_op"]))
        print("gold distribution per type:")
        for t, dist in stats["gold_dist"].items():
            print(f"  {t}: {dict(dist)}")

    print(f"\n=== GATE ({GATE_PATH}) ===")
    print(f"cases={gate_stats['cases']}")
    print("per kind:", dict(gate_stats["per_kind"]))

    # ---- adapter validation ----
    sys.path.insert(0, str(ROOT / "finetuning" / "train"))
    import adapter  # noqa: E402

    def check_adapter(path):
        total, skipped = 0, 0
        for row in adapter.read_jsonl(path):
            total += 1
            if adapter.adapt_row(row) is None:
                skipped += 1
        return total, skipped

    for name, path in (("train", TRAIN_PATH), ("eval", EVAL_PATH), ("gate", GATE_PATH)):
        total, skipped = check_adapter(path)
        print(f"\nAdapter check ({name}): {total} rows, {skipped} skipped")

    # ---- duplicate check across files ----
    def keyset(path):
        return {dedup_key(json.loads(line)) for line in open(path, encoding="utf-8") if line.strip()}

    train_keys, eval_keys, gate_keys = keyset(TRAIN_PATH), keyset(EVAL_PATH), keyset(GATE_PATH)
    print(f"\nDuplicate check: train∩eval={len(train_keys & eval_keys)} "
          f"train∩gate={len(train_keys & gate_keys)} eval∩gate={len(eval_keys & gate_keys)}")

    # ---- sample rows ----
    LOCAL.mkdir(exist_ok=True)
    rng = random.Random(1)
    all_rows = list(open(TRAIN_PATH, encoding="utf-8"))
    sample_lines = rng.sample(all_rows, 40)
    with open(SAMPLES_PATH, "w", encoding="utf-8") as fh:
        for line in sample_lines:
            row = json.loads(line)
            fh.write(json.dumps(row, indent=2, ensure_ascii=False) + "\n\n")
    print(f"\nWrote 40 sample rows to {SAMPLES_PATH}")


if __name__ == "__main__":
    main()
