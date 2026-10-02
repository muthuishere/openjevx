#!/usr/bin/env python3
"""Rule-reading drills (v0.5.2): make the model read the rule's number, not remember it.

v0.5.0 was confidently wrong on the basics gates in a few families (llmresults/13-v0.5.2-gate-misses.md):
it answered with the TRAIN threshold of a rule when the gate stated another (min balance Rs 5000 judged
against Rs 1000, credit score 750 against 700), it called 0.1 C "at or below freezing", it failed meetings
that start on the half hour and dates after the train range, and it said a deadline that is today or later
"has already passed". The basics train set states one threshold per rule, so the model never had to read it.

Families (gold always comes from evaluating the rule):
  thresholds  every everyday numeric rule (gen_basics.NUMERIC) with many thresholds instead of one,
              never the gate's threshold for that rule
  decimals    decimal readings against whole-number thresholds (not freezing/boiling: the gate checks
              that this transfers to them)
  dates       the date rules over 2020-2035 with same-day and +-1 day cases, plus negated questions
              (not for passport/subscription: the gate asks those negated)
  bare_dates  a date compared with today with no rule text, never on one of the gate's "today"s
  meetings    meetings on the quarter hour, 30-120 minutes, back-to-back cases, a negated question
  fields      single-field checks: missing (null), status equals, list contains, a number against a
              threshold stated in the question
Any row whose state a gate or test file already asks about (leak_check's state test, any question type)
is dropped, so the gate and eval files stay exactly as they are.

Output: <data>/train/rule_reading_train.jsonl (see finetuning/paths.py)
"""
import datetime as dt
import json
import random
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
import gen_basics as gb  # noqa: E402
from datavalidate.leak_check import state_seqs  # noqa: E402

OUT = paths.TRAIN / "rule_reading_train.jsonl"
SEED = 20261002
YN = gb.YN
FIXED_T = {"water_boil", "freezing", "stock_out"}  # physical facts / "is zero": keep their number
DECIMAL_OK = {"fever", "battery_low", "storage_full", "free_shipping", "discount_min", "baggage", "cabin_bag",
              "speeding", "pass_mark", "attendance", "cpu_alert", "error_rate", "latency_slo", "coverage",
              "bmi_over", "budget", "min_balance", "atm_limit", "rain_umbrella", "fuel_low", "tyre_pressure",
              "overtime", "parking", "blood_sugar", "gst_invoice", "delivery_radius", "aqi", "screen_time"}
GATE_TODAYS = {gb.GATE_TODAY.isoformat(), "2026-09-29", "2026-01-15", "2026-12-01", "2026-06-10"}

# Negated questions for the date rules, with True when the answer is the opposite of the rule holding.
DATE_NEG = {
    "parcel_late": [("Is the parcel still on time?", True), ("Has the parcel not yet missed its promised date?", True)],
    "bill_overdue": [("Is the bill still within its due date?", True), ("Is the bill not yet overdue?", True)],
    "warranty": [("Has the warranty expired?", True), ("Has the warranty run out?", True)],
    "deadline": [("Is the task still on time?", True), ("Is the deadline still ahead?", True)],
}


def row(domain, state, instr, gold, criteria=None):
    return {"source": f"our-cases-rulereading/{domain}", "domain": f"rulereading/{domain}", "state": state,
            "questions": {"q": {"type": "noul", "instructions": instr, "criteria": criteria or dict(YN)}},
            "gold": {"q": "true" if gold else "false"}}


def nice(v, lo, hi, rng):
    """A threshold people would write: round to a step that suits its size half the time."""
    span = hi - lo
    if rng.random() < 0.5 and span >= 100:
        step = 10 ** (len(str(int(span))) - 2)
        v = round(v / step) * step
    return min(max(v, lo), hi)


def thresholds(rng):
    rows = []
    for rid, dom, rule, field, unit, t0, op, (lo, hi), phrasings, noun in gb.NUMERIC:
        if rid in FIXED_T:
            continue
        gate_t = gb.GATE_T.get(rid)
        is_float = isinstance(t0, float)
        use = phrasings[:-1]
        ts = set()
        for _ in range(500):  # 24 thresholds, fewer when the range is small (approvals 0-6)
            if len(ts) == 24:
                break
            t = round(rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo)), 1) if is_float else \
                int(nice(rng.randint(lo + 1, hi - 1), lo + 1, hi - 1, rng))
            if t != gate_t:
                ts.add(t)
        for t in sorted(ts):
            step = 0.1 if is_float else 1
            big = max(step, round((hi - lo) / 20, 1 if is_float else 0))
            vals = [t - step, t, t + step, t - 2 * step, t + 2 * step, t - big, t + big,
                    rng.uniform(lo, hi * 1.3) if is_float else rng.randint(lo, int(hi * 1.3))]
            if rid in DECIMAL_OK and not is_float:
                vals += [t - 0.5, t + 0.5, t - 0.1, t + 0.1, round(rng.uniform(lo, hi), 1)]
            for v in vals:
                v = round(v, 1) if (is_float or v != int(v)) else int(v)
                if v < 0 and lo >= 0:
                    continue
                tt = int(t) if not is_float and t == int(t) else t
                state = {"rule": rule.format(t=tt), noun: {field: v}}
                q = rng.choice(use)
                if rng.random() < 0.3:
                    state, q = {noun: {field: v}}, f"{q} (Rule: {rule.format(t=tt)})"
                rows.append(row(f"thresholds/{rid}", gb.render(state, rng), q, gb.compare(v, op, t)))
    return rows


def dates(rng):
    rows = []
    lo, hi = dt.date(2020, 1, 1), dt.date(2035, 12, 31)
    for rid, dom, rule, field, phrasings, kind in gb.DATE_RULES:
        use = [(p, False) for p in phrasings[:-1]] + DATE_NEG.get(rid, [])
        for _ in range(450):
            today = lo + dt.timedelta(days=rng.randint(0, (hi - lo).days))
            if today.isoformat() in GATE_TODAYS:
                continue
            off = rng.choice([0, 0, -1, 1, -2, 2, rng.randint(-7, 7), rng.randint(-60, 60), rng.randint(-800, 800)])
            d = today + dt.timedelta(days=off)
            holds = today > d if kind == "after" else today <= d
            q, neg = rng.choice(use)
            state = {"rule": rule, "today": today.isoformat(), field: d.isoformat()}
            rows.append(row(f"dates/{rid}", gb.render(state, rng), q, holds != neg))
    return rows


# (entity, field, how a question names the field)
BARE = [("task", "deadline", "deadline"), ("invoice", "due_date", "due date"), ("order", "ship_by", "ship-by date"),
        ("licence", "expires_on", "expiry date"), ("ticket", "sla_deadline", "SLA deadline"),
        ("trial", "ends_on", "end date"), ("coupon", "valid_until", "last valid date"),
        ("project", "milestone_date", "milestone date")]
# (template, test on (deadline d, today t))
BARE_Q = [
    ("Has the {n}'s {f} already passed?", lambda d, t: d < t),
    ("Today is {t}. Is {d} already in the past?", lambda d, t: d < t),
    ("Is the {f} {d} today or later?", lambda d, t: d >= t),
    ("Is today ({t}) after the {f}?", lambda d, t: t > d),
    ("Is the {f} today?", lambda d, t: d == t),
    ("Is the {n}'s {f} still in the future?", lambda d, t: d > t),
    ("Today is {t}: is the {f} {d} on or before today?", lambda d, t: d <= t),
    ("Is the {n} past its {f}?", lambda d, t: t > d),
]


def bare_dates(rng):
    rows = []
    lo, hi = dt.date(2020, 1, 1), dt.date(2035, 12, 31)
    for _ in range(4000):
        today = lo + dt.timedelta(days=rng.randint(0, (hi - lo).days))
        if rng.random() < 0.3:  # near a month end, so +-days cross a month
            today = today.replace(day=1) - dt.timedelta(days=rng.randint(0, 2))
        if today.isoformat() in GATE_TODAYS:
            continue
        off = rng.choice([0, 0, 0, -1, 1, 1, -2, 2, rng.randint(-10, 10), rng.randint(-45, 45), rng.randint(-500, 500)])
        d = today + dt.timedelta(days=off)
        n, f, name = rng.choice(BARE)
        tmpl, test = rng.choice(BARE_Q)
        state = {"today": today.isoformat(), n: {f: d.isoformat()}}
        q = tmpl.format(n=n, f=name, d=d.isoformat(), t=today.isoformat())
        rows.append(row("bare_dates", state, q, test(d.isoformat(), today.isoformat())))
    return rows


def meetings(rng):
    rows = []
    hm = lambda m: f"{m // 60:02d}:{m % 60:02d}"
    use = [("Do these two meetings clash?", False), ("Do the meetings overlap?", False),
           ("Do the two meetings overlap in time?", False), ("Can someone attend both meetings in full?", True)]
    for _ in range(2000):
        s1 = rng.randint(7, 17) * 60 + rng.choice([0, 15, 45]); d1 = rng.choice([30, 45, 60, 90, 120])
        d2 = rng.choice([30, 45, 60, 90, 120])
        s2 = s1 + rng.choice([-d2, d1, 0, 15, -15, 30, -30, 45, -45, 60, -60, d1 + 15, -d2 - 15, 180])
        if not 0 <= s2 < 24 * 60 - d2:
            continue
        clash = s1 < s2 + d2 and s2 < s1 + d1
        q, neg = rng.choice(use)
        state = {"rule": "Two meetings clash if their times overlap; back-to-back is not a clash.",
                 "meeting_a": f"{hm(s1)}-{hm(s1 + d1)}", "meeting_b": f"{hm(s2)}-{hm(s2 + d2)}"}
        rows.append(row("meetings", gb.render(state, rng), q, clash != neg))
    return rows


NULL_FIELDS = [("ticket", "owner", ["alice", "bob", "team-ops"]), ("order", "tracking_id", ["TRK-4411", "TRK-0093"]),
               ("user", "email", ["a@example.com", "ops@example.org"]), ("invoice", "po_number", ["PO-7781", "PO-1203"]),
               ("change", "reviewer", ["priya", "sam"]), ("host", "last_backup", ["2026-03-02", "2025-11-30"]),
               ("employee", "manager_id", ["E-1042", "E-2210"]), ("shipment", "carrier", ["DHL", "BlueDart"])]
STATUSES = [("order", ["pending", "shipped", "delivered", "returned", "cancelled"]),
            ("deploy", ["queued", "running", "failed", "succeeded", "rolled_back"]),
            ("ticket", ["open", "on_hold", "resolved", "closed", "reopened"]),
            ("payment", ["authorized", "captured", "refunded", "declined", "voided"]),
            ("application", ["submitted", "approved", "declined", "withdrawn", "rejected"])]
LISTS = [("user", "roles", ["admin", "editor", "viewer", "owner", "billing"]),
         ("issue", "labels", ["bug", "urgent", "docs", "feature", "ui", "backend"]),
         ("server", "regions", ["us-east", "eu-west", "ap-south", "eu-central"]),
         ("recipe", "allergens", ["nuts", "gluten", "dairy", "soy", "egg"]),
         ("account", "flags", ["trial", "vip", "suspended", "verified"])]
# (entity, field, unit, op, threshold range, value range, question with {t})
STATED = [("member", "age", "", ">=", (12, 70), (0, 110), "Is the member old enough? The minimum age is {t}."),
          ("driver", "age_years", "", ">=", (16, 25), (10, 99), "Can the driver rent the van? Drivers must be {t} or older."),
          ("host", "disk_pct", "%", ">", (70, 98), (0, 100), "Is the host's disk above the alert threshold? The threshold is {t}%."),
          ("queue", "depth", "", ">", (100, 5000), (0, 9000), "Is the queue backed up? Backed up means depth is greater than {t}."),
          ("product", "units_left", "", "<", (2, 50), (0, 300), "Should the product be restocked? Restock when units_left is below {t}."),
          ("pod", "restarts", "", ">=", (3, 10), (0, 40), "Is the pod crash-looping? It is when restarts are {t} or more."),
          ("cart", "items", "", "<=", (5, 30), (0, 60), "Is the cart within the item limit? The limit is {t} items."),
          ("api", "p99_ms", "ms", "<", (100, 1500), (10, 4000), "Is the API meeting its target? The target is p99 under {t} ms.")]


def fields(rng):
    rows = []
    for _ in range(700):
        e, f, vals = rng.choice(NULL_FIELDS)
        v = None if rng.random() < 0.5 else rng.choice(vals)
        q, test = rng.choice([(f"Is the {e}'s {f} field missing (null)?", lambda v: v is None),
                              (f"Is {f} set on the {e}?", lambda v: v is not None),
                              (f"Does the {e} have no {f.replace('_', ' ')}?", lambda v: v is None)])
        rows.append(row("fields/null", {e: {f: v}}, q, test(v)))
    for _ in range(900):
        e, sts = rng.choice(STATUSES)
        s, target = rng.choice(sts), rng.choice(sts)
        if rng.random() < 0.4:
            target = s
        q = rng.choice([f'Does the {e}\'s status equal "{target}"?', f'Is the {e} status exactly "{target}"?'])
        rows.append(row("fields/status", {e: {"status": s}}, q, s == target))
    for _ in range(900):
        e, f, pool = rng.choice(LISTS)
        items = rng.sample(pool, rng.randint(0, len(pool) - 1))
        target = rng.choice(items) if items and rng.random() < 0.5 else rng.choice(pool)
        q = rng.choice([f'Does the {e}\'s {f} list contain "{target}"?', f'Is "{target}" one of the {e}\'s {f}?'])
        rows.append(row("fields/list", {e: {f: items}}, q, target in items))
    for _ in range(2400):
        e, f, unit, op, (tlo, thi), (vlo, vhi), tmpl = rng.choice(STATED)
        t = rng.randint(tlo, thi)
        v = rng.choice([t, t - 1, t + 1, t - 2, t + 2, rng.randint(vlo, vhi), vhi, vlo])
        if not vlo <= v <= vhi:
            continue
        rows.append(row(f"fields/stated/{e}", {e: {f: v}}, tmpl.format(t=t), gb.compare(v, op, t)))
    return rows


def forbidden():
    """State sequences of every gate and test file, so no drill asks about a state they ask about."""
    keys = set()
    for path in sorted(paths.GATE.glob("*.jsonl")) + sorted(paths.EVAL.glob("*.jsonl")):
        with open(path) as fh:
            for line in fh:
                try:
                    keys |= state_seqs(json.loads(line)["state"])
                except Exception:
                    pass
    return keys


def main():
    rng = random.Random(SEED)
    fams = {"thresholds": thresholds(rng), "dates": dates(rng), "bare_dates": bare_dates(rng),
            "meetings": meetings(rng), "fields": fields(rng)}
    forbid = forbidden()
    seen, out, dropped = set(), [], Counter()
    for name, rows in fams.items():
        for r in rows:
            k = json.dumps([r["state"], r["questions"]], sort_keys=True)
            if k in seen:
                continue
            seen.add(k)
            if state_seqs(r["state"]) & forbid:
                dropped[name] += 1
                continue
            out.append(r)
    rng.shuffle(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    per = Counter(r["domain"].split("/")[1] for r in out)
    yes = sum(r["gold"]["q"] == "true" for r in out)
    print(f"rule_reading_train: {len(out)} questions, yes {yes} / no {len(out) - yes}; per family {dict(per)}; "
          f"dropped (state in a gate/test file) {dict(dropped)} -> {OUT}")


if __name__ == "__main__":
    main()
