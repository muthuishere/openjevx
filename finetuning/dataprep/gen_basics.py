#!/usr/bin/env python3
"""Everyday basics: the small rules people check all the time (ADR 0004, item H).

Each rule states its threshold in the state, gives the facts, and asks whether the rule holds.
Values cluster around the threshold (just below / on / just above) plus far values, both
polarities, several phrasings. Gold is computed by evaluating the rule, never written by hand.
The last phrasing of every rule is reserved for the gate file, so the gate tests the rule,
not a memorised sentence. No gate state is ever a training state either (a reworded question about
a trained state is still a leak): the gate uses its own thresholds (GATE_T), opening hours
(GATE_HOURS), a "today" outside the train range, half-hour meetings and, for the physical constants,
decimal readings the train split never uses; any gate case whose facts still match a train case
is dropped (checked on the underlying facts, whether train rendered them as JSON or English).

Outputs: <data>/train/basics_train.jsonl, <data>/gate/basics_gate.jsonl (see finetuning/paths.py)
"""
import datetime as dt
import json
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
YN = {"false": "no", "true": "yes"}

# (id, domain, rule text, fact path, unit, threshold, operator, value range, [phrasings...], state noun)
# operator: ">=", ">", "<=", "<", "=="; phrasings are yes/no questions about the person/thing.
NUMERIC = [
    ("driving_licence", "transport", "You can apply for a driving licence at {t} or older.", "age", "years", 18, ">=", (5, 80),
     ["Can this person apply for a driving licence?", "Is this person old enough to get a driving licence?",
      "Is this person eligible for a driving licence by age?", "Does this person meet the minimum age for a licence?"], "person"),
    ("voting", "civic", "You can vote at {t} or older.", "age", "years", 18, ">=", (8, 90),
     ["Can this person vote?", "Is this person old enough to vote?", "Is this person eligible to vote by age?"], "person"),
    ("car_rental", "travel", "You must be at least {t} to rent a car.", "age", "years", 21, ">=", (16, 75),
     ["Can this person rent a car?", "Is this person old enough to rent a car?", "Does this person meet the car rental age?"], "customer"),
    ("senior_discount", "retail", "Senior citizen discount applies from age {t}.", "age", "years", 60, ">=", (30, 95),
     ["Does the senior discount apply?", "Is this customer a senior citizen for the discount?", "Should this customer get the senior discount?"], "customer"),
    ("child_ticket", "travel", "Children under {t} travel free.", "age", "years", 5, "<", (0, 15),
     ["Does this child travel free?", "Is the ticket free for this child?", "Is this passenger young enough to travel free?"], "passenger"),
    ("movie_rating", "entertainment", "An A-rated film needs a viewer aged {t} or older.", "age", "years", 18, ">=", (10, 60),
     ["Can this viewer watch the A-rated film?", "Is this viewer allowed into the A-rated film?", "Is this viewer old enough for the film?"], "viewer"),
    ("retirement", "hr", "Retirement age is {t}.", "age", "years", 60, ">=", (35, 75),
     ["Has this employee reached retirement age?", "Is this employee due to retire?", "Is this employee at or past retirement age?"], "employee"),
    ("fever", "health", "A body temperature of {t} C or more is a fever.", "temperature_c", "C", 38.0, ">=", (35.5, 41.0),
     ["Does this person have a fever?", "Is this temperature a fever?", "Should this reading be treated as a fever?"], "patient"),
    ("battery_low", "devices", "Battery below {t}% counts as low.", "battery_percent", "%", 20, "<", (0, 100),
     ["Is the battery low?", "Should the phone warn about low battery?", "Is this battery level low?"], "phone"),
    ("storage_full", "devices", "Storage at {t}% or more is nearly full.", "storage_used_percent", "%", 90, ">=", (10, 100),
     ["Is storage nearly full?", "Should the device warn that storage is nearly full?", "Is the disk nearly full?"], "device"),
    ("free_shipping", "ecommerce", "Free shipping on orders of Rs {t} or more.", "cart_total_rs", "Rs", 500, ">=", (50, 3000),
     ["Does this order get free shipping?", "Is shipping free for this cart?", "Does the free-shipping rule apply?"], "order"),
    ("discount_min", "retail", "The coupon needs a minimum purchase of Rs {t}.", "cart_total_rs", "Rs", 999, ">=", (100, 5000),
     ["Can the coupon be applied?", "Does this cart meet the coupon minimum?", "Is this purchase big enough for the coupon?"], "cart"),
    ("baggage", "travel", "Checked bags may weigh at most {t} kg.", "bag_weight_kg", "kg", 23, "<=", (5, 40),
     ["Is this bag within the weight limit?", "Can this bag be checked without extra charge?", "Does the bag meet the weight rule?"], "bag"),
    ("cabin_bag", "travel", "Cabin bags may weigh at most {t} kg.", "cabin_bag_kg", "kg", 7, "<=", (1, 15),
     ["Is this cabin bag allowed?", "Is the cabin bag within the limit?", "Can this bag go in the cabin?"], "bag"),
    ("speeding", "transport", "The speed limit here is {t} km/h.", "speed_kmh", "km/h", 50, ">", (10, 120),
     ["Is the driver speeding?", "Is this car over the speed limit?", "Is this speed above the limit?"], "car"),
    ("pass_mark", "education", "The pass mark is {t} out of 100.", "score", "marks", 35, ">=", (0, 100),
     ["Did the student pass?", "Is this score a pass?", "Has the student cleared the exam?"], "student"),
    ("attendance", "education", "Students need at least {t}% attendance to sit the exam.", "attendance_percent", "%", 75, ">=", (30, 100),
     ["Can the student sit the exam?", "Does the student meet the attendance rule?", "Is attendance high enough for the exam?"], "student"),
    ("password_length", "software", "Passwords need at least {t} characters.", "password_length", "characters", 8, ">=", (1, 30),
     ["Is the password long enough?", "Does this password meet the length rule?", "Is the password length valid?"], "password"),
    ("cpu_alert", "software", "Alert when CPU is above {t}%.", "cpu_percent", "%", 90, ">", (5, 100),
     ["Should the CPU alert fire?", "Is CPU above the alert threshold?", "Does this CPU reading need an alert?"], "server"),
    ("error_rate", "software", "Page on-call when the error rate is {t}% or higher.", "error_rate_percent", "%", 5, ">=", (0, 60),
     ["Should on-call be paged?", "Is the error rate high enough to page?", "Does this error rate breach the paging rule?"], "service"),
    ("latency_slo", "software", "The latency target is p95 under {t} ms.", "p95_latency_ms", "ms", 300, "<", (50, 2000),
     ["Is the service meeting its latency target?", "Is p95 latency within the target?", "Is latency OK?"], "service"),
    ("coverage", "software", "Merges need test coverage of at least {t}%.", "coverage_percent", "%", 80, ">=", (20, 100),
     ["Can this PR merge on coverage?", "Is coverage high enough to merge?", "Does coverage meet the bar?"], "pull_request"),
    ("approvals", "software", "A pull request needs at least {t} approvals.", "approvals", "approvals", 2, ">=", (0, 6),
     ["Does the PR have enough approvals?", "Can the PR merge on approvals?", "Is the approval count sufficient?"], "pull_request"),
    ("stock_out", "retail", "An item is out of stock when stock is {t}.", "stock_units", "units", 0, "==", (0, 600),
     ["Is the item out of stock?", "Is this product sold out?", "Should the item show as out of stock?"], "item"),
    ("reorder", "retail", "Reorder when stock falls below {t} units.", "stock_units", "units", 20, "<", (0, 200),
     ["Should this item be reordered?", "Is stock below the reorder level?", "Is it time to reorder?"], "item"),
    ("bmi_over", "health", "A BMI of {t} or more is overweight.", "bmi", "", 25.0, ">=", (15.0, 40.0),
     ["Is this BMI in the overweight range?", "Is this person overweight by BMI?", "Does this BMI count as overweight?"], "person"),
    ("budget", "finance", "The monthly budget is Rs {t}.", "spent_rs", "Rs", 20000, ">", (1000, 50000),
     ["Is the budget exceeded?", "Has spending gone over budget?", "Is this month over budget?"], "account"),
    ("min_balance", "finance", "The account must keep a minimum balance of Rs {t}.", "balance_rs", "Rs", 1000, "<", (0, 20000),
     ["Is the balance below the minimum?", "Will a low-balance charge apply?", "Is the account under its minimum balance?"], "account"),
    ("atm_limit", "finance", "Daily ATM withdrawal limit is Rs {t}.", "withdrawn_today_rs", "Rs", 25000, "<=", (0, 60000),
     ["Is this withdrawal within the daily limit?", "Can this amount be withdrawn today?", "Is the ATM total within limit?"], "customer"),
    ("water_boil", "science", "Water boils at {t} C at sea level.", "water_temperature_c", "C", 100, ">=", (20, 120),
     ["Is the water boiling?", "Has the water reached boiling point?", "Is this water at boiling temperature?"], "pot"),
    ("freezing", "science", "Water freezes at {t} C or below.", "temperature_c", "C", 0, "<=", (-20, 20),
     ["Will water freeze at this temperature?", "Is it cold enough for water to freeze?", "Is this at or below freezing?"], "weather"),
    ("rain_umbrella", "daily", "Take an umbrella when the chance of rain is {t}% or more.", "rain_chance_percent", "%", 60, ">=", (0, 100),
     ["Should you take an umbrella?", "Is an umbrella needed today?", "Is the rain chance high enough for an umbrella?"], "forecast"),
    ("aqi", "health", "Air quality index above {t} is unhealthy.", "aqi", "", 150, ">", (10, 400),
     ["Is the air unhealthy?", "Is the AQI in the unhealthy range?", "Should people limit outdoor activity?"], "city"),
    ("elevator", "daily", "The lift carries at most {t} people.", "people", "people", 8, "<=", (1, 15),
     ["Can everyone take the lift together?", "Is the lift within capacity?", "Is the group within the lift limit?"], "group"),
    ("fuel_low", "transport", "Refuel when the tank is below {t}%.", "fuel_percent", "%", 15, "<", (0, 100),
     ["Should the driver refuel?", "Is the fuel low?", "Is it time to fill the tank?"], "car"),
    ("tyre_pressure", "transport", "Tyre pressure below {t} psi is too low.", "tyre_psi", "psi", 30, "<", (15, 45),
     ["Is the tyre pressure too low?", "Do the tyres need air?", "Is this tyre under-inflated?"], "car"),
    ("hotel_checkin", "travel", "Hotel check-in opens at {t}:00.", "arrival_hour", "hour", 14, ">=", (0, 23),
     ["Can the guest check in now?", "Is check-in open at this hour?", "Has check-in opened?"], "guest"),
    ("loan_age", "finance", "Loan applicants must be at least {t}.", "age", "years", 21, ">=", (15, 70),
     ["Can this person apply for the loan?", "Is the applicant old enough for a loan?", "Does the applicant meet the loan age rule?"], "applicant"),
    ("credit_score", "finance", "Loans need a credit score of at least {t}.", "credit_score", "", 700, ">=", (300, 900),
     ["Does the applicant meet the credit score rule?", "Is the credit score high enough?", "Does the score qualify for the loan?"], "applicant"),
    ("overtime", "hr", "Hours above {t} per week are overtime.", "hours_worked", "hours", 48, ">", (10, 80),
     ["Did this employee work overtime?", "Are there overtime hours this week?", "Does this week include overtime?"], "employee"),
    ("leave_balance", "hr", "Leave can be approved only if the balance covers it: balance is {t} days.", "days_requested", "days", 12, "<=", (1, 30),
     ["Can this leave be approved from the balance?", "Is the balance enough for this leave?", "Does the balance cover the request?"], "employee"),
    ("parking", "daily", "Parking is free for the first {t} minutes.", "parked_minutes", "minutes", 30, "<=", (1, 240),
     ["Is parking free for this stay?", "Does this car park for free?", "Is this stay inside the free period?"], "car"),
    ("screen_time", "daily", "The daily screen-time limit is {t} minutes.", "screen_minutes", "minutes", 120, ">", (0, 400),
     ["Is the screen-time limit exceeded?", "Has the child gone over screen time?", "Is screen time over the limit?"], "child"),
    ("steps_goal", "health", "The daily step goal is {t} steps.", "steps", "steps", 10000, ">=", (500, 20000),
     ["Was the step goal met?", "Did this person hit the step goal?", "Is the step count at goal?"], "person"),
    ("blood_sugar", "health", "Fasting blood sugar of {t} mg/dL or more is high.", "fasting_sugar_mg_dl", "mg/dL", 126, ">=", (60, 300),
     ["Is the fasting blood sugar high?", "Is this reading in the high range?", "Should this sugar reading be flagged?"], "patient"),
    ("gst_invoice", "business", "Invoices above Rs {t} need a GST number.", "invoice_rs", "Rs", 50000, ">", (1000, 200000),
     ["Does this invoice need a GST number?", "Must a GST number be on this invoice?", "Is a GST number required here?"], "invoice"),
    ("delivery_radius", "ecommerce", "We deliver within {t} km.", "distance_km", "km", 10, "<=", (0, 40),
     ["Can this address get delivery?", "Is this address inside the delivery radius?", "Is delivery available here?"], "address"),
]

DATE_RULES = [
    ("parcel_late", "ecommerce", "A parcel is late if today is after its promised date.", "promised_date",
     ["Is the parcel late?", "Has the parcel missed its promised date?", "Is this delivery overdue?"], "after"),
    ("bill_overdue", "finance", "A bill is overdue if today is after its due date.", "due_date",
     ["Is the bill overdue?", "Has the payment due date passed?", "Is this bill past due?"], "after"),
    ("warranty", "retail", "The warranty is valid up to and including its end date.", "warranty_end_date",
     ["Is the warranty still valid?", "Is the product still under warranty?", "Can a warranty claim be made today?"], "on_or_before"),
    ("passport_valid", "travel", "A passport is valid up to and including its expiry date.", "expiry_date",
     ["Is the passport valid today?", "Can this passport be used today?", "Is the passport unexpired?"], "on_or_before"),
    ("deadline", "work", "A task is overdue if today is after its deadline.", "deadline",
     ["Has the deadline passed?", "Is the task overdue?", "Is it past the deadline?"], "after"),
    ("subscription", "software", "A subscription is active up to and including its renewal date.", "renewal_date",
     ["Is the subscription still active?", "Is the plan active today?", "Has the subscription not yet lapsed?"], "on_or_before"),
]

TIME_RULES = [
    ("store_open", "retail", "The store is open from {o}:00 to {c}:00 (closing time excluded).", 10, 21,
     ["Is the store open now?", "Can a customer shop at this time?", "Is the shop open at this hour?"]),
    ("bank_hours", "finance", "The bank branch is open from {o}:00 to {c}:00 (closing time excluded).", 9, 16,
     ["Is the branch open now?", "Can a customer visit the bank now?", "Is the bank open at this time?"]),
    ("quiet_hours", "daily", "Quiet hours run from {o}:00 to {c}:00 (end excluded).", 22, 24,
     ["Is it quiet hours now?", "Should noise be kept down now?", "Is this within quiet hours?"]),
    ("support_hours", "software", "Support is staffed from {o}:00 to {c}:00 (end excluded).", 9, 18,
     ["Is support available now?", "Can a customer reach support now?", "Is support staffed at this hour?"]),
]

# Gate-only thresholds: the gate states a rule the model never saw with these numbers. Rules that
# state a physical fact (boiling, freezing) or zero stock keep their number and use unseen values.
GATE_T = {
    "driving_licence": 17, "voting": 21, "car_rental": 25, "senior_discount": 65, "child_ticket": 3,
    "movie_rating": 16, "retirement": 58, "fever": 37.5, "battery_low": 15, "storage_full": 85,
    "free_shipping": 499, "discount_min": 1499, "baggage": 20, "cabin_bag": 10, "speeding": 60,
    "pass_mark": 40, "attendance": 80, "password_length": 12, "cpu_alert": 85, "error_rate": 2,
    "latency_slo": 250, "coverage": 70, "approvals": 1, "reorder": 50, "bmi_over": 30.0,
    "budget": 15000, "min_balance": 5000, "atm_limit": 20000, "rain_umbrella": 50, "aqi": 100,
    "elevator": 6, "fuel_low": 10, "tyre_pressure": 32, "hotel_checkin": 12, "loan_age": 18,
    "credit_score": 750, "overtime": 40, "leave_balance": 8, "parking": 60, "screen_time": 90,
    "steps_goal": 8000, "blood_sugar": 100, "gst_invoice": 20000, "delivery_radius": 5,
}
GATE_DECIMAL = {"water_boil", "freezing"}  # integer readings in train, one-decimal readings in the gate
GATE_HOURS = {"store_open": (11, 20), "bank_hours": (10, 15), "quiet_hours": (21, 24), "support_hours": (8, 17)}
GATE_TODAY = dt.date(2028, 9, 29)  # train "today" is 2026-09-29 +/- 500 days


def compare(v, op, t):
    return {">=": v >= t, ">": v > t, "<=": v <= t, "<": v < t, "==": v == t}[op]


def near_values(t, lo, hi, is_float, rng):
    step = 0.1 if is_float else 1
    vals = [t - step, t, t + step, t - 2 * step, t + 2 * step, rng.uniform(lo, hi) if is_float else rng.randint(lo, hi),
            rng.uniform(lo, hi) if is_float else rng.randint(lo, hi), lo, hi]
    vals = [round(v, 1) if is_float else int(v) for v in vals]
    return [v for v in vals if lo <= v <= hi]


def render(state, rng):
    if rng.random() < 0.25:  # some states as plain English
        return "; ".join(f"{k.replace('_', ' ')}: {json.dumps(v) if not isinstance(v, str) else v}" for k, v in state.items())
    return state


def numeric_rows(rng, gate):
    rows = []
    for rid, dom, rule, field, unit, t, op, (lo, hi), phrasings, noun in NUMERIC:
        use = phrasings[-1:] if gate else phrasings[:-1]
        if gate:
            t = GATE_T.get(rid, t)
        is_float = isinstance(t, float) or (gate and rid in GATE_DECIMAL)
        vals = near_values(t, lo, hi, is_float, rng) if gate else [
            v for _ in range(25) for v in near_values(t, lo, hi, is_float, rng)]
        if gate and rid in GATE_DECIMAL:  # 99.9 / 100.1, never a whole number the train split uses
            vals = [v if v != int(v) else round(v + 0.5 if v + 0.5 <= hi else v - 0.5, 1) for v in vals]
        for v in vals:
            q = rng.choice(use)
            state = {"rule": rule.format(t=t), noun: {field: v}}
            if not gate and rng.random() < 0.3:  # rule inside the question instead of the state
                state = {noun: {field: v}}
                q = f"{q} (Rule: {rule.format(t=t)})"
            gold = compare(v, op, t)
            rows.append({"source": f"our-cases-basics/{dom}/{rid}", "domain": f"basics/{dom}/{rid}",
                         "state": render(state, rng) if not gate else state,
                         "questions": {"q": {"type": "noul", "instructions": q, "criteria": dict(YN)}},
                         "gold": {"q": "true" if gold else "false"}, "_facts": (rid, t, v)})
    return rows


def date_rows(rng, gate):
    rows = []
    base = dt.date(2026, 9, 29)
    for rid, dom, rule, field, phrasings, kind in DATE_RULES:
        use = phrasings[-1:] if gate else phrasings[:-1]
        offsets = [-1, 0, 1, -30, 30] if gate else [o for _ in range(40) for o in (-1, 0, 1, rng.randint(-400, 400))]
        for off in offsets:
            today = base + dt.timedelta(days=rng.randint(-500, 500)) if not gate else GATE_TODAY
            the_date = today + dt.timedelta(days=off)
            gold = today > the_date if kind == "after" else today <= the_date
            state = {"rule": rule, "today": today.isoformat(), field: the_date.isoformat()}
            rows.append({"source": f"our-cases-basics/{dom}/{rid}", "domain": f"basics/{dom}/{rid}",
                         "state": state if gate else render(state, rng),
                         "questions": {"q": {"type": "noul", "instructions": rng.choice(use), "criteria": dict(YN)}},
                         "gold": {"q": "true" if gold else "false"}, "_facts": (rid, str(today), str(the_date))})
    return rows


def time_rows(rng, gate):
    rows = []
    for rid, dom, rule, o, c, phrasings in TIME_RULES:
        use = phrasings[-1:] if gate else phrasings[:-1]
        if gate:
            o, c = GATE_HOURS[rid]
        hours = [o - 1, o, c - 1, c, (o + c) // 2] if gate else [h for _ in range(30) for h in (o - 1, o, c - 1, c, rng.randint(0, 23))]
        for h in hours:
            h %= 24
            minute = 0 if gate else rng.choice([0, 0, 15, 30, 45, 59])
            gold = o <= h < c
            state = {"rule": rule.format(o=o, c=c), "time_now": f"{h:02d}:{minute:02d}"}
            rows.append({"source": f"our-cases-basics/{dom}/{rid}", "domain": f"basics/{dom}/{rid}",
                         "state": state if gate else render(state, rng),
                         "questions": {"q": {"type": "noul", "instructions": rng.choice(use), "criteria": dict(YN)}},
                         "gold": {"q": "true" if gold else "false"}, "_facts": (rid, o, c, h, minute)})
    return rows


def meeting_rows(rng, gate):
    rows = []
    phrasings = ["Do these two meetings clash?", "Do the meetings overlap?", "Is there a calendar conflict?"]
    use = phrasings[-1:] if gate else phrasings[:-1]
    hm = lambda m: f"{m // 60:02d}:{m % 60:02d}"
    for _ in range(10 if gate else 1500):
        if gate:  # gate meetings start on the half hour; train meetings are all on the hour
            s1 = rng.randint(8, 16) * 60 + 30; d1 = rng.choice([60, 120])
            s2 = rng.choice([s1 - d1, s1 + d1, s1, s1 + 30, s1 - 30, s1 + 60, s1 - 60, s1 + 180]); d2 = rng.choice([60, 90, 120])
        else:
            s1 = rng.randint(8, 16); d1 = rng.choice([1, 2])
            s2 = rng.choice([s1 - d1, s1 + d1, s1, s1 + 1, s1 - 1, s1 + 3]); d2 = rng.choice([1, 2])
            s1, d1, s2, d2 = s1 * 60, d1 * 60, s2 * 60, d2 * 60
        gold = s1 < s2 + d2 and s2 < s1 + d1
        state = {"rule": "Two meetings clash if their times overlap; back-to-back is not a clash.",
                 "meeting_a": f"{hm(s1)}-{hm(s1 + d1)}", "meeting_b": f"{hm(s2)}-{hm(s2 + d2)}"}
        rows.append({"source": "our-cases-basics/work/meeting_clash", "domain": "basics/work/meeting_clash",
                     "state": state if gate else render(state, rng),
                     "questions": {"q": {"type": "noul", "instructions": rng.choice(use), "criteria": dict(YN)}},
                     "gold": {"q": "true" if gold else "false"}, "_facts": ("meeting", s1, d1, s2, d2)})
    return rows


def dedupe(rows):
    seen, out = set(), []
    for r in rows:
        k = json.dumps([r["state"], r["questions"]], sort_keys=True)
        if k not in seen:
            seen.add(k); out.append(r)
    return out


def main():
    train = dedupe(numeric_rows(random.Random(11), False) + date_rows(random.Random(12), False)
                   + time_rows(random.Random(13), False) + meeting_rows(random.Random(14), False))
    gate = dedupe(numeric_rows(random.Random(21), True) + date_rows(random.Random(22), True)
                  + time_rows(random.Random(23), True) + meeting_rows(random.Random(24), True))
    # No gate case may share its facts with a train case, however train worded or rendered it.
    train_facts = {r["_facts"] for r in train}
    dropped = sum(r["_facts"] in train_facts for r in gate)
    gate = [r for r in gate if r["_facts"] not in train_facts]
    print(f"basics_gate: dropped {dropped} cases whose facts are in train")
    for r in train + gate:
        del r["_facts"]
    gate_keys = {json.dumps([r["state"], r["questions"]], sort_keys=True) for r in gate}
    train = [r for r in train if json.dumps([r["state"], r["questions"]], sort_keys=True) not in gate_keys]
    random.Random(5).shuffle(train)
    for name, rows in (("basics_train", train), ("basics_gate", gate)):
        out = (paths.GATE if name.endswith("_gate") else paths.TRAIN) / f"{name}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        yes = sum(r["gold"]["q"] == "true" for r in rows)
        print(f"{name}: {len(rows)} questions, yes {yes} / no {len(rows) - yes}, rules {len({r['source'] for r in rows})}")


if __name__ == "__main__":
    main()
