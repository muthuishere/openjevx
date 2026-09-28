#!/usr/bin/env python3
"""Generate a large, deterministic-gold test suite for OpenJevX and score it.

Every case has a mechanically-verifiable correct answer derived from the
`state` text itself (numeric comparison, explicit stated fact, counting,
simple conjunction, etc). No external dataset is used -- huggingface.co is
blocked by this sandbox's network policy, so the official
LocalLLaMA/typed-decisions benchmark used in the project's own ADRs could
not be fetched. This suite targets the same *kind* of task (typed choice /
score / noul decisions over a provided state) with self-checkable labels.
"""
import json
import random

random.seed(20260928)

CASES = []  # each: {"id":..., "category":..., "state":..., "questions": {...}, "gold": {...}}

def add(cat, state, questions, gold):
    CASES.append({
        "id": f"{cat}-{len(CASES):05d}",
        "category": cat,
        "state": state,
        "questions": questions,
        "gold": gold,
    })

# ---------------------------------------------------------------------------
# Category A: numeric comparison (noul) -- "does X exceed Y"
# Split into "clear" (>15% gap) and "close" (<=5% gap) sub-buckets.
# ---------------------------------------------------------------------------
SUBJECTS_AB = [
    ("invoice total", "approved budget", "$"),
    ("account balance", "minimum required balance", "$"),
    ("current temperature", "safety threshold temperature", "C"),
    ("server CPU usage", "alert threshold", "%"),
    ("candidate years of experience", "required years of experience", " years"),
    ("shipment weight", "maximum allowed weight", " kg"),
    ("page load time", "SLA limit", " ms"),
    ("patient heart rate", "normal upper limit", " bpm"),
]
for i in range(220):
    subj, ref, unit = SUBJECTS_AB[i % len(SUBJECTS_AB)]
    clear = i % 3 != 0
    base = random.randint(50, 5000)
    if clear:
        gap = base * random.uniform(0.2, 0.8)
    else:
        gap = base * random.uniform(0.01, 0.05)
    higher_is_a = i % 2 == 0
    a = base + gap if higher_is_a else base
    b = base if higher_is_a else base + gap
    a, b = round(a, 2), round(b, 2)
    state = f"The {subj} is {a}{unit}. The {ref} is {b}{unit}."
    gold_true = a > b
    add(
        "A_numeric_compare_clear" if clear else "A_numeric_compare_close",
        state,
        {"exceeds": {"type": "noul", "instructions": f"Does the {subj} exceed the {ref}?"}},
        {"exceeds": "true" if gold_true else "false"},
    )

# ---------------------------------------------------------------------------
# Category B: explicit stated boolean fact + negation (noul)
# ---------------------------------------------------------------------------
FACTS = [
    ("door", "locked"), ("server", "online"), ("invoice", "paid"),
    ("account", "verified"), ("shipment", "delivered"), ("ticket", "resolved"),
    ("employee", "on leave"), ("contract", "signed"), ("backup", "completed"),
    ("test suite", "passing"), ("firewall", "enabled"), ("license", "expired"),
]
for i in range(180):
    subj, adj = FACTS[i % len(FACTS)]
    negated = i % 2 == 1
    if negated:
        state = f"The {subj} is not {adj}."
        gold = "false"
    else:
        state = f"The {subj} is {adj}."
        gold = "true"
    add(
        "B_explicit_fact",
        state,
        {"fact": {"type": "noul", "instructions": f"Is the {subj} {adj}?"}},
        {"fact": gold},
    )

# ---------------------------------------------------------------------------
# Category C: choice from an explicitly labeled value in text
# ---------------------------------------------------------------------------
CHOICE_SETS = [
    ("status", ["open", "closed", "pending", "cancelled"]),
    ("weather", ["sunny", "rainy", "cloudy", "snowy"]),
    ("priority", ["low", "medium", "high", "critical"]),
    ("department", ["sales", "engineering", "finance", "support"]),
    ("shirt color", ["red", "blue", "green", "black"]),
    ("day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
]
for i in range(180):
    field, options = CHOICE_SETS[i % len(CHOICE_SETS)]
    correct = options[i % len(options)]
    state = f"The current {field} is '{correct}'."
    add(
        "C_explicit_choice",
        state,
        {"pick": {"type": "choice", "instructions": f"What is the current {field}?",
                  "criteria": {o: "" for o in options}}},
        {"pick": correct},
    )

# ---------------------------------------------------------------------------
# Category D: ordinal bucket / score from an explicit count
# ---------------------------------------------------------------------------
COUNT_SUBJECTS = [
    "unresolved support tickets", "failed login attempts", "items in the cart",
    "overdue invoices", "open pull requests", "pending approvals",
]
def bucket(n):
    if n == 0:
        return 0
    if n <= 3:
        return 1
    if n <= 10:
        return 2
    return 3
for i in range(180):
    subj = COUNT_SUBJECTS[i % len(COUNT_SUBJECTS)]
    n = [0, 1, 2, 3, 5, 8, 10, 15, 30, 100][i % 10]
    state = f"There are {n} {subj}."
    gold_bucket = bucket(n)
    add(
        "D_count_bucket",
        state,
        {"level": {"type": "score",
                   "instructions": f"How many {subj} are there?",
                   "criteria": ["none (0)", "few (1-3)", "some (4-10)", "many (11+)"]}},
        {"level": str(gold_bucket)},
    )

# ---------------------------------------------------------------------------
# Category E: cheapest-of-three choice (simple arithmetic over stated values)
# ---------------------------------------------------------------------------
for i in range(160):
    prices = random.sample(range(50, 2000), 3)
    a, b, c = prices
    state = f"Vendor A costs ${a}. Vendor B costs ${b}. Vendor C costs ${c}."
    names = ["A", "B", "C"]
    cheapest = names[prices.index(min(prices))]
    add(
        "E_cheapest_of_three",
        state,
        {"cheapest": {"type": "choice", "instructions": "Which vendor is cheapest?",
                      "criteria": {"A": "", "B": "", "C": ""}}},
        {"cheapest": cheapest},
    )

# ---------------------------------------------------------------------------
# Category F: two-fact conjunction (noul) -- both conditions must hold
# ---------------------------------------------------------------------------
for i in range(180):
    temp = random.randint(15, 40)
    humidity = random.randint(30, 95)
    state = f"The temperature is {temp} degrees C and the humidity is {humidity} percent."
    gold = (temp > 30) and (humidity > 70)
    add(
        "F_conjunction",
        state,
        {"hot_and_humid": {"type": "noul",
                            "instructions": "Is it both hot (over 30C) and humid (over 70 percent)?"}},
        {"hot_and_humid": "true" if gold else "false"},
    )

# ---------------------------------------------------------------------------
# Category G: no-context world-knowledge control (state carries none of the
# answer -- tests whether the model hallucinates confident answers when it
# has no grounding, vs. correctly showing low/near-uniform confidence)
# ---------------------------------------------------------------------------
TRIVIA = [
    ("What is the capital of France?", {"Paris": "", "Berlin": "", "Madrid": "", "Rome": ""}, "Paris"),
    ("What is 2 + 2?", {"3": "", "4": "", "5": "", "22": ""}, "4"),
    ("Is water composed of hydrogen and oxygen?", None, "true"),
    ("What is the largest planet in the solar system?", {"Earth": "", "Jupiter": "", "Mars": "", "Venus": ""}, "Jupiter"),
    ("Is the sky green?", None, "false"),
]
for i in range(160):
    q, options, gold = TRIVIA[i % len(TRIVIA)]
    state = ""  # deliberately empty -- no grounding provided
    if options is None:
        add("G_no_context_trivia_noul", state,
            {"ans": {"type": "noul", "instructions": q}}, {"ans": gold})
    else:
        add("G_no_context_trivia_choice", state,
            {"ans": {"type": "choice", "instructions": q, "criteria": options}}, {"ans": gold})

print(f"generated {len(CASES)} cases", flush=True)
total_decisions = sum(len(c["questions"]) for c in CASES)
print(f"generated {total_decisions} decisions", flush=True)

with open("cases.jsonl", "w") as f:
    for c in CASES:
        f.write(json.dumps(c) + "\n")
