"""Small hand-checked probe of the four target uses: tool calling, reranking, diagnosing, programming."""
import json, sys, urllib.request

MODELS = {"base (not fine-tuned)": 21129, "v0.3": 21118, "v0.4": 21126}
YN = {"true": "yes, the statement holds", "false": "no, the statement does not hold"}


def ask(port, state, questions):
    body = json.dumps({"state": state, "questions": questions}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", body)))["answers"]


TOOLS = [
    ("What's the weather in Paris tomorrow, in celsius?",
     ["get_weather(city, date, unit)", "get_time(city)", "search_web(query)"],
     {"a": 'get_weather(city="Paris", date="tomorrow", unit="celsius")', "b": 'get_weather(city="Paris", date="today", unit="celsius")',
      "c": 'get_weather(city="London", date="tomorrow", unit="celsius")', "d": 'get_time(city="Paris")'}),
    ("Refund $25 on order 8841.",
     ["refund_order(order_id, amount)", "get_order(order_id)", "cancel_order(order_id)"],
     {"a": 'refund_order(order_id="8841", amount=25)', "b": 'refund_order(order_id="8814", amount=25)',
      "c": 'cancel_order(order_id="8841")', "d": 'refund_order(order_id="8841", amount=250)'}),
    ("Email alice@acme.com that the deploy is done.",
     ["send_email(to, subject, body)", "send_slack(channel, text)"],
     {"a": 'send_email(to="alice@acme.com", subject="Deploy done", body="The deploy is done.")',
      "b": 'send_email(to="bob@acme.com", subject="Deploy done", body="The deploy is done.")',
      "c": 'send_slack(channel="#deploys", text="The deploy is done.")'}),
    ("List the open pull requests in the payments repo.",
     ["list_prs(repo, state)", "merge_pr(repo, number)"],
     {"a": 'list_prs(repo="payments", state="open")', "b": 'list_prs(repo="payments", state="closed")',
      "c": 'merge_pr(repo="payments", number=1)'}),
    ("Convert 100 US dollars to euros.",
     ["convert(amount, from_currency, to_currency)"],
     {"a": 'convert(amount=100, from_currency="USD", to_currency="EUR")', "b": 'convert(amount=100, from_currency="EUR", to_currency="USD")',
      "c": 'convert(amount=10, from_currency="USD", to_currency="EUR")'}),
]

RERANK = [
    ("how do I reset my password", 0, [
        "To reset your password, open Settings > Security, click 'Reset password' and follow the link we email you.",
        "Invoices are generated on the 1st of every month and can be downloaded from the Billing page.",
        "Passwords must be at least 12 characters and include a number and a symbol.",
        "We are investigating an issue where some users cannot log in from the mobile app.",
        "Release 4.2 adds dark mode and faster search."]),
    ("python read a file line by line", 0, [
        "with open('data.txt') as fh:\n    for line in fh:\n        print(line.rstrip())",
        "BufferedReader br = new BufferedReader(new FileReader(\"data.txt\")); String line; while ((line = br.readLine()) != null) {}",
        "with open('out.txt', 'w') as fh:\n    fh.write('hello')",
        "squares = [x * x for x in range(10)]",
        "npm install --save lodash"]),
    ("why does my docker container exit immediately", 0, [
        "A container stops when its main process (PID 1) exits. If CMD runs a short command or a daemon that backgrounds itself, the container exits right away; run the process in the foreground.",
        "To install Docker on Ubuntu, add the apt repository and run apt-get install docker-ce.",
        "Use ports: in docker-compose.yml to publish a container port to the host.",
        "A Kubernetes pod stays Pending when no node has enough CPU or memory.",
        "Use multi-stage builds to make your image smaller."]),
]

CAUSES = {"database": "the database is down or unreachable", "bad_deploy": "a recent code deploy introduced a bug",
          "capacity": "the service is overloaded (too much traffic for its resources)",
          "certificate": "an expired or invalid TLS certificate", "network": "a network or DNS problem"}
DIAG = [
    ({"alerts": ["orders-api: 'connection refused' to db-primary:5432", "db-primary pod CrashLoopBackOff", "db node disk 100% full"],
      "recent_changes": "none in 24h"}, "database"),
    ({"alerts": ["errors started 2 minutes after deploying orders-api v2.3.1", "NullPointerException at OrderService.java:88"],
      "infrastructure": "all hosts healthy, CPU 30%"}, "bad_deploy"),
    ({"metrics": {"p95_latency": "200ms -> 4s", "cpu_all_pods": "98%", "traffic": "5x normal after a marketing email"},
      "recent_changes": "none", "errors": "timeouts only"}, "capacity"),
    ({"errors": ["SSL handshake failed: certificate has expired"], "started": "exactly at 00:00 UTC", "recent_changes": "none"}, "certificate"),
    ({"errors": ["DNS resolution failed for api.payments.internal (every service)"], "recent_changes": "network ACL updated 5 minutes ago"}, "network"),
]

CODE = [
    ("def print_all(a):\n    for i in range(len(a) + 1):\n        print(a[i])", "This code has a bug.", True),
    ("def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s", "This code has a bug.", False),
    ('def find(name):\n    q = "SELECT * FROM users WHERE name = \'" + name + "\'"\n    return db.execute(q)', "This code is vulnerable to SQL injection.", True),
    ('def find(name):\n    return db.execute("SELECT * FROM users WHERE name = %s", (name,))', "This code is vulnerable to SQL injection.", False),
    ("function isMissing(user) {\n  if (user = null) { return true; }\n  return false;\n}", "This code has a bug.", True),
    ('f, err := os.Open(path)\nif err != nil {\n    return err\n}\ndefer f.Close()', "This code has a bug.", False),
    ("def average(xs):\n    # xs can be an empty list\n    return sum(xs) / len(xs)", "This code can crash on valid input.", True),
]


def run(port):
    r = {k: [] for k in ("tool_pick", "tool_verify", "rerank_mrr", "diagnose", "code")}
    for req, tools, cands in TOOLS:
        st = {"request": req, "available_tools": tools}
        a = ask(port, st, {"q": {"type": "choice", "instructions": "Which call correctly fulfils the request (right tool and right arguments)?", "criteria": cands}})["q"]
        r["tool_pick"].append((a["choice"] == "a", a["probabilities"]["a"]))
        for key, gold in (("a", True), ("b", False)):
            v = ask(port, dict(st, proposed_call=cands[key]), {"q": {"type": "noul", "instructions": "The proposed call is correct for the request: right tool and right argument values.", "criteria": YN}})["q"]["noul"]
            r["tool_verify"].append(((v >= .5) == gold, v if gold else 1 - v))
    for query, rel, passages in RERANK:
        scores = [ask(port, {"query": query, "passage": p}, {"q": {"type": "noul", "instructions": "The passage answers the query.", "criteria": YN}})["q"]["noul"] for p in passages]
        rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(rel) + 1
        r["rerank_mrr"].append((rank == 1, 1 / rank))
    for st, gold in DIAG:
        a = ask(port, st, {"q": {"type": "choice", "instructions": "What is the most likely root cause?", "criteria": CAUSES}})["q"]
        r["diagnose"].append((a["choice"] == gold, a["probabilities"][gold]))
    for code, stmt, gold in CODE:
        v = ask(port, {"code": code}, {"q": {"type": "noul", "instructions": stmt, "criteria": YN}})["q"]["noul"]
        r["code"].append(((v >= .5) == gold, v if gold else 1 - v))
    return r


out = {}
for name, port in MODELS.items():
    out[name] = run(port)
json.dump(out, open("/Users/muthuishere/muthu/gitworkspace/openjevx/.local/usecase/probe_results.json", "w"), indent=1)
labels = {"tool_pick": "Tool calling: pick the right call (5)", "tool_verify": "Tool calling: verify a call right/wrong-arg (10)",
          "rerank_mrr": "Reranking: right passage ranked #1 (3 queries)", "diagnose": "Diagnosing: root cause (5)", "code": "Programming: bug / vuln (7)"}
print(f"{'':50s}" + "".join(f"{m:>24s}" for m in MODELS))
for k, lab in labels.items():
    row = f"{lab:50s}"
    for m in MODELS:
        xs = out[m][k]; ok = sum(x[0] for x in xs); conf = sum(x[1] for x in xs) / len(xs)
        extra = f"MRR {conf:.2f}" if k == "rerank_mrr" else f"p(right) {conf:.2f}"
        row += f"{f'{ok}/{len(xs)}  {extra}':>24s}"
    print(row)
