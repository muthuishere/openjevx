"""Score a JSONL test file against an OpenJevX server: accuracy, jevx-usable, confidently wrong.

usage: gate_eval.py URL FILE [FILE...]      (rows: {state, questions, gold:{qid: label}})
jevx defaults: noul yes >= 0.8, no <= 0.2; choice/score usable when top probability >= 0.6.
"""
import json, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor


def label(g):
    return str(g.get("label", g) if isinstance(g, dict) else g).lower()


def verdict(t, a):
    p = a.get("probabilities") or {}
    if t == "noul":
        y = a["noul"]; pred = "true" if y >= .5 else "false"
        sure = "true" if y >= .8 else ("false" if y <= .2 else None)
        return pred, sure
    top = max(p, key=p.get); return top.lower(), (top.lower() if p[top] >= .6 else None)


def run(URL, r):
    body = json.dumps({"state": r["state"], "questions": r["questions"]}).encode()
    try:
        ans = json.load(urllib.request.urlopen(urllib.request.Request(URL, body), timeout=120))["answers"]
    except Exception as e:
        return []
    out = []
    for qid, q in r["questions"].items():
        gold = label(r["gold"][qid]); t = q["type"]
        if t == "score" and isinstance(q.get("criteria"), list) and not gold.isdigit():
            continue
        pred, sure = verdict(t, ans[qid])
        out.append((t, pred == gold, sure == gold, sure is not None and sure != gold))
    return out


def score(url, path):
    """{n, accuracy, confident_right, confident_wrong} for one test file."""
    rows = [json.loads(l) for l in open(path)]
    with ThreadPoolExecutor(4) as ex:
        res = [x for rs in ex.map(lambda r: run(url, r), rows) for x in rs]
    n = len(res) or 1
    return {"n": len(res), "accuracy": sum(r[1] for r in res) / n,
            "confident_right": sum(r[2] for r in res) / n, "confident_wrong": sum(r[3] for r in res) / n}


if __name__ == "__main__":
    for path in sys.argv[2:]:
        m = score(sys.argv[1], path)
        print(f"{path.split('/')[-1]:36s} n={m['n']:6d}  accuracy {m['accuracy']:6.1%}  "
              f"jevx-usable & right {m['confident_right']:6.1%}  confidently WRONG {m['confident_wrong']:5.1%}", flush=True)
