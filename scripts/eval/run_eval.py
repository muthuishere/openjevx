#!/usr/bin/env python3
import json
import time
import urllib.request
import urllib.error
import base64

URL = "http://127.0.0.1:21118/v1/systemone"
AUTH = base64.b64encode(b"admin:adminadmin").decode()

def call(state, questions):
    body = json.dumps({"state": state, "questions": questions}).encode()
    req = urllib.request.Request(URL, data=body, method="POST",
                                  headers={"Content-Type": "application/json",
                                           "Authorization": f"Basic {AUTH}"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            elapsed = time.perf_counter() - started
            return json.loads(resp.read()), elapsed, None
    except urllib.error.HTTPError as e:
        return None, time.perf_counter() - started, f"HTTP {e.code}: {e.read().decode()[:200]}"
    except Exception as e:
        return None, time.perf_counter() - started, str(e)

def predicted_label(qtype, answer):
    if qtype == "choice":
        return str(answer["choice"])
    if qtype == "noul":
        return "true" if answer["noul"] >= 0.5 else "false"
    if qtype == "score":
        probs = answer["probabilities"]
        return max(probs, key=probs.get)
    return None

def confidence_for_gold(qtype, answer, gold):
    if qtype == "noul":
        return answer["probabilities"].get(gold, 1 - answer["noul"])
    return answer["probabilities"].get(str(gold), 0.0)

def main():
    cases = [json.loads(line) for line in open("cases.jsonl")]
    results = []
    errors = []
    latencies = []
    for i, case in enumerate(cases):
        resp, elapsed, err = call(case["state"], case["questions"])
        latencies.append(elapsed)
        if err:
            errors.append({"id": case["id"], "error": err})
            continue
        answers = resp["answers"]
        for qid, question in case["questions"].items():
            qtype = question["type"]
            gold = case["gold"][qid]
            answer = answers[qid]
            pred = predicted_label(qtype, answer)
            correct = (pred == str(gold))
            conf_in_pred = answer.get("confidence", None)
            conf_in_gold = confidence_for_gold(qtype, answer, str(gold))
            results.append({
                "id": case["id"], "category": case["category"], "qtype": qtype,
                "state": case["state"], "instructions": question["instructions"],
                "gold": gold, "pred": pred, "correct": correct,
                "confidence_in_pred": conf_in_pred, "prob_mass_on_gold": conf_in_gold,
                "probabilities": answer.get("probabilities", {}),
            })
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(cases)} cases done", flush=True)

    with open("results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open("errors.jsonl", "w") as f:
        for e in errors:
            f.write(json.dumps(e) + "\n")

    n = len(results)
    correct = sum(r["correct"] for r in results)
    print(f"\n=== OVERALL ===")
    print(f"decisions: {n}  correct: {correct}  accuracy: {correct/n:.4f}")
    print(f"cases with errors: {len(errors)}")
    print(f"latency p50={sorted(latencies)[len(latencies)//2]*1000:.1f}ms "
          f"p95={sorted(latencies)[int(len(latencies)*0.95)]*1000:.1f}ms "
          f"mean={sum(latencies)/len(latencies)*1000:.1f}ms")

    by_cat = {}
    by_type = {}
    for r in results:
        by_cat.setdefault(r["category"], [0, 0])
        by_cat[r["category"]][1] += 1
        by_cat[r["category"]][0] += int(r["correct"])
        by_type.setdefault(r["qtype"], [0, 0])
        by_type[r["qtype"]][1] += 1
        by_type[r["qtype"]][0] += int(r["correct"])

    print("\n=== BY CATEGORY ===")
    for cat in sorted(by_cat):
        c, t = by_cat[cat]
        print(f"{cat:35s} {c:4d}/{t:4d}  {c/t:.4f}")

    print("\n=== BY QUESTION TYPE ===")
    for qt in sorted(by_type):
        c, t = by_type[qt]
        print(f"{qt:10s} {c:4d}/{t:4d}  {c/t:.4f}")

    # Calibration: mean predicted-confidence vs empirical accuracy, bucketed
    print("\n=== CALIBRATION (confidence in the predicted answer vs actual accuracy) ===")
    buckets = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in buckets:
        bucket_results = [r for r in results if r["confidence_in_pred"] is not None and lo <= r["confidence_in_pred"] < hi]
        if not bucket_results:
            continue
        acc = sum(r["correct"] for r in bucket_results) / len(bucket_results)
        mean_conf = sum(r["confidence_in_pred"] for r in bucket_results) / len(bucket_results)
        print(f"conf[{lo:.1f},{hi:.1f}) n={len(bucket_results):4d}  mean_conf={mean_conf:.3f}  empirical_acc={acc:.3f}  gap={mean_conf-acc:+.3f}")

    # Brier score for noul questions (binary calibration quality)
    noul_results = [r for r in results if r["qtype"] == "noul"]
    if noul_results:
        brier = sum((r["probabilities"].get("true", 0.5) - (1.0 if r["gold"] == "true" else 0.0)) ** 2 for r in noul_results) / len(noul_results)
        print(f"\nBrier score (noul, lower=better, 0.25=coin-flip baseline): {brier:.4f}")

    summary = {
        "n_decisions": n, "n_correct": correct, "accuracy": correct / n,
        "n_errors": len(errors),
        "latency_ms": {
            "p50": sorted(latencies)[len(latencies)//2] * 1000,
            "p95": sorted(latencies)[int(len(latencies)*0.95)] * 1000,
            "mean": sum(latencies)/len(latencies) * 1000,
        },
        "by_category": {k: {"correct": v[0], "total": v[1], "accuracy": v[0]/v[1]} for k, v in by_cat.items()},
        "by_type": {k: {"correct": v[0], "total": v[1], "accuracy": v[0]/v[1]} for k, v in by_type.items()},
        "brier_noul": brier if noul_results else None,
    }
    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
