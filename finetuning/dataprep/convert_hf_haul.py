#!/usr/bin/env python3
"""Convert downloaded HF datasets into openjevx premise/hypothesis JSONL (ADR 0004 slots A, B, E).

No model labels: bug/label ground truth comes from real fixes and real review outcomes.
Writes to <data>/incoming/. Run: ./.venv-dl/bin/python finetuning/dataprep/convert_hf_haul.py
"""
import pyarrow.parquet as pq
import glob, json, random, hashlib, argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
IN = str(paths.RAW / "hf")
OUT = str(paths.INCOMING)
random.seed(7)


def wr(name, rows):
    with open(os.path.join(OUT, name), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(name, len(rows))


MAXLEN = 4000


def clip(s, n=MAXLEN):
    s = s or ""
    return s if len(s) <= n else s[: n // 2] + "\n...\n" + s[-n // 2 :]


def buggy_pair(buggy, fixed, src):
    bug = buggy.strip()
    fix = fixed.strip()
    if not bug or not fix or len(bug) > 6000 or len(fix) > 6000:
        return []
    out = []
    key = hashlib.sha1((bug + fix).encode()).hexdigest()[:12]
    out.append({"premise": f"Code:\n```java\n{bug}\n```",
                "hypothesis": "This code has a bug.", "label": 1,
                "source": src, "key": key, "image": ""})
    out.append({"premise": f"Code:\n```java\n{fix}\n```",
                "hypothesis": "This code has a bug.", "label": 0,
                "source": src, "key": key + "-neg", "image": ""})
    return out


def convert_bugs():
    rows = []
    for f in sorted(glob.glob(f"{IN}/ASSERT-KTH__megadiff-single-function/data/*.parquet")):
        t = pq.read_table(f, columns=["buggy_function", "fixed_function"])
        for b, fx in zip(t.column("buggy_function").to_pylist(), t.column("fixed_function").to_pylist()):
            rows += buggy_pair(b, fx, "hdiff-megadiff-sf")
    wr("hdiff-megadiff-sf.jsonl", rows)
    rows = []
    for f in sorted(glob.glob(f"{IN}/ASSERT-KTH__repairllama-datasets/data/ir1xor1/test.jsonl".replace("test.jsonl", "train.jsonl"),
                              ))[:2]:
        pass


def convert_repairllama(limit=100000):
    rows = []
    seen = set()
    for f in ["data/ir1xor1/train.jsonl"]:
        with open(f"{IN}/ASSERT-KTH__repairllama-datasets/{f}", buffering=1 << 20) as fh:
            for i, line in enumerate(fh):
                if len(rows) >= limit:
                    break
                d = json.loads(line)
                bug, fix = d["input"].strip(), d["output"].strip()
                if not bug or not fix or len(bug) > 6000 or len(fix) > 6000 or bug == fix:
                    continue
                key = hashlib.sha1((bug + fix).encode()).hexdigest()[:12]
                if key in seen:
                    continue
                seen.add(key)
                rows += buggy_pair(bug, fix, "hdiff-repairllama")
    wr("hdiff-repairllama.jsonl", rows)


def synthesize_row(after, changed_false_src, before, before_lines, after_lines, after_changed, repo, pr, lang, q):
    return None


def convert_codereview(limit=120000):
    rows = []
    for f in sorted(glob.glob(f"{IN}/ronantakizawa__github-codereview/data/train/*.parquet")):
        if len(rows) >= limit:
            break
        t = pq.read_table(f)
        for b, a, c, q, ct, neg, repo, pr in zip(
            t.column("before_code").to_pylist(), t.column("after_code").to_pylist(),
            t.column("reviewer_comment").to_pylist(), t.column("quality_score").to_pylist(),
            t.column("comment_type").to_pylist(), t.column("is_negative").to_pylist(),
            t.column("repo_name").to_pylist(), t.column("pr_number").to_pylist(),
        ):
            if len(rows) >= limit:
                break
            if not b or not a or not c or q is None or q < 0.6:
                continue
            if len(b) > 5000 or len(a) > 5000 or len(c) > 700:
                continue
            pre = (f"Repo {repo} PR #{pr}. Code under review:\n"
                   f"```\n{b.strip()}\n```\nReviewer comment: {c.strip()}\n")
            key = hashlib.sha1(f"{repo}#{pr}{b}{c}".encode()).hexdigest()[:12]
            if ct and ct != "none":
                cat = f"The comment flags a real problem in this code (type: {ct})."
                rows.append({"premise": pre, "hypothesis": cat, "label": 1, "source": "codereview-typed", "key": key, "image": ""})
    wr("codereview-typed.jsonl", rows)


def mutate_args(call, rng):
    """Deterministic one-argument corruptions for tool-call hard negatives."""
    muts = []
    for k, v in list(call["arguments"].items()):
        if isinstance(v, str):
            m = dict(call); m["arguments"] = dict(call["arguments"]); m["arguments"][k] = v + "-x"
            muts.append(("wrong value", m))
            m2 = dict(call); m2["arguments"] = dict(call["arguments"]); m2["arguments"][k] = len(v)
            muts.append(("wrong type", m2))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            m = dict(call); m["arguments"] = dict(call["arguments"]); m["arguments"][k] = v + 1
            muts.append(("wrong value", m))
            m3 = dict(call); m3["arguments"] = dict(call["arguments"]); m3["arguments"][k] = str(v)
            muts.append(("wrong type", m3))
        elif isinstance(v, (int, float)):
            pass
        elif isinstance(v, bool):
            m = dict(call); m["arguments"] = dict(call["arguments"]); m["arguments"][k] = not v
            muts.append(("wrong value", m))
        elif isinstance(v, list) and v:
            m = dict(call); m["arguments"] = dict(call["arguments"]); m["arguments"][k] = v[:-1]
            muts.append(("shorter list", m))
    # missing required-arg style: drop first arg
    if len(call["arguments"]) > 1:
        k0 = list(call["arguments"])[0]
        m = dict(call); m["arguments"] = {k: v for k, v in call["arguments"].items() if k != k0}
        muts.append(("missing field", m))
    return muts[:3]


TOOLKEY = "toolcall-apigen"


def convert_toolcalls(limit=40000):
    rows = []
    f = glob.glob(f"{IN}/argilla__apigen-function-calling/data/*.parquet")[0]
    t = pq.read_table(f)
    queries = t.column("query").to_pylist()
    tools = t.column("tools").to_pylist()
    answers = t.column("answers").to_pylist()
    idx = list(range(len(queries)))
    random.shuffle(idx)
    for i in idx:
        if len(rows) >= limit:
            break
        try:
            ql, tl, al = queries[i], json.loads(tools[i]), json.loads(answers[i])
        except Exception:
            continue
        if not al or not tl:
            continue
        tool_names = set()
        for tt in tl:
            tt = tt.get("function", tt)
            tool_names.add(tt.get("name"))
        def call_str(c):
            args = ", ".join(f"{k}={json.dumps(v)}" for k, v in (c.get("arguments") or {}).items())
            return f"{c['name']}({args})"
        correct = al[0] if len(al) == 1 and isinstance(al[0].get("arguments"), dict) else None
        if not correct:
            continue
        pre = f"User request: {json.dumps(ql)[:1500] if isinstance(ql, str) else json.dumps(str(ql))[:1500]}\n"
        pre += "Available tools: " + (json.dumps(tl)[:2000] if isinstance(tl, str) else json.dumps(tl))[:2000] + "\n"
        key = hashlib.sha1(f"{ql}".encode()).hexdigest()[:12]
        rows.append({"premise": pre, "hypothesis": f"The correct call is {call_str(correct)}.",
                     "label": 1, "source": "toolcall-apigen", "key": key, "image": ""})
        for kind, m in mutate_args(correct, random):
            if m["name"] not in tool_names and kind != "wrong tool":
                continue
            rows.append({"premise": pre, "hypothesis": f"The correct call is {call_str(m)}.",
                         "label": 0, "source": "toolcall-apigen-neg", "key": key + "-" + kind[:4], "image": ""})
    wr("toolcall-apigen.jsonl", rows)


def convert_crafe():
    rows = []
    for f in sorted(glob.glob(f"{IN}/TuringEnterprises__CRAVE/*.parquet")):
        t = pq.read_table(f)
        for p, lab, repo, pr in zip(t.column("patch").to_pylist(), t.column("label").to_pylist(),
                                    t.column("repo").to_pylist(), t.column("pr_number").to_pylist()):
            if not p or lab not in ("APPROVE", "REQUEST_CHANGES") or len(p) > 6000:
                continue
            pre = f"Repo {repo} PR #{pr} diff:\n{p.strip()[:5500]}"
            key = hashlib.sha1(f"{repo}{pr}{p}".encode()).hexdigest()[:12]
            rows.append({"premise": pre, "hypothesis": "This change was accepted by the human reviewer.",
                         "label": 1 if lab == "APPROVE" else 0, "source": "prverdict-crave", "key": key, "image": ""})
    wr("prverdict-crave.jsonl", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*", default=None)
    a = ap.parse_args()
    todo = a.only or ["bugs", "repairllama", "codereview", "toolcalls", "crafe"]
    if "bugs" in todo:
        convert_bugs()
    if "repairllama" in todo:
        convert_repairllama()
    if "codereview" in todo:
        convert_codereview()
    if "toolcalls" in todo:
        convert_toolcalls()
    if "crafe" in todo:
        convert_crafe()
