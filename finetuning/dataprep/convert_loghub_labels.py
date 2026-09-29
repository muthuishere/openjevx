#!/usr/bin/env python3
"""Turn loghub sets that carry real labels (BGL, OpenStack) into noul training rows.

BGL: every line of the full BGL.log starts with the operators' alert tag ("-" = no alert,
anything else = an alert they flagged). The tag is the gold and is stripped from the state.
Messages are deduplicated (at most MAX_COPIES per message once timestamps and node ids are
removed, at most TEMPLATE_CAP per number-masked template), split train/eval by template hash, and
balanced 50/50 by downsampling normals. Every alert is FATAL/FAILURE level but so are ~500k normal
lines, so half the normals are drawn from non-INFO lines to stop "FATAL" being the answer.

OpenStack: loghub's anomaly_labels.txt names the VM instances with injected anomalies in
openstack_abnormal.log. Rows are one per instance; written only if there are at least
MIN_POSITIVES anomalous instances (there are 4 in the published set, so today it is skipped).

usage: convert_loghub_labels.py [--full DIR]   (DIR holds BGL/BGL.log and OpenStack/ from Zenodo 8196385)
Writes incoming/public/{bgl_logs,openstack_logs}_{train,eval}.jsonl.
"""
import argparse, collections, hashlib, json, random, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

SEED = 20260929
EVAL_FRAC = 0.10
MAX_COPIES = 3
TEMPLATE_CAP = 400  # the alerts are ~88 templates; without a cap 7 socket-error templates dominate
CAP_PER_CLASS = 30000
MIN_POSITIVES = 50
OS_LINES = 15

BGL_Q = ["Is this an alert an operator should act on?",
         "Should a system operator take action on this log line?",
         "Does this Blue Gene/L log line report a problem that needs an operator?",
         "Would an operator flag this line as an alert?",
         "Is this log line a real alert rather than routine noise?"]
OS_Q = ["Did something go wrong with this VM instance?",
        "Should an operator investigate this VM instance's log lines?",
        "Do these OpenStack log lines show a problem with the instance?",
        "Is this instance's lifecycle abnormal?"]

HEX_NUM = re.compile(r"0x[0-9a-fA-F]+|\b[0-9a-fA-F]{6,}\b|\d+")


def is_eval(key):
    digest = hashlib.blake2b(str(key).encode(), digest_size=8, key=b"openjevx-loghub").digest()
    return int.from_bytes(digest, "big") / 2 ** 64 < EVAL_FRAC


def row(name, rng, state, phrasings, gold):
    return {"source": f"our-cases-public/{name}", "domain": f"public/{name}", "state": state,
            "questions": {"q1": {"type": "noul", "instructions": rng.choice(phrasings),
                                 "criteria": {"true": "yes", "false": "no"}}},
            "gold": {"q1": "true" if gold else "false"}}


def write(name, split, rows):
    out = paths.INCOMING / "public" / f"{name}_{split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    pos = sum(r["gold"]["q1"] == "true" for r in rows)
    print(f"{out}: {len(rows)} rows ({pos} true / {len(rows) - pos} false)")


def bgl(full, rng):
    log = full / "BGL" / "BGL.log"
    if not log.exists():
        sys.exit(f"missing {log} (download BGL.zip from zenodo record 8196385)")
    copies, per_template = collections.Counter(), collections.Counter()
    pools = {True: [], False: []}
    template_labels = collections.defaultdict(set)
    rejects = collections.Counter()
    with open(log, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(" ", 9)
            if len(parts) < 10 or not parts[9].strip():
                rejects["short line"] += 1
                continue
            alert = parts[0] != "-"
            comp, level, msg = parts[7], parts[8], parts[9].strip()
            dedup = (comp, level, msg)
            template = HEX_NUM.sub("<*>", f"{comp} {level} {msg}")
            template_labels[template].add(alert)
            if copies[dedup] >= MAX_COPIES:
                rejects["duplicate message"] += 1
                continue
            copies[dedup] += 1
            if per_template[template] >= TEMPLATE_CAP:
                rejects["template cap"] += 1
                continue
            per_template[template] += 1
            pools[alert].append((template, " ".join(parts[1:]), level != "INFO"))
    mixed = sum(len(v) == 2 for v in template_labels.values())
    print(f"bgl: {len(pools[True])} alert / {len(pools[False])} normal after dedup; "
          f"{len(template_labels)} templates, {mixed} carry both labels; rejects {dict(rejects)}")
    splits = {}
    for split in ("train", "eval"):
        want_eval = split == "eval"
        pos = [x for x in pools[True] if is_eval(x[0]) == want_eval]
        neg = [x for x in pools[False] if is_eval(x[0]) == want_eval]
        cap = CAP_PER_CLASS if split == "train" else int(CAP_PER_CLASS * EVAL_FRAC)
        n = min(len(pos), len(neg), cap)
        hard = [x for x in neg if x[2]]
        easy = [x for x in neg if not x[2]]
        k = min(len(hard), n // 2)
        negs = rng.sample(hard, k) + rng.sample(easy, n - k)
        chosen = [(x, True) for x in rng.sample(pos, n)] + [(x, False) for x in negs]
        rng.shuffle(chosen)
        print(f"bgl {split}: {n} alerts, {k} non-INFO normals, {n - k} INFO normals")
        splits[split] = [row("bgl_logs", rng, {"system": "BGL (Blue Gene/L supercomputer)", "log": x[1]},
                             BGL_Q, gold) for x, gold in chosen]
        write("bgl_logs", split, splits[split])


def openstack(full, rng):
    base = full / "OpenStack"
    labels_file, log = base / "anomaly_labels.txt", base / "openstack_abnormal.log"
    if not labels_file.exists() or not log.exists():
        print(f"openstack: skipped, missing {labels_file} or {log}")
        return
    bad = set(re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", labels_file.read_text()))
    lines = collections.defaultdict(list)
    for name in ("openstack_abnormal.log", "openstack_normal1.log", "openstack_normal2.log"):
        with open(base / name, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                m = re.search(r"\[instance: ([0-9a-f-]{36})\]", line)
                if m and len(lines[m.group(1)]) < OS_LINES:
                    lines[m.group(1)].append(line.rstrip("\n"))
    pos = [i for i in lines if i in bad]
    neg = [i for i in lines if i not in bad]
    if len(pos) < MIN_POSITIVES:
        print(f"openstack: skipped, only {len(pos)} labelled anomalous instances "
              f"(vs {len(neg)} normal); need {MIN_POSITIVES}")
        return
    n = min(len(pos), len(neg))
    chosen = [(i, True) for i in pos[:n]] + [(i, False) for i in rng.sample(neg, n)]
    for split in ("train", "eval"):
        rows = [row("openstack_logs", rng, {"system": "OpenStack", "instance": i, "log": "\n".join(lines[i])},
                    OS_Q, gold) for i, gold in chosen if is_eval(i) == (split == "eval")]
        rng.shuffle(rows)
        write("openstack_logs", split, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=str(paths.RAW / "loghub-full"))
    a = ap.parse_args()
    full = Path(a.full)
    bgl(full, random.Random(SEED))
    openstack(full, random.Random(SEED + 1))


if __name__ == "__main__":
    main()
