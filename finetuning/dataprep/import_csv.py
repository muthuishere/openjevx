#!/usr/bin/env python3
"""Turn your own decisions (a CSV) into OpenJevX training rows.

usage: import_csv.py FILE.csv --name NAME [--skip-bad] [--add-to-config [--repeat 3]] [--dry-run]

CSV (UTF-8, comma, header row); one decision per row:
  state     required  JSON object or plain text: the facts the decision depends on
  question  required  the instruction, e.g. "Is the parcel late?"
  type      optional  noul (yes/no, default) | choice | score
  options   choice: "billing|engineering|sales" or "billing=payments and charges|engineering=bugs"
            score:  "low|medium|high" (lowest first)
            noul:   empty, or "true=...|false=..." (what yes and no mean)
  answer    required  noul: true/false/yes/no/1/0 · choice: one option key · score: level name or index (0 = lowest)
  split     optional  train | test | gate (default: 90% train / 10% gate, fixed by a hash of state+question)
  source    optional  a tag for where the row came from (default your-data)

Writes <data>/train/NAME_train.jsonl, <data>/eval/NAME_eval.jsonl, <data>/gate/NAME_gate.jsonl
(<data> is ~/openjevx/data or $OPENJEVX_DATA). Refuses to write if any row is bad, unless --skip-bad.
--add-to-config adds the files to ~/.config/openjevx/config.json (backed up first) so `task all` uses them.
"""
import argparse, csv, hashlib, json, os, shutil, sys, time
from collections import Counter, defaultdict
from pathlib import Path

FT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FT))
sys.path.insert(0, str(FT / "train"))
import paths  # noqa: E402
import adapter  # noqa: E402

TYPES = ("noul", "choice", "score")
SPLITS = ("train", "test", "gate")
YES, NO = ("true", "yes", "1", "y"), ("false", "no", "0", "n")
MIN_ROWS, MAX_SHARE = 20, 0.8


class RowError(Exception):
    pass


def parse_options(text):
    """'a|b' or 'a=desc|b=desc' -> ordered dict {key: description}."""
    out = {}
    for part in [p.strip() for p in (text or "").split("|") if p.strip()]:
        key, _, desc = part.partition("=")
        key, desc = key.strip(), desc.strip()
        if not key:
            raise RowError(f"option '{part}' has no name before '='")
        if key in out:
            raise RowError(f"option '{key}' is listed twice")
        out[key] = desc or key
    return out


def parse_state(text):
    text = (text or "").strip()
    if not text:
        raise RowError("state is empty: put the facts the decision depends on here, e.g. {\"days_late\": 3}")
    if text.startswith("{"):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as e:
            raise RowError(f"state starts with '{{' but is not valid JSON ({e.msg} at char {e.pos}); "
                           "check quotes (use \"double quotes\", and double them inside a CSV cell) or write plain text")
        if not isinstance(value, dict) or not value:
            raise RowError("state JSON must be a non-empty object")
        return value
    return text


def default_split(state_text, question):
    digest = hashlib.sha256(f"{state_text}\n{question}".encode("utf-8")).digest()
    return "gate" if int.from_bytes(digest[:8], "big") % 10 == 0 else "train"


def build_row(rec):
    """One CSV record (dict) -> (split, training row). Raises RowError with a fix-it message."""
    get = lambda k: (rec.get(k) or "").strip()
    state = parse_state(get("state"))
    question = get("question")
    if not question:
        raise RowError("question is empty: write the instruction, e.g. \"Is the parcel late?\"")
    qtype = (get("type") or "noul").lower()
    if qtype in ("yes/no", "yesno", "bool", "boolean"):
        qtype = "noul"
    if qtype not in TYPES:
        raise RowError(f"type '{qtype}' is unknown: use noul (yes/no), choice or score")
    answer = get("answer")
    if not answer:
        raise RowError("answer is empty")
    options = parse_options(get("options"))
    q = {"type": qtype, "instructions": question}
    if qtype == "noul":
        extra = set(options) - {"true", "false"}
        if extra:
            raise RowError(f"a noul question takes options only as 'true=...|false=...', not {sorted(extra)}")
        if options and set(options) != {"true", "false"}:
            raise RowError("give both 'true=...' and 'false=...' or leave options empty")
        if options:
            q["criteria"] = {"true": options["true"], "false": options["false"]}
        low = answer.lower()
        if low in YES:
            gold = "true"
        elif low in NO:
            gold = "false"
        else:
            raise RowError(f"answer '{answer}' is not yes/no: use true/false, yes/no or 1/0")
    elif qtype == "choice":
        if len(options) < 2:
            raise RowError("a choice question needs at least 2 options, e.g. billing|engineering|sales")
        if answer not in options:
            match = [k for k in options if k.lower() == answer.lower()]
            if not match:
                raise RowError(f"answer '{answer}' is not one of the options {list(options)}")
            answer = match[0]
        q["criteria"] = options
        gold = answer
    else:
        levels = list(options)
        if len(levels) < 2:
            raise RowError("a score question needs at least 2 levels, lowest first, e.g. low|medium|high")
        if answer.isdigit():
            if int(answer) >= len(levels):
                raise RowError(f"answer {answer} is past the last level (0..{len(levels) - 1}: {levels})")
            gold = str(int(answer))
        else:
            match = [i for i, k in enumerate(levels) if k.lower() == answer.lower()]
            if not match:
                raise RowError(f"answer '{answer}' is not a level name {levels} or index 0..{len(levels) - 1}")
            gold = str(match[0])
        q["criteria"] = levels
    split = (get("split") or "").lower()
    if split and split not in SPLITS:
        raise RowError(f"split '{split}' is unknown: use train, test or gate, or leave it empty")
    source = get("source") or "your-data"
    split = split or default_split(get("state"), question)
    return split, {"source": source, "domain": source, "state": state,
                   "questions": {"q1": q}, "gold": {"q1": gold}}


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = [h.strip().lower() for h in (reader.fieldnames or [])]
        missing = [c for c in ("state", "question", "answer") if c not in header]
        if missing:
            sys.exit(f"{path}: header is missing column(s) {missing}; the header row needs at least state,question,answer")
        reader.fieldnames = header
        for rec in reader:
            yield reader.line_num, rec


def convert(path):
    """-> ({split: [rows]}, [(line, error)])"""
    out, errors, seen = defaultdict(list), [], {}
    for line, rec in read_csv(path):
        if not any((v or "").strip() for v in rec.values() if isinstance(v, str)):
            continue
        if None in rec:
            errors.append((line, f"row has more cells than the header ({len(rec[None])} extra); "
                                 "wrap a cell containing commas in \"double quotes\""))
            continue
        try:
            split, row = build_row(rec)
        except RowError as e:
            errors.append((line, str(e)))
            continue
        key = (json.dumps(row["state"], sort_keys=True), row["questions"]["q1"]["instructions"])
        if key in seen and seen[key][1] != row["gold"]["q1"]:
            errors.append((line, f"same state and question as line {seen[key][0]} but a different answer; "
                                 "a fact that decides the answer is missing from the state"))
            continue
        seen[key] = (line, row["gold"]["q1"])
        out[split].append(row)
    return out, errors


def report(out):
    by_q = defaultdict(Counter)
    for rows in out.values():
        for r in rows:
            q = r["questions"]["q1"]
            label = r["gold"]["q1"]
            if q["type"] == "score":
                label = q["criteria"][int(label)]
            by_q[q["instructions"]][label] += 1
    print(f"\n{'question':60s} rows  labels")
    warnings = []
    for question, labels in sorted(by_q.items()):
        n = sum(labels.values())
        print(f"{question[:60]:60s} {n:4d}  " + ", ".join(f"{k}={v}" for k, v in labels.most_common()))
        if n < MIN_ROWS:
            warnings.append(f"'{question}': only {n} rows; aim for {MIN_ROWS}+ with near-miss pairs around the rule")
        top, count = labels.most_common(1)[0]
        if n > 1 and count / n > MAX_SHARE:
            warnings.append(f"'{question}': {count / n:.0%} of answers are '{top}'; add rows with the other answer(s)")
    for w in warnings:
        print("warning:", w)
    return warnings


def adapter_rejects(rows):
    return [r for r in rows if adapter.adapt_row(json.loads(json.dumps(r))) is None]


def targets(name):
    return {"train": paths.TRAIN / f"{name}_train.jsonl", "test": paths.EVAL / f"{name}_eval.jsonl",
            "gate": paths.GATE / f"{name}_gate.jsonl"}


def add_to_config(written, repeat, cfg_path=None):
    cfg_path = Path(cfg_path or os.environ.get("OPENJEVX_FT_CONFIG", Path.home() / ".config/openjevx/config.json"))
    if not cfg_path.exists():
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FT / "config.example.json", cfg_path)
        print(f"created {cfg_path} from config.example.json")
    cfg = json.loads(cfg_path.read_text())
    backup = cfg_path.with_name(f"{cfg_path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(cfg_path, backup)
    rel = {split: str(p.relative_to(paths.DATA)) for split, p in written.items()}

    def add(lst, item):
        if item not in lst:
            lst.append(item)

    d, v, g = cfg.setdefault("dataprep", {}), cfg.setdefault("datavalidate", {}), cfg.setdefault("gate", {})
    if "train" in rel:
        extra = d.setdefault("extra_train", [])
        extra[:] = [x for x in extra if x.get("file") != rel["train"]] + [{"file": rel["train"], "repeat": repeat}]
        add(v.setdefault("leak_train", []), rel["train"])
    if "test" in rel:
        add(d.setdefault("extra_eval", []), rel["test"])
        add(v.setdefault("leak_eval", []), rel["test"])
    if "gate" in rel:
        add(g.setdefault("files", []), rel["gate"])
        add(v.setdefault("leak_eval", []), rel["gate"])
    tmp = cfg_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2) + "\n")
    os.replace(tmp, cfg_path)
    print(f"updated {cfg_path} (backup: {backup.name}): " + ", ".join(f"{k} -> {p}" for k, p in rel.items()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--name", required=True, help="short name for the output files, e.g. mydata")
    ap.add_argument("--skip-bad", action="store_true", help="write the good rows even if some rows are bad")
    ap.add_argument("--add-to-config", action="store_true", help="add the written files to your fine-tune config")
    ap.add_argument("--repeat", type=int, default=3, help="how many times the train file is repeated in the mix (default 3)")
    ap.add_argument("--dry-run", action="store_true", help="check and report, write nothing")
    args = ap.parse_args(argv)
    if not args.name.replace("-", "").replace("_", "").isalnum():
        sys.exit("--name: use letters, digits, - and _ only")

    out, errors = convert(args.csv)
    total = sum(len(r) for r in out.values())
    print(f"{args.csv}: {total} good rows, {len(errors)} bad")
    for line, msg in errors:
        print(f"  line {line}: {msg}")
    report(out)
    rejects = [r for rows in out.values() for r in adapter_rejects(rows)]
    if rejects:
        print(f"adapter rejected {len(rejects)} row(s) (the trainer would drop them):")
        for r in rejects[:10]:
            print("  ", json.dumps(r)[:200])
    else:
        print("adapter: every row accepted")
    if errors and not args.skip_bad:
        sys.exit(f"\nnot written: fix the {len(errors)} bad row(s) above, or pass --skip-bad to write the good ones")
    if not total:
        sys.exit("nothing to write")
    if args.dry_run:
        return
    written = {}
    for split, path in targets(args.name).items():
        rows = out.get(split, [])
        if not rows:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for r in rows:
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")
        written[split] = path
        print(f"wrote {len(rows):5d} {split:5s} rows -> {path}")
    if "gate" not in written:
        print("note: no gate rows; mark some rows split=gate so you can measure the model on your rules")
    if args.add_to_config:
        add_to_config(written, args.repeat)


if __name__ == "__main__":
    main()
