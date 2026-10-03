#!/usr/bin/env python3
"""Tests for import_csv.py. Run: python finetuning/dataprep/test_import_csv.py (or pytest)."""
import csv, json, os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import import_csv as ic  # noqa: E402
import adapter  # noqa: E402  (import_csv puts finetuning/train on the path)

HEADER = ["state", "question", "type", "options", "answer", "split", "source"]


def write_csv(rows, header=HEADER):
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def one(**kw):
    rec = {"state": '{"x": 1}', "question": "Q?", "type": "", "options": "", "answer": "yes", "split": "", "source": ""}
    rec.update(kw)
    return ic.build_row(rec)


def test_noul():
    split, row = one(answer="No", options="true=late|false=on time")
    assert row["gold"] == {"q1": "false"}
    assert row["questions"]["q1"] == {"type": "noul", "instructions": "Q?", "criteria": {"true": "late", "false": "on time"}}
    assert row["source"] == "your-data" and row["state"] == {"x": 1}
    assert adapter.adapt_row(row)["gold"]["q1"]["label"] == "false"


def test_choice():
    _, row = one(type="choice", options="billing=money|engineering=bugs|sales", answer="Engineering")
    assert row["gold"]["q1"] == "engineering"
    assert row["questions"]["q1"]["criteria"] == {"billing": "money", "engineering": "bugs", "sales": "sales"}
    assert adapter.adapt_row(row)["gold"]["q1"]["probabilities"]["engineering"] == 0.9


def test_score_by_name_and_index():
    _, a = one(type="score", options="low|medium|high", answer="high")
    _, b = one(type="score", options="low|medium|high", answer="2")
    assert a["gold"]["q1"] == b["gold"]["q1"] == "2"
    assert a["questions"]["q1"]["criteria"] == ["low", "medium", "high"]
    assert adapter.adapt_row(a)["gold"]["q1"]["probabilities"]["2"] == 0.9


def test_plain_text_state():
    _, row = one(state="Promised Friday, came Monday.")
    assert row["state"] == "Promised Friday, came Monday."


def bad(**kw):
    try:
        one(**kw)
    except ic.RowError as e:
        return str(e)
    raise AssertionError(f"expected an error for {kw}")


def test_bad_rows():
    assert "not yes/no" in bad(answer="maybe")
    assert "not one of the options" in bad(type="choice", options="a|b", answer="c")
    assert "at least 2" in bad(type="choice", options="a", answer="a")
    assert "past the last level" in bad(type="score", options="low|high", answer="5")
    assert "not valid JSON" in bad(state="{x: 1}")
    assert "state is empty" in bad(state="")
    assert "question is empty" in bad(question="")
    assert "unknown" in bad(type="rating")
    assert "unknown" in bad(split="dev")
    assert "true=" in bad(options="late|early")


def test_file_errors_and_conflicts():
    path = write_csv([
        ['{"d": 1}', "Late?", "", "", "yes", "", ""],
        ['{"d": 1}', "Late?", "", "", "no", "", ""],       # same facts, other answer
        ['{"d": 2}', "Late?", "", "", "perhaps", "", ""],  # bad answer
        ['{"d": 3}', "Late?", "", "", "no", "test", ""],
    ])
    out, errors = ic.convert(path)
    assert [line for line, _ in errors] == [3, 4], errors
    assert "different answer" in errors[0][1]
    assert len(out["test"]) == 1


def test_split_deterministic():
    a = [one(state=json.dumps({"n": i}))[0] for i in range(400)]
    b = [one(state=json.dumps({"n": i}))[0] for i in range(400)]
    assert a == b
    gate = a.count("gate") / len(a)
    assert 0.05 < gate < 0.15, gate
    assert one(split="TEST")[0] == "test"


def test_example_csv_and_cli():
    example = HERE.parent / "examples" / "decisions.csv"
    out, errors = ic.convert(example)
    assert not errors, errors
    rows = [r for rs in out.values() for r in rs]
    assert len(rows) >= 35
    # The managed (SageMaker) fine-tune needs 100 trainable items; its smoke shard is the train rows x2, so the
    # example must carry at least 50 train rows or the first job a user tries with it is refused.
    assert len(out["train"]) >= 50, len(out["train"])
    assert {r["questions"]["q1"]["type"] for r in rows} == {"noul", "choice", "score"}
    assert not ic.adapter_rejects(rows)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "config.json"
        env = {**os.environ, "OPENJEVX_DATA": str(Path(tmp) / "data"), "OPENJEVX_FT_CONFIG": str(cfg)}
        run = lambda *a: subprocess.run([sys.executable, HERE / "import_csv.py", example, "--name", "ex", *a],
                                        env=env, capture_output=True, text=True)
        r = run("--add-to-config")
        assert r.returncode == 0, r.stdout + r.stderr
        assert (Path(tmp) / "data/train/ex_train.jsonl").exists() and (Path(tmp) / "data/gate/ex_gate.jsonl").exists()
        assert run("--add-to-config", "--repeat", "5").returncode == 0
        c = json.loads(cfg.read_text())
        mine = [x for x in c["dataprep"]["extra_train"] if x["file"] == "train/ex_train.jsonl"]
        assert mine == [{"file": "train/ex_train.jsonl", "repeat": 5}]
        assert c["gate"]["files"].count("gate/ex_gate.jsonl") == 1
        assert "gate/ex_gate.jsonl" in c["datavalidate"]["leak_eval"]
        assert list(Path(tmp).glob("config.json.bak-*"))
        badcsv = write_csv([['{"d": 1}', "Late?", "", "", "perhaps", "", ""]])
        r = subprocess.run([sys.executable, HERE / "import_csv.py", badcsv, "--name", "bad"], env=env,
                           capture_output=True, text=True)
        assert r.returncode != 0 and "line 2" in r.stdout and not (Path(tmp) / "data/train/bad_train.jsonl").exists()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
