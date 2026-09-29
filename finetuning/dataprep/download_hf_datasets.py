#!/usr/bin/env python3
"""Download curated Hugging Face datasets for openjevx v0.5 training.

Matches the categories in docs/adr/0004-training-data-for-v0.5.md.
Outputs one JSONL file per dataset to <data>/raw/public-<date>/ and appends provenance
to <data>/raw/public-<date>/SOURCES.md.

Run with:
    python3 finetuning/dataprep/download_hf_datasets.py
"""

from __future__ import annotations

import gzip
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
# Downloads are source data, not training rows: they go under raw/, one folder per day.
INCOMING = paths.RAW / f"public-{datetime.now(timezone.utc):%Y-%m-%d}"
SOURCES = INCOMING / "SOURCES.md"
TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# Curated datasets keyed by ADR category.
# Each entry: (hf_id, config_or_None, max_rows_or_None, converter_callable)
DATASETS: dict[str, list[tuple[str, str | None, int | None, Any]]] = {
    "A": [
        # Real bugs with fixes / vulnerable vs safe code
        ("bstee615/bigvul", None, 50_000, "convert_bigvul"),
        ("mcanoglu/defect-detection", None, 50_000, "convert_defect_detection"),
        ("CyberNative/Code_Vulnerability_Security_DPO", None, 50_000, "convert_cybernative"),
    ],
    "B": [
        # Code review comments
        ("Qodo/PR-Review-Bench", None, 20_000, "download_qodo_pr_review_bench"),
    ],
    "E": [
        # Tool-call / agent traces
        ("lambda/hermes-agent-reasoning-traces", "kimi", 10_000, "convert_hermes"),
        ("lambda/hermes-agent-reasoning-traces", "glm-5.1", 10_000, "convert_hermes"),
        ("trace-commons/agent-traces", None, 5_000, "convert_trace_commons"),
        ("Exgentic/agent-llm-traces", None, 5_000, "convert_exgentic"),
    ],
    "F": [
        # Search / rerank judgments
        ("microsoft/ms_marco", "v1.1", 100_000, "convert_ms_marco"),
        ("code-search-net/code_search_net", "python", 100_000, "convert_codesearchnet"),
    ],
    "G": [
        # Domain rules as written by people
        ("huggingface/policy-docs", None, None, "convert_policy_docs"),
    ],
}


def jsonl_path(category: str, source: str, config: str | None) -> Path:
    safe = source.replace("/", "-")
    suffix = f"-{config}" if config else ""
    return INCOMING / f"{category}-{safe}{suffix}-{TODAY}.jsonl"


def write_jsonl(path: Path, rows: list[dict[str, Any]], compress: bool = True) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_path = path.with_suffix(path.suffix + ".gz") if compress else path
    opener = gzip.open if compress else open
    count = 0
    with opener(out_path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------

def convert_bigvul(ds, max_rows: int | None) -> list[dict]:
    rows = []
    for i, ex in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        rows.append({
            "category": "A",
            "source": "bstee615/bigvul",
            "task": "vulnerability_detection",
            "buggy_code": str(ex.get("before_fix", "") or ""),
            "fixed_code": str(ex.get("after_fix", "") or ""),
            "label": int(ex.get("vul", 0)) if ex.get("vul") is not None else None,
            "cve": str(ex.get("CVE_ID", "") or ""),
            "cwe": str(ex.get("CWE_ID", "") or ""),
            "raw": {k: v for k, v in ex.items()},
        })
    return rows


def convert_defect_detection(ds, max_rows: int | None) -> list[dict]:
    rows = []
    for i, ex in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        rows.append({
            "category": "A",
            "source": "mcanoglu/defect-detection",
            "task": "defect_detection",
            "code": str(ex.get("code", "") or ""),
            "label": int(ex.get("label", 0)) if ex.get("label") is not None else None,
            "raw": {k: v for k, v in ex.items()},
        })
    return rows


def convert_cybernative(ds, max_rows: int | None) -> list[dict]:
    rows = []
    for i, ex in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        rows.append({
            "category": "A",
            "source": "CyberNative/Code_Vulnerability_Security_DPO",
            "task": "vulnerability_repair",
            "instruction": str(ex.get("prompt", "") or ex.get("instruction", "") or ""),
            "vulnerable_code": str(ex.get("vulnerable", "") or ""),
            "fixed_code": str(ex.get("fixed", "") or ""),
            "raw": {k: v for k, v in ex.items()},
        })
    return rows


def convert_pr_review_bench(ds, max_rows: int | None) -> list[dict]:
    rows = []
    for i, ex in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        rows.append({
            "category": "B",
            "source": "Qodo/PR-Review-Bench",
            "task": "code_review",
            "diff": str(ex.get("diff", "") or ""),
            "comment": str(ex.get("comment", "") or ""),
            "label": int(ex.get("label", 0)) if ex.get("label") is not None else None,
            "raw": {k: v for k, v in ex.items()},
        })
    return rows


def convert_hermes(ds, max_rows: int | None) -> list[dict]:
    rows = []
    for i, ex in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        rows.append({
            "category": "E",
            "source": "lambda/hermes-agent-reasoning-traces",
            "task": "tool_call_trace",
            "tools": ex.get("tools", ""),
            "conversations": ex.get("conversations", []),
            "category_meta": str(ex.get("category", "") or ""),
            "subcategory": str(ex.get("subcategory", "") or ""),
            "raw": {k: v for k, v in ex.items()},
        })
    return rows


def convert_trace_commons(ds, max_rows: int | None) -> list[dict]:
    rows = []
    for i, ex in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        rows.append({
            "category": "E",
            "source": "trace-commons/agent-traces",
            "task": "agent_trace",
            "harness": str(ex.get("harness", "") or ""),
            "session_id": str(ex.get("session_id", "") or ""),
            "prompt": ex.get("prompt", []),
            "messages": ex.get("messages", []),
            "tools": ex.get("tools", []),
            "raw": {k: v for k, v in ex.items()},
        })
    return rows


def convert_exgentic(ds, max_rows: int | None) -> list[dict]:
    rows = []
    for i, ex in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        rows.append({
            "category": "E",
            "source": "Exgentic/agent-llm-traces",
            "task": "agent_trace",
            "trace_id": str(ex.get("trace_id", "") or ""),
            "harness": str(ex.get("harness", "") or ""),
            "model": str(ex.get("model", "") or ""),
            "messages": ex.get("messages", []),
            "tool_calls": ex.get("tool_calls", []),
            "tool_results": ex.get("tool_results", []),
            "raw": {k: v for k, v in ex.items()},
        })
    return rows


def convert_ms_marco(ds, max_rows: int | None) -> list[dict]:
    rows = []
    for i, ex in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        passages = ex.get("passages", {}) or {}
        is_selected = passages.get("is_selected", []) if isinstance(passages, dict) else []
        passage_text = passages.get("passage_text", []) if isinstance(passages, dict) else []
        rows.append({
            "category": "F",
            "source": "microsoft/ms_marco",
            "task": "search_rerank",
            "query": str(ex.get("query", "") or ""),
            "query_type": str(ex.get("query_type", "") or ""),
            "passage_texts": passage_text,
            "relevance": is_selected,
            "answers": ex.get("answers", []),
            "raw": {k: v for k, v in ex.items()},
        })
    return rows


def convert_codesearchnet(ds, max_rows: int | None) -> list[dict]:
    rows = []
    for i, ex in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        rows.append({
            "category": "F",
            "source": "code-search-net/code_search_net",
            "task": "code_search_rerank",
            "language": str(ex.get("language", "") or ""),
            "func_name": str(ex.get("func_name", "") or ""),
            "docstring": str(ex.get("docstring", "") or ex.get("func_documentation", "") or ""),
            "code": str(ex.get("func_code", "") or ex.get("code", "") or ""),
            "raw": {k: v for k, v in ex.items()},
        })
    return rows


def convert_policy_docs(ds, max_rows: int | None) -> list[dict]:
    try:
        import pdfplumber
    except ImportError:
        pdfplumber = None  # type: ignore

    rows = []
    for i, ex in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        pdf_obj = ex.get("pdf")
        text = ""
        if pdfplumber and pdf_obj is not None:
            try:
                # pdf_obj may be a file-like object or bytes path
                if hasattr(pdf_obj, "seek"):
                    pdf_obj.seek(0)
                    data = pdf_obj.read()
                else:
                    data = bytes(pdf_obj)
                with pdfplumber.open(io.BytesIO(data)) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            except Exception as e:
                text = f"[pdf extraction failed: {e}]"
        rows.append({
            "category": "G",
            "source": "huggingface/policy-docs",
            "task": "domain_rules",
            "title": str(ex.get("title", "") or ""),
            "text": text,
            "url": str(ex.get("url", "") or ""),
            "date": str(ex.get("date", "") or ""),
            "raw": {k: (str(v) if not isinstance(v, bytes) else "<bytes>") for k, v in ex.items() if k != "pdf"},
        })
    return rows


def download_qodo_pr_review_bench(_ds, max_rows: int | None) -> list[dict]:
    from huggingface_hub import hf_hub_download
    file_path = hf_hub_download(
        repo_id="Qodo/PR-Review-Bench",
        filename="git_code_review_bench_100_w_open_prs.jsonl",
        repo_type="dataset",
    )
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            ex = json.loads(line)
            rows.append({
                "category": "B",
                "source": "Qodo/PR-Review-Bench",
                "task": "code_review",
                "repo": str(ex.get("repo", "") or ""),
                "pr_url": str(ex.get("pr_url_to_review", "") or ""),
                "num_issues": ex.get("num_of_issues"),
                "issues": ex.get("issues", []),
                "raw": {k: v for k, v in ex.items()},
            })
    return rows


CONVERTERS = {
    "convert_bigvul": convert_bigvul,
    "convert_defect_detection": convert_defect_detection,
    "convert_cybernative": convert_cybernative,
    "convert_pr_review_bench": convert_pr_review_bench,
    "download_qodo_pr_review_bench": download_qodo_pr_review_bench,
    "convert_hermes": convert_hermes,
    "convert_trace_commons": convert_trace_commons,
    "convert_exgentic": convert_exgentic,
    "convert_ms_marco": convert_ms_marco,
    "convert_codesearchnet": convert_codesearchnet,
    "convert_policy_docs": convert_policy_docs,
}


def load_one(hf_id: str, config: str | None, max_rows: int | None):
    kwargs: dict[str, Any] = {"trust_remote_code": False}
    if config:
        kwargs["name"] = config
    ds = load_dataset(hf_id, split="train", **kwargs)
    total = len(ds)
    return ds, total


def main() -> int:
    INCOMING.mkdir(parents=True, exist_ok=True)
    if not SOURCES.exists():
        SOURCES.write_text("# Data sources for v0.5 training\n\n")

    summary_lines: list[str] = []

    for category, entries in DATASETS.items():
        for hf_id, config, max_rows, converter_name in entries:
            print(f"\n[{category}] {hf_id}" + (f" ({config})" if config else ""))
            try:
                if converter_name.startswith("download_"):
                    ds = None
                    total = None
                else:
                    ds, total = load_one(hf_id, config, max_rows)
            except Exception as e:
                print(f"  ERROR loading: {e}")
                summary_lines.append(
                    f"- **{category}** `{hf_id}` config={config!r}: FAILED ({e})"
                )
                continue

            converter = CONVERTERS[converter_name]
            rows = converter(ds, max_rows)
            path = jsonl_path(category, hf_id, config)
            count = write_jsonl(path, rows)
            size_mb = path.with_suffix(path.suffix + ".gz").stat().st_size / (1024 * 1024)

            total_str = str(total) if total is not None else "raw-file"
            meta = {
                "category": category,
                "hf_id": hf_id,
                "config": config,
                "split": "train",
                "rows_written": count,
                "rows_available": total,
                "max_rows_setting": max_rows,
                "output": str(path.with_suffix(path.suffix + ".gz")),
                "size_mb": round(size_mb, 2),
                "date": TODAY,
                "license": "see dataset card",
            }
            print(f"  wrote {count}/{total_str} rows -> {meta['output']} ({size_mb:.1f} MB)")

            summary_lines.append(
                f"- **{category}** `{hf_id}` config={config!r}: {count}/{total_str} rows -> `{meta['output']}`"
            )

    # Append to SOURCES.md
    with SOURCES.open("a", encoding="utf-8") as f:
        f.write(f"\n## Download run {TODAY}\n\n")
        for line in summary_lines:
            f.write(line + "\n")

    print("\nDone. Summary written to", SOURCES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
