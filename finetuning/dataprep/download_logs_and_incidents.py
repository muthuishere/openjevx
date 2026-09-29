#!/usr/bin/env python3
"""Download real production logs and public incident postmortems for v0.5.

Sources:
- Loghub (logpai/loghub): real system logs from Zenodo
- danluu/post-mortems: curated public postmortems
- icco/postmortems: structured postmortem metadata
- aifi-intelligence/ai-failure-intelligence: HF dataset of AI failures

Outputs JSONL to <data>/raw/public-<date>/ and appends to SOURCES.md.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
# Downloads are source data, not training rows: they go under raw/, one folder per day.
INCOMING = paths.RAW / f"public-{datetime.now(timezone.utc):%Y-%m-%d}"
SOURCES = INCOMING / "SOURCES.md"
TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# Smaller, labeled Loghub datasets first (real production/lab logs).
LOGHUB_DATASETS = {
    "OpenStack": "https://zenodo.org/records/8196385/files/OpenStack.tar.gz?download=1",
    "Hadoop": "https://zenodo.org/records/8196385/files/Hadoop.zip?download=1",
    "BGL": "https://zenodo.org/records/8196385/files/BGL.zip?download=1",
    "HDFS_v1": "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1",
    "Linux": "https://zenodo.org/records/8196385/files/Linux.tar.gz?download=1",
    "Apache": "https://zenodo.org/records/8196385/files/Apache.tar.gz?download=1",
    "Mac": "https://zenodo.org/records/8196385/files/Mac.tar.gz?download=1",
    "Zookeeper": "https://zenodo.org/records/8196385/files/Zookeeper.tar.gz?download=1",
}

POSTMORTEM_REPOS = {
    "danluu": "https://github.com/danluu/post-mortems.git",
    "icco": "https://github.com/icco/postmortems.git",
}


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


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} -> {dest.name}")
    urllib.request.urlretrieve(url, dest)


def extract_archive(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as z:
            z.extractall(dest_dir)
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(dest_dir)


def convert_loghub(name: str, url: str) -> tuple[Path, int]:
    """Download a Loghub archive and emit one JSONL row per log line."""
    tmp = Path(tempfile.gettempdir()) / "openjevx-loghub"
    archive = tmp / f"{name}.{'zip' if 'zip' in url else 'tar.gz'}"
    extract_dir = tmp / name
    download(url, archive)
    extract_archive(archive, extract_dir)

    rows: list[dict] = []
    label_files = list(extract_dir.rglob("*anomal*")) + list(extract_dir.rglob("*label*"))
    labels: dict[str, int] = {}
    for lf in label_files:
        if lf.is_file() and lf.suffix in (".csv", ".txt", ""):
            try:
                with lf.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line or "," not in line:
                            continue
                        parts = line.split(",")
                        key = parts[0].strip()
                        val = parts[-1].strip().lower()
                        labels[key] = 1 if val in ("anomaly", "abnormal", "alert", "fail", "failure", "1") else 0
            except Exception:
                pass

    log_files = [p for p in extract_dir.rglob("*") if p.is_file() and "readme" not in p.name.lower() and p.suffix in (".log", ".txt", "")]
    # Prefer .log files, then structured csv, skip big archives.
    log_files = sorted(log_files, key=lambda p: (not p.suffix.endswith(".log"), p.stat().st_size))

    max_per_dataset = 200_000
    for lf in log_files:
        if len(rows) >= max_per_dataset:
            break
        try:
            with lf.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if len(rows) >= max_per_dataset:
                        break
                    text = line.strip()
                    if not text:
                        continue
                    # Try to infer label from block id etc.
                    label = None
                    for key, val in labels.items():
                        if key in text:
                            label = val
                            break
                    rows.append({
                        "category": "D",
                        "source": f"logpai/loghub/{name}",
                        "task": "log_classification",
                        "log_file": str(lf.relative_to(extract_dir)),
                        "text": text,
                        "label": label,
                    })
        except Exception as e:
            print(f"    warn reading {lf}: {e}")

    out = INCOMING / f"D-loghub-{name}-{TODAY}.jsonl"
    count = write_jsonl(out, rows)
    return out, count


def clone_or_pull(repo_url: str, dest: Path) -> None:
    if dest.exists():
        subprocess.run(["git", "-C", str(dest), "pull"], check=False, capture_output=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(dest)], check=False, capture_output=True)


def convert_danluu_postmortems(repo_dir: Path) -> tuple[Path, int]:
    readme = repo_dir / "README.md"
    rows: list[dict] = []
    if readme.exists():
        text = readme.read_text(encoding="utf-8", errors="ignore")
        # Categories are ## headings; entries are [Company](url). Description
        current_category = "Uncategorized"
        for line in text.splitlines():
            heading = re.match(r"^##\s+(.+?)\s*$", line)
            if heading:
                current_category = heading.group(1).strip()
                continue
            m = re.match(r"^\[([^\]]+)\]\(([^)]+)\)\.\s*(.*)$", line)
            if m:
                title, url, description = m.groups()
                rows.append({
                    "category": "C",
                    "source": "danluu/post-mortems",
                    "task": "incident_postmortem",
                    "category_label": current_category,
                    "company": title.strip(),
                    "title": title.strip(),
                    "url": url.strip(),
                    "text": description.strip(),
                })
    out = INCOMING / f"C-danluu-post-mortems-{TODAY}.jsonl"
    count = write_jsonl(out, rows)
    return out, count


def convert_icco_postmortems(repo_dir: Path) -> tuple[Path, int]:
    data_dir = repo_dir / "data"
    json_dir = repo_dir / "tmp" / "json"
    rows: list[dict] = []
    files = list(data_dir.glob("*.md")) if data_dir.exists() else []
    if json_dir.exists():
        files.extend(json_dir.glob("*.json"))
    for f in files:
        try:
            if f.suffix == ".json":
                obj = json.loads(f.read_text(encoding="utf-8"))
                rows.append({
                    "category": "C",
                    "source": "icco/postmortems",
                    "task": "incident_postmortem",
                    "title": obj.get("title", ""),
                    "company": obj.get("company", ""),
                    "categories": obj.get("categories", []),
                    "url": obj.get("url", ""),
                    "description": obj.get("description", ""),
                    "text": "",
                })
            else:
                text = f.read_text(encoding="utf-8", errors="ignore")
                rows.append({
                    "category": "C",
                    "source": "icco/postmortems",
                    "task": "incident_postmortem",
                    "title": f.stem,
                    "company": "",
                    "text": text,
                })
        except Exception as e:
            print(f"    warn reading {f}: {e}")
    out = INCOMING / f"C-icco-postmortems-{TODAY}.jsonl"
    count = write_jsonl(out, rows)
    return out, count


def convert_ai_failure_hf() -> tuple[Path, int]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets library not installed; skipping aifi-intelligence/ai-failure-intelligence")
        return INCOMING / "placeholder", 0
    ds = load_dataset("aifi-intelligence/ai-failure-intelligence", split="train", trust_remote_code=False)
    rows = []
    for ex in ds:
        rows.append({
            "category": "C",
            "source": "aifi-intelligence/ai-failure-intelligence",
            "task": "ai_failure_incident",
            "title": str(ex.get("title", "") or ""),
            "description": str(ex.get("description", "") or ""),
            "root_cause": str(ex.get("root_cause", "") or ""),
            "categories": ex.get("categories", []),
            "severity": ex.get("severity", None),
            "recommendations": ex.get("recommendations", []),
            "raw": {k: v for k, v in ex.items()},
        })
    out = INCOMING / f"C-aifi-ai-failure-intelligence-{TODAY}.jsonl"
    count = write_jsonl(out, rows)
    return out, count


def main() -> int:
    INCOMING.mkdir(parents=True, exist_ok=True)
    if not SOURCES.exists():
        SOURCES.write_text("# Data sources for v0.5 training\n\n")

    summary_lines: list[str] = []

    # Logs
    for name, url in LOGHUB_DATASETS.items():
        print(f"\n[D] loghub/{name}")
        try:
            out, count = convert_loghub(name, url)
            size_mb = out.with_suffix(out.suffix + ".gz").stat().st_size / (1024 * 1024)
            print(f"  wrote {count} rows -> {out.with_suffix(out.suffix + '.gz').relative_to(paths.DATA)} ({size_mb:.1f} MB)")
            summary_lines.append(
                f"- **D** `logpai/loghub/{name}`: {count} rows -> `{out.with_suffix(out.suffix + '.gz').relative_to(paths.DATA)}`"
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            summary_lines.append(f"- **D** `logpai/loghub/{name}`: FAILED ({e})")

    # Postmortems
    tmp_repos = Path(tempfile.gettempdir()) / "openjevx-repos"
    for repo_name, repo_url in POSTMORTEM_REPOS.items():
        print(f"\n[C] {repo_name}/post-mortems")
        try:
            repo_dir = tmp_repos / repo_name
            clone_or_pull(repo_url, repo_dir)
            if repo_name == "danluu":
                out, count = convert_danluu_postmortems(repo_dir)
            else:
                out, count = convert_icco_postmortems(repo_dir)
            size_mb = out.with_suffix(out.suffix + ".gz").stat().st_size / (1024 * 1024)
            print(f"  wrote {count} rows -> {out.with_suffix(out.suffix + '.gz').relative_to(paths.DATA)} ({size_mb:.1f} MB)")
            summary_lines.append(
                f"- **C** `{repo_url}`: {count} rows -> `{out.with_suffix(out.suffix + '.gz').relative_to(paths.DATA)}`"
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            summary_lines.append(f"- **C** `{repo_url}`: FAILED ({e})")

    # AI failure HF dataset
    print("\n[C] aifi-intelligence/ai-failure-intelligence")
    try:
        out, count = convert_ai_failure_hf()
        if count:
            size_mb = out.with_suffix(out.suffix + ".gz").stat().st_size / (1024 * 1024)
            print(f"  wrote {count} rows -> {out.with_suffix(out.suffix + '.gz').relative_to(paths.DATA)} ({size_mb:.1f} MB)")
            summary_lines.append(
                f"- **C** `aifi-intelligence/ai-failure-intelligence`: {count} rows -> `{out.with_suffix(out.suffix + '.gz').relative_to(paths.DATA)}`"
            )
    except Exception as e:
        print(f"  ERROR: {e}")
        summary_lines.append(f"- **C** `aifi-intelligence/ai-failure-intelligence`: FAILED ({e})")

    with SOURCES.open("a", encoding="utf-8") as f:
        f.write(f"\n## Logs & incidents run {TODAY}\n\n")
        for line in summary_lines:
            f.write(line + "\n")

    print("\nDone. Summary appended to", SOURCES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
