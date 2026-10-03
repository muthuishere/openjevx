"""Licence filter: the model trains only on data whose licence allows commercial use.

tasksource rows are kept only when the dataset's own `license_use` is in config licence.tasksource_allow. The
column is `commercial`, `non-commercial` or `unspecified` (the tasksource-jev-typed-decisions card: the most
restrictive of the card and DPI licences). Rows in train_openjevx.jsonl carry their tasksource subset as `domain`
(build_master.py), so each row's `license_use` is looked up from the dataset's parquet files by that subset. Building
the map fails if a subset ever has two values, and a row whose subset is missing from the map is dropped (fail
closed). Sources named in licence.exclude_sources (e.g. MS MARCO, non-commercial research only) are dropped whole.

  python3 finetuning/dataprep/licence.py     # writes <data>/work/licence/tasksource_license_use.json
"""
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

RAW = paths.RAW / "tasksource-jev-typed-decisions" / "data"
MAP_PATH = paths.WORK / "licence" / "tasksource_license_use.json"


def build_map(out=MAP_PATH):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(RAW / "*.parquet")))
    if not files:
        sys.exit(f"licence: no tasksource parquet files under {RAW}")
    uses = defaultdict(set)
    for f in files:
        t = pq.read_table(f, columns=["source", "license_use"]).to_pydict()
        for s, u in zip(t["source"], t["license_use"]):
            uses[s].add(u)
    mixed = {s: sorted(u) for s, u in uses.items() if len(u) != 1}
    if mixed:
        sys.exit(f"licence: tasksource subsets with more than one license_use: {mixed}")
    m = {s: next(iter(u)) for s, u in sorted(uses.items())}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=1))
    print(f"licence: {len(m)} tasksource subsets {dict(Counter(m.values()))} -> {out}")
    return m


class Filter:
    """keep(row) -> True to train on it; excluded rows are counted for the manifest."""

    def __init__(self, cfg, lmap):
        self.allow = set(cfg["tasksource_allow"])
        self.exclude = dict(cfg.get("exclude_sources", {}))
        self.lmap = lmap
        self.cfg = cfg
        self.out = {}
        self.kept = {}

    def _count(self, split, key, row, subset=None):
        s = self.out.setdefault(split, {"rows": 0, "decisions": 0, "by_class": {}})
        c = s["by_class"].setdefault(key, {"rows": 0, "decisions": 0, "subsets": Counter()})
        n = len(row.get("questions") or {}) or 1
        for d in (s, c):
            d["rows"] += 1
            d["decisions"] += n
        if subset:
            c["subsets"][subset] += 1

    def keep(self, row, split):
        src = row.get("source", "")
        if src in self.exclude:
            self._count(split, src, row)
            return False
        if src.startswith("tasksource"):
            use = self.lmap.get(row.get("domain"), "unknown subset")
            if use not in self.allow:
                self._count(split, f"tasksource {use}", row, row.get("domain"))
                return False
        group = src if src.startswith("our-cases-public") else src.split("/")[0]
        k = self.kept.setdefault(split, Counter())
        k[group + " rows"] += 1
        k[group + " decisions"] += len(row.get("questions") or {}) or 1
        return True

    def manifest(self):
        out = {"policy": self.cfg, "kept": {split: dict(sorted(c.items())) for split, c in self.kept.items()}}
        for split, s in self.out.items():
            out[split] = {"rows": s["rows"], "decisions": s["decisions"],
                          "by_class": {k: {"rows": v["rows"], "decisions": v["decisions"],
                                           "subsets": dict(v["subsets"].most_common())}
                                       for k, v in sorted(s["by_class"].items())}}
        return out


if __name__ == "__main__":
    build_map()
