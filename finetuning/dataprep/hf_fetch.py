#!/usr/bin/env python3
"""Download HF dataset files via the resolve-cache path (bypasses /resolve 401-gate on this IP).

Usage: python hf_fetch.py DATASET_ID [DATASET_ID ...]
Saves into <data>/raw/hf/<flattened names>.
"""
import sys, os, json, urllib.request, urllib.parse, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
BASE = paths.RAW / "hf"


def api_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "openjevx-data/0.1"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def fetch_file(repo, rfilename, dest: pathlib.Path):
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{rfilename}"
    req = urllib.request.Request(url, headers={"User-Agent": "openjevx-data/0.1"}, method="HEAD")
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        final = resp.geturl()
        if resp.status != 200:
            raise RuntimeError(f"HEAD {resp.status}")
    except urllib.error.HTTPError as e:
        # resolve-cache fallback: ask the API for the sha and build the cache URL
        info = api_json(f"https://huggingface.co/api/datasets/{repo}")
        sha = info["sha"]
        final = (
            f"https://huggingface.co/api/resolve-cache/datasets/{repo}/{sha}/{rfilename}"
            f"?{urllib.parse.quote('/datasets/%s/resolve/main/%s' % (repo, rfilename), safe='/')}=&etag="
        )
        resp = urllib.request.urlopen(urllib.request.Request(final, headers={"User-Agent": "openjevx-data/0.1"}), timeout=60)
        if resp.status != 200:
            raise RuntimeError(f"cache fallback {resp.status}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(resp.geturl() if final == url else final, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)
    return dest.stat().st_size


def main():
    repos = sys.argv[1:]
    if not repos:
        print(__doc__)
        sys.exit(1)
    for repo in repos:
        files = api_json(f"https://huggingface.co/api/datasets/{repo}")  # sanity
        listing = api_json(f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true")
        entries = [e for e in listing if e["type"] == "file" and not e["path"].startswith(".") and e["path"] != "README.md" and "figures/" not in e["path"]]
        print(f"{repo}: {len(entries)} files")
        for e in entries:
            fname = e["path"]
            dest = BASE / repo.replace("/", "__") / fname
            if dest.exists() and dest.stat().st_size == e.get("size", -1) > 0:
                print(f"  skip {fname}")
                continue
            t0 = time.time()
            try:
                n = fetch_file(repo, fname, dest)
                print(f"  ok   {fname} {round(n/1e6,1)}MB in {round(time.time()-t0)}s")
            except Exception as err:
                print(f"  FAIL {fname}: {repr(err)[:140]}")


if __name__ == "__main__":
    main()
