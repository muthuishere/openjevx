#!/usr/bin/env python3
"""EVAL-ONLY log-triage gate from OUR OWN infrastructure logs. Never training data.

Sources (collected read-only by collect_ownlogs.sh into <data>/raw/ownlogs/<kind>.txt, one line each):
  openjevx-server    server.log of our openjevx server runs (local gate/scratch runs)
  openjevx-pipeline  ~/openjevx/data/work/*.log, runtime logs of our fine-tuning / gate pipeline
  kamal-proxy        kamal-proxy ACCESS lines from our deemwar servers, reduced on the server to
                     level/msg/service/method/status/duration (client IPs, hosts, paths, queries gone)
  ci-openjevx        GitHub Actions logs of muthuishere/openjevx
  ci-pgbx            GitHub Actions test runs of pgbx (deemwar-products/pgbx)
Excluded at collection and again here: anything matching the client markers in
~/.config/openjevx/ownlogs-exclude.txt (outside the repo), and reqsume
lines that are not pure proxy access lines.

Scrub (scrub(); the collector pipes every line through `sec sh -c cat` first, which redacts every
value in the sec vault, then through `--scrub <kind>`; read() scrubs again before labelling):
  IPv4/IPv6 -> <ip>, emails -> <email>, UUIDs -> <uuid>, JWT / Bearer / key=value secrets / long
  hex (>=24) / token-like base64 (>=24, mixed letters+digits) -> <token>, URL host -> <host> (kept
  only if it is an openjevx host), URL path -> <path>, query strings dropped, other non-openjevx
  hostnames -> <host>, home dirs -> ~. Lines that still match a client marker or mention a password
  are dropped, not kept.

Label (rule only, never a model). q1 "Should an on-call engineer act on this log line?"
  DROP  ambiguous: WARN/deprecation/retry, HTTP 4xx, TLS handshake noise, partial or short lines,
        lines matching nothing below.
  TRUE  real failure: level ERROR/FATAL/CRITICAL/panic, uncaught exception / traceback head,
        HTTP 5xx access line, crash or non-zero exit, connection refused, OOM, failed
        job/test/build step ("N failed" with N>0, FAILED, ##[error], rustc error[E..]).
  FALSE clearly routine: INFO/DEBUG with no failure word, HTTP 2xx/3xx, health checks, startup
        banners, "listening on", passing tests/steps (test ... ok, "0 failed", Finished, Compiling).
Rows are deduplicated by state, capped per number-masked template, and balanced per source.

usage: gen_ownlogs_gate.py                 read raw/ownlogs/*.txt, write gate/ownlogs_gate.jsonl
       gen_ownlogs_gate.py --scrub KIND    filter: scrub stdin lines of KIND to stdout (collector)
"""
import argparse, collections, json, os, random, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

SEED = 20261004
IN_DIR = paths.RAW / "ownlogs"
OUT = paths.GATE / "ownlogs_gate.jsonl"
KINDS = ["openjevx-server", "openjevx-pipeline", "kamal-proxy", "ci-openjevx", "ci-pgbx"]
CAP_PER_LABEL = 60      # per source and label
TEMPLATE_CAP = 2        # rows per number-masked template
MAX_LEN = 400
QUESTION = "Should an on-call engineer act on this log line?"
CRITERIA = {"true": "yes", "false": "no"}

_EX = Path(os.environ.get("OWNLOGS_EXCLUDE", Path.home() / ".config/openjevx/ownlogs-exclude.txt"))
if not _EX.is_file():  # fail closed: without the client markers nothing may be labelled
    sys.exit(f"gen_ownlogs_gate: {_EX} is missing (one client/user-content marker per line)")
_WORDS = [w.strip() for w in _EX.read_text().splitlines() if w.strip() and not w.startswith("#")]
EXCLUDE = re.compile("|".join([re.escape(w) for w in _WORDS] + ["password", "passwd"]), re.I)
ANSI = re.compile(r"(?:\x1b|\^\[)\[[0-9;?]*[A-Za-z]")
GH_PREFIX = re.compile(r"^(?:[^\t]*\t[^\t]*\t)?\ufeff?\d{4}-\d\d-\d\dT[\d:.]+Z ?")  # gh --log job/step/time
SCRIPT_ECHO = re.compile(r"^(?:\x1b|\^\[)\[36;1m")  # GitHub Actions echoing a step's script, not a log event
SUBS = [
    (re.compile(r"eyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]+"), "<token>"),
    (re.compile(r"\bbearer\s+\S+", re.I), "<token>"),
    (re.compile(r"\b(?:authorization|x-api-key|api[_-]?key|secret|access[_-]?key|token|signature|sig|credential)s?"
                r"[\"']?\s*[=:]\s*[\"']?[^\s\"',&}]+", re.I), "<token>"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b|\bgh[pousr]_[A-Za-z0-9]{20,}\b|\bsk-[A-Za-z0-9_-]{16,}"), "<token>"),
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b"), "<email>"),
    (re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"), "<uuid>"),
    (re.compile(r"(?:/Users|/home)/(?!runner\b)[\w.-]+"), "~"),
    (re.compile(r"muthuishere", re.I), "<user>"),
]
URL = re.compile(r"\b([a-z][a-z0-9+.-]*)://([^/\s\"'<>]+)(/[^\s\"'<>?#]*)?(\?[^\s\"'<>#]*)?(#\S*)?", re.I)
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b")
IPV6 = re.compile(r"\[?(?<![\w:])(?=[0-9a-fA-F:]*[a-fA-F])(?=[0-9a-fA-F:]*\d)"
                  r"(?:[0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}(?![\w:])\]?(?::\d{1,5})?")
HOST = re.compile(r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:com|net|org|io|ai|dev|app|cloud|co|in|xyz|"
                  r"me|tech|site|online|eu|de|uk|us|info|biz|pages|workers)\b(?!\.)", re.I)
HEX = re.compile(r"\b[0-9a-fA-F]{24,}\b")
B64 = re.compile(r"(?<![\w/.-])[A-Za-z0-9+_-]{24,}={0,2}(?![\w/.-])")
QUERY = re.compile(r"(\"query\"\s*:\s*)\"[^\"]*\"")


def tokenish(m):
    s = m.group(0)
    digits, upper, lower = any(c.isdigit() for c in s), any(c.isupper() for c in s), any(c.islower() for c in s)
    return "<token>" if digits and upper and lower or (digits and (upper or lower) and len(s) >= 32) else s


def keep_host(h):
    return "openjevx" in h.lower()


def scrub(line):
    """Return the scrubbed line, or None if it must not be kept."""
    line = GH_PREFIX.sub("", line.lstrip("\ufeff"))
    if SCRIPT_ECHO.match(line):
        return None
    s = ANSI.sub("", line).replace("\r", "").rstrip("\n").strip()
    if not s or EXCLUDE.search(s):
        return None
    for rx, rep in SUBS:
        s = rx.sub(rep, s)
    s = URL.sub(lambda m: f"{m.group(1)}://{m.group(2) if keep_host(m.group(2)) else '<host>'}"
                          + ("/<path>" if m.group(3) and m.group(3) != "/" else ""), s)
    s = QUERY.sub(r"\1\"\"", s)
    s = IPV4.sub("<ip>", s)
    s = IPV6.sub("<ip>", s)
    s = HOST.sub(lambda m: m.group(0) if keep_host(m.group(0)) else "<host>", s)
    s = HEX.sub("<token>", s)
    s = B64.sub(tokenish, s)
    if EXCLUDE.search(s) or "token=" in s.lower():
        return None
    return s[:MAX_LEN]


# ---- the label rule -------------------------------------------------------------------------
LEVEL = r"(?:^|[\s\[\"'=(|:])"
AMBIG = re.compile(
    LEVEL + r"warn(?:ing)?(?!s)\b|deprecat|\bretry|\bretrying|\bretried|will retry|TLS handshake error|"
    r"\bskipp(?:ed|ing)\b|\bignored\b|\bflaky\b|"
    r"reserve cache|operation was cancell?ed|\bcancell?ed\b|"   # Actions cache contention / superseded runs
    r"\bexp=\S+\s+got=|^sample:|\"source\":\s*\"our-cases|"          # our reports quoting dataset rows
    r"\battempt [2-9]\d*\b|"
    r"^\s*\"[^\"]+\"\s*:\s*[^{\[]*$|"                                # a key/value fragment of a JSON report
    r"^\s*(?:if-no-files-found|continue-on-error|fail-fast|fail-on-\w+):", re.I)  # step inputs, not events
STATUS = re.compile(r"(?:\"status\"\s*:\s*|\bstatus[=: ]\s*|\bHTTP/\d(?:\.\d)?\"?\s+)([1-5]\d\d)\b")
FAIL_CASED = re.compile(
    LEVEL + r"(?:ERROR|FATAL|CRITICAL|CRIT|PANIC)\b|\bFAILED\b|^\s*FAIL\b|\bOOM(?:Killed)?\b|"
    r"^\s*(?:[\w.]+\.)?[A-Z]\w*(?:Error|Exception)\b[:(]")
FAIL_ANY = re.compile(
    r"\blevel[=:]\s*\"?(?:error|fatal|critical|panic)\b|\"level\"\s*:\s*\"(?:error|fatal|critical|panic)\"|"
    r"panicked at|\bpanic:|Traceback \(most recent call last\)|Exception in thread|\buncaught\b|"
    r"unhandled (?:exception|rejection)|exit(?:ed)? (?:with )?(?:code|status)\s*:?\s*[1-9]\d*|"
    r"returned non-zero exit|non-zero exit|died with <Signals|segmentation fault|core dumped|out of memory|"
    r"cannot allocate memory|no space left on device|connection refused|ECONNREFUSED|##\[error\]|"
    r"\berror\[E\d{4}\]|^\s*error(?:\[\w+\])?:|\b[1-9]\d* failed\b|test result: FAILED|\bbuild failed\b|"
    r"could not compile|\bfailed to\b|^\s*fatal:|[\"']status[\"']\s*:\s*[\"']failed[\"']|"
    r"\b(?:job|step|run|build|task|provider|test|deploy(?:ment)?)s? failed\b", re.I)
OK = re.compile(
    r"(?:^|[\[\"'=(|]|\d\d(?::\d\d)+(?:[.,]\d+)?Z?\s)(?:INFO|DEBUG|TRACE)\b|\blevel[=:]\s*\"?(?:info|debug|trace)\b|\"level\"\s*:\s*\"(?:info|debug|trace)\"|"
    r"\bhealth(?:y|check|z)?\b|listening on|\blistening\b|\bstarted\b|\bstarting\b|\bready\b|"
    r"\bOpenJevX \w+ at\b|^\S+ \S+ (?:device \w+|model \S+ version \S+)|"   # our server's startup banner
    r"test result: ok|^\s*test \S+ \.\.\. ok\b|\b0 failed\b|\bpassed\b|^\s*Finished\b|^\s*Compiling\b|"
    r"^\s*Downloaded\b|^\s*Installing\b|Successfully|\bsucceeded\b|(?<!no )\bsuccess\b|\bcompleted?\b|\bdone\b|"
    r"##\[group\]|^\s*Run \S|\bok\b", re.I)
FAIL_WORD = re.compile(r"fail|error|exception|refused|denied|timeout|timed out|crash|panic|kill|abort|fatal|"
                       r"unavailable|unreachable|traceback|corrupt|lost|cannot|can't|unable|invalid", re.I)
PASS_STRONG = re.compile(  # passing-test / clean-summary lines whose text may still contain "fail"
    r"^\s*ok \d+ - |^\s*test \S+ \.\.\. ok\s*$|test result: ok\b|^#\s*fail\s+0\s*$|\bERRORS\s+0\s*$")
FAIL_STRONG = re.compile(r"^\s*not ok \d+|^#\s*fail\s+[1-9]\d*\s*$|\bERRORS\s+[1-9]\d*\s*$")
NO_ERRORS = re.compile(r"\b(?:no|0|zero) (?:errors?|failures?)\b|\berrors?[=:]\s*0\b", re.I)


def label(text):
    """True / False per the rule, or None to drop. Order matters: ambiguity first, then failure."""
    if len(text) < 15 or not re.search(r"[A-Za-z]{3}", text):
        return None
    if AMBIG.search(text):
        return None
    if FAIL_STRONG.search(text):
        return True
    if PASS_STRONG.search(text):
        return False
    st = STATUS.search(text)
    if st:
        code = int(st.group(1))
        if code >= 500:
            return True
        if code >= 400:
            return None
        if not FAIL_WORD.search(text):
            return False
    if FAIL_CASED.search(text) or FAIL_ANY.search(text):
        if NO_ERRORS.search(text):      # "0 errors", "no failures": a summary, not a failure
            return None
        return True
    if OK.search(text) and not FAIL_WORD.search(text):
        return False
    return None


def template(text):
    st = STATUS.search(text)  # keep the HTTP status so 2xx and 5xx of one route are different templates
    text = re.sub(r"our-cases-\S+", "<case>", text)
    return re.sub(r"\d+", "#", text) + (f" status={st.group(1)}" if st else "")


def read(kind):
    f = IN_DIR / f"{kind}.txt"
    if not f.exists():
        return None
    out = []
    with open(f, encoding="utf-8", errors="replace") as h:
        for line in h:
            s = scrub(line)
            if s is not None:
                out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrub", metavar="KIND", help="scrub stdin to stdout (used by collect_ownlogs.sh)")
    a = ap.parse_args()
    if a.scrub:
        for line in sys.stdin:
            s = scrub(line)
            if s is not None and not (a.scrub != "kamal-proxy" and re.search("reqsume", s, re.I)):
                print(s)
        return
    rng = random.Random(SEED)
    rows, seen, report = [], set(), {}
    for kind in KINDS:
        lines = read(kind)
        if lines is None:
            report[kind] = "missing"
            continue
        pools, per_tmpl, dropped = {True: [], False: []}, collections.Counter(), 0
        for s in lines:
            if kind != "kamal-proxy" and re.search("reqsume", s, re.I):
                dropped += 1
                continue
            g = label(s)
            if g is None:
                dropped += 1
                continue
            state = {"service": kind, "log": s}
            key = json.dumps(state, sort_keys=True)
            t = (kind, template(s))
            if key in seen or per_tmpl[t] >= TEMPLATE_CAP:
                continue
            seen.add(key)
            per_tmpl[t] += 1
            pools[g].append(state)
        kept = {g: rng.sample(p, min(len(p), CAP_PER_LABEL)) for g, p in pools.items()}
        report[kind] = {"collected": len(lines), "dropped": dropped, "unique_true": len(pools[True]),
                        "unique_false": len(pools[False]), "true": len(kept[True]), "false": len(kept[False])}
        for g, states in kept.items():
            for st in states:
                rows.append({"source": f"our-ownlogs/{kind}", "domain": f"ownlogs/{kind}", "state": st,
                             "questions": {"q1": {"type": "noul", "instructions": QUESTION, "criteria": dict(CRITERIA)}},
                             "gold": {"q1": "true" if g else "false"}})
    rng.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    pos = sum(r["gold"]["q1"] == "true" for r in rows)
    for kind, rep in report.items():
        print(f"{kind}: {rep}")
    print(f"{OUT}: {len(rows)} rows ({pos} true / {len(rows) - pos} false)")


if __name__ == "__main__":
    main()
