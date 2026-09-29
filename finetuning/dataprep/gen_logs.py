#!/usr/bin/env python3
"""Deterministic, rule-labelled log-triage generator for OpenJevX.

Three log kinds in their real formats with documented codes:
  db/*   PostgreSQL server log (SQLSTATE), MySQL client/server errors, slow-query logs
  app/*  Spring/Java stack traces, Python tracebacks, Node, Go panics, nginx, JSON logs
  fe/*   browser console errors, failed network requests, Sentry-style events

Row contract (same as gen_it_worker.py):
    {"source": "our-cases-logs/<kind>/<family>", "domain": "logs/<kind>/<family>",
     "state": <dict OR plain-English string>,
     "questions": {"<qid>": {"type", "instructions", "criteria"}}, "gold": {"<qid>": "<label>"}}

The state holds the log text plus the context facts. Every gold label is computed by the
rules below from the template's fixed attributes and those facts; nothing is hand-written.

RULE TABLE (evaluated in this order)
  template attrs: team, kind (config|code_bug|capacity|network|auth|data|None=not asked),
                  base severity 0-3, transient (retry helps), db_root (True/False/None=not asked),
                  noise (intrinsically harmless line), rate_gated (only matters in volume)
  thr = alert_threshold_per_5m (stated in the state), n = errors_last_5m
  R1 noise   = template.noise OR matches_known_benign_pattern
               OR (rate_gated AND n < thr) OR (transient AND retry_succeeded AND n < thr)
  R2 real    = NOT noise                          ("real failure someone should act on")
  R3 sev     = 0 if noise else base (+1 if n >= thr OR users_affected >= 100), max 3;
               environment != production -> sev capped at 1 (medium)
  R4 page    = production AND real AND (sev == 3 OR (sev == 2 AND service_criticality == critical))
               (dev/staging never page)
  R5 deploy  = real AND started_after_latest_deploy AND kind in {code_bug, config}
  R6 action  = ignore if not real; roll_back if production AND deploy AND sev >= 2;
               page if page; else ticket
  R7 team    = template.team; PostgreSQL 28P01 from an external IP at/over thr -> security
  R8 kind    = template.kind ;  R9 retry = template.transient ;  R10 db_root = template.db_root
  Noise templates are not asked team/kind/db_root/deploy/retry (no meaningful answer).

Near-miss twins: ~half of the cases are emitted with a twin that changes ONE fact (environment,
known-benign flag, deploy timing, retry result, count across the stated threshold, criticality or
source IP) and asks the questions whose answer flips.

Held-out families (eval only): db/mysql, app/go_panic, fe/sentry.
Phrasings: last template of every question family is gate-only, the one before it eval-only.
No eval or gate case asks about a state train already asked about (leak_check's state-level check).

Usage:  python3 finetuning/dataprep/gen_logs.py [--test-only]   (--test-only keeps the train file)
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
from datavalidate.leak_check import in_keys, state_keys  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = paths.TRAIN / "logs_train.jsonl"
EVAL_PATH = paths.EVAL / "logs_eval.jsonl"
GATE_PATH = paths.GATE / "logs_gate.jsonl"
SAMPLES_PATH = paths.WORK / "samples" / "logs_samples.txt"

TRAIN_SEED, EVAL_SEED, GATE_SEED = 20260929, 47110929, 90210929
TRAIN_Q, EVAL_Q, GATE_CASES = 60000, 6000, 300
HELD_OUT = {("db", "mysql"), ("app", "go_panic"), ("fe", "sentry")}

SERVICES = ["checkout-service", "auth-service", "payments-api", "search-service", "orders-service",
            "user-api", "inventory-service", "billing-worker", "notifications-service", "reporting-api"]
WEB_APPS = ["web-storefront", "admin-console", "customer-portal", "checkout-web", "docs-site"]
TABLES = ["orders", "users", "invoices", "payments", "cart_items", "sessions", "shipments"]
DBS = ["shop", "billing", "accounts", "catalog"]
INTERNAL_IPS = ["10.0.3.7", "10.0.2.15", "10.0.4.12", "10.0.1.14", "172.16.5.21"]
EXTERNAL_IPS = ["203.0.113.7", "198.51.100.23", "192.0.2.44", "203.0.113.150", "198.51.100.91"]
CRITICALITY = ["critical", "standard", "internal-tool"]


def A(rng, seq):
    return rng.choice(seq)


def ts_pg(rng):
    return f"2026-09-{rng.randint(1, 29):02d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:" \
           f"{rng.randint(0, 59):02d}.{rng.randint(0, 999):03d} UTC"


def ts_iso(rng):
    return f"2026-09-{rng.randint(1, 29):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:" \
           f"{rng.randint(0, 59):02d}.{rng.randint(0, 999999):06d}Z"


def ts_nginx(rng):
    return f"2026/09/{rng.randint(1, 29):02d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"


def ts_clf(rng):
    return f"{rng.randint(1, 29):02d}/Sep/2026:{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d} +0000"


# ---------------------------------------------------------------------------
# log templates: each returns log text; attrs decide the rules
# ---------------------------------------------------------------------------

TEMPLATES = defaultdict(list)  # (kind, family) -> [template dict]


def tpl(kind, fam, name, team, fkind, base, transient=None, db_root=None, noise=False,
        rate_gated=False):
    def deco(fn):
        TEMPLATES[(kind, fam)].append(dict(name=name, fn=fn, team=team, kind=fkind, base=base,
                                           transient=transient, db_root=db_root, noise=noise,
                                           rate_gated=rate_gated))
        return fn
    return deco


def pg(rng, level, msg, extra=()):
    prefix = f"{ts_pg(rng)} [{rng.randint(1000, 65000)}] "
    if rng.random() < 0.5:
        prefix += f"{A(rng, ['app_rw', 'app', 'reporting'])}@{A(rng, DBS)} "
    lines = [f"{prefix}{level}:  {msg}"] + [f"{prefix}{k}:  {v}" for k, v in extra]
    return "\n".join(lines)


# ---- PostgreSQL ----
@tpl("db", "postgres", "23505", "backend", "data", 1, transient=False, db_root=False)
def _(rng, c):
    tbl, col, con = A(rng, [("users", "email", "users_email_key"), ("orders", "id", "orders_pkey"),
                            ("payments", "idempotency_key", "payments_idempotency_key_key")])
    val = f"u{rng.randint(1000, 99999)}@example.com" if col == "email" else rng.randint(1000, 99999)
    return pg(rng, "ERROR", f'duplicate key value violates unique constraint "{con}"',
              [("DETAIL", f"Key ({col})=({val}) already exists."),
               ("STATEMENT", f"INSERT INTO {tbl} ({col}, created_at) VALUES ($1, now())")])


@tpl("db", "postgres", "40P01", "backend", "code_bug", 1, transient=True, rate_gated=True)
def _(rng, c):
    a, b = rng.randint(1000, 9000), rng.randint(9001, 20000)
    tx = rng.randint(100000, 999999)
    return pg(rng, "ERROR", "deadlock detected",
              [("DETAIL", f"Process {a} waits for ShareLock on transaction {tx}; blocked by process {b}.\n"
                          f"\tProcess {b} waits for ShareLock on transaction {tx - 2}; blocked by process {a}."),
               ("HINT", "See server log for query details."),
               ("CONTEXT", f'while updating tuple (12,{rng.randint(1, 40)}) in relation "{A(rng, TABLES)}"')])


@tpl("db", "postgres", "40001", "backend", "code_bug", 1, transient=True, rate_gated=True)
def _(rng, c):
    return pg(rng, "ERROR", "could not serialize access due to concurrent update",
              [("STATEMENT", f"UPDATE {A(rng, TABLES)} SET status = $1 WHERE id = $2")])


@tpl("db", "postgres", "53100", "database", "capacity", 3, transient=False, db_root=True)
def _(rng, c):
    return pg(rng, "ERROR", f'could not extend file "base/16384/{rng.randint(16000, 99999)}": No space left on device',
              [("HINT", "Check free disk space.")])


@tpl("db", "postgres", "53200", "database", "capacity", 2, transient=False, db_root=True)
def _(rng, c):
    return pg(rng, "ERROR", "out of memory",
              [("DETAIL", f'Failed on request of size {A(rng, [8192, 16384, 1048576])} in memory context "ExecutorState".')])


@tpl("db", "postgres", "53300", "database", "capacity", 2, transient=False, db_root=True)
def _(rng, c):
    return pg(rng, "FATAL", A(rng, ["sorry, too many clients already",
                                    "remaining connection slots are reserved for non-replication superuser connections"]))


@tpl("db", "postgres", "57P01", "database", None, 1, transient=True, rate_gated=True)
def _(rng, c):
    return pg(rng, "FATAL", "terminating connection due to administrator command")


@tpl("db", "postgres", "57014", "backend", "capacity", 1, transient=False, rate_gated=True)
def _(rng, c):
    return pg(rng, "ERROR", "canceling statement due to statement timeout",
              [("STATEMENT", f"SELECT * FROM {A(rng, TABLES)} WHERE lower(email) = $1 ORDER BY created_at DESC")])


@tpl("db", "postgres", "42P01", "backend", "code_bug", 2, transient=False, db_root=False)
def _(rng, c):
    t = A(rng, ["invoices_v2", "user_prefs", "order_events", "feature_flags"])
    return pg(rng, "ERROR", f'relation "{t}" does not exist at character 15',
              [("STATEMENT", f"SELECT * FROM {t} WHERE account_id = $1")])


@tpl("db", "postgres", "42703", "backend", "code_bug", 2, transient=False, db_root=False)
def _(rng, c):
    col = A(rng, ["discount_code", "archived_at", "tenant_id", "shipping_tier"])
    return pg(rng, "ERROR", f'column "{col}" does not exist at character 8',
              [("STATEMENT", f"SELECT {col}, id FROM {A(rng, TABLES)} WHERE id = $1")])


@tpl("db", "postgres", "28P01", "infra", "auth", 1, transient=False, db_root=False, rate_gated=True)
def _(rng, c):
    user = A(rng, ["app_rw", "postgres", "reporting", "admin"])
    return pg(rng, "FATAL", f'password authentication failed for user "{user}"',
              [("DETAIL", f'Connection matched pg_hba.conf line {rng.randint(80, 120)}: '
                          f'"host all all 0.0.0.0/0 scram-sha-256"')]) + \
        f"\n(connection from {c['source_ip']})"


@tpl("db", "postgres", "XX000", "database", None, 2, transient=False, db_root=True)
def _(rng, c):
    return pg(rng, "ERROR", f"cache lookup failed for relation {rng.randint(16000, 99999)}")


@tpl("db", "postgres", "PANIC", "database", "capacity", 3, transient=False, db_root=True)
def _(rng, c):
    return pg(rng, "PANIC", f'could not write to file "pg_wal/xlogtemp.{rng.randint(100, 9999)}": No space left on device') + \
        "\n" + pg(rng, "LOG", "server process (PID 4121) was terminated by signal 6: Aborted")


@tpl("db", "postgres", "checkpoint", "database", None, 0, noise=True)
def _(rng, c):
    if rng.random() < 0.5:
        return pg(rng, "LOG", A(rng, ["checkpoint starting: time", "checkpoint starting: wal"]))
    b = rng.randint(200, 9000)
    return pg(rng, "LOG", f"checkpoint complete: wrote {b} buffers ({b / 163.84:.1f}%); 0 WAL file(s) added, "
                          f"0 removed, {rng.randint(0, 4)} recycled; write={rng.uniform(1, 270):.3f} s, "
                          f"sync=0.004 s, total={rng.uniform(1, 270):.3f} s")


@tpl("db", "postgres", "autovacuum", "database", None, 0, noise=True)
def _(rng, c):
    t = A(rng, TABLES)
    return pg(rng, "LOG", f'automatic vacuum of table "{A(rng, DBS)}.public.{t}": index scans: 1\n'
                          f"\tpages: 0 removed, {rng.randint(100, 90000)} remain, 0 skipped due to pins\n"
                          f"\ttuples: {rng.randint(10, 9000)} removed, {rng.randint(1000, 900000)} remain")


# ---- MySQL (held out) ----
def mysql_client(rng, line):
    return f"{ts_iso(rng)} {A(rng, SERVICES)} db error: {line}"


@tpl("db", "mysql", "1205", "backend", "code_bug", 1, transient=True, rate_gated=True)
def _(rng, c):
    return mysql_client(rng, "ERROR 1205 (HY000): Lock wait timeout exceeded; try restarting transaction")


@tpl("db", "mysql", "1213", "backend", "code_bug", 1, transient=True, rate_gated=True)
def _(rng, c):
    return mysql_client(rng, "ERROR 1213 (40001): Deadlock found when trying to get lock; try restarting transaction")


@tpl("db", "mysql", "1040", "database", "capacity", 2, transient=False, db_root=True)
def _(rng, c):
    return mysql_client(rng, "ERROR 1040 (08004): Too many connections")


@tpl("db", "mysql", "2013", "database", "network", 1, transient=True, rate_gated=True)
def _(rng, c):
    return mysql_client(rng, "ERROR 2013 (HY000): Lost connection to MySQL server during query")


@tpl("db", "mysql", "diskfull", "database", "capacity", 3, transient=False, db_root=True)
def _(rng, c):
    return f"{ts_iso(rng)} {rng.randint(1, 900)} [ERROR] mysqld: Disk full (/tmp/#sql_{rng.randint(1000, 9999)}_0.MYI); " \
           "waiting for someone to free some space... (errno: 28 \"No space left on device\")"


@tpl("db", "mysql", "aborted", "database", None, 0, noise=True)
def _(rng, c):
    return f"{ts_iso(rng)} {rng.randint(1000, 90000)} [Note] Aborted connection {rng.randint(1000, 90000)} to db: " \
           f"'{A(rng, DBS)}' user: 'app' host: '{A(rng, INTERNAL_IPS)}' (Got an error reading communication packets)"


@tpl("db", "mysql", "page_cleaner", "database", None, 0, noise=True)
def _(rng, c):
    return f"{ts_iso(rng)} 0 [Note] InnoDB: page_cleaner: 1000ms intended loop took {rng.randint(1100, 9000)}ms. " \
           "The settings might not be optimal. (flushed=200 and evicted=0, during the time.)"


# ---- slow query logs ----
@tpl("db", "slow_query", "mysql_slow", "database", "capacity", 1, transient=False, db_root=True, rate_gated=True)
def _(rng, c):
    return (f"# Time: {ts_iso(rng)}\n# User@Host: app[app] @  [{A(rng, INTERNAL_IPS)}]  Id: {rng.randint(100, 99999)}\n"
            f"# Query_time: {rng.uniform(5, 60):.6f}  Lock_time: 0.000112 Rows_sent: 1  Rows_examined: {rng.randint(10**6, 9 * 10**6)}\n"
            f"SET timestamp={rng.randint(1790000000, 1795000000)};\n"
            f"SELECT * FROM {A(rng, TABLES)} WHERE lower(email) = 'x@example.com';")


@tpl("db", "slow_query", "pg_duration", "database", "capacity", 1, transient=False, db_root=True, rate_gated=True)
def _(rng, c):
    return pg(rng, "LOG", f"duration: {rng.uniform(5000, 60000):.3f} ms  statement: SELECT o.* FROM orders o "
                          f"JOIN {A(rng, TABLES)} t ON t.order_id = o.id WHERE o.created_at > now() - interval '90 days'")


# ---- Java / Spring ----
def spring(rng, level, logger, msg):
    return f"{ts_pg(rng)[:-4]} {level} 1 --- [nio-8080-exec-{rng.randint(1, 20)}] {logger} : {msg}"


@tpl("app", "java", "npe", "backend", "code_bug", 2, transient=False, db_root=False)
def _(rng, c):
    return (spring(rng, "ERROR", "o.a.c.c.C.[.[.[/].[dispatcherServlet]",
                   "Servlet.service() for servlet [dispatcherServlet] in context with path [] threw exception "
                   "[Request processing failed; nested exception is java.lang.NullPointerException] with root cause") +
            "\n\njava.lang.NullPointerException: Cannot invoke \"com.acme.orders.Customer.getAddress()\" because "
            "\"customer\" is null\n\tat com.acme.orders.OrderService.ship(OrderService.java:88)\n"
            "\tat com.acme.orders.OrderController.ship(OrderController.java:41)")


@tpl("app", "java", "oom", "backend", "capacity", 3, transient=False, db_root=False)
def _(rng, c):
    return (f'Exception in thread "http-nio-8080-exec-{rng.randint(1, 20)}" java.lang.OutOfMemoryError: Java heap space\n'
            "\tat java.base/java.util.Arrays.copyOf(Arrays.java:3537)\n"
            "\tat com.acme.reports.ExportService.buildCsv(ExportService.java:112)")


@tpl("app", "java", "hikari", "backend", "capacity", 2, transient=True, rate_gated=True)
def _(rng, c):
    return (spring(rng, "WARN", "o.h.engine.jdbc.spi.SqlExceptionHelper", "SQL Error: 0, SQLState: 08001") + "\n" +
            spring(rng, "ERROR", "o.h.engine.jdbc.spi.SqlExceptionHelper",
                   "HikariPool-1 - Connection is not available, request timed out after 30000ms.") +
            "\njava.sql.SQLTransientConnectionException: HikariPool-1 - Connection is not available, request timed out after 30000ms.\n"
            "\tat com.zaxxer.hikari.pool.HikariPool.createTimeoutException(HikariPool.java:696)")


@tpl("app", "java", "started", "backend", None, 0, noise=True)
def _(rng, c):
    return spring(rng, "INFO", "c.a.Application", f"Started Application in {rng.uniform(3, 30):.3f} seconds (JVM running for {rng.uniform(4, 35):.3f})")


# ---- Python ----
@tpl("app", "python", "keyerror", "backend", "code_bug", 2, transient=False, db_root=False)
def _(rng, c):
    return (f"ERROR {ts_pg(rng)[:-4]} log django.request: Internal Server Error: /api/checkout\n"
            "Traceback (most recent call last):\n  File \"/app/orders/views.py\", line 42, in checkout\n"
            "    sku = payload[\"sku\"]\nKeyError: 'sku'")


@tpl("app", "python", "pg_refused", "database", None, 3, transient=False, db_root=True)
def _(rng, c):
    return ("Traceback (most recent call last):\n  File \"/app/db.py\", line 18, in connect\n"
            "    return psycopg2.connect(DSN)\n"
            "psycopg2.OperationalError: connection to server at \"db-primary\" (10.0.2.5), port 5432 failed: Connection refused\n"
            "\tIs the server running on that host and accepting TCP/IP connections?")


@tpl("app", "python", "redis_refused", "infra", "network", 2, transient=False, db_root=False)
def _(rng, c):
    return ("Traceback (most recent call last):\n  File \"/usr/local/lib/python3.12/site-packages/redis/connection.py\", "
            "line 357, in connect\n    sock = self.retry.call_with_retry(\n"
            "ConnectionRefusedError: [Errno 111] Connection refused\n\n"
            "redis.exceptions.ConnectionError: Error 111 connecting to redis-cache:6379. Connection refused.")


@tpl("app", "python", "timeout", "backend", "network", 1, transient=True, db_root=False, rate_gated=True)
def _(rng, c):
    return (f"{ts_pg(rng)[:-4]} ERROR [payments] upstream call failed\nTraceback (most recent call last):\n"
            "  File \"/app/payments/client.py\", line 77, in capture\n    resp = await client.post(url, json=body)\n"
            "httpx.ReadTimeout: timed out\n"
            f"INFO:     {A(rng, INTERNAL_IPS)}:{rng.randint(30000, 60000)} - \"POST /v1/capture HTTP/1.1\" 504 Gateway Timeout")


@tpl("app", "python", "uvicorn_ok", "backend", None, 0, noise=True)
def _(rng, c):
    return f"INFO:     {A(rng, INTERNAL_IPS)}:{rng.randint(30000, 60000)} - \"GET /healthz HTTP/1.1\" 200 OK"


# ---- Node ----
@tpl("app", "node", "unhandled", "backend", "code_bug", 2, transient=False, db_root=False)
def _(rng, c):
    return ("node:internal/process/promises:288\n            triggerUncaughtException(err, true /* fromPromise */);\n"
            "            ^\n\n[UnhandledPromiseRejection: This error originated either by throwing inside of an async "
            "function without a catch block, or by rejecting a promise which was not handled with .catch(). "
            "The promise rejected with the reason \"undefined\".] {\n  code: 'ERR_UNHANDLED_REJECTION'\n}")


@tpl("app", "node", "econnrefused", "infra", "network", 2, transient=False, db_root=False)
def _(rng, c):
    ip = A(rng, INTERNAL_IPS)
    return (f"Error: connect ECONNREFUSED {ip}:6379\n    at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1555:16) {{\n"
            f"  errno: -111,\n  code: 'ECONNREFUSED',\n  syscall: 'connect',\n  address: '{ip}',\n  port: 6379\n}}")


@tpl("app", "node", "etimedout", "infra", "network", 1, transient=True, db_root=False, rate_gated=True)
def _(rng, c):
    ip = A(rng, INTERNAL_IPS)
    return (f"Error: connect ETIMEDOUT {ip}:443\n    at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1555:16) {{\n"
            f"  errno: -110,\n  code: 'ETIMEDOUT',\n  syscall: 'connect',\n  address: '{ip}',\n  port: 443\n}}")


@tpl("app", "node", "heap_oom", "backend", "capacity", 3, transient=False, db_root=False)
def _(rng, c):
    return ("<--- Last few GCs --->\n\n[1:0x5f1c2a0]  4821345 ms: Mark-Compact 2046.9 (2082.4) -> 2045.1 (2083.2) MB\n\n"
            "FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory")


# ---- Go (held out) ----
@tpl("app", "go_panic", "nil_deref", "backend", "code_bug", 3, transient=False, db_root=False)
def _(rng, c):
    return ("panic: runtime error: invalid memory address or nil pointer dereference\n"
            f"[signal SIGSEGV: segmentation violation code=0x1 addr=0x0 pc=0x{rng.randint(0x400000, 0x4fffff):x}]\n\n"
            f"goroutine {rng.randint(1, 900)} [running]:\nmain.(*Handler).ServeHTTP(0x0, {{0x7f1c20, 0xc0001a2000}}, 0xc000190100)\n"
            "\t/app/handler.go:88 +0x1c")


@tpl("app", "go_panic", "index", "backend", "code_bug", 3, transient=False, db_root=False)
def _(rng, c):
    n = rng.randint(1, 9)
    return (f"panic: runtime error: index out of range [{n}] with length {n}\n\ngoroutine {rng.randint(1, 900)} [running]:\n"
            "main.parseItems(...)\n\t/app/items.go:41")


@tpl("app", "go_panic", "map_writes", "backend", "code_bug", 3, transient=False, db_root=False)
def _(rng, c):
    return (f"fatal error: concurrent map writes\n\ngoroutine {rng.randint(1, 900)} [running]:\n"
            "main.(*Cache).Set(...)\n\t/app/cache.go:27")


@tpl("app", "go_panic", "deadline", "backend", "network", 1, transient=True, db_root=False, rate_gated=True)
def _(rng, c):
    return (f'time="{ts_iso(rng)}" level=error msg="rpc failed" '
            'err="rpc error: code = DeadlineExceeded desc = context deadline exceeded"')


# ---- nginx ----
@tpl("app", "nginx", "504", "backend", None, 2, transient=True, rate_gated=True)
def _(rng, c):
    cl = A(rng, EXTERNAL_IPS)
    return (f"{ts_nginx(rng)} [error] 31#31: *{rng.randint(1000, 99999)} upstream timed out (110: Connection timed out) "
            f"while reading response header from upstream, client: {cl}, server: api.example.com, "
            f"request: \"GET /v1/orders HTTP/1.1\", upstream: \"http://{A(rng, INTERNAL_IPS)}:8080/v1/orders\", host: \"api.example.com\"\n"
            f"{cl} - - [{ts_clf(rng)}] \"GET /v1/orders HTTP/1.1\" 504 562 \"-\" \"Mozilla/5.0\"")


@tpl("app", "nginx", "502", "backend", None, 3, transient=False)
def _(rng, c):
    cl = A(rng, EXTERNAL_IPS)
    return (f"{ts_nginx(rng)} [error] 31#31: *{rng.randint(1000, 99999)} connect() failed (111: Connection refused) "
            f"while connecting to upstream, client: {cl}, server: api.example.com, request: \"POST /v1/cart HTTP/1.1\", "
            f"upstream: \"http://{A(rng, INTERNAL_IPS)}:8080/v1/cart\", host: \"api.example.com\"\n"
            f"{cl} - - [{ts_clf(rng)}] \"POST /v1/cart HTTP/1.1\" 502 552 \"-\" \"Mozilla/5.0\"")


@tpl("app", "nginx", "499", "backend", "capacity", 1, transient=False, db_root=False, rate_gated=True)
def _(rng, c):
    return f"{A(rng, EXTERNAL_IPS)} - - [{ts_clf(rng)}] \"GET /v1/search?q=shoes HTTP/1.1\" 499 0 \"-\" \"Mozilla/5.0\""


@tpl("app", "nginx", "200", "backend", None, 0, noise=True)
def _(rng, c):
    return (f"{A(rng, EXTERNAL_IPS)} - - [{ts_clf(rng)}] \"GET /static/app.css HTTP/1.1\" "
            f"{A(rng, [200, 304])} {rng.randint(0, 90000)} \"https://app.example.com/\" \"Mozilla/5.0\"")


# ---- structured JSON ----
def jlog(rng, c, level, msg, status, latency, **extra):
    d = {"ts": ts_iso(rng), "level": level, "msg": msg, "service": c["service"],
         "trace_id": f"{rng.getrandbits(64):016x}", "status": status, "latency_ms": latency}
    d.update(extra)
    return json.dumps(d)


@tpl("app", "json_logs", "500", "backend", "code_bug", 2, transient=False, db_root=False)
def _(rng, c):
    return jlog(rng, c, "error", "unhandled exception in request handler", 500, rng.randint(5, 200),
                error="TypeError: 'NoneType' object is not subscriptable")


@tpl("app", "json_logs", "upstream_timeout", "backend", "network", 1, transient=True, db_root=False, rate_gated=True)
def _(rng, c):
    return jlog(rng, c, "error", "payment provider call timed out", 504, rng.randint(30000, 30100),
                error="context deadline exceeded")


@tpl("app", "json_logs", "retrying", "backend", "network", 1, transient=True, db_root=False, rate_gated=True)
def _(rng, c):
    return jlog(rng, c, "warn", "retrying request", 503, rng.randint(100, 3000), attempt=rng.randint(2, 3))


@tpl("app", "json_logs", "ok", "backend", None, 0, noise=True)
def _(rng, c):
    return jlog(rng, c, "info", "request completed", 200, rng.randint(3, 250))


# ---- browser console ----
@tpl("fe", "console", "typeerror", "frontend", "code_bug", 2, transient=False, db_root=False)
def _(rng, c):
    p = A(rng, ["map", "length", "id", "price"])
    comp = A(rng, ["ProductList", "CartSummary", "OrderTable", "UserMenu"])
    return (f"Uncaught TypeError: Cannot read properties of undefined (reading '{p}')\n"
            f"    at {comp} (main.{rng.getrandbits(24):06x}.js:2:{rng.randint(1000, 90000)})")


@tpl("fe", "console", "chunkload", "frontend", "config", 2, transient=False, db_root=False)
def _(rng, c):
    n = rng.randint(10, 999)
    return (f"ChunkLoadError: Loading chunk {n} failed.\n"
            f"(error: https://app.example.com/static/js/{n}.{rng.getrandbits(24):06x}.chunk.js)")


@tpl("fe", "console", "cors", "frontend", "config", 2, transient=False, db_root=False)
def _(rng, c):
    return ("Access to fetch at 'https://api.example.com/v1/cart' from origin 'https://app.example.com' has been "
            "blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present on the requested resource. "
            "If an opaque response serves your needs, set the request's mode to 'no-cors' to fetch the resource with CORS disabled.")


@tpl("fe", "console", "csp", "frontend", "config", 1, transient=False, db_root=False)
def _(rng, c):
    return ("Refused to load the script 'https://cdn.example-analytics.com/tag.js' because it violates the following "
            "Content Security Policy directive: \"script-src 'self'\". Note that 'script-src-elem' was not explicitly set, "
            "so 'script-src' is used as a fallback.")


@tpl("fe", "console", "promise_fetch", "frontend", "network", 1, transient=True, db_root=False, rate_gated=True)
def _(rng, c):
    return "Uncaught (in promise) TypeError: Failed to fetch\n    at loadCart (main.js:1:48213)"


@tpl("fe", "console", "hydration", "frontend", "code_bug", 1, transient=False, db_root=False)
def _(rng, c):
    return A(rng, ["Error: Hydration failed because the initial UI does not match what was rendered on the server.",
                   f"Warning: Text content did not match. Server: \"{rng.randint(1, 5)} items\" Client: \"{rng.randint(6, 9)} items\""])


@tpl("fe", "console", "react_key", "frontend", None, 0, noise=True)
def _(rng, c):
    return ("Warning: Each child in a list should have a unique \"key\" prop.\n\n"
            f"Check the render method of `{A(rng, ['CartItems', 'OrderTable', 'NavMenu'])}`. "
            "See https://reactjs.org/link/warning-keys for more information.")


@tpl("fe", "console", "devtools", "frontend", None, 0, noise=True)
def _(rng, c):
    return ("Download the React DevTools for a better development experience: https://reactjs.org/link/react-devtools")


# ---- browser network ----
def fe_req(rng, method, path, status):
    return f"{method} https://api.example.com{path} {status}"


@tpl("fe", "network", "refused", "infra", "network", 3, transient=False, db_root=False)
def _(rng, c):
    return f"GET https://api.example.com/v1/{A(rng, ['orders', 'cart', 'profile'])} net::ERR_CONNECTION_REFUSED"


@tpl("fe", "network", "401", "backend", "auth", 1, transient=False, db_root=False, rate_gated=True)
def _(rng, c):
    return fe_req(rng, "GET", "/v1/profile", "401 (Unauthorized)")


@tpl("fe", "network", "403", "backend", "auth", 1, transient=False, db_root=False, rate_gated=True)
def _(rng, c):
    return fe_req(rng, "POST", "/v1/admin/export", "403 (Forbidden)")


@tpl("fe", "network", "404", "frontend", "code_bug", 1, transient=False, db_root=False)
def _(rng, c):
    return fe_req(rng, "GET", f"/v1/{A(rng, ['prodcts', 'user/prefrences', 'ordrs'])}", "404 (Not Found)")


@tpl("fe", "network", "500", "backend", "code_bug", 2, transient=False, db_root=False)
def _(rng, c):
    return fe_req(rng, "POST", "/v1/checkout", "500 (Internal Server Error)")


@tpl("fe", "network", "503", "infra", "capacity", 2, transient=True, db_root=False, rate_gated=True)
def _(rng, c):
    return fe_req(rng, "GET", "/v1/search?q=shoes", "503 (Service Unavailable)")


# ---- Sentry-style events (held out); titles reuse console errors ----
SENTRY_TITLES = [
    ("TypeError: Cannot read properties of null (reading 'addEventListener')", "frontend", "code_bug", 2, False, False),
    ("ChunkLoadError: Loading chunk 482 failed.", "frontend", "config", 2, False, False),
    ("TypeError: Failed to fetch", "frontend", "network", 1, True, True),
    ("Error: Hydration failed because the initial UI does not match what was rendered on the server.",
     "frontend", "code_bug", 1, False, False),
]
for _t, _team, _kind, _base, _tr, _rg in SENTRY_TITLES:
    def _mk(title):
        def fn(rng, c):
            return json.dumps({"title": title, "level": "error", "culprit": f"{A(rng, ['ProductList', 'app/cart'])}",
                               "count": c["errors_last_5m"], "userCount": c["users_affected"],
                               "release": f"web@4.{rng.randint(1, 20)}.{rng.randint(0, 9)}",
                               "environment": c["environment"], "platform": "javascript"})
        return fn
    tpl("fe", "sentry", _t[:20], _team, _kind, _base, transient=_tr, db_root=False, rate_gated=_rg)(_mk(_t))


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------

def evaluate(t, f):
    n, thr = f["errors_last_5m"], f["alert_threshold_per_5m"]
    prod = f["environment"] == "production"
    noise = (t["noise"] or f["matches_known_benign_pattern"] or (t["rate_gated"] and n < thr)
             or (bool(t["transient"]) and f["retry_succeeded"] and n < thr))
    real = not noise
    if noise:
        sev = 0
    else:
        sev = min(3, t["base"] + (1 if (n >= thr or f["users_affected"] >= 100) else 0))
        if not prod:
            sev = min(sev, 1)
    page = prod and real and (sev == 3 or (sev == 2 and f["service_criticality"] == "critical"))
    deploy = real and f["started_after_latest_deploy"] and t["kind"] in ("code_bug", "config")
    if not real:
        action = "ignore"
    elif prod and deploy and sev >= 2:
        action = "roll_back"
    elif page:
        action = "page"
    else:
        action = "ticket"
    team = t["team"]
    if t["name"] == "28P01" and f.get("source_ip") in EXTERNAL_IPS and n >= thr:
        team = "security"
    return dict(noise=noise, real=real, sev=sev, page=page, deploy=deploy, action=action, team=team)


# ---------------------------------------------------------------------------
# questions: phrasings[-1] gate only, [-2] eval only
# ---------------------------------------------------------------------------

PHR = {
    "page": ["Should on-call be paged for this?", "Does this need to wake up the on-call engineer?",
             "Is this worth paging someone for right now?", "Would you page on-call over this log?",
             "Page the on-call engineer for this?"],
    "real": ["Is this a real failure someone should act on?", "Does this log show a genuine problem that needs action?",
             "Is somebody going to have to do something about this?", "Is this an actual failure rather than expected behaviour?",
             "Is this a real problem that needs someone's attention?"],
    "noise": ["Is this just noise that can be ignored?", "Can we safely ignore this log?",
              "Is this harmless background noise?", "Is it fine to leave this alone?",
              "Is this log safe to ignore?"],
    "db_root": ["Is the database the root cause here?", "Is the problem in the database itself?",
                "Does this point at the database as the source of the failure?",
                "Is the database where this failure originates?", "Is the database to blame for this?"],
    "deploy": ["Is this caused by the latest deploy?", "Did the most recent release introduce this?",
               "Is the last deploy responsible for this error?", "Would you blame the latest deployment for this?",
               "Did the latest deploy cause this?"],
    "retry": ["Would retrying likely fix this?", "Is this the kind of error that goes away on a retry?",
              "Would simply retrying the operation probably succeed?", "Is a retry likely to get past this error?",
              "Would a retry probably work here?"],
    "team": ["Which team should handle this?", "Who owns this problem?", "Which team should this be routed to?",
             "Route this to which team?", "Which team needs to pick this up?"],
    "kind": ["What kind of failure is this?", "Classify this failure.", "What category does this failure fall into?",
             "What type of problem is this?", "Which kind of failure does this log show?"],
    "action": ["What is the first thing to do?", "What should happen first?", "What's the right first response?",
               "What should on-call do first about this?", "What is the first step here?"],
    "sev": ["How severe is this?", "Rate the severity of this log.", "What severity would you give this?",
            "How bad is this?", "What severity is this?"],
}
NOUL_TEXT = {"page": ("Yes, page on-call.", "No, don't page."), "real": ("Yes, it's a real failure.", "No."),
             "noise": ("Yes, it's noise.", "No, it matters."), "db_root": ("Yes, the database.", "No, not the database."),
             "deploy": ("Yes, the deploy caused it.", "No."), "retry": ("Yes, a retry should work.", "No, retrying won't help.")}
TEAMS = {"frontend": "Frontend team.", "backend": "Backend team.", "database": "Database team / DBAs.",
         "infra": "Infrastructure / platform team.", "security": "Security team."}
KINDS = {"config": "Configuration problem.", "code_bug": "Bug in the code.", "capacity": "Capacity / resource exhaustion.",
         "network": "Network / connectivity.", "auth": "Authentication / authorization.", "data": "Bad or conflicting data."}
ACTIONS = {"page": "Page on-call now.", "ticket": "Open a ticket for normal working hours.",
           "roll_back": "Roll back the latest deploy.", "ignore": "Ignore it."}
SEV_LEVELS = ["Low: no action needed.", "Medium: fix during normal hours.",
              "High: needs attention soon.", "Critical: act now."]


def phr(rng, split, key):
    p = PHR[key]
    pool = [p[-1]] if split == "gate" else (p[:-1] if split == "eval" else p[:-2])
    return A(rng, pool)


def gold_map(t, r):
    g = {"page": r["page"], "real": r["real"], "noise": r["noise"], "action": r["action"], "sev": r["sev"]}
    if not t["noise"]:
        g["team"] = r["team"]
        if t["kind"] is not None:
            g["kind"] = t["kind"]
            g["deploy"] = r["deploy"]
        if t["transient"] is not None:
            g["retry"] = t["transient"]
        if t["db_root"] is not None:
            g["db_root"] = t["db_root"]
    return g


def make_question(rng, split, key, g):
    instr = phr(rng, split, key)
    if key in NOUL_TEXT:
        yes, no = NOUL_TEXT[key]
        return "noul", instr, {"false": no, "true": yes}, "true" if g else "false"
    if key == "sev":
        return "score", instr, list(SEV_LEVELS), str(g)
    opts = {"team": TEAMS, "kind": KINDS, "action": ACTIONS}[key]
    keys = list(opts)
    rng.shuffle(keys)
    return "choice", instr, {k: opts[k] for k in keys}, g


# ---------------------------------------------------------------------------
# facts + cases
# ---------------------------------------------------------------------------

def sample_facts(rng, kind, t, gate=False):
    thr = A(rng, [10, 20, 25, 50, 100])
    f = {"environment": A(rng, ["production"] * 6 + ["staging"] * 2 + ["development"]),
         "service": A(rng, WEB_APPS if kind == "fe" else SERVICES),
         "service_criticality": A(rng, CRITICALITY),
         "alert_threshold_per_5m": thr,
         "started_after_latest_deploy": rng.random() < 0.55,
         "matches_known_benign_pattern": rng.random() < 0.06,
         "retry_succeeded": rng.random() < 0.3}
    # counts: sit near the stated threshold half the time
    if rng.random() < 0.5:
        n = thr + A(rng, [-3, -2, -1, 0, 1, 2, 5])
    else:
        n = A(rng, [1, 1, 2, 3, rng.randint(1, thr * 20)])
    f["errors_last_5m"] = max(1, n)
    f["users_affected"] = A(rng, [0, rng.randint(1, 99), rng.randint(100, 5000)]) if kind != "db" else \
        A(rng, [0, 0, rng.randint(1, 99), rng.randint(100, 5000)])
    if t["name"] == "28P01":
        f["source_ip"] = A(rng, EXTERNAL_IPS if rng.random() < 0.6 else INTERNAL_IPS)
        if rng.random() < 0.5:
            f["errors_last_5m"] = rng.randint(f["alert_threshold_per_5m"], f["alert_threshold_per_5m"] * 30)
    if gate:  # obvious: both extremes, no threshold sitting
        loud = rng.random() < 0.5
        f["environment"] = "production" if loud else A(rng, ["development", "staging", "production"])
        f["errors_last_5m"] = thr * rng.randint(10, 40) if loud else 1
        f["users_affected"] = rng.randint(500, 20000) if loud else 0
        f["matches_known_benign_pattern"] = False
        f["retry_succeeded"] = False if loud else f["retry_succeeded"]
        f["service_criticality"] = "critical" if loud else A(rng, CRITICALITY)
    return f


FLIPS = ["environment", "matches_known_benign_pattern", "started_after_latest_deploy", "retry_succeeded",
         "errors_last_5m", "service_criticality", "users_affected", "source_ip"]


def flip(rng, f):
    f2 = dict(f)
    k = A(rng, [x for x in FLIPS if x in f])
    if k == "environment":
        f2[k] = A(rng, ["staging", "development"]) if f[k] == "production" else "production"
    elif k == "errors_last_5m":
        thr = f["alert_threshold_per_5m"]
        f2[k] = max(1, thr - rng.randint(1, 3)) if f[k] >= thr else thr + rng.randint(0, 3)
    elif k == "users_affected":
        f2[k] = rng.randint(0, 99) if f[k] >= 100 else rng.randint(100, 3000)
    elif k == "service_criticality":
        f2[k] = "standard" if f[k] == "critical" else "critical"
    elif k == "source_ip":
        f2[k] = A(rng, INTERNAL_IPS if f[k] in EXTERNAL_IPS else EXTERNAL_IPS)
    else:
        f2[k] = not f[k]
    return f2


def _flatten_english(state):
    parts = []
    for k, v in state.items():
        if k == "log":
            continue
        parts.append(f"{k.replace('_', ' ')} is {str(v).lower() if isinstance(v, bool) else v}")
    return "Log:\n" + state["log"] + "\n\nContext: " + "; ".join(parts) + "."


def build_state(rng, kind, fam, t, f):
    text = t["fn"](rng, f)
    state = {"log_source": {"db": "database", "app": "backend", "fe": "browser"}[kind] + f"/{fam}", "log": text}
    state.update(f)
    state["ref"] = f"INC-{rng.randint(100000, 999999)}"
    return state


def to_row(kind, fam, state, questions, gold):
    return {"source": f"our-cases-logs/{kind}/{fam}", "domain": f"logs/{kind}/{fam}",
            "state": state, "questions": questions, "gold": gold}


def assemble(rng, split, g, keys):
    questions, gold = {}, {}
    for i, key in enumerate(keys):
        qt, instr, crit, lab = make_question(rng, split, key, g[key])
        qid = f"{key}_{i}"
        questions[qid] = {"type": qt, "instructions": instr, "criteria": crit}
        gold[qid] = lab
    return questions, gold


def maybe_english(rng, state, split):
    return _flatten_english(state) if split != "gate" and rng.random() < 0.25 else state


def balanced_keys(rng, g, k, bal):
    """Prefer yes/no questions whose answer here is the under-represented one so far."""
    def w(key):
        if not isinstance(g[key], bool) or bal is None:
            return 1.0
        c = bal[key]
        return 0.25 if c[str(g[key]).lower()] > c[str(not g[key]).lower()] + 20 else 1.0
    keys, pool = [], list(g)
    while len(keys) < k:
        key = rng.choices(pool, weights=[w(x) for x in pool])[0]
        keys.append(key)
        pool.remove(key)
    return keys


def make_cases(rng, split, kind, fam, bal=None):
    t = A(rng, TEMPLATES[(kind, fam)])
    if fam == "postgres" and rng.random() < 0.12:  # the only security route; give it enough support
        t = next(x for x in TEMPLATES[(kind, fam)] if x["name"] == "28P01")
    if t["noise"] and split != "gate" and rng.random() < 0.5:  # noise lines are easy; halve them
        t = A(rng, [x for x in TEMPLATES[(kind, fam)] if not x["noise"]])
    f = sample_facts(rng, kind, t, gate=(split == "gate"))
    g = gold_map(t, evaluate(t, f))
    out = []
    if split != "gate" and rng.random() < 0.5 and not t["noise"]:
        for _ in range(6):
            f2 = flip(rng, f)
            g2 = gold_map(t, evaluate(t, f2))
            diff = [k for k in g if g[k] != g2.get(k)]
            if diff:
                others = [k for k in g if k not in diff]
                keys = diff[:3] + rng.sample(others, min(len(others), rng.randint(1, 2)))
                rng.shuffle(keys)
                s1, s2 = build_state(rng, kind, fam, t, f), build_state(rng, kind, fam, t, f2)
                s2["log"] = s1["log"] if "sentry" != fam else s2["log"]
                for s, gg in ((s1, g), (s2, g2)):
                    q, gl = assemble(rng, split, gg, keys)
                    out.append(to_row(kind, fam, maybe_english(rng, s, split), q, gl))
                return out
    k = 3 if split == "gate" else rng.randint(3, min(5, len(g)))
    keys = balanced_keys(rng, g, k, bal)
    q, gl = assemble(rng, split, g, keys)
    out.append(to_row(kind, fam, maybe_english(rng, build_state(rng, kind, fam, t, f), split), q, gl))
    return out


def train_state_keys(path):
    """(state, question type) keys of every train question, compared as leak_check does."""
    keys = set()
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        for q in row["questions"].values():
            keys.update(state_keys(row["state"], q.get("type")))
    return keys


def in_train(row, forbid):
    return in_keys(row["state"], [q.get("type") for q in row["questions"].values()], forbid)


def generate(path, seed, split, fams, target_q=None, target_cases=None, forbid=frozenset()):
    """forbid: train state keys; an eval/gate case about a trained state is dropped."""
    rng = random.Random(seed)
    seen, nq, nc, i = set(), 0, 0, 0
    fam_count, bal = Counter(), defaultdict(Counter)
    samples = defaultdict(list)
    with open(path, "w", encoding="utf-8") as fh:
        while (target_q and nq < target_q) or (target_cases and nc < target_cases):
            kind, fam = fams[i % len(fams)]
            i += 1
            for row in make_cases(rng, split, kind, fam, bal):
                dk = json.dumps([row["state"], row["questions"]], sort_keys=True)
                if dk in seen or (forbid and in_train(row, forbid)):
                    continue
                seen.add(dk)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                nc += 1
                fam_count[f"{kind}/{fam}"] += 1
                if len(samples[kind]) < 3 and rng.random() < 0.02:
                    samples[kind].append(row)
                for qid, q in row["questions"].items():
                    nq += 1
                    bal[qid.rsplit("_", 1)[0]][row["gold"][qid]] += 1
    return dict(cases=nc, questions=nq, fams=fam_count, bal=bal, samples=samples)


def main():
    all_fams = sorted(TEMPLATES)
    train_fams = [k for k in all_fams if k not in HELD_OUT]
    for p in (TRAIN_PATH, EVAL_PATH, GATE_PATH, SAMPLES_PATH):
        p.parent.mkdir(parents=True, exist_ok=True)
    # --test-only keeps the train file a model was trained on and redoes eval + gate against it.
    stats = {} if "--test-only" in sys.argv else {
        "train": generate(TRAIN_PATH, TRAIN_SEED, "train", train_fams, target_q=TRAIN_Q)}
    forbid = train_state_keys(TRAIN_PATH)
    stats["eval"] = generate(EVAL_PATH, EVAL_SEED, "eval", all_fams, target_q=EVAL_Q, forbid=forbid)
    stats["gate"] = generate(GATE_PATH, GATE_SEED, "gate", all_fams, target_cases=GATE_CASES, forbid=forbid)
    for name, s in stats.items():
        print(f"\n=== {name}: cases={s['cases']} questions={s['questions']}")
        print("  families:", dict(sorted(s["fams"].items())))
        for key, c in sorted(s["bal"].items()):
            print(f"  {key:8s} {dict(sorted(c.items()))}")

    sys.path.insert(0, str(ROOT / "finetuning" / "train"))
    import adapter  # noqa: E402
    for p in (TRAIN_PATH, EVAL_PATH, GATE_PATH):
        total = rej = 0
        for row in adapter.read_jsonl(p):
            total += 1
            a = adapter.adapt_row(row)
            if a is None or len(a["gold"]) != len(row["questions"]):
                rej += 1
        print(f"adapter {p.name}: {total} rows, {rej} rejected")

    with open(SAMPLES_PATH, "w", encoding="utf-8") as fh:
        for kind, rows in stats.get("train", {"samples": {}})["samples"].items():
            for row in rows:
                fh.write(json.dumps(row, indent=2, ensure_ascii=False) + "\n\n")
    print(f"samples -> {SAMPLES_PATH}")


if __name__ == "__main__":
    main()
