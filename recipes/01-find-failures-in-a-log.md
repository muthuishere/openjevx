# Find the failures in a log

Yes/no per log line: keep only the lines an on-call engineer would act on.

## Input

`app.log`:

```
2026-09-27 10:03:12 INFO  request served /checkout 200 in 84ms
2026-09-27 10:03:25 WARN  slow query 1.2s on orders_by_user
2026-09-27 10:03:40 ERROR payment gateway timeout after 30s (order 4021)
2026-09-27 10:03:51 INFO  cache warmed 1200 keys
2026-09-27 10:04:02 FATAL db connection refused: too many clients
2026-09-27 10:04:10 INFO  healthcheck ok
```

## Call the local server

```bash
curl -s localhost:21118/v1/systemone -d '{"state": "2026-09-27 10:03:12 INFO  request served /checkout 200 in 84ms", "questions": {"act": {"type": "noul", "instructions": "Is this line an error or failure an on-call engineer would act on?"}}}'
```

Real answer for the first input (model 0.5.2, server 0.5.7, CPU):

```json
{"answers":{"act":{"action":{"act_probability":1},"answer_confidence":0.9217,"confidence":0.9217,"noul":0.0783,"probabilities":{"false":0.9217,"true":0.0783},"type":"noul"}},"model":"openjevx","usage":{"input_tokens":64,"output_tokens":0,"server_ms":838.86}}
```

## Same thing with jevx

```bash
jevx filter --profile openjevx --no-context "Is this line an error or failure an on-call engineer would act on?" < app.log
```

Real output:

```
2026-09-27 10:03:40 ERROR payment gateway timeout after 30s (order 4021)
2026-09-27 10:04:02 FATAL db connection refused: too many clients
```

## Scores for every line

```bash
jevx ask --profile openjevx --no-context --lines app.log --noul act="Is this line an error or failure an on-call engineer would act on?"
```

Real output:

```
VERDICT  P     INPUT
no       0.08  2026-09-27 10:03:12 INFO  request served /checkout 200 in 84ms
no       0.10  2026-09-27 10:03:25 WARN  slow query 1.2s on orders_by_user
yes      0.90  2026-09-27 10:03:40 ERROR payment gateway timeout after 30s (order 4021)
no       0.09  2026-09-27 10:03:51 INFO  cache warmed 1200 keys
yes      0.90  2026-09-27 10:04:02 FATAL db connection refused: too many clients
no       0.09  2026-09-27 10:04:10 INFO  healthcheck ok
```

The table output needs jevx v0.11.0 or newer.

## What to do with the answer

Open the code behind the kept lines; skip the rest. Read `noul` as the probability of yes: jevx's defaults are yes at 0.8 or above and no at 0.2 or below; in between, ask a narrower question or check it yourself.

## How the local model did

Right on every line: the ERROR and FATAL lines are yes (0.90) and jevx filter keeps exactly those two; the INFO lines are no (0.08-0.09). The slow-query WARN is now a clear no (0.10), so if you want warnings too, ask a second question for them.
