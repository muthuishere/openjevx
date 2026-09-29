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
while IFS= read -r l; do
  jq -nc --arg l "$l" '{state: $l, questions: {"act": {"type": "noul", "instructions": "Is this line an error or failure an on-call engineer would act on?"}}}' \
  | curl -s localhost:21118/v1/systemone -d @- | jq -c --arg l "$l" '{p: .answers.act.noul, line: $l}'
done < app.log
```

Real answer (openjevx v0.4.0 8-bit model, CPU):

```json
{"p":0.1866,"line":"2026-09-27 10:03:12 INFO  request served /checkout 200 in 84ms"}
{"p":0.7359,"line":"2026-09-27 10:03:25 WARN  slow query 1.2s on orders_by_user"}
{"p":0.8356,"line":"2026-09-27 10:03:40 ERROR payment gateway timeout after 30s (order 4021)"}
{"p":0.1212,"line":"2026-09-27 10:03:51 INFO  cache warmed 1200 keys"}
{"p":0.8385,"line":"2026-09-27 10:04:02 FATAL db connection refused: too many clients"}
{"p":0.1654,"line":"2026-09-27 10:04:10 INFO  healthcheck ok"}
```

## Same thing with jevx

```bash
jevx filter --profile openjevx --no-context "Is this line an error or failure an on-call engineer would act on?" < app.log
```

Real output:

```
(no output: jevx kept no line at or above its yes threshold)
```

## What to do with the answer

Open the code behind the kept lines; skip the rest. Read `noul` as the probability of yes: jevx's defaults are yes at 0.8 or above and no at 0.2 or below; in between, ask a narrower question or check it yourself.

## How the local model did

Right on ERROR and FATAL (0.84), but it also scores the slow-query WARN at 0.74, which hosted Jev leaves out. jevx filter kept no line at all on this model: its yes threshold is stricter than these scores. jevx wraps your question in its own prompt and applies its yes/no thresholds, so its numbers differ from the raw curl call.
