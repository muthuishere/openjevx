# Secret in a file before writing it

Yes/no on a line before it is written to disk or a commit.

## Call the local server

```bash
curl -s localhost:21118/v1/systemone -d '{"state": "DATABASE_URL=postgres://app:Pr0d-p4ss@db.internal:5432/app", "questions": {"secret": {"type": "noul", "instructions": "Does this line contain a password?"}}}' | jq -c .answers
```

Real answer (openjevx v0.4.0 8-bit model, CPU):

```json
{"secret":{"action":{"act_probability":1},"answer_confidence":0.8297,"confidence":0.8297,"noul":0.1703,"probabilities":{"false":0.8297,"true":0.1703},"type":"noul"}}
```

## Same thing with jevx

```bash
jevx is --profile openjevx --no-context "Does this line contain a password?" --in "DATABASE_URL=postgres://app:Pr0d-p4ss@db.internal:5432/app"
```

Real output:

```
unsure 0.34
```

## What to do with the answer

On yes, replace the value with an environment-variable reference before writing. On unsure, treat it as yes: a missed secret costs more than a false alarm.

## How the local model did

Missed on this model: P(yes) 0.17 and jevx says unsure 0.34. Hosted Jev says yes 0.99. For secrets use a pattern scanner; do not rely on this answer.
