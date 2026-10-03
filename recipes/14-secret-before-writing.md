# Secret in a file before writing it

Yes/no on a line before it is written to disk or a commit.

## Call the local server

The calls send `OPENJEVX_API_KEY`, and the `jevx` lines use the `openjevx` profile that carries it: [the API key](README.md).

```bash
curl -s -H "Authorization: Bearer $OPENJEVX_API_KEY" localhost:21118/v1/systemone -d '{"state": "DATABASE_URL=postgres://app:Pr0d-p4ss@db.internal:5432/app", "questions": {"secret": {"type": "noul", "instructions": "Does this line contain a password?"}}}' | jq -c .answers
```

Real answer (model 0.5.2, server 0.5.7, CPU):

```json
{"secret":{"action":{"act_probability":1},"answer_confidence":0.799,"confidence":0.799,"noul":0.201,"probabilities":{"false":0.799,"true":0.201},"type":"noul"}}
```

## Same thing with jevx

```bash
jevx is --profile openjevx --no-context "Does this line contain a password?" --in "DATABASE_URL=postgres://app:Pr0d-p4ss@db.internal:5432/app"
```

Real output:

```
unsure 0.20
```

## What to do with the answer

On yes, replace the value with an environment-variable reference before writing. On unsure, treat it as yes: a missed secret costs more than a false alarm.

## How the local model did

Missed on this model: P(yes) 0.20, right on jevx's no line, so jevx says unsure. Hosted Jev says yes 0.99. For secrets use a pattern scanner; do not rely on this answer.
