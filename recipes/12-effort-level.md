# Do it here or hand it to a smaller model?

Rating: how much reasoning a task needs. `score` is the expected level (0 = first level).

## Call the local server

The calls send `OPENJEVX_API_KEY`, and the `jevx` lines use the `openjevx` profile that carries it: [the API key](README.md).

```bash
curl -s -H "Authorization: Bearer $OPENJEVX_API_KEY" localhost:21118/v1/systemone -d '{"state": "Rename the variable foo to bar in utils.py", "questions": {"effort": {"type": "score", "instructions": "How much reasoning does this coding task need?", "criteria": ["trivial", "moderate", "hard"]}}}' | jq -c .answers
```

Real answer (model 0.5.2, server 0.5.7, CPU):

```json
{"effort":{"action":{"act_probability":1},"answer_confidence":0.68,"confidence":0.68,"probabilities":{"0":0.68,"1":0.2744,"2":0.0455},"score":0.3655,"type":"score"}}
```

## Same thing with jevx

```bash
jevx ask --profile openjevx --no-context --in "Rename the variable foo to bar in utils.py" --score effort="How much reasoning does this coding task need?|trivial;moderate;hard"
```

Real output:

```
effort           trivial    0.68
```

## What to do with the answer

Delegate low-effort work to a smaller model; keep the hard ones.

## How the local model did

Right: trivial (level 0 at 0.68, expected level 0.37), and jevx says trivial 0.68. Hosted Jev says trivial 0.94.
