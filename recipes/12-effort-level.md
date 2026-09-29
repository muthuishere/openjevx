# Do it here or hand it to a smaller model?

Rating: how much reasoning a task needs. `score` is the expected level (0 = first level).

## Call the local server

```bash
curl -s localhost:21118/v1/systemone -d '{"state": "Rename the variable foo to bar in utils.py", "questions": {"effort": {"type": "score", "instructions": "How much reasoning does this coding task need?", "criteria": ["trivial", "moderate", "hard"]}}}' | jq -c .answers
```

Real answer (openjevx v0.4.0 8-bit model, CPU):

```json
{"effort":{"action":{"act_probability":1},"answer_confidence":0.5348,"confidence":0.5348,"probabilities":{"0":0.5348,"1":0.4179,"2":0.0472},"score":0.5124,"type":"score"}}
```

## Same thing with jevx

```bash
jevx ask --profile openjevx --no-context --in "Rename the variable foo to bar in utils.py" --score effort="How much reasoning does this coding task need?|trivial;moderate;hard"
```

Real output:

```
effort           unsure     0.57
```

## What to do with the answer

Delegate low-effort work to a smaller model; keep the hard ones.

## How the local model did

Leans trivial (level 0 at 0.53, expected level 0.51) but not confidently; jevx says unsure. Hosted Jev says trivial 0.94.
