# Pick a value among candidates

Pick-one: code parses the candidate dates, the model chooses which one is the due date.

## Input

`invoice.txt`:

```
INVOICE 2026-117  Issued: 2026-09-01
Consulting services, September .......... 1,250.00 EUR
Total due: 1,250.00 EUR
Payment terms: due within 45 days of the issue date.
```

## Call the local server

The calls send `OPENJEVX_API_KEY`, and the `jevx` lines use the `openjevx` profile that carries it: [the API key](README.md).

```bash
curl -s -H "Authorization: Bearer $OPENJEVX_API_KEY" localhost:21160/v1/systemone -d '{"state": "INVOICE 2026-117  Issued: 2026-09-01\nConsulting services, September .......... 1,250.00 EUR\nTotal due: 1,250.00 EUR\nPayment terms: due within 45 days of the issue date.", "questions": {"due": {"type": "choice", "instructions": "Which date is the payment due date?", "criteria": {"a": "2026-09-01", "b": "2026-10-16", "c": "2026-10-31"}}}}' | jq -c .answers
```

Real answer (model 0.5.2, server 0.5.7, CPU):

```json
{"due":{"action":{"act_probability":1},"answer_confidence":0.898,"choice":"a","confidence":0.898,"probabilities":{"a":0.898,"b":0.047,"c":0.055},"type":"choice"}}
```

## Same thing with jevx

```bash
jevx pick --profile openjevx --no-context "Which date is the payment due date?" a=2026-09-01 b=2026-10-16 c=2026-10-31 --in @invoice.txt
```

Real output:

```
a 0.90
```

## What to do with the answer

Use the chosen candidate. Compute the candidates in code (45 days after 2026-09-01 is 2026-10-16); the model judges text, it does not do date arithmetic.

## How the local model did

Wrong on this model, and confidently: it picks `a` (the issue date, 0.90 from curl and jevx) instead of `b` (2026-10-16). Hosted Jev picks `b 1.00`. Do not use the local model for this one; compute due dates in code.
