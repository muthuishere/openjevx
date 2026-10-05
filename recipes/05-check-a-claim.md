# Check a claim against the source

Yes/no: does the document say what you are about to write?

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
curl -s -H "Authorization: Bearer $OPENJEVX_API_KEY" localhost:21160/v1/systemone -d '{"state": "INVOICE 2026-117  Issued: 2026-09-01\nConsulting services, September .......... 1,250.00 EUR\nTotal due: 1,250.00 EUR\nPayment terms: due within 45 days of the issue date.", "questions": {"claim": {"type": "noul", "instructions": "Does the invoice say the total is 1,500 EUR?"}}}' | jq -c .answers
```

Real answer (model 0.5.2, server 0.5.7, CPU):

```json
{"claim":{"action":{"act_probability":1},"answer_confidence":0.8829,"confidence":0.8829,"noul":0.1171,"probabilities":{"false":0.8829,"true":0.1171},"type":"noul"}}
```

## Same thing with jevx

```bash
jevx is --profile openjevx --no-context "Does the invoice say the total is 1,500 EUR?" < invoice.txt
```

Real output:

```
no 0.12
```

## What to do with the answer

If the answer is no, do not write the figure; re-read the source.

## How the local model did

Right: no (P(yes) 0.12).
