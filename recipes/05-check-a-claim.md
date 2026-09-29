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

```bash
curl -s localhost:21118/v1/systemone -d '{"state": "INVOICE 2026-117  Issued: 2026-09-01\nConsulting services, September .......... 1,250.00 EUR\nTotal due: 1,250.00 EUR\nPayment terms: due within 45 days of the issue date.", "questions": {"claim": {"type": "noul", "instructions": "Does the invoice say the total is 1,500 EUR?"}}}' | jq -c .answers
```

Real answer (openjevx v0.4.0 8-bit model, CPU):

```json
{"claim":{"action":{"act_probability":1},"answer_confidence":0.8777,"confidence":0.8777,"noul":0.1223,"probabilities":{"false":0.8777,"true":0.1223},"type":"noul"}}
```

## Same thing with jevx

```bash
jevx is --profile openjevx --no-context "Does the invoice say the total is 1,500 EUR?" < invoice.txt
```

Real output:

```
no 0.07
```

## What to do with the answer

If the answer is no, do not write the figure; re-read the source.

## How the local model did

Right: no (P(yes) 0.12).
