# Which tool should handle this?

Pick-one among function names (function calling from natural language).

## Call the local server

```bash
curl -s localhost:21118/v1/systemone -d '{"state": "I want my money back for order 88", "questions": {"tool": {"type": "choice", "instructions": "Which function should handle this request?", "criteria": {"refund_payment": "refund a charge", "create_invoice": "create a new invoice", "update_email": "change the billing email"}}}}' | jq -c .answers
```

Real answer (model 0.5.2, server 0.5.7, CPU):

```json
{"tool":{"action":{"act_probability":1},"answer_confidence":0.971,"choice":"refund_payment","confidence":0.971,"probabilities":{"create_invoice":0.0149,"refund_payment":0.971,"update_email":0.014},"type":"choice"}}
```

## Same thing with jevx

```bash
jevx pick --profile openjevx --no-context "Which function should handle this request?" refund_payment="refund a charge" create_invoice="create a new invoice" update_email="change the billing email" --in "I want my money back for order 88"
```

Real output:

```
refund_payment 0.97
```

## What to do with the answer

Call the chosen function; extract its arguments (order 88) in code.

## How the local model did

Right: refund_payment 0.97.
