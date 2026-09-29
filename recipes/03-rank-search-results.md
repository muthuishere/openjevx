# Rank search results

Yes/no per result, sorted by the probability of yes.

## Input

`results.txt`:

```
Pricing of the enterprise plan
Refund policy for annual plans
Cancelling a subscription and getting money back
Changing your avatar
```

## Call the local server

```bash
curl -s localhost:21118/v1/systemone -d '{"state": "Pricing of the enterprise plan", "questions": {"hit": {"type": "noul", "instructions": "Does this page answer: how do I get my money back?"}}}'
```

Real answer for the first input (openjevx v0.4.0 8-bit model, CPU):

```json
{"answers":{"hit":{"action":{"act_probability":1},"answer_confidence":0.8412,"confidence":0.8412,"noul":0.1588,"probabilities":{"false":0.8412,"true":0.1588},"type":"noul"}},"model":"openjevx","usage":{"input_tokens":46,"output_tokens":0}}
```

## Same thing with jevx

```bash
jevx rank --profile openjevx --no-context "Does this page answer: how do I get my money back?" --top 3 < results.txt
```

Real output:

```
0.78  Cancelling a subscription and getting money back
0.29  Refund policy for annual plans
0.16  Pricing of the enterprise plan
```

## What to do with the answer

Open the top results only. The order matters more than the absolute numbers.

## How the local model did

The best hit ranks first, but "Refund policy for annual plans" scores only 0.29, so use the order, not a fixed cut-off.
