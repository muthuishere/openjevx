# Sort PR review comments

Pick-one per review comment, four ways.

## Input

`reviews.txt`:

```
nit: rename x to count
This loop never terminates when the list is empty
LGTM, nice work
Could we reuse the retry helper here instead of a new one?
Looks good to me
This builds the SQL with string concatenation from user input: SQL injection
```

## Call the local server

```bash
curl -s localhost:21118/v1/systemone -d '{"state": "nit: rename x to count", "questions": {"kind": {"type": "choice", "instructions": "What kind of review comment is this?", "criteria": {"must": "a real bug or risk that must be fixed", "should": "a reasonable change request", "nit": "style or naming only", "none": "praise or approval"}}}}'
```

Real answer for the first input (openjevx v0.4.0 8-bit model, CPU):

```json
{"answers":{"kind":{"action":{"act_probability":1},"answer_confidence":0.5296,"choice":"nit","confidence":0.5296,"probabilities":{"must":0.0552,"nit":0.5296,"none":0.1049,"should":0.3103},"type":"choice"}},"model":"openjevx","usage":{"input_tokens":53,"output_tokens":0}}
```

## Same thing with jevx

```bash
jevx ask --profile openjevx --no-context --lines reviews.txt --choice kind="What kind of review comment is this?|must=a real bug or risk that must be fixed;should=a reasonable change request;nit=style or naming only;none=praise or approval"
```

Real output:

```
VERDICT  P     INPUT
nit      0.62  nit: rename x to count
must     0.80  This loop never terminates when the list is empty
none     0.77  LGTM, nice work
unsure   0.40  Could we reuse the retry helper here instead of a new one?
unsure   0.59  Looks good to me
must     0.86  This builds the SQL with string concatenation from user input: SQL injection
```

The table output needs jevx v0.11.0 or newer.

## What to do with the answer

Handle `must` first, reply to `should`, batch the nits.

## How the local model did

Four of six are right: the nit (0.62), the endless loop (`must`, 0.80), LGTM (`none`, 0.77) and the SQL injection (`must`, 0.86). The retry-helper request (0.40) and "Looks good to me" (0.59) come out unsure.
