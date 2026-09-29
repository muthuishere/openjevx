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
{"answers":{"kind":{"action":{"act_probability":1},"answer_confidence":0.4024,"choice":"should","confidence":0.4024,"probabilities":{"must":0.113,"nit":0.3363,"none":0.1483,"should":0.4024},"type":"choice"}},"model":"openjevx","usage":{"input_tokens":53,"output_tokens":0}}
```

## Same thing with jevx

```bash
jevx ask --profile openjevx --no-context --lines reviews.txt --choice kind="What kind of review comment is this?|must=a real bug or risk that must be fixed;should=a reasonable change request;nit=style or naming only;none=praise or approval"
```

Real output:

```
VERDICT  P     INPUT
unsure   0.43  nit: rename x to count
unsure   0.51  This loop never terminates when the list is empty
none     0.67  LGTM, nice work
unsure   0.43  Could we reuse the retry helper here instead of a new one?
none     0.63  Looks good to me
must     0.67  This builds the SQL with string concatenation from user input: SQL injection
```

## What to do with the answer

Handle `must` first, reply to `should`, batch the nits.

## How the local model did

The two `none` lines and the SQL injection (`must`, 0.67) are right; three lines come out unsure. jevx wraps your question in its own prompt and applies its yes/no thresholds, so its numbers differ from the raw curl call above.
