# OpenJevX recipes

Eighteen everyday decisions an agent or a script hands to OpenJevX, each as a call you can run against your own
server. Start the server (`task run`, the Docker image, or a release binary), then copy any call below. The
server listens on `localhost:21118` by default. These pages are also in the Docker image as files, in
`/usr/share/openjevx/recipes`, and on the server at `/recipes`.

Start with [what you get](what-you-get.md): model size, measured CPU latency, and the three question types.

## The API key

From server 0.5.7, `POST /v1/systemone` needs `Authorization: Bearer <key>` whenever the server listens on anything
other than loopback (the Docker image, a cloud server). Every call on these pages sends it from `OPENJEVX_API_KEY`:

```bash
export OPENJEVX_API_KEY=$(cat /path/to/openjevx.api-key)   # the file the server names at startup
```

The key is the one you set (`OPENJEVX_API_KEY` or `"api_key"` in `openjevx.json`), or the one the server generated:
its startup log says `api key: required on /v1/systemone ... (from /path/openjevx.api-key)`, beside `openjevx.json`
(in the Docker image, `/data/openjevx.api-key`). On `127.0.0.1` with no key set the server needs none and ignores
the header, so the calls work unchanged there.

For `jevx`, put the header in the profile once; every `jevx` call on these pages uses `--profile openjevx`, and
jevx reads the variable at call time:

```bash
jevx profile add openjevx http://localhost:21118/v1/systemone --model openjevx-0.5.2 \
  --header 'Authorization: Bearer $OPENJEVX_API_KEY'
```

## The request

Every recipe is one `POST /v1/systemone` with a `state` (the text or JSON the question is about) and named
`questions`. There are three question types:

| type | ask | criteria | read the answer from |
|---|---|---|---|
| `noul` | yes/no | none (optional `{"true": "...", "false": "..."}`) | `noul`: probability of yes |
| `choice` | pick one | `{"key": "description", ...}` | `choice` and `confidence` |
| `score` | rating | `["lowest", ..., "highest"]` | `score` (expected level, 0 = first) and `probabilities` |

## Every answer here is real

Each recipe was run on 2026-10-03 against a local OpenJevX server (model 0.5.2, server 0.5.7, CPU, Apple M5 Pro)
and the output was pasted unedited, including the wrong and unsure ones; two runs gave the same answers. Each page
ends with how the local model did. The `jevx` lines ran through jevx v0.11.0 with `--profile openjevx` and
`--no-context` (so your agent files do not change the question) and `--fresh` (so no cached answer was reused;
`jevx judge` has no cache flag); the tables and `--fresh` need jevx v0.11.0 or newer. The "hosted Jev" answers
quoted on some pages were recorded once, earlier, and were not re-run. Your inputs will score differently: copy the pattern, not the numbers, and test on your own data.

## Recipes

| recipe | decision |
|---|---|
| [Find the failures in a log](01-find-failures-in-a-log.md) | Yes/no per log line: keep only the lines an on-call engineer would act on. |
| [Route a ticket queue](02-route-a-ticket-queue.md) | Pick-one per ticket: which team owns it. |
| [Rank search results](03-rank-search-results.md) | Yes/no per result, sorted by the probability of yes. |
| [Pick a value among candidates](04-pick-a-value.md) | Pick-one: code parses the candidate dates, the model chooses which one is the due date. |
| [Check a claim against the source](05-check-a-claim.md) | Yes/no: does the document say what you are about to write? |
| [Before asking the user "A or B?"](06-before-asking-a-or-b.md) | Pick-one: what would the user want next, given what they already said. |
| [Is this command dangerous?](07-is-this-command-dangerous.md) | Yes/no about a command before running it. |
| [Prompt injection in a fetched page](08-prompt-injection.md) | Yes/no: does fetched text try to instruct an AI assistant? |
| [Which file to open first](09-which-file-first.md) | Yes/no per file path, sorted by the probability of yes. |
| [Flaky test or real bug?](10-flaky-or-real-bug.md) | Pick-one per CI failure. |
| [Sort PR review comments](11-sort-review-comments.md) | Pick-one per review comment, four ways. |
| [Do it here or hand it to a smaller model?](12-effort-level.md) | Rating: how much reasoning a task needs. |
| [Question or instruction?](13-question-or-instruction.md) | Pick-one: answer the message, or change code? |
| [Secret in a file before writing it](14-secret-before-writing.md) | Yes/no on a line before it is written to disk or a commit. |
| [Did the build succeed?](15-did-the-build-succeed.md) | Yes/no over build output before reporting "done". |
| [Which tool should handle this?](16-which-tool.md) | Pick-one among function names (function calling from natural language). |
| [Several judgements in one call](17-several-judgements.md) | One request, three questions of the three types. |
| [Would the user accept this turn?](18-would-the-user-accept.md) | Yes/no and rating about the agent's own final message, before handing back. |

## Reading the result

| you get | the agent should |
|---|---|
| `noul` at or above 0.8, or `confidence` at or above 0.6 for a choice or score | act on it, and say so in its report |
| `noul` at or below 0.2 | act on the no |
| anything in between | rewrite the question more narrowly, check it itself, or ask the user |
| an HTTP error | report that the call failed; never treat it as a no |

These are the `jevx` CLI's default thresholds. When the answer is unsure, the fix is usually a narrower question:
name the exact thing you are looking for, say what counts as yes, and let code do arithmetic and dates.

jevx caches answers for 7 days, keyed by the profile's model name. After upgrading the OpenJevX model, run
`jevx cache clear`, or give the profile a versioned model name (for example `openjevx-0.5.2`), so you do not get the
old model's answers back.
