# ADR 0005 — Agent reports go in `llmresults/`

Status: proposed
Date: 2026-09-29

## Decision

Every agent report, probe and result file goes in `llmresults/`. The name describes the content,
not the tool that wrote it. Files are numbered `NN-topic.md`. `llmresults/*.jsonl` is gitignored.

## To do (separately)

- Move any reports still sitting in the old, tool-named folder into `llmresults/` and delete it.
- Point running agent sessions and `AGENTS.md` at `llmresults/`.
- Decide whether the `.md` reports are committed, so the site's "Source:" lines point at files a
  reader can open.

`site/tests/site.test.mjs` fails the build if any built page mentions the old folder name.
