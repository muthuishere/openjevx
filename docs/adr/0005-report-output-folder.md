# ADR 0005 — Agent reports go in `llmresults/`, not `clauderesults/`

Status: proposed
Date: 2026-09-29

## Context

Agent sessions write their run reports, probes and result CSVs into a folder at the repo root.
That folder was named `clauderesults/`, after the tool that happened to write it. The public site
cites those files as sources, so the name leaked onto the site.

The site and `.gitignore` now say `llmresults/`, and the local folder was renamed. But sessions
still running with the old path kept writing new files to `clauderesults/`
(`09-usecase-probe.md`, `probe.py`, `probe_results.json`), and ADR 0004 cites
`clauderesults/09-usecase-probe.md`. So the folder keeps coming back and the reports end up in
two places.

## Decision

- Every agent report, probe and result file goes in `llmresults/`. The name describes the content,
  not the tool that wrote it.
- `clauderesults/` is retired. Nothing new is written there.
- Files in `llmresults/` are numbered `NN-topic.md`, continuing the existing sequence.
- `llmresults/*.jsonl` stays gitignored. Whether the `.md` reports are committed is a separate call.

## To do (separately)

- Move what is left in `clauderesults/` into `llmresults/`, then delete `clauderesults/`.
- Update ADR 0004's source path to `llmresults/09-usecase-probe.md`.
- Tell any running agent session, and any agent instructions (`AGENTS.md`), to write to
  `llmresults/`.
- Decide whether the `.md` reports are committed, so the site's "Source:" lines point at files
  a reader can open.

The site test (`site/tests/site.test.mjs`) already fails the build if any page mentions
`clauderesults`.
