# Report 3 — Status snapshot

- **Released:** v0.3.0 live on GitHub (4-bit weight-only model), local server on port 21118
  healthy. v0.1.0 and v0.2.0 stay published for anyone still on them.
- **opencode + OpenRouter test generator:** finished on its own — 2,911 cases, 9,549 labelled
  questions, 76.9% agreement between the opencode generator and the OpenRouter blind labeller.
  Stopped short of the 10,000 target (consecutive free-model calls started failing — rate limits,
  most likely); nothing crashed.
- **Not yet run at that point:** the 9,549-question opencode set and the full 2,000-question
  public test, both against the exact shipped v0.3.0 server. (Both completed after this status —
  see report 4.)
- **Uncommitted at that point:** `scripts/quantize_w8.py` (the 8-bit build script, kept for the
  record even though 4-bit ships).
- **Waiting on the owner:** whether to rent the RTX 4090 on vast.ai for the fine-tune — the only
  step not yet started as of this report.
