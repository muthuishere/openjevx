# ADR 0007 — Managed fine-tuning: bring your data, get your decision model

Status: proposed
Date: 2026-09-29

## Context

ADR 0006 teaches a developer to fine-tune OpenJevX on their own data themselves. Most teams will
not: gathering rules out of documents, writing labelled questions with near-miss cases, checking
for leakage, renting a GPU and judging whether the result is good enough is days of specialist
work. The value is the model at the end: a small 8-bit ONNX (≤ 750 MB) that makes *their*
decisions (is this worth paging, which team owns this, does this discount apply, is this 4-day
week's deadline missed) in milliseconds, on their own hardware, with no per-call LLM cost.

We already own the whole pipeline (`finetuning/`): dataprep → datavalidate (leakage) → package →
a rented GPU box that trains, quantizes, uploads to R2 and destroys itself → gate → 8-bit ONNX
plus a trainable checkpoint. The service is that pipeline run for a customer, with the data work
done by us.

## Decision (proposed)

Offer **"bring your data, get your model"** as a managed service:

1. **Customer brings data.** Documents (policies, runbooks, pricing, terms, READMEs), code
   (validation logic, config), records (tickets with the team that took them, alerts with whether
   anyone acted, logs around real incidents, past decisions with outcomes). Upload to a
   per-customer private R2 prefix, or grant read access to a bucket/repo.
2. **We do the data work.**
   - *Chunk and extract*: split documents and code into rule candidates and decision points.
   - *Rules*: turn each rule into an executable check (threshold, date, membership, routing table).
     **The customer confirms every rule** before it labels anything (a review screen, not a chat).
   - *Label*: generate questions around each rule's boundary with near-miss twins, computed by
     evaluating the rule; map real records to questions with their real outcome as the label.
     A model may draft rules and wording; it never supplies a label.
   - *Hold out* a gate set per decision (customer-reviewed, never trained on) plus our basics gate.
3. **We train** with the existing one-job pipeline (smoke → full → gate) on our GPU provider.
4. **We deliver**: the 8-bit ONNX, the gate report (per decision: right & confident, confidently
   wrong, abstain rate), the openjevx server/Docker image to run it, and the trainable checkpoint
   so the next round starts from it. The customer runs it themselves; we never serve their traffic.
5. **Re-train loop**: customer sends back corrections (wrong answers with the right one) → next
   version. Versioned models, same gate, must not regress.

## Options

**A. Delivery**
- A1 (pick): customer downloads the ONNX + server image, runs it on their own machines.
- A2: we host an endpoint per customer. More revenue per customer, but we hold their traffic and
  their data longer; conflicts with the "owned decisions" pitch. Later, as an add-on.

**B. How much the customer does**
- B1 (pick): customer uploads data and confirms rules in a review UI; we do everything else.
- B2: fully done-for-you (we also confirm rules, from interviews). Higher price, slower, for enterprises.
- B3: self-serve only (ADR 0006 toolkit). Free tier / open source path.

**C. Pricing shape**
- C1 (pick to test): per model version (setup + per retrain), priced on decisions covered and
  rows labelled. GPU cost per run is ~$2–3, so price is almost all data work.
- C2: subscription with N retrains/month. C3: enterprise contract.

**D. Data handling**
- Per-customer private R2 prefix, encrypted at rest, signed links only; GPU boxes get time-limited
  links and destroy themselves (already true). Retention: delete raw uploads N days after delivery
  unless the customer opts into retrain storage. DPA/terms before the first paid customer.
- Customer data is never used to train anyone else's model or ours.

**E. What we build first**
- E1 (pick): a thin job layer over `finetuning/` — customer, dataset, rule set, run, model
  version, gate report — plus the rule-review screen. Everything else is the pipeline we have.
- E2: full web app first. Too much before the first customer.

## Consequences

- The hard, valuable part (rules → labels → gate) becomes the product; training is a commodity step.
- Needs: tenant isolation in storage and runs, a rule-review UI, per-customer gates, run history,
  billing, a DPA. Our basics gate stays mandatory for every customer model.
- Risk: customer data quality. Mitigation: the gate is the contract — we ship only when the
  customer's own gate passes, and say plainly when it does not.
- The self-hosted version of the same thing is ADR 0008.

## Open questions

- First design partner and their first three decisions.
- Minimum data per decision to promise a result (start: 20 confirmed rules or 500 real outcomes).
- Whether rule extraction reads code as well as prose in v1 (same open item as ADR 0006).
