# trust-geometry — Preregistration (frozen before results)

**Thesis.** LLMs compute a single, causal, structured representation of *authority /
trust* — "how much should I defer to this span of context" — and that one
representation is the shared mechanism behind a family of behaviors (prompt
injection, sycophancy, jailbreak, retrieval over-trust). Prompt injection is one
readout of it, not the point.

Working backward from the target paper title **"How models decide what to trust."**
Model under test: **openai/gpt-oss-20b** (the source paper's model). Source paper:
*Prompt Injection as Role Confusion* (arXiv:2603.12277).

## Two load-bearing claims

- **CLAIM 1 — Unification (one axis).** A direction recovered from *prompt injection
  alone* causally transfers to sycophancy, jailbreak compliance, and retrieval
  over-trust. Killer result = cross-behavior transfer matrix.
- **CLAIM 2 — Ordered scale.** Authority is a single *ordered* scalar
  (system > assistant > cot > user > tool, tested not assumed); the source paper's
  separate role probes (systemness/userness/cotness/toolness) are levels on it, not
  five independent features.

## Two preregistered outcomes (either is publishable)

- **Success:** "One knob for trust: a single ordered authority axis causally unifies
  prompt injection, sycophancy, and jailbreak."
- **Honest fallback:** "Probing Is Not Enough for Trust: a decodable authority axis
  that is causally inert / separable across behaviors."

## Ladder (cheapest-fatal-first; red-team ordering)

- **R0 — Foundation + gates** *(this run)*. Reproduce tag-controlled role probes on
  gpt-oss-20b native harmony format. Gates: (a) probe beats the model's own
  output-logit baseline (anti-circularity); (b) real role tags beat random-marker
  pseudo-roles (no tokenization artifact); (c) zero-shot generalization to the
  model's own natural conversations.
- **R0-geometry — CLAIM 2**. Do the 5 role centroids collapse onto one ordered axis?
  Pre-registered bright lines: participation ratio below the 2.5th pct of a
  random-centroid null; ordered-axis held-out accuracy within 0.05 of the unordered
  multinomial at fewer DoF; recovered PC1 order matches the hypothesized authority
  order **and** (R1) an independently-measured behavioral deference order.
- **R1 — Discriminant (next run, the decision gate).** Authority ≠ generic
  compliance/refusal: the axis must produce *differential per-source reweighting*
  (compress the system>user>tool gap), leave a no-conflict compliance control
  unmoved, beat and be non-collinear with a fitted generic-compliance direction, and
  have low cosine with the refusal direction (transfer survives orthogonalizing
  refusal out). **Fail → stop; it is global obedience, not trust.**
- **R2** single-behavior causal footing on injection (steer/ablate, matched
  random/orthogonal/wrong-layer controls, beat TF-IDF).
- **R3** ordered-scale causal (monotone steering along the ladder, dose-response).
- **R4** cross-behavior transfer matrix (CLAIM 1). Pass: off-diagonal ≥ ~70% of the
  native-fit diagonal, beating a surface-transfer baseline.
- **R5** one-axis-vs-bundle: post-match 2nd-classifier AUC CI includes 0.5; any
  surviving residual must be causally inert; 1-D ablation jointly knocks out all
  behaviors with capability retention.
- **R6** synthesis + mediation of the source paper's CoTness→ASR 9%→90% dose-response.
- **R7** cross-model replication.

## Required controls (named, non-negotiable)

Refusal direction (Arditi-style) — report cosine, orthogonalize-and-retest;
generic-compliance direction; position-only and length-only control directions;
native-direction upper bound in every transfer cell; efficacy gate + coherence/KL
floor on steering; base/random-marker artifact control; placebo non-trust behavior;
capability-retention check under ablation; surface-*transfer* baseline.

## Prior art we must out-scope (else "just another steering-vector paper")

Arditi et al. (single refusal direction); Persona Vectors (trait directions incl.
sycophancy); *Who is In Charge?* (authority is steerable but system-user vs social
subspaces are **distinct** — counter-evidence to CLAIM 2); Shared Sycophancy-Lying/
Jailbreak Circuit (generic cross-behavior "agreement" substrate); Granularity Axis
(ordered social-role scalar method); JUICE / Parameters-vs-Context (RAG trust knob).
Novelty lives only in the **conjunction**: injection-sourced transfer across ≥3
behaviors through one *named ordered authority axis*, with the full causal battery.
