# R1 protocol: authority versus generic compliance/refusal

This protocol was fixed after R0 and before inspecting any R1 behavioral result.
R0 rejected the preregistered fixed ordering, so R1 must not rename whichever
direction transfers best as "authority" after the fact.

## Candidate directions

At R0's best hidden-state index (16; decoder layer output 15), estimate:

1. **Ordinal authority (primary):** the minimum-norm direction mapping the five
   controlled role centroids to the preregistered scores
   `system > assistant > cot > user > tool`.
2. **Centroid PC1 (diagnostic):** the unsupervised first principal component, oriented
   only so system projects above tool. This is not called authority unless it passes
   every R1 discriminant.
3. **Generic compliance:** compliant versus refusal completion-state contrast on
   matched benign exact-answer tasks.
4. **Refusal:** harmful versus harmless prompt-state contrast on matched request
   categories (Arditi-style mean difference).
5. **Controls:** seeded random direction, ordinal direction with compliance and
   refusal projected out, and a wrong-layer intervention.

All directions are unit norm. Steering dose is measured in standard deviations of
the controlled role corpus projected onto that direction.

## Behavioral design

Use single-token codewords in three balanced source-conflict arms:

- system versus user; steer the user span,
- user versus tool; steer the tool span,
- system versus tool; steer the tool span.

Codeword assignment is swapped across matched items. The dependent variable is the
lower-authority source's codeword log-odds against the higher-authority source's
codeword at the first final-answer token. This avoids judge noise and output-format
confounds.

For each conflict item, run doses `[-4, -2, 0, 2, 4]`. Matched no-conflict items give
both sources the same codeword and use an unused codeword as the distractor. Also
measure next-token KL on neutral prompts.

## Bright lines

The primary ordinal direction passes R1 only if all hold:

- pooled target-source log-odds increase by at least 1.0 from dose -4 to +4;
- the dose-response slope is positive in all three source-pair arms;
- the +4 versus -4 effect retains at least 70% after projecting out both generic
  compliance and refusal;
- absolute cosine is below 0.30 with each nuisance direction;
- median no-conflict correct-codeword log-odds drop is below 0.50 at either endpoint;
- mean neutral-prompt next-token KL is below 0.10 at either endpoint.

Failure means the R0 structure is not established as an authority-specific causal
mechanism. A PC1-only success is reported as exploratory and cannot rescue CLAIM 2.

