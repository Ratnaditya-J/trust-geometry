# trust-geometry

Do LLMs compute a single, ordered, causal representation of **authority / trust** —
"how much should I defer to this span of context" — that unifies prompt injection,
sycophancy, jailbreak, and retrieval over-trust? Working backward from the title
*"How models decide what to trust."*

Model under test: `openai/gpt-oss-20b`. Source paper: *Prompt Injection as Role
Confusion* (arXiv:2603.12277). See [`prereg/PREREGISTRATION.md`](prereg/PREREGISTRATION.md)
for the frozen thesis, the two load-bearing claims, the ladder, and the pre-registered
pass/fail bright lines.

## Layout
- `src/trust_geometry/harmony_roles.py` — tag-controlled role corpus in gpt-oss native
  harmony format, with position-matched content spans; harmony parser.
- `src/trust_geometry/extract.py` — GPU activation/logit extraction + generation.
- `src/trust_geometry/analysis.py` — probes, gates, and the CLAIM-2 ordered-scale
  geometry battery (pure numpy/sklearn, unit-tested on synthetic ground truth).
- `scripts/run_r0.py` — R0 + geometry driver (runs on GPU pod).
- `scripts/pod_bootstrap.sh` — pod entrypoint; pushes results to branch `r0-run`.
- `data/neutral_c4.jsonl` — fixed 320-sequence C4 neutral-text sample (reproducible).

## Status
R0 completed on gpt-oss-20b on 2026-07-11. Role is strongly decodable and the
controlled centroids are unusually low-dimensional, but the preregistered ordered
authority scale fails: the 1-D ordered classifier reaches 0.442 versus 0.989 for the
multinomial probe, and PC1 does not recover the hypothesized role order. See
[`results/R0_SUMMARY.md`](results/R0_SUMMARY.md). R1's causal authority-vs-compliance
discriminant is next.
