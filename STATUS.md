# trust-geometry — Project Summary & Status

_Last updated: 2026-07-11 after the first successful GPU run._

---

## 1. Motivation — where this came from

This project grew out of a close reading of **"Prompt Injection as Role Confusion"**
(arXiv:2603.12277) and its LessWrong write-up. The paper trains linear *role probes*
on gpt-oss-20b and argues that prompt injection happens because models infer "who is
speaking" from **style, not the true role tag** — injected text that *sounds like* a
trusted role occupies the same representational space as that role.

The seed insight (yours) was that the paper's argument **slides from correlation to
causation**: from "the same probe readout fires" → "it's the same internal feature" →
"it's the same causal mechanism." A shared linear-probe projection is not the same
thing as a shared latent feature, which is not the same thing as a shared causal
function. That gap is real, and the paper never closes it (it does **zero**
activation-space interventions — all its evidence is probe readouts plus text-level
ablations).

Working backward from that, we asked what would actually be **frontier** here. The
conclusion: a prompt-injection *defense* is catch-up work (capability/provenance
systems like CaMeL, dual-LLM, instruction-hierarchy training already exist). The
genuinely open, frontier-shaped question is the **science** the paper left on the
table.

---

## 2. Goal — the thesis

> **LLMs compute a single, causal, structured representation of "authority / trust" —
> how much to defer to a span of context — and that one representation is the shared
> mechanism behind a family of behaviors: prompt injection, sycophancy, jailbreak,
> and retrieval over-trust. Injection is one readout of it, not the point.**

Target paper title to earn: **"How models decide what to trust."**

Two load-bearing claims (frozen in `prereg/PREREGISTRATION.md`):

- **CLAIM 1 — Unification ("one axis").** A direction recovered from *prompt injection
  alone* causally transfers to sycophancy, jailbreak, and RAG over-trust. Killer
  result = a cross-behavior transfer matrix.
- **CLAIM 2 — Ordered scale.** Authority is a single *ordered* scalar
  (system > assistant > cot > user > tool, tested not assumed); the paper's separate
  role probes (systemness/userness/cotness/toolness) are *levels* on it, not five
  independent features.

Two **preregistered outcomes**, both publishable:
- **Success:** "One knob for trust: a single ordered authority axis causally unifies
  prompt injection, sycophancy, and jailbreak."
- **Honest fallback:** "Probing Is Not Enough for Trust: a decodable authority axis
  that is causally inert / separable across behaviors."

---

## 3. How we designed it (multi-agent workflow + prereg)

A workflow fanned out: **literature/novelty recon** (11 priors with URLs), a
**repo-capability audit** of the existing portfolio, **three independent design
ladders**, then an adversarial **novelty judge** and **red-team**.

Key conclusions that shaped the plan:
- **Novelty lives only in the conjunction.** Each individual piece is already owned —
  single refusal direction (Arditi), trait/persona vectors (Anthropic), steerable
  authority representation but with *distinct* subspaces (*Who is In Charge?* — direct
  counter-evidence to CLAIM 2), a generic cross-behavior "agreement" substrate (Shared
  Sycophancy-Jailbreak Circuit), an ordered social-role axis method (Granularity Axis),
  a RAG-trust knob (JUICE). What's unclaimed is injection-sourced transfer across ≥3
  behaviors through **one named ordered authority axis**, with the full causal battery.
- **The deadliest failure is a mislabeled true positive:** if the axis is really
  generic "comply / don't refuse," you'll get a beautiful unified transfer result that
  says nothing about *trust*. So the **discriminant** (authority ≠ generic compliance)
  is promoted to an early, load-bearing gate.
- **Ladder is ordered cheapest-fatal-first** (R0…R7), so the cheapest test that can
  kill the thesis runs before any expensive compute.

---

## 4. What we built and validated

The science/analysis layer was unit-tested on synthetic ground truth, then R0 was run
successfully on gpt-oss-20b on an H100.

| File | What it is | Status |
|------|-----------|--------|
| `data/neutral_c4.jsonl` | Fixed 320-seq C4 neutral-text sample (reproducible) | ✅ built |
| `src/trust_geometry/harmony_roles.py` | Tag-controlled role corpus in gpt-oss **native harmony format**, content positions matched across roles (the paper's positional control), byte-exact content spans; + harmony parser | ✅ tested |
| `src/trust_geometry/analysis.py` | 5-way role probes (grouped CV), **3 gates** (logit-baseline anti-circularity, random-marker artifact, zero-shot generalization), and the **CLAIM-2 ordered-scale geometry battery** (participation ratio vs random-centroid null, ordinal-vs-multinomial, recovered order) | ✅ unit-tested: correctly labels an ordered-line arrangement "one axis" and orthogonal clusters "bundle" |
| `src/trust_geometry/extract.py` | GPU activation / logit-feature extraction + real-conversation generation | ✅ GPU-run |
| `scripts/run_r0.py` | R0 + geometry driver | ✅ completed on gpt-oss-20b |
| `scripts/pod_main.py`, `src/trust_geometry/github_io.py` | Pure-Python pod entrypoint + GitHub-API results channel (heartbeats/results, no git/apt on pod) | ✅ written |
| `prereg/PREREGISTRATION.md` | Frozen thesis, claims, ladder, bright-line pass/fail criteria | ✅ |
| `README.md` | Overview | ✅ |

**Verified facts about the target model:** gpt-oss-20b = 24 layers, hidden 2880, MoE;
its harmony format routes a user "system" prompt to the `developer` role while the true
`system` role is the top-authority slot (matters for the ordered-scale claim).

---

## 5. GPU run: infrastructure post-mortem and R0 result

Six earlier RunPod attempts failed because of container-launch plumbing:

1. Image entrypoint (Jupyter/SSH) swallowed the start command → pod idle 50 min.
2. Official image had no `/workspace` dir + `set -e` → instant crash-loop.
3. `apt-get install git` unreliable in the container → clone failed → loop.
4. Dockerhub `pytorch/pytorch` image stalled pulling (uncached on host).
5. `python` not on PATH (only `python3`) with entrypoint cleared → exec-fail loop.
6. Fixed (`bash → base64 → python3`, cached RunPod image) but **killed before verifying**
   at the user's (correct) instruction to stop burning money.

On 2026-07-11, the required cheap launch validation passed on an A40, followed by a
successful full H100 R0 run. One H100 bootstrap retry was needed because the image's
Debian-installed `cryptography` package had no pip RECORD; installing it once with
`--ignore-installed` fixed the environment. The successful R0 driver completed in
570 seconds. All pods were terminated after artifact retrieval; nothing is billing.

**Process lesson (the real mistake):** repeatedly attached an **expensive H100 to debug
a launch mechanism**. Should have validated "can a cheap container run my code and
report back" on a near-free instance in ~2 minutes *before* ever touching a 20B model.
The known-good recipe is: **RunPod cached image + `bash` to base64-decoded `python3`
plus HTTP status/artifact channel + hard self-termination watchdog.**

### R0 result

- Best five-way role probe: **0.9888** at layer 16 (chance 0.20).
- Logit-baseline gate: **pass** (hidden 0.9888 vs logits 0.3680).
- Random-marker gate: **pass** (real 0.9888 vs marker 0.7808).
- Natural-conversation zero-shot accuracy: **0.5729** over 96 spans.
- Geometry is strongly low-dimensional: participation ratio **1.1909** and PC1
  variance **0.9142**, both far outside the random-centroid null.
- **CLAIM 2's preregistered ordered-scale test fails:** ordered accuracy 0.4416 vs
  multinomial 0.9888; recovered order `user, tool, system, cot, assistant` does not
  match `system, assistant, cot, user, tool` in either orientation.

Interpretation: R0 finds a strongly decodable, unusually low-dimensional role
structure, but not the claimed ordered authority scale. Calling PC1 an authority or
trust axis would be premature. Full metrics and hashes are in
`results/R0_SUMMARY.md` and `results/r0_results.json`.

---

## 6. What's pending

**Immediate (R1 — the discriminant decision gate):**
- Build the matched source-conflict/no-conflict behavioral suite.
- Construct candidate layer-16 directions without relabeling PC1 after seeing the
  outcome: preregistered ordinal contrast, unsupervised PC1, generic compliance, and
  refusal, plus random/orthogonal/wrong-layer controls.
- Test whether steering changes which source wins under conflict while leaving
  no-conflict task completion and coherence intact.
- Measure cosine and orthogonalized transfer against compliance and refusal.
- **Fail R1 -> stop:** the low-dimensional role signal is not authority-specific.

**Then the ladder (`prereg/PREREGISTRATION.md`), cheapest-fatal-first:**
- **R1 — Discriminant (decision gate):** authority ≠ generic compliance/refusal —
  differential per-source reweighting, unmoved no-conflict control, non-collinear with
  fitted compliance + refusal directions. **Fail → stop.**
- **R2:** single-behavior causal footing on injection (steer/ablate, matched controls,
  beat TF-IDF).
- **R3:** ordered-scale *causal* (monotone steering along the ladder, dose-response).
- **R4:** cross-behavior transfer matrix (CLAIM 1, killer result).
- **R5:** one-axis-vs-bundle adjudication (residual-distinguishability + joint ablation).
- **R6:** synthesis + mediation of the paper's CoTness→ASR 9%→90% dose-response.
- **R7:** cross-model replication.

**Biggest not-yet-built component:** the **behavioral suites** for sycophancy,
jailbreak-compliance, and retrieval over-trust (needed for R4). No repo currently
elicits/measures these; `exit-friction` is a reusable crossed-design template and RAMP
points to external jailbreak corpora, but these are net-new builds and are the true gate
on the unification claim.

**Infra status:** launch validation and the H100 path are proven. Reuse the HTTP
status/artifact channel and the `cryptography --ignore-installed` bootstrap repair.
