#!/usr/bin/env python3
"""
R0 + CLAIM-2 driver (runs on the RunPod pod).

Stages:
  1. Build tag-controlled role corpus (250 base x 5 roles), native harmony format.
  2. Extract per-layer content activations + logit features for gpt-oss-20b.
  3. Build random-marker control corpus; extract.
  4. Generate real conversations; extract activations for natural role spans.
  5. Analysis: layer sweep role probes, logit-baseline gate, marker gate,
     zero-shot generalisation, CLAIM-2 geometry battery.
  6. Write results.json + small arrays; caller pushes to git.

Everything heavy stays on the pod; only compact results leave.
"""
import os, sys, json, time, traceback
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trust_geometry.harmony_roles import build_variants, parse_harmony_spans, ROLES, ROLE_WRAP
from trust_geometry import analysis as A

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "r0_log.txt"), "w")
def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()

N_BASE = int(os.environ.get("TG_N_BASE", "250"))
MAX_CT = int(os.environ.get("TG_MAX_CT", "96"))

# fixed real user prompts for zero-shot generalisation (diverse, neutral)
REAL_PROMPTS = [
    "How do I grow tomatoes indoors as a beginner?",
    "Explain how a suspension bridge stays up.",
    "What's a good weekly plan to start running?",
    "Summarize why the sky is blue.",
    "How does compound interest work?",
    "Give me three tips for better sleep.",
    "What causes the seasons to change?",
    "How do vaccines train the immune system?",
    "Explain the difference between RAM and storage.",
    "What's a simple recipe for lentil soup?",
    "How do noise-cancelling headphones work?",
    "Why do onions make you cry when cut?",
    "Explain how GPS finds your location.",
    "What is the water cycle?",
    "How do I change a flat bike tire?",
    "Describe how bread rises.",
    "What makes a good password?",
    "How does a refrigerator keep food cold?",
    "Explain photosynthesis simply.",
    "What are tips for a first job interview?",
    "How do airplanes generate lift?",
    "What is the greenhouse effect?",
    "How do I start composting at home?",
    "Explain how Wi-Fi transmits data.",
    "What causes ocean tides?",
    "How does a camera focus an image?",
    "Give a beginner plan to learn guitar.",
    "Why does ice float on water?",
    "How do solar panels make electricity?",
    "Explain how memory works in the brain.",
]

def main():
    import torch
    from trust_geometry.extract import load, activations_for, generate_real_conversations
    tok, model = load(token=os.environ.get("HF_TOKEN"))
    log("model loaded. num layers reported by config:", model.config.num_hidden_layers)

    neutral = [json.loads(l) for l in open(os.path.join(os.path.dirname(__file__), "..", "data", "neutral_c4.jsonl"))][:N_BASE]
    log(f"neutral base sequences: {len(neutral)}")

    # ---- stage 1+2: role corpus ----
    seqs, spans, ys, groups = [], [], [], []
    for row in neutral:
        for rs in build_variants(tok, row["id"], row["text"], max_content_tokens=MAX_CT):
            seqs.append(rs.input_ids); spans.append((rs.content_start, rs.content_end))
            ys.append(ROLES.index(rs.role)); groups.append(row["id"])
    y = np.array(ys); groups = np.array(groups)
    log(f"role corpus: {len(seqs)} sequences; extracting activations...")
    t0 = time.time()
    means, logit_feats = activations_for(model, seqs, spans)
    log(f"role activations {means.shape} logit_feats {logit_feats.shape} in {time.time()-t0:.0f}s")
    Lp1 = means.shape[1]
    acts_by_layer = {L: means[:, L, :] for L in range(Lp1)}

    # ---- stage 3: random-marker control ----
    import random; random.seed(0)
    marker_tokens = [" banana", " concrete", " velvet", " orbit", " lantern"]
    mk_ids = [tok.encode(m, add_special_tokens=False) for m in marker_tokens]
    mk_ids = [m[0] for m in mk_ids]
    filler = tok.encode(" the", add_special_tokens=False)[0]
    msg = A_MESSAGE = tok.encode("<|message|>", add_special_tokens=False)
    end = tok.encode("<|end|>", add_special_tokens=False)
    mseqs, mspans, my, mgroups = [], [], [], []
    maxpre = 1 + len(msg)  # marker + <|message|>
    for row in neutral:
        cids = tok.encode(row["text"], add_special_tokens=False)[:MAX_CT]
        for k in range(5):
            ids = [filler] * (maxpre - (1 + len(msg))) + [mk_ids[k]] + msg + cids + end
            cs = 1 + len(msg)
            mseqs.append(ids); mspans.append((cs, cs + len(cids)))
            my.append(k); mgroups.append(row["id"])
    my = np.array(my); mgroups = np.array(mgroups)
    log(f"marker corpus: {len(mseqs)} sequences; extracting...")
    mmeans, _ = activations_for(model, mseqs, mspans)
    marker_by_layer = {L: mmeans[:, L, :] for L in range(mmeans.shape[1])}

    # ---- stage 4: real conversations (zero-shot generalisation) ----
    log("generating real conversations...")
    convos = generate_real_conversations(tok, model, REAL_PROMPTS, max_new_tokens=180)
    zs_seqs, zs_spans, zs_y, zs_groups = [], [], [], []
    for ci, c in enumerate(convos):
        for sp in parse_harmony_spans(c["ids"], tok):
            if sp["end"] - sp["start"] < 3:
                continue
            zs_seqs.append(c["ids"]); zs_spans.append((sp["start"], sp["end"]))
            zs_y.append(ROLES.index(sp["role"])); zs_groups.append(ci)
    log(f"zero-shot spans: {len(zs_seqs)} (roles present: {sorted(set(zs_y))})")
    zs_means = None
    if zs_seqs:
        zs_means, _ = activations_for(model, zs_seqs, zs_spans)
    zs_y = np.array(zs_y)

    # ---- stage 5: analysis ----
    log("training role probes (layer sweep)...")
    probes = A.train_role_probes(acts_by_layer, y, groups)
    best = probes["best_layer"]
    log(f"best layer {best} acc {probes['best_acc']:.3f} (chance {probes['chance']:.2f})")

    gate_logit = A.logit_baseline_gate(acts_by_layer[best], logit_feats, y, groups)
    gate_marker = A.marker_control_gate(probes["best_acc"], marker_by_layer, my, mgroups, best)
    log(f"logit-gate: hidden {gate_logit['hidden_acc']:.3f} vs logit {gate_logit['logit_acc']:.3f} passes={gate_logit['passes']}")
    log(f"marker-gate: real {gate_marker['real_acc']:.3f} vs marker {gate_marker['marker_acc']:.3f} passes={gate_marker['passes']}")

    # zero-shot generalisation: train on ALL neutral at best layer, apply to real convo tokens
    zs = {}
    if zs_means is not None and len(zs_y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(acts_by_layer[best])
        clf = LogisticRegression(C=0.5, max_iter=3000).fit(sc.transform(acts_by_layer[best]), y)
        zpred = clf.predict(sc.transform(zs_means[:, best, :]))
        zs["overall_acc"] = float((zpred == zs_y).mean())
        zs["per_role_recall"] = {ROLES[c]: float((zpred[zs_y == c] == c).mean())
                                 for c in sorted(set(zs_y.tolist()))}
        proba = clf.predict_proba(sc.transform(zs_means[:, best, :]))
        cot_idx = ROLES.index("cot")
        if cot_idx in clf.classes_:
            col = list(clf.classes_).index(cot_idx)
            zs["mean_cotness_by_true_role"] = {
                ROLES[c]: float(proba[zs_y == c, col].mean()) for c in sorted(set(zs_y.tolist()))}
        log(f"zero-shot generalisation acc {zs['overall_acc']:.3f}")

    log("CLAIM-2 geometry battery...")
    geo = A.geometry_battery(acts_by_layer[best], y, groups)
    log(f"participation_ratio {geo['participation_ratio']:.2f} (null_lo {geo['pr_null_lo']:.2f}) below_null={geo['pr_below_null']}")
    log(f"ordinal {geo['ordinal']['ordered_acc']:.3f} vs multinomial {geo['ordinal']['multinomial_acc']:.3f} ordered_sufficient={geo['ordinal']['ordered_is_sufficient']}")
    log(f"recovered order {geo['recovered_order_pc1']} matches_hyp={geo['order_matches']}")

    results = {
        "model": "openai/gpt-oss-20b", "n_base": len(neutral), "max_content_tokens": MAX_CT,
        "num_layers_plus1": Lp1, "roles": ROLES, "hyp_order": A.HYP_ORDER,
        "probes": probes, "gate_logit": gate_logit, "gate_marker": gate_marker,
        "zero_shot": zs, "geometry": geo,
        "layer_sweep": {int(L): probes["per_layer"][L]["acc"] for L in probes["per_layer"]},
    }
    json.dump(results, open(os.path.join(OUT, "r0_results.json"), "w"), indent=2)
    # small arrays for later R1 direction construction
    cent = np.stack([acts_by_layer[best][y == c].mean(0) for c in range(5)])
    np.savez(os.path.join(OUT, "r0_arrays.npz"),
             centroids_best=cent, best_layer=best,
             layer_accs=np.array([probes["per_layer"][L]["acc"] for L in sorted(probes["per_layer"])]))
    log("DONE. results written.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
        raise
