"""
Analysis core for trust-geometry R0 + CLAIM 2 (ordered-scale) geometry.

Pure numpy / scipy / sklearn so it is unit-testable without a GPU. The GPU side
(extract.py) only produces activation arrays; all statistics live here.

R0 deliverables:
  - train_role_probes:   5-way multinomial role probes per layer, grouped CV
                         (leave-base-sequence-out, so no base text leaks train->test).
  - logit_baseline_gate: anti-circularity control (latent-horizon) — the hidden-state
                         probe must beat a probe fit on the model's own output logits.
  - marker_control_gate: random-marker control (red-team) — arbitrary rare-token
                         "roles" must NOT separate as well as real role tags, else
                         separability is a trivial tokenization artifact.

CLAIM 2 (ordered scalar) geometry battery, on per-role content-activation centroids:
  - effective_dimensionality (participation ratio) vs a random-centroid null.
  - pc1_variance vs random-centroid null.
  - ordinal_vs_multinomial: does a single ORDERED axis (cumulative-link / threshold
    model) explain held-out role assignment about as well as the unordered multinomial,
    at fewer degrees of freedom? If yes -> authority behaves as one ordered scalar.
  - recovered_order: ordering of role centroids along the fitted authority axis,
    compared to the hypothesised system>assistant>cot>user>tool.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROLES = ["system", "user", "cot", "assistant", "tool"]
# hypothesised authority ordering (tested, not assumed by the geometry itself)
HYP_ORDER = ["system", "assistant", "cot", "user", "tool"]


def _cv_accuracy(X, y, groups, C=0.5, n_splits=5, seed=0):
    """Grouped-CV multinomial LR accuracy + per-class recall. Groups = base ids."""
    ng = len(np.unique(groups))
    n_splits = min(n_splits, ng)
    gkf = GroupKFold(n_splits=n_splits)
    accs, per = [], {r: [] for r in range(len(np.unique(y)))}
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        accs.append(float((pred == y[te]).mean()))
        for c in np.unique(y[te]):
            m = y[te] == c
            per[int(c)].append(float((pred[m] == c).mean()))
    return float(np.mean(accs)), float(np.std(accs)), {c: float(np.mean(v)) for c, v in per.items() if v}


def train_role_probes(acts_by_layer: dict[int, np.ndarray], y: np.ndarray,
                      groups: np.ndarray, C=0.5) -> dict:
    """acts_by_layer[layer] -> (N, d) mean-pooled content activations. y in [0..4]."""
    chance = 1.0 / len(np.unique(y))
    layers = sorted(acts_by_layer)
    per_layer = {}
    for L in layers:
        acc, sd, per = _cv_accuracy(acts_by_layer[L], y, groups, C=C)
        per_layer[L] = {"acc": acc, "acc_std": sd, "per_role": per}
    best = max(layers, key=lambda L: per_layer[L]["acc"])
    return {"chance": chance, "per_layer": per_layer, "best_layer": int(best),
            "best_acc": per_layer[best]["acc"]}


def logit_baseline_gate(hidden_best: np.ndarray, logit_feats: np.ndarray,
                        y: np.ndarray, groups: np.ndarray, C=0.5) -> dict:
    """Anti-circularity: hidden-state probe must BEAT a probe on the model's own
    output logits (logit_feats = e.g. top-k next-token logit features at content).
    """
    h_acc, _, _ = _cv_accuracy(hidden_best, y, groups, C=C)
    l_acc, _, _ = _cv_accuracy(logit_feats, y, groups, C=C)
    return {"hidden_acc": h_acc, "logit_acc": l_acc, "margin": h_acc - l_acc,
            "passes": h_acc > l_acc + 0.02}


def marker_control_gate(real_acc: float, marker_acts_by_layer: dict[int, np.ndarray],
                        y: np.ndarray, groups: np.ndarray, best_layer: int, C=0.5) -> dict:
    """Random-marker control: arbitrary rare tokens as pseudo-roles. If they separate
    ~as well as real role tags, the 'role geometry' is a trivial tokenization artifact.
    """
    m_acc, _, _ = _cv_accuracy(marker_acts_by_layer[best_layer], y, groups, C=C)
    return {"real_acc": real_acc, "marker_acc": m_acc, "gap": real_acc - m_acc,
            "passes": real_acc > m_acc + 0.05}


# ---------------- CLAIM 2: ordered-scale geometry ----------------

def _centroids(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack([X[y == c].mean(0) for c in range(len(ROLES))])  # (5, d)


def effective_dimensionality(cent: np.ndarray) -> float:
    """Participation ratio of centroid spread: (sum lambda)^2 / sum(lambda^2).
    ~1 => centroids lie on one line (single axis). Up to K-1 for K roles.
    """
    Xc = cent - cent.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    if lam.sum() == 0:
        return 0.0
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def pc1_variance(cent: np.ndarray) -> float:
    Xc = cent - cent.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    v = s ** 2
    return float(v[0] / v.sum())


def random_centroid_null(d: int, k: int, n: int = 2000, seed=0) -> dict:
    """Null: k random unit centroids in d dims. Distribution of participation ratio
    and PC1 variance, to judge whether the real centroids are unusually 1-D/ordered.
    """
    rng = np.random.default_rng(seed)
    prs, pc1s = [], []
    for _ in range(n):
        c = rng.standard_normal((k, d))
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        prs.append(effective_dimensionality(c))
        pc1s.append(pc1_variance(c))
    return {"pr_mean": float(np.mean(prs)), "pr_lo": float(np.percentile(prs, 2.5)),
            "pc1_mean": float(np.mean(pc1s)), "pc1_hi": float(np.percentile(pc1s, 97.5))}


def ordinal_vs_multinomial(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                           order: list[str], C=0.5, seed=0) -> dict:
    """Compare a single ORDERED-axis model against the unordered multinomial on
    held-out data. The ordered model: project activations onto ONE axis (LDA-style
    fit to the given ordinal role rank), then fit K-1 ordered thresholds; predict the
    nearest rank. If the 1-D ordered model's held-out accuracy is close to the full
    multinomial's, authority behaves as a single ordered scalar.
    Returns held-out accuracies + the DoF each model uses.
    """
    rank = np.array([order.index(ROLES[int(c)]) for c in y], dtype=float)  # 0..4 target rank
    d = X.shape[1]
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    ord_acc, mult_acc = [], []
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        # ordered axis: regress rank on activations (1 direction), then threshold
        w, *_ = np.linalg.lstsq(np.c_[Xtr, np.ones(len(Xtr))], rank[tr], rcond=None)
        proj_tr = np.c_[Xtr, np.ones(len(Xtr))] @ w
        proj_te = np.c_[Xte, np.ones(len(Xte))] @ w
        # thresholds = midpoints between sorted per-rank means (K-1 params)
        centers = np.array([proj_tr[rank[tr] == r].mean() for r in range(5)])
        thr = (centers[:-1] + centers[1:]) / 2
        pred_rank = np.searchsorted(thr, proj_te)
        # map predicted rank back to role class
        rank_to_role = {order.index(ROLES[c]): c for c in range(5)}
        pred_ord = np.array([rank_to_role[int(r)] for r in pred_rank])
        ord_acc.append(float((pred_ord == y[te]).mean()))
        # unordered multinomial
        clf = LogisticRegression(C=C, max_iter=2000).fit(Xtr, y[tr])
        mult_acc.append(float((clf.predict(Xte) == y[te]).mean()))
    return {"ordered_acc": float(np.mean(ord_acc)), "multinomial_acc": float(np.mean(mult_acc)),
            "gap": float(np.mean(mult_acc) - np.mean(ord_acc)),
            "ordered_dof": d + 1 + 4, "multinomial_dof": 5 * (d + 1),
            "ordered_is_sufficient": float(np.mean(mult_acc) - np.mean(ord_acc)) < 0.05}


def recovered_order(cent: np.ndarray) -> list[str]:
    """Order roles by their projection on PC1 of the centroid cloud."""
    Xc = cent - cent.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[0]
    return [ROLES[i] for i in np.argsort(proj)]


def geometry_battery(X_best: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     order: list[str] = HYP_ORDER) -> dict:
    cent = _centroids(X_best, y)
    d = X_best.shape[1]
    null = random_centroid_null(d, 5)
    pr = effective_dimensionality(cent)
    pc1 = pc1_variance(cent)
    ordm = ordinal_vs_multinomial(X_best, y, groups, order)
    rec = recovered_order(cent)
    return {
        "participation_ratio": pr, "pr_null_mean": null["pr_mean"], "pr_null_lo": null["pr_lo"],
        "pr_below_null": pr < null["pr_lo"],
        "pc1_variance": pc1, "pc1_null_hi": null["pc1_hi"], "pc1_above_null": pc1 > null["pc1_hi"],
        "ordinal": ordm,
        "recovered_order_pc1": rec, "hypothesised_order": order,
        "order_matches": rec == order or rec == order[::-1],
    }
