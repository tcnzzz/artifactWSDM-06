"""Core numerics: exposure, the constraint check, exact repair, FA*IR, groups."""
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

GROUP_NAMES = "ABDEF"   # no "C": candidate ids are C01, C02, ... (an earlier run with a group "C" confused the models)

# ---------------------------------------------------------------- provider groups
ML_BINS = {2: [1990], 3: [1980, 1990], 5: [1980, 1990, 1994, 1997]}   # release-year cut points
MIND_INFO = {"news", "finance", "sports", "weather", "autos", "health"}
MIND_BINS3 = [{"news", "finance", "sports", "weather"}, {"autos", "health", "lifestyle", "travel", "foodanddrink"}]


def group_of(item, dataset, G):
    """Provider-group proxy from real metadata (release period / news category)."""
    if dataset == "ml1m":
        return int(np.searchsorted(ML_BINS[G], item["year"], side="right"))
    if G == 2:
        return 0 if item["category"] in MIND_INFO else 1
    if G == 3:
        return 0 if item["category"] in MIND_BINS3[0] else (1 if item["category"] in MIND_BINS3[1] else 2)
    raise ValueError(G)


# ---------------------------------------------------------------------- exposure
def weights(k, gamma=None):
    """Position weights. gamma=None: logarithmic 1/log2(j+1); else geometric gamma^(j-1)."""
    j = np.arange(1, k + 1)
    return 1.0 / np.log2(j + 1) if gamma is None else gamma ** (j - 1.0)


def floors(groups, rho):
    """Per-group exposure-share floors rho * n_g / k, truncated to 3 decimals."""
    k = len(groups)
    return {g: math.floor(rho * groups.count(g) / k * 1000 + 1e-9) / 1000 for g in sorted(set(groups))}


def shares(groups_in_rank_order, w=None):
    k = len(groups_in_rank_order)
    w = weights(k) if w is None else w
    out = {}
    for g, wj in zip(groups_in_rank_order, w):
        out[g] = out.get(g, 0.0) + wj
    tot = w.sum()
    return {g: v / tot for g, v in out.items()}


def margin(groups_in_rank_order, fl, w=None):
    """min_g (share_g - floor_g); the slate satisfies the constraint iff margin >= 0."""
    sh = shares(groups_in_rank_order, w)
    return min(sh.get(g, 0.0) - f for g, f in fl.items())


def satisfied(groups_in_rank_order, fl, w=None):
    return margin(groups_in_rank_order, fl, w) >= -1e-9


# ---------------------------------------------------------------- exact repair
def repair(scores, groups, fl, weight_list=None):
    """Exposure-constrained assignment (integer LP, Singh & Joachims-style constraint):
    maximise sum_j w_j * score(item at j) subject to share_g >= floor_g for every
    weight vector in weight_list (several vectors = robust repair).
    Returns a list of item indices in rank order, or None if infeasible."""
    k = len(scores)
    w0 = weights(k)
    weight_list = [w0] if weight_list is None else weight_list
    c = -(np.outer(np.asarray(scores, float), w0)).ravel()          # x[i, j] flattened
    A, lb = [], []
    for i in range(k):                                               # each item once
        r = np.zeros((k, k)); r[i, :] = 1; A.append(r.ravel()); lb.append(1)
    for j in range(k):                                               # each position once
        r = np.zeros((k, k)); r[:, j] = 1; A.append(r.ravel()); lb.append(1)
    n_eq = len(A)
    for w in weight_list:
        for g, f in fl.items():
            r = np.zeros((k, k))
            for i in range(k):
                if groups[i] == g:
                    r[i, :] = w
            A.append(r.ravel()); lb.append(f * w.sum() - 1e-9)
    A = np.array(A); lb = np.array(lb); ub = np.full(len(lb), np.inf); ub[:n_eq] = 1
    res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=np.ones(k * k), bounds=Bounds(0, 1))
    if not res.success:
        return None
    x = res.x.reshape(k, k).round()
    return [int(np.argmax(x[:, j])) for j in range(k)]


def maxmin_slack(groups, rho, w=None):
    """Largest t such that some permutation of this slate has share_g - floor_g >= t
    for every group g.  Objective-free: it depends only on the multiset of groups,
    k, rho and the position weights, not on the relevance scores or the model output.
    t* < 0 means that no ranking can meet the floor; t* >= 0 is the best guaranteed
    clearance.  We binary-search t and solve a fixed-t integer feasibility problem
    with HiGHS (scipy.optimize.milp); the direct max-t formulation occasionally
    triggers a HiGHS solve error, while the fixed-t model is stable.
    """
    k = len(groups)
    w = weights(k) if w is None else np.asarray(w, float)
    W = w.sum()
    gs = sorted(set(groups))
    ng = {g: groups.count(g) for g in gs}
    fl = floors(groups, rho)
    G, nx = len(gs), len(gs) * k
    A, lo, hi = [], [], []
    for j in range(k):                                               # one group per position
        row = np.zeros(nx)
        for gi in range(G):
            row[gi * k + j] = 1
        A.append(row); lo.append(1); hi.append(1)
    for gi, g in enumerate(gs):                                      # each group's count
        row = np.zeros(nx)
        for j in range(k):
            row[gi * k + j] = 1
        A.append(row); lo.append(ng[g]); hi.append(ng[g])
    A = np.array(A); lo = np.array(lo); hi = np.array(hi)
    integ = np.ones(nx)

    def feasible(t):
        # add share_g - floor_g >= t for every group
        rows = []
        for gi, g in enumerate(gs):
            row = np.zeros(nx)
            for j in range(k):
                row[gi * k + j] = -w[j] / W
            rows.append(row)
        A_t = np.vstack([A, np.array(rows)])
        lo_t = np.concatenate([lo, np.full(G, -np.inf)])
        hi_t = np.concatenate([hi, -np.array([fl[g] for g in gs]) - t])
        res = milp(np.zeros(nx), constraints=LinearConstraint(A_t, lo_t, hi_t),
                   integrality=integ, bounds=Bounds(0, 1))
        if res.success:
            return True
        if res.status == 2:                                          # infeasible
            return False
        raise RuntimeError("maxmin_slack: HiGHS failed on groups=%r rho=%r t=%r"
                           % (groups, rho, t))

    t_lo, t_hi = -0.1, 0.3
    while not feasible(t_lo):
        t_lo -= 0.2
    while feasible(t_hi):
        t_hi += 0.2
    for _ in range(40):                                              # < 1e-10 bracket
        mid = (t_lo + t_hi) / 2.0
        if feasible(mid):
            t_lo = mid
        else:
            t_hi = mid
    return float((t_lo + t_hi) / 2.0)


# ------------------------------------------------------------------------ FA*IR
def fair_rerank(scores, groups, protected, p, alpha=0.1):
    """FA*IR (Zehlike et al. 2017) through the authors' released fairsearchcore package."""
    import fairsearchcore as fsc
    from fairsearchcore import fail_prob
    from fairsearchcore.models import FairScoreDoc
    key = fail_prob.LegalAssignmentKey
    if not getattr(key, "_patched", False):          # the released __hash__ returns a float on py3.13
        orig = key.__hash__
        key.__hash__ = lambda self: int(orig(self))
        key._patched = True
    k = len(scores)
    p = min(max(p, 0.02), 0.98)
    fair = fsc.Fair(k, p, alpha)
    docs = [FairScoreDoc(i, float(scores[i]), groups[i] == protected) for i in range(k)]
    docs.sort(key=lambda d: -d.score)
    return [d.id for d in fair.re_rank(docs)]


# ---------------------------------------------------------------------- utility
def ndcg(labels_in_rank_order):
    lab = np.asarray(labels_in_rank_order, float)
    if lab.sum() == 0:
        return float("nan")
    w = weights(len(lab))
    return float((lab * w).sum() / (np.sort(lab)[::-1] * w).sum())


def score_utility(scores_in_rank_order):
    """DCG of the base-recommender scores relative to the relevance-sorted order."""
    s = np.asarray(scores_in_rank_order, float)
    w = weights(len(s))
    return float((s * w).sum() / (np.sort(s)[::-1] * w).sum())
