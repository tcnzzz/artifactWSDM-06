"""Cluster bootstrap for the pooled token-probability AUROC.

The point estimate is the stratified AUROC used in the paper: pairs are formed
within a dataset x cell setting, and settings are weighted by their number of
positive--negative pairs.  The interval resamples slate IDs *within each
dataset* (all variants of a slate move together), which is the unit at which the
slates were sampled.  Saved records only; no model call.
"""
import math

import numpy as np

CELLS = ["base", "rho070", "rho080", "rho095", "k05", "k20", "G3", "G5",
         "order_random", "order_reverse", "paraphrase", "minbudget", "reason", "history"]


def clustered_ci(rows, probs, models, cells=CELLS, B=2000, seed=20261115):
    """Return one dict per model: n, slates, auroc, ci_low, ci_high, bootstrap, seed."""
    out = []
    for model in models:
        pts = {}
        for (m, ds, cell, sid), p in probs.items():
            # Only the two reported datasets; the ptrue tree also holds the
            # mind_tied / mind_nanscores diagnostic variants, which the paper
            # does not pool.
            if m != model or ds not in ("ml1m", "mind") or cell not in cells:
                continue
            total = p["p_true"] + p["p_false"]
            if total > 0:
                pts[ds, cell, sid] = p["p_true"] / total
        d = [r for r in rows
             if r["model"] == model and r["valid"]
             and (r["dataset"], r["cell"], r["sid"]) in pts]
        ids = {ds: sorted({r["sid"] for r in d if r["dataset"] == ds})
               for ds in ("ml1m", "mind")}
        maps = {ds: {sid: i for i, sid in enumerate(v)} for ds, v in ids.items()}
        groups = []
        for ds in ids:
            for cell in cells:
                c = sorted([r for r in d if r["dataset"] == ds and r["cell"] == cell],
                           key=lambda r: pts[ds, cell, r["sid"]])
                if not c:
                    continue
                scores = np.array([pts[ds, cell, r["sid"]] for r in c])
                y = np.array([not bool(r["violated"]) for r in c])
                indices = np.array([maps[ds][r["sid"]] for r in c])
                starts = np.r_[0, np.flatnonzero(np.diff(scores)) + 1]
                groups.append((ds, y, indices, starts))

        def auc(weights):
            u = pairs = 0.0
            for ds, y, indices, starts in groups:
                w = weights[ds][indices]
                pos = np.add.reduceat(w * y, starts)
                neg = np.add.reduceat(w * (1 - y), starts)
                u += np.sum(pos * (np.cumsum(neg) - neg / 2))
                pairs += pos.sum() * neg.sum()
            return u / pairs if pairs else float("nan")

        point = auc({ds: np.ones(len(v)) for ds, v in ids.items()})
        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(B):
            weights = {ds: np.bincount(rng.integers(0, len(v), len(v)), minlength=len(v))
                       for ds, v in ids.items()}
            boot.append(auc(weights))
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        out.append(dict(model=model, n=len(d), slates={ds: len(v) for ds, v in ids.items()},
                        auroc=float(point), ci_low=float(lo), ci_high=float(hi),
                        bootstrap=B, seed=seed))
    return out


def _floors(gs, rho):
    return {g: math.floor(rho * gs.count(g) / len(gs) * 1000 + 1e-9) / 1000 for g in set(gs)}


def _margin(gs, fl):
    w = [1 / math.log2(j + 2) for j in range(len(gs))]
    return min(sum(x for g0, x in zip(gs, w) if g0 == g) / sum(w) - f for g, f in fl.items())


def one_sided_rates(rows):
    """One-sided-floor violations, and the same outputs against every group floor.

    The one-sided prompt states only the initially most under-exposed group's
    floor.  Re-evaluating the unchanged ranking against all floors separates a
    wording effect from a change in the requirement.
    """
    out = []
    for model in ("qwen2.5:7b", "llama3.1:8b", "gemma2:9b"):
        rr = [r for r in rows
              if r["model"] == model and r["dataset"] == "ml1m"
              and r["cell"] == "minbudget" and r["valid"]]
        out.append(dict(model=model, n=len(rr),
                        one_sided=sum(bool(r["violated"]) for r in rr),
                        all_groups=sum(_margin(r["rank_groups"], _floors(r["rank_groups"], r["rho"])) < -1e-9
                                       for r in rr)))
    return out
