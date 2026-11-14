"""Calibrated verifier for the case where the constraint cannot be checked at emission time.

SYNTHETIC FACTOR (labelled as such in the paper): every session has a hidden browsing
patience gamma drawn from a finite set GAMMAS; true exposure weights are gamma^(j-1).
The operator knows the set but neither the session's gamma nor its distribution, so the
true violation label is only available afterwards (calibration data).
Slates and claims are the real LLM outputs in results/raw/.

Data: the default cell (rho = 0.9, k = 10, two groups), one model and dataset at a time.
Policies: trust the claim, nominal check, robust check, always repair, and conformal risk control
(CRC) on the stated confidence, the token probability of the claim, the nominal margin, the
worst-case margin, and logistic scores fitted on a separate training third (margins only, and
margins plus claim, confidence and token probability). Escalated slates receive the robust exact
repair (or the nominal repair where no robust one exists). Writes results/verifier.csv and
results/verifier_shift.csv.
"""
import json
import os
import zlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fairlib import margin, ndcg, repair, weights

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GAMMAS = [0.5, 0.6, 0.7, 0.8, 0.9]
PRIORS = {"uniform": [0.2] * 5, "impatient": [0.4, 0.3, 0.15, 0.1, 0.05], "patient": [0.05, 0.1, 0.15, 0.3, 0.4]}
ALPHAS = [0.05, 0.10]
SPLITS = 500
SEED = 20261005
RHO_CELLS = ["rho050", "rho070", "rho080", "base", "rho095"]


def crc_threshold(score, y, alpha):
    """Smallest lambda such that (n * Rhat(lambda) + 1) / (n + 1) <= alpha, where a slate is
    accepted iff score > lambda and the loss is y * 1[accepted]. Returns +inf if none.
    Rhat is non-increasing in lambda, so the feasible set is an upper interval; candidates are the
    observed scores (descending) followed by -inf."""
    n = len(score)
    s_sorted = np.sort(score)[::-1]
    y_sorted = np.asarray(y, float)[np.argsort(-score, kind="stable")]
    cum = np.concatenate([[0.0], np.cumsum(y_sorted)])          # cum[m] = violations among top m
    cand = np.concatenate([np.unique(score)[::-1], [-np.inf]])
    m = np.searchsorted(-s_sorted, -cand, side="left")            # number of scores > cand
    feasible = (cum[m] + 1) / (n + 1) <= alpha
    if not feasible[0]:
        return np.inf
    last = len(feasible) if feasible.all() else int(np.argmin(feasible))
    return float(cand[last - 1])


def load(prior, dataset, model, cells=("base",)):
    """Rows for one model x dataset: the LLM slate, its hidden-patience violation label, numeric
    margins, the claim, the token probability of the claim, and the fallback used on escalation
    (robust exact repair over all patience values, or nominal exact repair if none exists)."""
    md = model.replace(":", "_")
    rows = []
    for cell in cells:
        p = os.path.join(ROOT, "results", "raw", md, dataset, f"{cell}.jsonl")
        if not os.path.exists(p):
            continue
        pt = {}
        pp = os.path.join(ROOT, "results", "ptrue", md, dataset, f"{cell}.jsonl")
        if os.path.exists(pp):
            for line in open(pp):
                x = json.loads(line)
                tot = x["p_true"] + x["p_false"]
                pt[x["sid"]] = x["p_true"] / tot if tot > 0 else np.nan
        for line in open(p):
            r = json.loads(line)
            if not r.get("valid"):
                continue
            by = {c["cid"]: c for c in r["cands"]}
            base = sorted(r["cands"], key=lambda c: c["base_rank"])
            bg, bs = [c["group"] for c in base], [c["score"] for c in base]
            fl = r["floors"]
            k = len(bg)
            nominal_rep = repair(bs, bg, fl)
            if nominal_rep is None:
                continue
            robust_rep = repair(bs, bg, fl, [weights(k, gm) for gm in GAMMAS])
            fb = robust_rep if robust_rep is not None else nominal_rep
            g = [by[c]["group"] for c in r["ranking"]]
            fg = [bg[i] for i in fb]
            rng = np.random.default_rng(zlib.crc32(f"{SEED}-{prior}-{r['sid']}-{cell}".encode()))
            gi = rng.choice(len(GAMMAS), p=PRIORS[prior])
            ms = [margin(g, fl, weights(k, gm)) for gm in GAMMAS]
            conf = r["confidence"] if isinstance(r.get("confidence"), (int, float)) else 50.0
            claim = 0.0 if r.get("claim") is False else 1.0
            labels = [c["label"] for c in base]
            rows.append({"model": model, "cell": cell, "sid": r["sid"], "rho": r["rho"],
                         "y": float(ms[gi] < -1e-9),
                         "fb_viol": float(margin(fg, fl, weights(k, GAMMAS[gi])) < -1e-9),
                         "fb_robust": robust_rep is not None,
                         "nominal": margin(g, fl), "robust": min(ms), **{f"m{j}": m for j, m in enumerate(ms)},
                         "claim": claim, "conf": conf, "claim_score": conf + 1000 * claim,
                         "ptrue": pt.get(r["sid"], np.nan),
                         "ndcg": ndcg([by[c]["label"] for c in r["ranking"]]) if sum(labels) else np.nan,
                         "fb_ndcg": ndcg([labels[i] for i in fb]) if sum(labels) else np.nan})
    return pd.DataFrame(rows)


FEATS = ["nominal", "robust", "m0", "m1", "m2", "m3", "m4"]
FEATS_CLAIM = FEATS + ["claim", "conf", "ptrue"]


def fit_score(train, feats):
    if train.y.nunique() < 2:
        return lambda d: d["robust"].values
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    clf.fit(train[feats].fillna(0.5).values, train.y.values)
    return lambda d: -clf.decision_function(d[feats].fillna(0.5).values)      # higher = safer


def outcome(te, acc, split, policy, alpha):
    y, fb = te.y.values, te.fb_viol.values
    lab = ~np.isnan(te.ndcg.values)
    served_ndcg = np.where(acc, te.ndcg.values, te.fb_ndcg.values)[lab]
    return {"split": split, "policy": policy, "alpha": alpha, "risk": float((y * acc).mean()),
            "escalated": float(1 - acc.mean()),
            "viol_among_accepted": float((y * acc).sum() / acc.sum()) if acc.sum() else np.nan,
            "served_violation": float(np.where(acc, y, fb).mean()),
            "served_ndcg": float(served_ndcg.mean()) if lab.any() else np.nan}


def evaluate(df, rng, splits=SPLITS):
    out = []
    n = len(df)
    has_pt = df.ptrue.notna().mean() > 0.9
    for sp in range(splits):
        perm = rng.permutation(n)
        tr, ca, te = (df.iloc[perm[: n // 3]], df.iloc[perm[n // 3: 2 * n // 3]], df.iloc[perm[2 * n // 3:]])
        scores = {"claim": lambda d: d["claim_score"].values, "nominal": lambda d: d["nominal"].values,
                  "robust": lambda d: d["robust"].values, "learned": fit_score(tr, FEATS)}
        if has_pt:
            scores["ptrue"] = lambda d: d["ptrue"].fillna(0.5).values
            scores["learned_claim"] = fit_score(tr, FEATS_CLAIM)
        fixed = {"trust_claim": te.claim.values > 0.5, "nominal_check": te.nominal.values >= -1e-9,
                 "robust_check": te.robust.values >= -1e-9, "always_repair": np.zeros(len(te), bool)}
        for name, acc in fixed.items():
            out.append(outcome(te, acc, sp, name, np.nan))
        for alpha in ALPHAS:
            for name, fn in scores.items():
                lam = crc_threshold(fn(ca), ca.y.values, alpha)
                out.append(outcome(te, fn(te) > lam, sp, "crc_" + name, alpha))
    return pd.DataFrame(out)


def summarise(res, **tags):
    g = res.groupby(["policy", "alpha"], dropna=False)
    s = g.agg(risk=("risk", "mean"), risk_p95=("risk", lambda x: np.percentile(x, 95)),
              escalated=("escalated", "mean"), esc_lo=("escalated", lambda x: np.percentile(x, 2.5)),
              esc_hi=("escalated", lambda x: np.percentile(x, 97.5)),
              viol_among_accepted=("viol_among_accepted", "mean"),
              served_violation=("served_violation", "mean"), served_ndcg=("served_ndcg", "mean"),
              risk_se=("risk", lambda x: x.std() / np.sqrt(len(x)))).reset_index()
    for k, v in tags.items():
        s[k] = v
    return s


MODELS = ["qwen2.5:7b", "llama3.1:8b", "gemma2:9b"]
CELLSETS = ["base"]

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    allres, shift = [], []
    for dataset in ("ml1m", "mind"):
        for model, cell in [(m, c) for m in MODELS for c in CELLSETS]:
            for prior in PRIORS:
                if prior != "uniform" and cell != "base":
                    continue
                d = load(prior, dataset, model, (cell,))
                if len(d) < 60:
                    continue
                s = summarise(evaluate(d, rng, SPLITS if prior == "uniform" else 200), dataset=dataset, prior=prior, cell=cell,
                              model=model, n=len(d), base_rate=d.y.mean(), claim_rate=d.claim.mean(),
                              robust_feasible=d.fb_robust.mean(), fb_violation=d.fb_viol.mean())
                allres.append(s)
            if cell != "base":
                continue
            # patience shift: calibrate with uniform patience, deploy to impatient users
            du, di = load("uniform", dataset, model), load("impatient", dataset, model)
            if len(du) < 60:
                continue
            r2 = np.random.default_rng(SEED + 1)
            for sp in range(SPLITS):
                perm = r2.permutation(len(du))
                ca, te = perm[: len(du) // 2], perm[len(du) // 2:]
                for alpha in ALPHAS:
                    lam = crc_threshold(du.robust.values[ca], du.y.values[ca], alpha)
                    for tgt, dt in (("uniform", du), ("impatient", di)):
                        acc = dt.robust.values[te] > lam
                        shift.append({"dataset": dataset, "model": model, "test": tgt, "alpha": alpha, "split": sp,
                                      "risk": float((dt.y.values[te] * acc).mean()), "escalated": float(1 - acc.mean())})
    V = pd.concat(allres)
    V.to_csv(os.path.join(ROOT, "results", "verifier.csv"), index=False)
    sh = pd.DataFrame(shift).groupby(["dataset", "model", "test", "alpha"]).agg(
        risk=("risk", "mean"), escalated=("escalated", "mean"), splits=("split", "nunique")).reset_index()
    sh.to_csv(os.path.join(ROOT, "results", "verifier_shift.csv"), index=False)
    pd.set_option("display.width", 250)
    print(V[(V.prior == "uniform") & ((V.alpha == 0.1) | V.alpha.isna())].round(3).to_string())
    print(sh.round(3))
