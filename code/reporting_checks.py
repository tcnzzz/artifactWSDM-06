"""Reporting checks used by the manuscript and by the standalone verifier.

Everything here is computed from saved records only:
  results/checked.jsonl        one parsed output per line
  results/ptrue/*/*/*.jsonl    teacher-forced p(true)/p(false) of the claim
  results/feedback_claims.csv  first/final rounds of the retry and memory loops

The module has no side effects; code/make_assets.py turns the returned
dictionary into macros and tables, and code/verify_reporting_checks.py
recomputes it and asserts the published headline values.
"""
from collections import Counter, defaultdict
import math

import numpy as np

MODELS = ["qwen2.5:7b", "llama3.1:8b", "gemma2:9b"]
CELLS = ["base", "rho070", "rho080", "rho095", "k05", "k20", "G3", "G5",
         "order_random", "order_reverse", "paraphrase", "minbudget", "reason", "history"]


def load_probs(root):
    """(model, dataset, cell, sid) -> {'p_true': ..., 'p_false': ...}."""
    import glob
    import json
    import os
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "results", "ptrue", "*", "*", "*.jsonl"))):
        model = os.path.basename(os.path.dirname(os.path.dirname(p))).replace("_", ":")
        dataset = os.path.basename(os.path.dirname(p))
        cell = os.path.basename(p)[:-6]
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            out[(model, dataset, cell, r["sid"])] = r
    return out


def _pair_stats(recs, scores):
    """Mann-Whitney numerator/denominator with ties handled by grouping."""
    y = np.array([not bool(r["violated"]) for r in recs], dtype=float)
    s = np.array(scores, dtype=float)
    if not len(y) or not y.any() or y.all():
        return 0.0, 0.0
    order = np.argsort(s, kind="mergesort")
    s, y = s[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    pos = np.add.reduceat(y, starts)
    neg = np.add.reduceat(1.0 - y, starts)
    return float(np.sum(pos * (np.cumsum(neg) - neg / 2.0))), float(pos.sum() * neg.sum())


def _auc(recs, scores):
    u, n = _pair_stats(recs, scores)
    return u / n if n else None


def _score(probs, r):
    p = probs.get((r["model"], r["dataset"], r["cell"], r["sid"]))
    if p is None:
        return None
    total = p["p_true"] + p["p_false"]
    return p["p_true"] / total if total > 0 else None


def _pooled(recs, probs):
    groups = defaultdict(list)
    for r in recs:
        groups[(r["dataset"], r["cell"])].append(r)
    u = n = 0.0
    for rr in groups.values():
        ss = [_score(probs, r) for r in rr]
        if any(s is None for s in ss):
            continue
        uu, nn = _pair_stats(rr, ss)
        u += uu
        n += nn
    return u / n if n else None


def metrics(recs):
    """Counts and rates of reporting errors on one set of records."""
    assert all(type(r["claim"]) is bool for r in recs)
    v = sum(bool(r["violated"]) for r in recs)
    c = len(recs) - v
    claims = sum(bool(r["claim"]) for r in recs)
    caught = sum(bool(r["violated"]) and not bool(r["claim"]) for r in recs)
    alarm = sum((not bool(r["violated"])) and not bool(r["claim"]) for r in recs)
    false = sum(bool(r["violated"]) and bool(r["claim"]) for r in recs)
    return dict(n=len(recs), violations=v, compliant=c, caught=caught,
                false_alarms=alarm, claims_compliant=claims, false_assurances=false,
                detection=caught / v if v else None,
                false_alarm_rate=alarm / c if c else None,
                false_assurance_rate=false / claims if claims else None)


def rounded_violation(r):
    """Verdict when the model sees the three-decimal weights from the prompt."""
    gs = r["rank_groups"]
    w = [round(1 / math.log2(i + 2), 3) for i in range(len(gs))]
    fl = {g: math.floor(r["rho"] * gs.count(g) / len(gs) * 1000 + 1e-9) / 1000
          for g in set(gs)}
    return any(sum(v for g0, v in zip(gs, w) if g0 == g) / sum(w) < f - 1e-9
               for g, f in fl.items())


def _scope(name, recs):
    changed = [r for r in recs if bool(r["violated"]) != rounded_violation(r)]
    rounded = [dict(r, violated=rounded_violation(r)) for r in recs]
    stable = [r for r in recs if r not in changed]
    return dict(scope=name, exact=metrics(recs), displayed=metrics(rounded),
                stable=metrics(stable), changed=len(changed),
                unique_slates=len({(r["dataset"], r["sid"]) for r in recs}),
                slates_by_dataset={ds: len({r["sid"] for r in recs if r["dataset"] == ds})
                                   for ds in ("ml1m", "mind")})


def compute(rows, probs, feedback_rows):
    """Return the same structure as the former revision-data/reporting-checks/checks.json."""
    key = lambda r: (r["model"], r["dataset"], r["cell"], r["sid"])
    by = {key(r): r for r in rows}
    assert len(by) == len(rows)

    coverage = []
    for model in MODELS + ["qwen2.5:14b", "llama3.3:70b"]:
        rr = [r for r in rows if r["model"] == model and r["valid"]
              and r["dataset"] in ("ml1m", "mind") and r["cell"] in CELLS]
        counts = Counter()
        included, both = [], []
        for r in rr:
            p = probs.get(key(r))
            if p is None:
                counts["missing_record"] += 1
                continue
            pt, pf = p["p_true"], p["p_false"]
            cat = "both" if pt > 0 and pf > 0 else "true_only" if pt > 0 else "false_only" if pf > 0 else "neither"
            counts[cat] += 1
            if pt + pf > 0:
                included.append(r)
            if cat == "both":
                both.append(r)
        coverage.append(dict(
            model=model, eligible=len(rr),
            **{k: counts[k] for k in ("both", "true_only", "false_only", "neither", "missing_record")},
            included=len(included), excluded=len(rr) - len(included),
            auroc=_pooled(included, probs), both_auroc=_pooled(both, probs)))

    audit = []
    for model in MODELS + ["qwen2.5:14b", "llama3.3:70b"]:
        for ds in ("ml1m", "mind"):
            rr = [r for r in rows if r["model"] == model and r["dataset"] == ds
                  and r["cell"] == "audit" and r["valid"] and r["feasible"]]
            if not rr:
                continue
            base = [by[model, ds, "base", r["sid"]] for r in rr]
            for x, y in zip(rr, base):
                assert y["valid"] and x["rank_groups"] == y["rank_groups"]
                assert x["rank_base"] == y["rank_base"] and x["violated"] == y["violated"]
            matched = [(x, y) for x, y in zip(rr, base)
                       if _score(probs, x) is not None and _score(probs, y) is not None]
            xx = [x for x, _ in matched]
            yy = [y for _, y in matched]
            audit.append(dict(model=model, dataset=ds, base=metrics(base), audit=metrics(rr),
                              probability_pairs=len(matched),
                              base_auroc=_auc(yy, [_score(probs, y) for y in yy]),
                              audit_auroc=_auc(xx, [_score(probs, x) for x in xx])))

    scopes = {
        "core_standard_feasible": [
            r for r in rows
            if r["model"] in MODELS and r["dataset"] in ("ml1m", "mind")
            and r["valid"] and r["feasible"] and r["prompt"] == "explicit"
            and not str(r["cell"]).startswith(("ablation", "count"))],
        "qwen3_default": [
            r for r in rows
            if r["model"] == "qwen3:8b" and r["dataset"] == "ml1m"
            and r["cell"] == "base" and r["valid"]],
    }
    rounding = [_scope(name, rr) for name, rr in scopes.items()]

    feedback = []
    for r in feedback_rows:
        if not r["cell"].startswith("feedback_retry"):
            continue
        n = int(r["n_valid"])
        v = int(r["n_violated"])
        c = int(r["n_satisfied"])
        false = round(float(r["claim_true_given_violated"]) * v)
        caught = v - false
        nf = int(r["n_claim_false"])
        alarm = nf - caught
        claims = n - nf
        feedback.append(dict(
            model=r["model"], cell=r["cell"], attempted=int(r["n"]),
            invalid=int(r["n_invalid"]), valid=n, violations=v, caught=caught,
            compliant=c, false_alarms=alarm, claims_compliant=claims,
            false_assurances=false,
            detection=caught / v if v else None,
            false_alarm_rate=alarm / c if c else None,
            false_assurance_rate=false / claims if claims else None,
            boolean_auroc=float(r["auroc_claim"]) if r["auroc_claim"] else None))

    infeasible = [r for r in rows if r["valid"] and not r["feasible"]
                  and r["cell"] == "G5" and r["model"] in MODELS]
    return dict(probability_coverage=coverage, matched_audits=audit,
                rounding=rounding, feedback=feedback,
                infeasible_core_G5={"n": len(infeasible),
                                    "claims_compliant": sum(bool(r["claim"]) for r in infeasible)})
