"""Turn saved results into everything the manuscript displays.

Writes manuscript/generated/numbers.tex (macros), the three reporting tables,
generated/writing_numbers.tex (clustered intervals), and the pgfplots dose
figure with its data file.  No model call, no new inference: every value comes
from results/*.csv, results/*.jsonl or results/ptrue/*.jsonl.
"""
import glob
import json
import os
import re
import sys

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reporting_checks  # noqa: E402
import writing_revision  # noqa: E402

ROOT = os.path.dirname(HERE)
# The paper source normally sits next to this directory as ../manuscript.
# MANUSCRIPT_DIR overrides it, so the checks can run against a temporary copy
# of the shipped generated/ files when the paper source is not present.
MS = os.environ.get("MANUSCRIPT_DIR") or os.path.join(os.path.dirname(ROOT), "manuscript")
GEN = os.path.join(MS, "generated")
FIG = os.path.join(MS, "figures")
FIGDATA = os.path.join(FIG, "data")
os.makedirs(GEN, exist_ok=True)
os.makedirs(FIGDATA, exist_ok=True)
MODELS = ["qwen2.5:7b", "llama3.1:8b", "gemma2:9b"]
SHORT = {"qwen2.5:7b": "qwen", "llama3.1:8b": "llama", "gemma2:9b": "gemma", "qwen2.5:14b": "qwenfourteen",
         "qwen3:8b": "qwenthree", "command-r:35b": "commandr", "llama3.3:70b": "llamaseventy",
         "qwen2.5:7b-fp16": "qwenfp"}
STRONG = ["qwen2.5:14b", "qwen3:8b", "command-r:35b", "llama3.3:70b", "qwen2.5:7b-fp16"]
MODEL_NAME = {"qwen2.5:7b": "Qwen2.5-7B", "llama3.1:8b": "Llama3.1-8B", "gemma2:9b": "Gemma2-9B",
              "qwen2.5:14b": "Qwen2.5-14B", "qwen3:8b": "Qwen3-8B, thinking", "command-r:35b": "Command-R-35B",
              "llama3.3:70b": "Llama3.3-70B", "qwen2.5:7b-fp16": "Qwen2.5-7B, FP16"}
DS_NAME = {"ml1m": "ML-1M", "mind": "MIND"}
S = pd.read_csv(os.path.join(ROOT, "results", "summary.csv"))
num = {}


def emit(key, value):
    """Register one LaTeX macro \\N{key}; value is converted to str."""
    num[key] = str(value)


# ------------------------------------------------------------------ B12 solver environment
# Constants of the pinned environment (requirements.txt and the bundled HiGHS),
# not result data; kept as macros so the paper contains no hand-typed digits.
emit("solver-version", "1.12.0")
emit("solver-k-default", "10")
emit("solver-hardware", "Apple M5 Max")


def pct(x, d=1):
    return "--" if pd.isna(x) else f"{100 * x:.{d}f}"


def pts(x, d=1):
    return "--" if pd.isna(x) else f"{100 * x:+.{d}f}".replace("-", "$-$")


def pval(p):
    if pd.isna(p):
        return "--"
    return "$<$.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def auc_s(r, key="auroc_tok", ci=True):
    if r is None or pd.isna(r[key]):
        return "--"
    return f"{r[key]:.2f} [{r[key + '_lo']:.2f}, {r[key + '_hi']:.2f}]" if ci else f"{r[key]:.2f}"


def get(model, ds, cell):
    r = S[(S.model == model) & (S.dataset == ds) & (S.cell == cell)]
    return None if r.empty else r.iloc[0]


# ------------------------------------------------------------------ B7 baseline heuristics
# frontier.csv is written by code/frontier.py; these macros are only used by the
# one-sentence baseline comparison in the results section.
FR = pd.read_csv(os.path.join(ROOT, "results", "frontier.csv"))
FR["baseline"] = FR["baseline"].fillna("")
G = FR[FR.baseline == "greedy"]
FA = FR[FR.baseline == "fair_ir"]
RR = FR[FR.baseline == "round_robin"]
# Exact-repair utility on the same slate set as the baselines (all feasible
# slates), aggregated to one mean per model x dataset x rho.  frontier.csv keeps
# the repair policy on the labelled subset used by the nDCG frontier, so using
# that row here would compare different slate sets.
_CHK = pd.read_json(os.path.join(ROOT, "results", "checked.jsonl"), lines=True)
_CELLS = ["rho050", "rho070", "rho080", "base", "rho095"]
_mask = (_CHK.model.isin(MODELS) & _CHK.dataset.isin(["ml1m", "mind"])
         & _CHK.cell.isin(_CELLS) & (_CHK.feasible == True))  # noqa: E712
RP = _CHK.loc[_mask].groupby(["model", "dataset", "cell"]).repair_util.mean().dropna()
if not G.empty:
    emit("heur-greedy-violation-max", pct(G.violation.max(), 0))
    emit("heur-greedy-util-min", f"{G.util.min():.3f}")
    emit("heur-greedy-util-max", f"{G.util.max():.3f}")
if not FA.empty:
    emit("heur-fair-max", pct(FA.violation.max(), 0))
    emit("heur-fair-util-min", f"{FA.util.min():.3f}")
if not RR.empty:
    emit("heur-rr-max", pct(RR.violation.max(), 0))
    emit("heur-rr-util-min", f"{RR.util.min():.3f}")
if not RP.empty:
    emit("heur-repair-util-min", f"{RP.min():.3f}")
    emit("heur-repair-util-max", f"{RP.max():.3f}")

# ------------------------------------------------------------------ B8 calibration
CAL = pd.read_csv(os.path.join(ROOT, "results", "calibration.csv"))
CP = CAL[(CAL.dataset == "pooled") & CAL.ece.notna()]
if not CP.empty:
    emit("cal-ece-min", f"{CP.ece.min():.2f}")
    emit("cal-ece-max", f"{CP.ece.max():.2f}")
    emit("cal-brier-min", f"{CP.brier.min():.2f}")
    emit("cal-brier-max", f"{CP.brier.max():.2f}")


# ------------------------------------------------------------------ per-cell macros
PCT_STATS = ("violation", "binding", "random_violation", "claim_true", "claim_true_given_violated",
             "claim_true_given_satisfied", "violation_binding", "violation_slack", "invalid", "unchanged",
             "fair_violation", "fair_changed", "claim_true_given_infeasible", "blocks", "ptrue_mean")
for _, r in S.iterrows():
    key = f"{SHORT[r.model]}-{r.dataset}-{r.cell}".replace("_", "")
    for stat in PCT_STATS:
        if stat in r and not pd.isna(r[stat]):
            num[f"{key}-{stat.replace('_', '')}"] = pct(r[stat])
            if stat + "_lo" in r and not pd.isna(r[stat + "_lo"]):
                num[f"{key}-{stat.replace('_', '')}-ci"] = f"[{pct(r[stat + '_lo'])}, {pct(r[stat + '_hi'])}]"
    for stat in ("util", "repair_util", "proj_util"):
        if stat in r and not pd.isna(r[stat]):
            num[f"{key}-{stat.replace('_', '')}pct"] = pct(r[stat])
    for stat in ("ndcg", "base_ndcg", "repair_ndcg", "proj_ndcg"):
        if stat in r and not pd.isna(r[stat]):
            num[f"{key}-{stat.replace('_', '')}"] = f"{r[stat]:.3f}"
    for stat in ("ndcg_minus_repair", "ndcg_minus_proj", "proj_minus_repair", "ndcg_minus_basendcg"):
        if stat in r and not pd.isna(r[stat]):
            num[f"{key}-{stat.replace('_', '')}"] = pts(r[stat])
            num[f"{key}-{stat.replace('_', '')}-ci"] = f"[{pts(r[stat + '_lo'])}, {pts(r[stat + '_hi'])}]"
            num[f"{key}-{stat.replace('_', '')}-neg"] = pts(-r[stat])
            num[f"{key}-{stat.replace('_', '')}-neg-ci"] = f"[{pts(-r[stat + '_hi'])}, {pts(-r[stat + '_lo'])}]"
    for stat in ("auroc", "auroc_tok"):
        if not pd.isna(r[stat]):
            num[f"{key}-{stat.replace('_', '')}"] = f"{r[stat]:.2f}"
            num[f"{key}-{stat.replace('_', '')}-ci"] = f"[{r[stat + '_lo']:.2f}, {r[stat + '_hi']:.2f}]"
    for c in ("n", "n_valid", "n_violated", "n_satisfied", "n_claim_false", "n_infeasible", "n_feasible",
              "n_invalid", "n_broken", "n_fixed", "conf_distinct", "n_with_label"):
        if c in r and not pd.isna(r[c]):
            num[f"{key}-{c.replace('_', '')}"] = str(int(r[c]))
    num[f"{key}-mcnemarp"] = pval(r.mcnemar_p)
    if not pd.isna(r.mcnemar_p):
        emit(f"{key}-mcnemarprel", "$<$ .001" if r.mcnemar_p < 0.001 else "$=$ " + pval(r.mcnemar_p))
    if not pd.isna(r.get("repair_ms_median", np.nan)):
        num[f"{key}-repairms"] = f"{r.repair_ms_median:.0f}"

# ------------------------------------------------------------------ projection evidence
PJ = S[(S.cell == "base") & S.dataset.isin(["ml1m", "mind"])
       & S.model.isin(["qwen2.5:7b", "llama3.1:8b", "gemma2:9b"])].dropna(
           subset=["proj_util", "repair_util", "proj_minus_repair"])
emit("proj-retain-min", pct((PJ.proj_util / PJ.repair_util).min()))
emit("proj-retain-max", pct((PJ.proj_util / PJ.repair_util).max()))
emit("proj-repair-gap-min", pts(PJ.proj_minus_repair.min()))
emit("proj-repair-gap-max", pts(PJ.proj_minus_repair.max()))

# ------------------------------------------------------------------ pooled statistics
MAIN = S[S.dataset.isin(["ml1m", "mind"]) & ~S.cell.str.startswith("ablation")]
E = MAIN[(MAIN.prompt == "explicit") & MAIN.model.isin(MODELS)]
num["explicit-outputs"] = f"{int(E.n_valid.sum()):,}"
num["explicit-violated"] = f"{int(E.n_violated.sum()):,}"
num["explicit-claimfalse"] = f"{int(E.n_claim_false.sum()):,}"
num["explicit-cells"] = str(len(E))
V2 = MAIN[MAIN.prompt.isin(["paraphrase", "minbudget", "history", "reason"]) & MAIN.model.isin(MODELS)]
num["variants-outputs"] = f"{int(V2.n_valid.sum()):,}"
num["variants-claimfalse"] = str(int(V2.n_claim_false.sum()))
num["all-outputs"] = f"{int(MAIN.n.sum()):,}"
num["invalid-max"] = pct(MAIN.invalid.max())

# ---- invalid-output rates (section 2, "at most X% in a cell"): invalid-max above includes Qwen3 G5 (n=13)
SMALL_MAIN = MAIN[MAIN.model.isin(MODELS)]            # three small models, reported cells (no ablations)
emit("invalid-max-small", pct(SMALL_MAIN.invalid.max()))
_q3 = get("qwen3:8b", "ml1m", "base")
emit("invalid-max-qwenthree-default", pct(_q3.invalid) if _q3 is not None else "--")

# ---- stronger models, default cell on ML-1M (abstract / intro: "14 to 80% against about 73%, claims 98 to 100%")
STR = [m for m in STRONG if m != "qwen2.5:7b-fp16" and get(m, "ml1m", "base") is not None]
if STR:
    SB = pd.DataFrame([get(m, "ml1m", "base") for m in STR])
    emit("strong-nmodels", len(STR))
    emit("strong-violation-min", pct(SB.violation.min(), 0))
    emit("strong-violation-max", pct(SB.violation.max(), 0))
    emit("strong-binding-mean", pct(SB.binding.mean(), 0))
    emit("strong-claim-min", pct(SB.claim_true.min(), 0))
    emit("strong-claim-max", pct(SB.claim_true.max(), 0))
    for m in STR:                                      # invalid outputs counted as failures
        r_ = get(m, "ml1m", "base")
        emit(f"{SHORT[m]}-ml1m-base-failinvalid", pct((r_.n_violated + r_.n_invalid) / r_.n_feasible))

# ---- Holm correction over the McNemar tests of Table 1 (default cells: 3 models x 2 datasets + strong models)
def holm_adjust(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def bh_adjust(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    q, running = [None] * len(ps), 1.0
    for rank in range(len(ps) - 1, -1, -1):
        i = order[rank]
        running = min(running, ps[i] * len(ps) / (rank + 1))
        q[i] = running
    return q


_fam = [(m, ds) for ds in ("ml1m", "mind") for m in MODELS] + [(m, "ml1m") for m in STR]
_fam = [(m, ds, get(m, ds, "base")) for m, ds in _fam]
_fam = [(m, ds, r_) for m, ds, r_ in _fam if r_ is not None and not pd.isna(r_.mcnemar_p)]
HOLM10, BH10, HOLMDS = {}, {}, {}
if _fam:
    _ps = [r_.mcnemar_p for _, _, r_ in _fam]
    _adj = holm_adjust(_ps)
    _bh = bh_adjust(_ps)
    for (m, ds, r_), pa, q in zip(_fam, _adj, _bh):
        emit(f"{SHORT[m]}-{ds}-base-mcnemarpholm", pval(pa))
        emit(f"{SHORT[m]}-{ds}-base-mcnemarbh", pval(q))
        HOLM10[(m, ds)], BH10[(m, ds)] = pa, q
    emit("holm-family-size", len(_fam))
    emit("holm-nsig-raw", sum(p < 0.05 for p in _ps))
    emit("holm-nsig-adj", sum(pa < 0.05 for pa in _adj))
    emit("holm-bh-nsig", sum(q < 0.05 for q in _bh))
    for ds in ("ml1m", "mind"):
        _sub = [(m, r_) for m, d, r_ in _fam if d == ds]
        if not _sub:
            continue
        _a = holm_adjust([r_.mcnemar_p for _, r_ in _sub])
        emit(f"holm-{ds}-family-size", len(_sub))
        emit(f"holm-{ds}-nsig", sum(pa < 0.05 for pa in _a))
        for (m, r_), pa in zip(_sub, _a):
            HOLMDS[(m, ds)] = pa
            emit(f"{SHORT[m]}-{ds}-base-mcnemarpholmdataset", pval(pa))
    _rows = []
    for m, ds, r_ in _fam:
        _rows.append({
            "family": "table1_10", "model": m, "dataset": ds, "cell": "base",
            "n_broken": int(r_.n_broken), "n_fixed": int(r_.n_fixed),
            "p_raw": r_.mcnemar_p, "p_holm_10": HOLM10[(m, ds)],
            "q_bh_10": BH10[(m, ds)], "p_holm_dataset": HOLMDS.get((m, ds)),
            "dataset_family_size": sum(1 for _, d, _ in _fam if d == ds),
        })
    pd.DataFrame(_rows).to_csv(os.path.join(ROOT, "results", "multiplicity.csv"), index=False)
# ---- max-min slack strata (B2): t* is the best clearance any ranking can guarantee
SL = pd.read_csv(os.path.join(ROOT, "results", "slack_strata.csv"))
emit("slack-delta", f'{SL.delta.iloc[0]:.2f}')


def _sl(scope, model, dataset, stratum, col):
    r = SL[(SL.scope == scope) & (SL.model == model) & (SL.dataset == dataset)
           & (SL.stratum == stratum)]
    return None if r.empty else r.iloc[0][col]


for _s in ("infeasible", "near", "wide"):
    for _col in ("n", "violation", "claim"):
        _v = _sl("exposure", "small", "pooled", _s, _col)
        if _v is not None and not pd.isna(_v):
            emit(f"slack-{_s}-{_col}", f"{int(_v):,}" if _col == "n" else f"{_v:.1f}")
    _v = _sl("exposure", "all", "pooled", _s, "n")
    if _v is not None and not pd.isna(_v):
        emit(f"slack-{_s}-n-all", f"{int(_v):,}")
for _m, _sh in SHORT.items():
    for _s in ("infeasible", "near", "wide"):
        for _col in ("n", "violation", "claim"):
            _v = _sl("exposure", _m, "pooled", _s, _col)
            if _v is not None and not pd.isna(_v):
                emit(f"slack-{_sh}-{_s}-{_col}", f"{int(_v):,}" if _col == "n" else f"{_v:.1f}")

# ---- counts quoted in prose (small-model pairs, Qwen3 reports of violation)
_small_fam = [(m, ds, get(m, ds, "base")) for ds in ("ml1m", "mind") for m in MODELS]
_small_fam = [(m, ds, r_) for m, ds, r_ in _small_fam if r_ is not None and not pd.isna(r_.mcnemar_p)]
emit("small-pairs", len(_small_fam))
emit("small-sig-pairs", sum(HOLM10.get((m, ds), r_.mcnemar_p) < 0.05 for m, ds, r_ in _small_fam))
if _q3 is not None:
    emit("qwenthree-ml1m-base-nclaimviolated", int(_q3.n_claim_false))
B = MAIN[(MAIN.cell == "base") & MAIN.model.isin(MODELS)]
num["base-auroctok-min"] = f"{B.auroc_tok.min():.2f}"
num["base-auroctok-max"] = f"{B.auroc_tok.max():.2f}"
num["base-auroc-min"] = f"{B.auroc.min():.2f}"
num["base-auroc-max"] = f"{B.auroc.max():.2f}"
num["base-claim-min"] = pct(B.claim_true.min())
num["base-confdistinct-min"] = str(int(B.conf_distinct.min()))
num["base-confdistinct-max"] = str(int(B.conf_distinct.max()))
num["explicit-auroctok-min"] = f"{E.auroc_tok.min():.2f}"
num["explicit-auroctok-max"] = f"{E.auroc_tok.max():.2f}"
Q = json.load(open(os.path.join(ROOT, "results", "recommender_quality.json")))
num["ml1m-knn-ndcg"] = f"{Q['ml1m']['ndcg10_itemknn']:.3f}"
num["ml1m-pop-ndcg"] = f"{Q['ml1m']['ndcg10_popularity']:.3f}"
num["mind-blend-ndcg"] = f"{Q['mind_v2']['ndcg10_blend']:.3f}"
num["mind-pop-ndcg"] = f"{Q['mind']['ndcg10_popularity']:.3f}"
num["mind-rand-ndcg"] = f"{Q['mind']['ndcg10_random']:.3f}"
num["mind-beta"] = f"{Q['mind_v2']['beta']:g}"
num["mind-distinct"] = f"{Q['mind_v2']['distinct_scores_top10_mean']:.1f}"
_tied = json.load(open(os.path.join(ROOT, "data", "slates_mind_tied.json")))
_dist = [len({round(it["score"], 2) for it in s_["items"][:10]}) for s_ in _tied]
num["mindtied-distinct"] = f"{np.mean(_dist):.1f}"
num["mindtied-fullytied"] = pct(np.mean(np.array(_dist) == 1))

# ------------------------------------------------------------------ Table 1: default cell (reporting columns)
def _paired_input(model, ds, cell="base"):
    """Input-order violation rate on the same valid feasible outputs as the table."""
    rr = _CHK[(_CHK.model == model) & (_CHK.dataset == ds) & (_CHK.cell == cell)
              & _CHK.valid & _CHK.feasible]
    return 100 * rr.binding.mean() if len(rr) else float("nan")


def _detected(model, ds, cell="base"):
    """Reported violations / actual violations on the valid feasible outputs."""
    rr = _CHK[(_CHK.model == model) & (_CHK.dataset == ds) & (_CHK.cell == cell)
              & _CHK.valid & _CHK.feasible]
    return f"{int((rr.violated.astype(bool) & (rr.claim == False)).sum())}/{int(rr.violated.astype(bool).sum())}"


def _auc_ci(r):
    a = auc_s(r)
    return a if a == "--" else a.replace(" [", r" \ci{[") + "}"


lines = []
for ds in ("ml1m", "mind", "strong"):
    first = True
    for m in (STRONG if ds == "strong" else MODELS):
        cell_ds = "ml1m" if ds == "strong" else ds
        r = get(m, cell_ds, "base")
        if r is None:
            continue
        lines.append(" & ".join([
            (DS_NAME["ml1m"] if ds == "strong" else DS_NAME[ds]) if first else "",
            MODEL_NAME[m],
            f"{int(r.n_valid)}/{int(r.n)}",
            f"{_paired_input(m, cell_ds):.1f}",
            f"{pct(r.violation)} \\ci{{[{pct(r.violation_lo)}, {pct(r.violation_hi)}]}}",
            pval(HOLM10.get((m, cell_ds), r.mcnemar_p)),
            pct(r.claim_true),
            _detected(m, cell_ds),
            _auc_ci(r),
        ]) + r" \\")
        first = False
    if ds in ("ml1m", "mind"):
        lines.append(r"\midrule")
open(os.path.join(GEN, "tab_base_reporting.tex"), "w").write(
    re.sub(r"\s*\\\\\s*$", "", "\n".join(lines)) + "\n")
# ------------------------------------------------------------------ Table 2: robustness (ML-1M)
ROWS = [("default", "base"), ("paraphrase", "paraphrase"), ("one-sided budget", "minbudget"),
        ("reasoning first", "reason"), ("user history", "history"), ("``be fair''", "befair"),
        ("audit (2nd call)", "audit"), None,
        ("count, top 5", "count"), ("count, matched", "countm"), None,
        ("$k=5$", "k05"), ("$k=20$", "k20"), ("$G=3$", "G3"), ("$G=5$", "G5"),
        ("random order", "order_random"), ("reversed order", "order_reverse")]
# Table 2 is its own family: Holm across every non-missing McNemar test it shows.
_t2 = []
for _row in ROWS:
    if _row is None:
        continue
    _label, _cell = _row
    for _m in MODELS:
        _x = get(_m, "ml1m", _cell)
        if _x is not None and not pd.isna(_x.mcnemar_p):
            _t2.append((_m, _cell, _x))
_t2_adj = holm_adjust([x.mcnemar_p for _, _, x in _t2])
T2 = {(m, cell): pa for (m, cell, _), pa in zip(_t2, _t2_adj)}
emit("tab2-family-size", len(_t2))
emit("tab2-nsig-adj", sum(pa < 0.05 for pa in _t2_adj))
_lab = {cell: label for label, cell in (r for r in ROWS if r)}
pd.DataFrame([{
    "family": "table2_45", "model": m, "dataset": "ml1m", "cell": cell,
    "label": _lab.get(cell, cell), "n_broken": int(x.n_broken), "n_fixed": int(x.n_fixed),
    "p_raw": x.mcnemar_p, "p_holm_table2": T2[(m, cell)],
    "significant_holm": bool(T2[(m, cell)] < 0.05),
    "sign": "" if T2[(m, cell)] >= 0.05 else ("+" if x.n_broken > x.n_fixed else "-"),
    "family_size": len(_t2),
} for m, cell, x in _t2]).to_csv(os.path.join(ROOT, "results", "multiplicity_table2.csv"), index=False)
FACTOR_ROWS = [
    ("default", "base"),
    ("paraphrase", "paraphrase"),
    ("reasoning first", "reason"),
    (None, None),                       # modified requirement
    ("one-sided floor", "minbudget"),
    ("count, top 5", "count"),
    ("count, matched", "countm"),
]
lines = []
for label, cell in FACTOR_ROWS:
    if cell is None:
        lines.append(r"\midrule\multicolumn{10}{@{}l}{\emph{Modified requirement}}\\")
        continue
    rs = [get(m, "ml1m", cell) for m in MODELS]
    if all(x is None for x in rs):
        continue
    cells = [label]
    for m, x in zip(MODELS, rs):
        if x is None:
            cells += ["--", "--", "--"]
            continue
        _p = T2.get((m, cell), x.mcnemar_p)
        sig = "" if pd.isna(_p) or _p >= 0.05 else (r"\worse" if x.n_broken > x.n_fixed else r"\better")
        claim = pct(x.claim_true, 0)
        cells += [pct(x.violation, 0) + sig,
                  rf"\ci{{{claim}}}" if claim == "100" else claim,
                  auc_s(x, ci=False)]
    lines.append(" & ".join(cells) + r" \\")
open(os.path.join(GEN, "tab_factors_reporting.tex"), "w").write(
    re.sub(r"\s*\\\\\s*$", "", "\n".join(lines)) + "\n")
# ------------------------------------------------------------------ feedback loops
CHK = pd.read_json(os.path.join(ROOT, "results", "checked.jsonl"), lines=True)
fb_rows = {}
for p in sorted(glob.glob(os.path.join(ROOT, "results", "raw", "*", "*", "feedback_*.jsonl"))):
    recs = [json.loads(line) for line in open(p)]
    if not recs:
        continue
    m, ds, mode = recs[0]["model"], recs[0]["dataset"], recs[0]["cell"].split("_")[1]
    first = [r["rounds"][0] for r in recs]
    last = [r["rounds"][-1] for r in recs]
    v0 = np.mean([bool(x["violated"]) for x in first if x["valid"]])
    v1 = np.mean([bool(x["violated"]) if x["valid"] else True for x in last])
    key = f"{SHORT[m]}-{ds}-fb{mode}"
    sids = {r["sid"] for r in recs}
    bb = CHK[(CHK.model == m) & (CHK.dataset == ds) & (CHK.cell == "base") & CHK.sid.isin(sids) & CHK.valid]
    num[f"{key}-basesame"] = pct(bb.violated.astype(float).mean())
    num[f"{key}-first"] = pct(v0)
    num[f"{key}-final"] = pct(v1)
    num[f"{key}-n"] = str(len(recs))
    num[f"{key}-calls"] = f"{np.mean([len(r['rounds']) for r in recs]):.1f}"
    fb_rows[(m, mode)] = (v1, np.mean([len(r["rounds"]) for r in recs]), bb.violated.astype(float).mean())

# ------------------------------------------------------------------ B10 feedback claims
# feedback_claims.py writes one row per model and cell; the paper uses the
# final-round claim rates and AUROC to say whether the exact check in the loop
# makes the claim less constant and more informative.
FBC = pd.read_csv(os.path.join(ROOT, "results", "feedback_claims.csv"))
for _, r in FBC.iterrows():
    key = f"fbclaim-{SHORT[r.model]}-{r.cell}".replace("_", "-")
    emit(f"{key}-violation", pct(r.violation))
    emit(f"{key}-claimtrue", pct(r.claim_true))
    emit(f"{key}-claimtrueviol", pct(r.claim_true_given_violated))
    emit(f"{key}-auroc", "--" if pd.isna(r.auroc_claim) else f"{r.auroc_claim:.2f}")
    emit(f"{key}-n", int(r.n))
    emit(f"{key}-nsatisfied", int(r.n_satisfied))
    emit(f"{key}-ninvalid", int(r.n_invalid))
for mode in ("retry", "memory"):
    F = FBC[FBC.cell == f"feedback_{mode}_final"]
    F0 = FBC[FBC.cell == f"feedback_{mode}_first"]
    if not F.empty:
        emit(f"fbclaim-{mode}-claimtrue-min", pct(F.claim_true.min()))
        emit(f"fbclaim-{mode}-claimtrue-max", pct(F.claim_true.max()))
        emit(f"fbclaim-{mode}-claimtrueviol-min", pct(F.claim_true_given_violated.min()))
        emit(f"fbclaim-{mode}-claimtrueviol-max", pct(F.claim_true_given_violated.max()))
        emit(f"fbclaim-{mode}-auroc-min", "--" if F.auroc_claim.isna().all() else f"{F.auroc_claim.min():.2f}")
        emit(f"fbclaim-{mode}-auroc-max", "--" if F.auroc_claim.isna().all() else f"{F.auroc_claim.max():.2f}")
        emit(f"fbclaim-{mode}-nsatisfied-min", int(F.n_satisfied.min()))
        emit(f"fbclaim-{mode}-n", int(F.n.max()))
        emit(f"fbclaim-{mode}-firstclaimtrue-min", pct(F0.claim_true.min()))
        emit(f"fbclaim-{mode}-firstclaimtrue-max", pct(F0.claim_true.max()))
        emit(f"fbclaim-{mode}-firstclaimtrueviol-min", pct(F0.claim_true_given_violated.min()))
        emit(f"fbclaim-{mode}-firstclaimtrueviol-max", pct(F0.claim_true_given_violated.max()))

# ------------------------------------------------------------------ labelled-subset sizes (repair table in the supplement)
for m in MODELS:
    r = get(m, "ml1m", "base")
    if r is not None:
        num[f"{SHORT[m]}-ml1m-base-nlabel"] = str(int(r.n_with_label))

# ------------------------------------------------------------------ direction of movement
dp = os.path.join(ROOT, "results", "direction.csv")
if os.path.exists(dp):
    for _, r in pd.read_csv(dp).iterrows():
        key = f"dir-{SHORT[r.model]}-{r.dataset}-{r.cell.replace('_', '')}"
        num[key + "-up"] = pct(r.moved_up)
        num[key + "-down"] = pct(r.moved_down)

# ------------------------------------------------------------------ count cells against the exposure floor
from fairlib import floors as _floors, satisfied as _sat  # noqa: E402
for m in MODELS:
    for cell in ("count", "countm"):
        p = os.path.join(ROOT, "results", "raw", m.replace(":", "_"), "ml1m", f"{cell}.jsonl")
        if not os.path.exists(p):
            continue
        out_ok, in_ok = [], []
        for line in open(p):
            r = json.loads(line)
            if not r.get("valid"):
                continue
            by = {c["cid"]: c["group"] for c in r["cands"]}
            fl = _floors([c["group"] for c in r["cands"]], 0.9)
            out_ok.append(_sat([by[i] for i in r["ranking"]], fl))
            in_ok.append(_sat([c["group"] for c in sorted(r["cands"], key=lambda c: c["base_rank"])], fl))
        num[f"{SHORT[m]}-ml1m-{cell}-expviol"] = pct(1 - np.mean(out_ok))
        num[f"{SHORT[m]}-ml1m-{cell}-expviolin"] = pct(1 - np.mean(in_ok))

# ------------------------------------------------------------------ reasoning traces
tp = os.path.join(ROOT, "results", "traces.csv")
if os.path.exists(tp):
    T = pd.read_csv(tp)
    T = T[T.dataset == "ml1m"]
    num["tr-stated"] = str(int(T.stated_all.sum()))
    num["tr-vc"] = str(int(T.violated_claimed.sum()))
    num["tr-off"] = str(int(T.stated_off.sum()))
    num["tr-sumgt1"] = str(int(T.stated_sum_gt1.sum()))
    num["tr-saysok"] = str(int(T.stated_says_ok_truth_not.sum()))
    num["tr-off-pct"] = pct(T.stated_off.sum() / T.stated_all.sum(), 0)
    num["tr-sumgt1-pct"] = pct(T.stated_sum_gt1.sum() / T.stated_all.sum(), 0)

# ------------------------------------------------------------------ informativeness against free features
ip = os.path.join(ROOT, "results", "informativeness.csv")
if os.path.exists(ip):
    I = pd.read_csv(ip)
    for _, r in I.iterrows():
        key = f"inf-{SHORT[r.model]}-{r.scope.replace('-', '')}-{r.score.replace('_', '')}"
        num[key] = f"{r.auroc:.2f}"
        num[key + "-ci"] = f"[{r.lo:.2f}, {r.hi:.2f}]"
    P = I[(I.scope == "pooled")]
    for score in ("input_ok", "switches"):
        q = P[(P.score == score) & P.model.isin(MODELS)]
        num[f"inf-pooled-{score.replace('_', '')}-min"] = f"{q.auroc.min():.2f}"
        num[f"inf-pooled-{score.replace('_', '')}-max"] = f"{q.auroc.max():.2f}"
    q = I[(I.scope.str.endswith("base")) & (I.score != "p") & I.model.isin(MODELS)]
    num["inf-base-free-min"] = f"{q.auroc.min():.2f}"
    num["inf-base-free-max"] = f"{q.auroc.max():.2f}"
    num["inf-pooled-cells"] = str(int(P[P.model == "qwen2.5:7b"].cells.max()))
    _cells = {}
    for m in P.model.unique():                         # every model that has pooled rows (small and strong)
        _c = P[(P.model == m) & (P.score == "p")].cells
        if len(_c):
            _cells[m] = int(_c.max())
            emit(f"inf-{SHORT[m]}-pooled-cells", _cells[m])
    _small = [_cells[m] for m in MODELS if m in _cells]
    if _small:
        emit("inf-pooled-cells-min", min(_small))
        emit("inf-pooled-cells-max", max(_small))
sp_ = os.path.join(ROOT, "results", "shortfall.csv")
if os.path.exists(sp_):
    SF = pd.read_csv(sp_)
    SF = SF[SF.model.isin(MODELS)]
    num["short-out-min"] = f"{SF.short_out.min():.3f}"
    num["short-out-max"] = f"{SF.short_out.max():.3f}"
    num["short-in-min"] = f"{SF.short_in.min():.3f}"
    num["short-in-max"] = f"{SF.short_in.max():.3f}"
tl = json.load(open(os.path.join(ROOT, "results", "tolerance.json")))
num["tol-n"] = f"{tl['outputs']:,}"
num["tol-same"] = f"{tl['same_verdict']:,}"
num["tol-near"] = str(tl["within_0.002_of_a_floor"])
num["tol-same-pct"] = pct(tl["same_verdict"] / tl["outputs"])

# ------------------------------------------------------------------ backend fidelity
F = json.load(open(os.path.join(ROOT, "results", "backend_fidelity.json")))
num["fid-total"] = str(sum(v["slates"] for v in F.values()))
num["fid-same"] = str(sum(v["same_ranking"] for v in F.values()))
num["fid-sameclaim"] = str(sum(v["same_claim"] for v in F.values()))
num["fid-sameviol"] = str(sum(v["same_violation"] for v in F.values()))

# ------------------------------------------------------------------ verifier
# NOTE (A1): the ver-* and sim-* macros below are retained for the artifact
# (results/verifier.csv, results/verifier_shift.csv) but are no longer used in
# the manuscript, which now only motivates the uncomputable case.
vp = os.path.join(ROOT, "results", "verifier.csv")
if os.path.exists(vp):
    V = pd.read_csv(vp)
    if "cell" in V:
        for _, r in V.iterrows():
            a = "" if pd.isna(r.alpha) else f"-a{int(round(100 * r.alpha)):02d}"
            key = f"ver-{r.dataset}-{r.cell}-{r.prior}-{SHORT[r.model]}-{r.policy.replace('_', '')}{a}"
            for stat in ("risk", "escalated", "viol_among_accepted", "served_violation"):
                num[f"{key}-{stat.replace('_', '')}"] = pct(r[stat])
            num[f"{key}-servedndcg"] = "--" if pd.isna(r.served_ndcg) else f"{r.served_ndcg:.3f}"
            base_key = f"ver-{r.dataset}-{r.cell}-{r.prior}-{SHORT[r.model]}"
            num[base_key + "-n"] = str(int(r.n))
            num[base_key + "-baserate"] = pct(r.base_rate)
            num[base_key + "-robustfeasible"] = pct(r.robust_feasible)
        # means over models (uniform prior)
        U = V[(V.prior == "uniform")]
        for (ds, cell, pol, al), g in U.groupby(["dataset", "cell", "policy", "alpha"], dropna=False):
            a = "" if pd.isna(al) else f"-a{int(round(100 * al)):02d}"
            key = f"ver-{ds}-{cell}-mean-{pol.replace('_', '')}{a}"
            for stat in ("risk", "escalated", "viol_among_accepted", "served_violation"):
                num[f"{key}-{stat.replace('_', '')}"] = pct(g[stat].mean())
            num[f"{key}-servedndcg"] = f"{g.served_ndcg.mean():.3f}"
            num[f"ver-{ds}-{cell}-mean-robustfeasible"] = pct(g.robust_feasible.mean())
sp = os.path.join(ROOT, "results", "verifier_shift.csv")
if os.path.exists(sp):
    SH = pd.read_csv(sp)
    if "model" in SH:
        for (ds, test, al), g in SH.groupby(["dataset", "test", "alpha"]):
            num[f"ver-shift-{ds}-{test}-a{int(round(100 * al)):02d}-risk"] = pct(g.risk.mean())
            num[f"ver-shift-{ds}-{test}-a{int(round(100 * al)):02d}-riskmax"] = pct(g.risk.max())
sim = pd.read_csv(os.path.join(ROOT, "results", "sim_crc.csv"))
num["sim-trials"] = "100{,}000"
num["sim-maxexcess"] = f"{(sim.risk - sim.upper).max():.4f}"
num["sim-settings"] = str(len(sim))

# ------------------------------------------------------------------ reporting checks and writing revision
# Both modules read only saved records: checked.jsonl, ptrue/*.jsonl and
# feedback_claims.csv.  They are the single source for the three reporting
# tables, the clustered intervals and the headline prose numbers.
ROWS = [json.loads(line) for line in open(os.path.join(ROOT, "results", "checked.jsonl"), encoding="utf-8")]
PROBS = reporting_checks.load_probs(ROOT)
RC = reporting_checks.compute(ROWS, PROBS, FBC.to_dict("records"))
WC = writing_revision.clustered_ci(ROWS, PROBS, reporting_checks.MODELS)

# ------------------------------------------------------------------ Table 4: separate audits (ML-1M)
AUDIT_NAME = {"qwen2.5:7b": "Qwen2.5-7B", "llama3.1:8b": "Llama3.1-8B",
              "gemma2:9b": "Gemma2-9B", "qwen2.5:14b": "Qwen2.5-14B"}
lines = []
for x in RC["matched_audits"]:
    if x["dataset"] != "ml1m":
        continue
    z = x["audit"]
    lines.append(f"{AUDIT_NAME[x['model']]} & {z['caught']}/{z['violations']} & "
                 f"{z['false_alarms']}/{z['compliant']} & "
                 f"{z['false_assurances']}/{z['claims_compliant']}" + r" \\")
open(os.path.join(GEN, "tab_audit_reporting.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ clustered token-probability intervals
# These live in their own file so the interval method is easy to find; the
# values still pass through emit() so number_map.py can trace them.
for r, name in zip(WC, ("qwen", "llama", "gemma")):
    emit(f"writing-{name}-pooled-ci", f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]")

# ------------------------------------------------------------------ Figure: dose-response in rho
# The JSON is the plotted data; the pgfplots file is a pure function of it, so
# the figure inside the paper can be rebuilt without matplotlib or a PDF.
RHO = [("rho050", 0.5), ("rho070", 0.7), ("rho080", 0.8), ("base", 0.9), ("rho095", 0.95)]
series = {"source": "Generated by artifact/code/make_assets.py from results/summary.csv.",
          "rho": [x for _, x in RHO], "panels": {}}
for ds, panel in (("ml1m", "ML-1M"), ("mind", "MIND")):
    pan = {"models": {}, "claimed": {}, "input": [], "random": []}
    ref = None
    for m in MODELS:
        d = pd.concat([S[(S.model == m) & (S.dataset == ds) & (S.cell == c)] for c, _ in RHO])
        d = d.sort_values("rho")
        if d.empty:
            continue
        pan["models"][MODEL_NAME[m]] = {
            "ci": [[round(100 * lo, 2), round(100 * hi, 2)]
                   for lo, hi in zip(d.violation_lo, d.violation_hi)],
            "viol": [round(100 * v, 2) for v in d.violation],
        }
        pan["claimed"][MODEL_NAME[m]] = [round(100 * (1 - c), 2) for c in d.claim_true]
        ref = d if ref is None else ref
    if ref is not None:
        pan["input"] = [round(100 * b, 2) for b in ref.binding]
        pan["random"] = [round(100 * rv, 2) for rv in ref.random_violation]
    series["panels"][panel] = pan
open(os.path.join(FIGDATA, "fig_dose_series.json"), "w").write(json.dumps(series, indent=1) + "\n")


def _dose_co(ys):
    return " ".join(f"({x:g},{y:.2f})" for x, y in zip(series["rho"], ys))


out = [r'''% Generated by artifact/code/make_assets.py -- do not edit by hand.
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 1, horizontal sep=0.35cm, y descriptions at=edge left},
  width=0.565\columnwidth, height=3.6cm,
  xmin=0.47, xmax=0.98, xtick={0.5,0.7,0.8,0.9}, ymin=-4, ymax=104, ytick={0,25,50,75,100},
  tick label style={font=\footnotesize}, label style={font=\footnotesize},
  title style={font=\small\bfseries, yshift=-1.3ex},
  axis line style={dInk!70, line width=0.5pt}, tick style={dInk!70}, major tick length=2pt,
  axis x line*=bottom, axis y line*=left, ymajorgrids, grid style={dInk!12, line width=0.3pt},
  xlabel={floor tightness $\rho$}, xlabel style={yshift=0.8ex},
  every axis plot/.append style={line width=1pt, mark size=1.5pt},
]''']
DOSE_STYLE = [("Qwen2.5-7B", "dQwen", "*"), ("Llama3.1-8B", "dLlama", "square*"),
              ("Gemma2-9B", "dGemma", "triangle*")]
for i, (panel, pan) in enumerate(series["panels"].items()):
    opts = f'title={{{panel}}}'
    if i == 0:
        opts += (r', ylabel={Actual violations (\%)}, ylabel style={yshift=-1.2ex},'
                 r' legend to name=doselegend, legend columns=2,'
                 r' legend style={draw=none, font=\footnotesize, row sep=-1pt,'
                 r' /tikz/every even column/.append style={column sep=7pt}}')
    out.append(rf'\nextgroupplot[{opts}]')
    for name, col, mk in DOSE_STYLE:
        m = pan["models"][name]
        lo, hi = [c[0] for c in m["ci"]], [c[1] for c in m["ci"]]
        out.append(rf'\addplot[draw=none, fill={col}, fill opacity=0.16, forget plot] coordinates {{{_dose_co(lo)} '
                   + " ".join(f"({x:g},{y:.2f})" for x, y in zip(series["rho"][::-1], hi[::-1]))
                   + r'} -- cycle;')
    out.append(rf'\addplot[dInk!65, dashed, line width=0.8pt, mark=none] coordinates {{{_dose_co(pan["input"])}}};')
    if i == 0:
        out.append(r'\addlegendentry{Input ranking}')
    out.append(rf'\addplot[dInk!45, dash dot, line width=0.8pt, mark=none] coordinates {{{_dose_co(pan["random"])}}};')
    if i == 0:
        out.append(r'\addlegendentry{Random permutation}')
    out.append(rf'\addplot[dCrim, densely dotted, line width=1.3pt, mark=none] coordinates {{{_dose_co([0] * len(series["rho"]))}}};')
    if i == 0:
        out.append(r'\addlegendentry{Model-reported violations: 0\%}')
    for name, col, mk in DOSE_STYLE:
        out.append(rf'\addplot[{col}, mark={mk}, mark options={{fill={col}}}] coordinates {{{_dose_co(pan["models"][name]["viol"])}}};')
        if i == 0:
            out.append(rf'\addlegendentry{{{name}}}')
out.append(r'''\end{groupplot}
\node[anchor=south, yshift=2.4ex] at ($(group c1r1.north west)!0.5!(group c2r1.north east)$) {\pgfplotslegendfromname{doselegend}};
\end{tikzpicture}''')
open(os.path.join(FIG, "fig_dose_pgf.tex"), "w").write("\n".join(out) + "\n")

# ------------------------------------------------------------------ headline prose numbers
core, q3 = RC["rounding"]
emit("head-core-n", f"{core['exact']['n']:,}")
emit("head-core-violated", f"{core['exact']['violations']:,}")
emit("head-core-violated-displayed", f"{core['displayed']['violations']:,}")
emit("head-core-claimfalse", f"{core['exact']['false_assurances']:,}")
emit("head-core-n-displayed", f"{core['displayed']['n']:,}")
emit("head-core-violated-displayed-stable", f"{core['stable']['violations']:,}")
emit("head-core-n-stable", f"{core['stable']['n']:,}")
emit("head-core-changed", core["changed"])
emit("head-core-slates", core["unique_slates"])
emit("head-core-slates-ml1m", core["slates_by_dataset"]["ml1m"])
emit("head-core-slates-mind", core["slates_by_dataset"]["mind"])
emit("head-q3-n", q3["exact"]["n"])
emit("head-q3-violated", q3["exact"]["violations"])
emit("head-q3-caught", q3["exact"]["caught"])
emit("head-q3-claimfalse", q3["exact"]["false_assurances"])
emit("head-q3-claims", q3["exact"]["claims_compliant"])
emit("head-q3-violated-displayed", q3["displayed"]["violations"])
emit("head-q3-caught-displayed", q3["displayed"]["caught"])
emit("head-q3-claimfalse-displayed", q3["displayed"]["false_assurances"])
emit("head-q3-changed", q3["changed"])
emit("head-q3-stable-n", q3["stable"]["n"])
emit("head-q3-stable-violated", q3["stable"]["violations"])
emit("head-q3-stable-caught", q3["stable"]["caught"])
emit("head-q3-recall", pct(q3["exact"]["caught"] / q3["exact"]["violations"]))
emit("head-q3-falseassurance", pct(q3["exact"]["false_assurances"] / q3["exact"]["claims_compliant"]))
emit("head-q3-falseassurance-displayed", pct(q3["displayed"]["false_assurances"] / q3["displayed"]["claims_compliant"]))
_q3r = get("qwen3:8b", "ml1m", "base")
emit("head-q3-attempted", int(_q3r.n))
emit("head-q3-invalid", int(_q3r.n_invalid))
emit("head-q3-failinvalid-displayed", pct((q3["displayed"]["violations"] + _q3r.n_invalid) / _q3r.n))
emit("head-q3-failinvalid-displayed-n", q3["displayed"]["violations"] + _q3r.n_invalid)
emit("head-infeasible-n", RC["infeasible_core_G5"]["n"])
emit("head-default-attempts-ml1m", int(get("qwen2.5:7b", "ml1m", "base").n))
emit("head-default-attempts-mind", int(get("qwen2.5:7b", "mind", "base").n))
for r_, name in zip(WC, ("qwen", "llama", "gemma")):
    emit(f"head-{name}-prob-n", f"{r_['n']:,}")

_l70 = get("llama3.3:70b", "ml1m", "base")
emit("head-l70-input", f"{_paired_input('llama3.3:70b', 'ml1m'):.1f}")
emit("head-l70-violated", pct(_l70.violation))
emit("head-l70-violated-n", int(_l70.n_violated))
emit("head-l70-caught", int(_l70.n_violated - _l70.n_claim_false))
emit("head-q3-input", f"{_paired_input('qwen3:8b', 'ml1m'):.1f}")

for x, name in zip(writing_revision.one_sided_rates(ROWS), ("qwen", "llama", "gemma")):
    emit(f"head-onesided-{name}", pct(x["one_sided"] / x["n"]))
    emit(f"head-allgroups-{name}", pct(x["all_groups"] / x["n"]))
emit("head-onesided-n", writing_revision.one_sided_rates(ROWS)[0]["n"])

_count_in = []
for cell in ("count", "countm"):
    rr = _CHK[(_CHK.dataset == "ml1m") & (_CHK.cell == cell) & _CHK.valid & _CHK.feasible]
    _v = rr.binding.mean()
    emit(f"head-{cell}-input", pct(_v, 0))
    _count_in.append(_v)
emit("head-count-input-min", pct(min(_count_in), 0))
emit("head-count-input-max", pct(max(_count_in), 0))
_count_v = [get(m, "ml1m", c).violation for m in MODELS for c in ("count", "countm")
            if get(m, "ml1m", c) is not None]
emit("head-count-output-min", pct(min(_count_v), 0))
emit("head-count-output-max", pct(max(_count_v), 0))

AU = {(x["model"], x["dataset"]): x for x in RC["matched_audits"]}
for key, name in (("qwen2.5:7b", "qwen"), ("qwen2.5:14b", "q14")):
    a = AU[(key, "ml1m")]
    emit(f"head-audit-{name}-n", a["audit"]["n"])
    emit(f"head-audit-{name}-base-caught", a["base"]["caught"])
    emit(f"head-audit-{name}-violated", a["audit"]["violations"])
    emit(f"head-audit-{name}-caught", a["audit"]["caught"])
    emit(f"head-audit-{name}-falsealarms", a["audit"]["false_alarms"])
    emit(f"head-audit-{name}-compliant", a["audit"]["compliant"])
    emit(f"head-audit-{name}-falseassurances", a["audit"]["false_assurances"])
    emit(f"head-audit-{name}-claims", a["audit"]["claims_compliant"])
    emit(f"head-audit-{name}-auroc-base", f"{a['base_auroc']:.2f}")
    emit(f"head-audit-{name}-auroc", f"{a['audit_auroc']:.2f}")

FB = {(x["model"], x["cell"]): x for x in RC["feedback"]}
for key, name in (("qwen2.5:7b", "qwen"), ("llama3.1:8b", "llama")):
    f0, f1 = FB[(key, "feedback_retry_first")], FB[(key, "feedback_retry_final")]
    emit(f"head-fb-{name}-first-failinvalid", pct((f0["violations"] + f0["invalid"]) / f0["attempted"]))
    emit(f"head-fb-{name}-final-failinvalid", pct((f1["violations"] + f1["invalid"]) / f1["attempted"]))
    for stat in ("valid", "violations", "caught", "false_alarms", "compliant",
                 "false_assurances", "claims_compliant"):
        emit(f"head-fb-{name}-final-{stat.replace('_', '')}", f1[stat])
    if f1["boolean_auroc"] is not None:
        emit(f"head-fb-{name}-final-auroc", f"{f1['boolean_auroc']:.2f}")

_qr = get("qwen2.5:7b", "ml1m", "base")
emit("head-repair-ms", f"{_qr.repair_ms_median:.0f}")
emit("head-repair-n", int(_qr.n_with_label))
emit("head-proj-gain", f"{-_qr.ndcg_minus_proj:+.3f}")
emit("head-proj-gain-ci", f"[{-_qr.ndcg_minus_proj_hi:.3f}, {-_qr.ndcg_minus_proj_lo:.3f}]")

with open(os.path.join(GEN, "writing_numbers.tex"), "w") as f:
    f.write("% generated by artifact/code/make_assets.py -- do not edit\n")
    for k, v in sorted(num.items()):
        if k.startswith("writing-"):
            f.write(f"\\expandafter\\def\\csname N@{k}\\endcsname{{{v}}}\n")
with open(os.path.join(GEN, "numbers.tex"), "w") as f:
    f.write("% generated by artifact/code/make_assets.py -- do not edit\n")
    for k, v in sorted(num.items()):
        if not k.startswith("writing-"):
            f.write(f"\\expandafter\\def\\csname N@{k}\\endcsname{{{v}}}\n")
print(len(num), "macros")
