"""Token-level probability of the claim: for every saved output, re-score the prompt plus the
model's own output up to '"satisfied":' and read P(' true') and P(' false') for the next token
(teacher forcing on the dedicated llama.cpp server, same weights). Writes
results/ptrue/<model>/<dataset>/<cell>.jsonl. Resumable.
Usage: python ptrue.py --model qwen2.5:7b"""
import argparse
import concurrent.futures as cf
import glob
import json
import math
import os
import re

import requests

import llm_rerank as L

ROOT = L.ROOT


def prompt_for(r, dataset):
    cell = L.CELLS[r["cell"]] if r["cell"] in L.CELLS else None
    if r["prompt"] == "audit":
        return L.build_audit_prompt(r["cands"], {**L.CELLS["base"], "rho": r["rho"]}, r["ranking"])[0]
    if r["prompt"] in ("reason", "paraphrase", "minbudget", "history", "countm"):
        if r["prompt"] == "history" and dataset not in L.HISTORY:
            L.HISTORY[dataset] = json.load(open(os.path.join(ROOT, "data", f"history_{dataset}.json")))
        return L.build_prompt_v2(r["cands"], cell, r["sid"], dataset)[0]
    return L.build_prompt(r["cands"], {**L.CELLS["base"], "rho": r["rho"], "k": r["k"], "G": r["G"],
                                       "order": r["order"], "prompt": r["prompt"]})[0]


def score(model, r, dataset):
    raw = r["raw"] or ""
    start = raw.index("</think>") if "</think>" in raw else 0      # thinking models: the answer part only
    m = re.compile(r'"satisfied"\s*:').search(raw, start)
    if m is None:
        return None
    msgs = [{"role": "system", "content": L.SYSTEM}, {"role": "user", "content": prompt_for(r, dataset)}]
    if model in L.JINJA_PORT:
        full = L.render_jinja(model, msgs) + raw[:m.end()]
        port = L.jinja_port(model)
    else:
        full = L.render(model, msgs) + raw[:m.end()]
        port = L.LLAMACPP_PORT[model]
    body = {"prompt": full, "n_predict": 1, "n_probs": 20, "temperature": 0, "cache_prompt": True}
    for _ in range(4):
        try:
            j = requests.post(f"http://localhost:{port}/completion", json=body, timeout=600).json()
            top = j["completion_probabilities"][0]["top_logprobs"]
            pt = sum(math.exp(t["logprob"]) for t in top if t["token"].strip() == "true")
            pf = sum(math.exp(t["logprob"]) for t in top if t["token"].strip() == "false")
            return {"sid": r["sid"], "p_true": pt, "p_false": pf, "tokens": j.get("tokens_evaluated")}
        except Exception:  # noqa: BLE001
            continue
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", default="", help="comma-separated cell files to score (default: all)")
    ap.add_argument("--backend", default="llamacpp", help="llamacpp (local) or llamacpp_cuda (GPU server)")
    a = ap.parse_args()
    L.BACKEND = a.backend
    md = a.model.replace(":", "_")
    PRIORITY = ["base", "rho095", "rho080", "rho070", "history", "reason", "paraphrase", "minbudget", "befair",
                "count", "countm", "G3", "G5", "k05", "k20", "order_random", "order_reverse", "rho050", "audit",
                "audit_rho080"]

    def order(p):
        cell = os.path.basename(p)[:-len(".jsonl")]
        ds = os.path.basename(os.path.dirname(p))
        return (ds == "mind_tied" or cell.startswith("ablation_"),
                PRIORITY.index(cell) if cell in PRIORITY else len(PRIORITY), ds)
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "raw", md, "*", "*.jsonl")), key=order):
        base = os.path.basename(p)
        if base.startswith(("feedback_", "fidelity_")) or "/mind_nanscores/" in p:
            continue
        if a.only and base[:-len(".jsonl")] not in a.only.split(","):
            continue
        dataset = os.path.basename(os.path.dirname(p))
        out = os.path.join(ROOT, "results", "ptrue", md, dataset, base)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        done = {json.loads(x)["sid"] for x in open(out)} if os.path.exists(out) else set()
        recs = [json.loads(x) for x in open(p)]
        recs = [r for r in recs if r["sid"] not in done and r.get("raw")]
        with cf.ThreadPoolExecutor(a.workers) as ex, open(out, "a") as f:
            for res in ex.map(lambda r: score(a.model, r, "mind" if dataset == "mind_tied" else dataset), recs):
                if res is not None:
                    f.write(json.dumps(res) + "\n"); f.flush()
        print(a.model, dataset, base, len(recs), flush=True)
