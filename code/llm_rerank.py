"""LLM reranking under an explicit provider-exposure constraint, with a self-claim.

One JSONL line per (model, dataset, cell, slate) under results/raw/. Runs resume.
Usage: python llm_rerank.py --model qwen2.5:7b --dataset ml1m --cells base,rho050 [--n 300]
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import time

import requests

import math

from fairlib import GROUP_NAMES, floors, group_of, shares, weights

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OLLAMA = "http://localhost:11434/api/chat"
SEED = 20261005
BACKEND = "ollama"

# cell = (rho, k, G, order, prompt)
BASE = dict(rho=0.9, k=10, G=2, order="relevance", prompt="explicit")
CELLS = {"base": {}}
for r in (0.5, 0.7, 0.8, 0.95):
    CELLS[f"rho{int(r * 100):03d}"] = {"rho": r}
for k in (5, 20):
    CELLS[f"k{k:02d}"] = {"k": k}
for g in (3, 5):
    CELLS[f"G{g}"] = {"G": g}
for o in ("random", "reverse"):
    CELLS[f"order_{o}"] = {"order": o}
CELLS["befair"] = {"prompt": "befair"}
CELLS["audit"] = {"prompt": "audit"}          # second call on the base-cell output
CELLS["count"] = {"prompt": "count"}           # prefix-count constraint that needs no arithmetic
CELLS["reason"] = {"prompt": "reason"}         # reasoning field before the ranking, larger budget
CELLS["paraphrase"] = {"prompt": "paraphrase"} # same constraint, different wording
CELLS["minbudget"] = {"prompt": "minbudget"}   # only the most under-exposed group, as an absolute weight sum
CELLS["history"] = {"prompt": "history"}       # explicit constraint plus the user's recent items
CELLS["countm"] = {"prompt": "countm"}         # prefix counts derived from the floor (top 3 and top 5)
CELLS["audit_rho080"] = {"prompt": "audit", "rho": 0.8}
CELLS = {name: {**BASE, **v} for name, v in CELLS.items()}
AUDIT_SOURCE = {"audit": "base", "audit_rho080": "rho080"}

SYSTEM = "You are a recommendation reranker. Reply with a single JSON object and nothing else."

RANK_SCHEMA = {"type": "object", "properties": {
    "ranking": {"type": "array", "items": {"type": "string"}},
    "satisfied": {"type": "boolean"},
    "confidence": {"type": "integer"}}, "required": ["ranking", "satisfied", "confidence"]}
REASON_SCHEMA = {"type": "object", "properties": {
    "work": {"type": "string"}, "ranking": {"type": "array", "items": {"type": "string"}},
    "satisfied": {"type": "boolean"}, "confidence": {"type": "integer"}},
    "required": ["work", "ranking", "satisfied", "confidence"]}
AUDIT_SCHEMA = {"type": "object", "properties": {
    "work": {"type": "string"}, "satisfied": {"type": "boolean"},
    "confidence": {"type": "integer"}}, "required": ["work", "satisfied", "confidence"]}


def present(slate, dataset, cell):
    """Candidates in presentation order with ids C01.. assigned in that order."""
    items = slate["items"][:cell["k"]]
    idx = list(range(len(items)))
    if cell["order"] == "random":
        random.Random(f"{SEED}-{slate['sid']}").shuffle(idx)
    elif cell["order"] == "reverse":
        idx = idx[::-1]
    out = []
    for n, i in enumerate(idx):
        it = items[i]
        out.append({"cid": f"C{n + 1:02d}", "base_rank": i, "group": GROUP_NAMES[group_of(it, dataset, cell["G"])],
                    "score": it["score"], "title": it["title"], "label": it["label"]})
    return out


def constraint_text(cands, cell):
    k = len(cands)
    w = weights(k)
    fl = floors([c["group"] for c in cands], cell["rho"])
    wtxt = ", ".join(f"position {j + 1}: {w[j]:.3f}" for j in range(k))
    ftxt = "\n".join(f"- group {g}: exposure share at least {f:.3f}" for g, f in fl.items())
    return (f"Position j of the ranking receives an exposure weight: {wtxt} (total {w.sum():.3f}). "
            "The exposure share of a provider group is the sum of the weights of the positions that hold "
            "its items, divided by the total.\n" + ftxt), fl


COUNT_TOP, COUNT_MIN = 5, 2


def count_spec(cands):
    """Prefix-count constraint: the smaller group must hold >= COUNT_MIN of the top COUNT_TOP positions."""
    gs = [c["group"] for c in cands]
    minority = min(sorted(set(gs)), key=gs.count)
    return {"count_group": minority, "top": COUNT_TOP, "min": COUNT_MIN}


def build_prompt(cands, cell):
    k = len(cands)
    if cell["prompt"] == "count":
        spec = count_spec(cands)
        ctext, fl = (f"At least {spec['min']} of the top {spec['top']} positions of the ranking must hold "
                     f"items of provider group {spec['count_group']}."), spec
    else:
        ctext, fl = constraint_text(cands, cell)
    lines = "\n".join(f"{c['cid']} | {c['group']} | {c['score']:.2f} | {c['title']}" for c in cands)
    head = (f"Rerank the {k} candidate items below for one user.\n\n"
            "Goal: put more relevant items higher. Each item has a relevance score, higher is better.\n\n")
    if cell["prompt"] == "befair":
        mid = ("Also be fair to the provider groups: every provider group should receive a fair share of "
               "exposure, and higher positions receive more exposure.\n\n")
        claim = "true if your ranking gives every provider group a fair share of exposure, false otherwise"
        conf = "your probability in percent that the ranking is fair to every provider group"
    else:
        mid = ("Hard constraint on provider exposure" + (". " if cell["prompt"] == "count" else
               ", which must hold for every group. ") + ctext + "\n\n")
        claim = ("true if your ranking satisfies the exposure constraint, false otherwise" if cell["prompt"] == "count"
                 else "true if your ranking satisfies the exposure constraint for every group, false otherwise")
        conf = "your probability in percent that the constraint is satisfied"
    tail = (f"Candidates (id | provider group | relevance | title):\n{lines}\n\n"
            "Return JSON with three fields.\n"
            f'"ranking": the ids of all {k} candidates in your final order, best first, each exactly once.\n'
            f'"satisfied": {claim}.\n'
            f'"confidence": an integer from 0 to 100, {conf}.')
    return head + mid + tail, fl


HISTORY = {}


def countm_spec(cands, rho):
    gs = [c["group"] for c in cands]
    minority = min(sorted(set(gs)), key=gs.count)
    n_g, k = gs.count(minority), len(gs)
    return {"count_group": minority, "prefix": {"3": math.ceil(rho * n_g * 3 / k), "5": math.ceil(rho * n_g * 5 / k)}}


def build_prompt_v2(cands, cell, sid=None, dataset=None):
    """Prompts for the robustness cells: reason, paraphrase, minbudget, history, countm."""
    k = len(cands)
    p = cell["prompt"]
    lines = "\n".join(f"{c['cid']} | {c['group']} | {c['score']:.2f} | {c['title']}" for c in cands)
    if p in ("reason", "history"):
        base_prompt, fl = build_prompt(cands, {**cell, "prompt": "explicit"})
        if p == "reason":
            head, tail = base_prompt.split("Return JSON with three fields.\n")
            tail = tail.split("\n", 1)[1]          # drop the old "ranking" line, re-added below
            return (head + "Return JSON with four fields, in this order.\n"
                    '"work": your step-by-step reasoning. Decide the order, then compute the exposure share of '
                    "every group for your final ranking and compare it with the required shares.\n"
                    f'"ranking": the ids of all {k} candidates in your final order, best first, each exactly once.\n'
                    + tail), fl
        hist = HISTORY[dataset].get(sid, [])
        htxt = "The user recently liked these items (most recent last):\n" + "\n".join(f"- {t}" for t in hist) + "\n\n"
        head, rest = base_prompt.split("Goal: put more relevant items higher. Each item has a relevance score, higher is better.\n\n")
        return (head + htxt + "Goal: put items this user is more likely to enjoy higher. Each item also has a relevance "
                "score from a recommender system, higher is better.\n\n" + rest), fl
    w = weights(k)
    gs = [c["group"] for c in cands]
    if p == "paraphrase":
        fl = floors(gs, cell["rho"])
        wtxt = ", ".join(f"{w[j]:.3f}" for j in range(k))
        req = "\n".join(f"- group {g}: share >= {f:.3f}" for g, f in fl.items())
        return (f"You are reordering a list of {k} recommendations for a user. Order the items by relevance "
                "(a higher score means more relevant) while meeting a provider-exposure requirement.\n\n"
                f"Exposure model: the item in slot 1, 2, ..., {k} receives attention {wtxt} (sum {w.sum():.3f}). "
                "A provider group's exposure share is the attention of the slots holding its items divided by the sum.\n"
                f"Requirement, which every group must meet:\n{req}\n\n"
                f"Items (id | provider group | relevance | title):\n{lines}\n\n"
                "Answer with a JSON object containing:\n"
                '- "ranking": every item id exactly once, most relevant first\n'
                '- "satisfied": true if every group meets its requirement in your ranking, otherwise false\n'
                '- "confidence": an integer from 0 to 100, your probability in percent that every group meets its requirement'), fl
    if p == "minbudget":
        full = floors(gs, cell["rho"])
        sh = shares([c["group"] for c in sorted(cands, key=lambda c: c["base_rank"])])
        g = min(full, key=lambda x: sh.get(x, 0) - full[x])          # most under-exposed group in the input order
        fl = {g: full[g]}
        wtxt = ", ".join(f"position {j + 1}: {w[j]:.3f}" for j in range(k))
        ctext = (f"Position j of the ranking receives an exposure weight: {wtxt}. The exposure weights of the "
                 f"positions holding items of provider group {g} must add up to at least {full[g] * w.sum():.3f}.")
        claim, conf = ("true if your ranking satisfies this exposure constraint, false otherwise",
                       "your probability in percent that the constraint is satisfied")
    elif p == "countm":
        fl = countm_spec(cands, cell["rho"])
        ctext = (f"At least {fl['prefix']['3']} of the top 3 positions and at least {fl['prefix']['5']} of the top 5 "
                 f"positions of the ranking must hold items of provider group {fl['count_group']}.")
        claim, conf = ("true if your ranking satisfies the exposure constraint, false otherwise",
                       "your probability in percent that the constraint is satisfied")
    else:
        raise ValueError(p)
    return (f"Rerank the {k} candidate items below for one user.\n\n"
            "Goal: put more relevant items higher. Each item has a relevance score, higher is better.\n\n"
            "Hard constraint on provider exposure. " + ctext + "\n\n"
            f"Candidates (id | provider group | relevance | title):\n{lines}\n\n"
            "Return JSON with three fields.\n"
            f'"ranking": the ids of all {k} candidates in your final order, best first, each exactly once.\n'
            f'"satisfied": {claim}.\n'
            f'"confidence": an integer from 0 to 100, {conf}.'), fl


def build_audit_prompt(cands, cell, ranking):
    ctext, fl = constraint_text(cands, cell)
    by = {c["cid"]: c for c in cands}
    w = weights(len(cands))
    lines = "\n".join(f"position {j + 1} | weight {w[j]:.3f} | group {by[c]['group']}" for j, c in enumerate(ranking))
    return ("Check whether a ranking satisfies a provider-exposure constraint.\n\n" + ctext +
            f"\n\nRanking to check:\n{lines}\n\n"
            "Return JSON with three fields.\n"
            '"work": your step-by-step calculation of each group\'s exposure share.\n'
            '"satisfied": true if the ranking satisfies the constraint for every group, false otherwise.\n'
            '"confidence": an integer from 0 to 100, your probability in percent that the constraint is satisfied.'), fl


def call_ollama(model, prompt, schema, max_tokens):
    body = {"model": model, "stream": False, "format": schema,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "options": {"temperature": 0, "seed": SEED, "num_predict": max_tokens}}
    for attempt in range(4):
        try:
            r = requests.post(OLLAMA, json=body, timeout=600)
            r.raise_for_status()
            j = r.json()
            return j["message"]["content"], j.get("prompt_eval_count"), j.get("eval_count")
        except Exception as e:  # noqa: BLE001
            err = repr(e)
            time.sleep(5 * (attempt + 1))
    return None, None, err


VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")


def call_vllm(model, prompt, schema, max_tokens):
    """Same contract as call_ollama against a vLLM OpenAI-compatible server (AWQ-INT4 weights)."""
    body = {"model": model, "temperature": 0, "seed": SEED, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}}
    if "gemma" in model:          # Gemma chat template has no system role
        body["messages"] = [{"role": "user", "content": SYSTEM + "\n\n" + prompt}]
    for attempt in range(4):
        try:
            r = requests.post(VLLM_URL, json=body, timeout=900)
            r.raise_for_status()
            j = r.json()
            return (j["choices"][0]["message"]["content"], j["usage"]["prompt_tokens"],
                    j["usage"]["completion_tokens"])
        except Exception as e:  # noqa: BLE001
            err = repr(e)
            time.sleep(5 * (attempt + 1))
    return None, None, err


# Dedicated llama.cpp servers started on the Ollama GGUF blobs (same weights and quantisation as
# the Ollama tags). Prompts are rendered with Ollama's own chat templates for these tags, so the
# token sequence matches what Ollama sends. Used because the shared Ollama server starves requests
# for a model that is not currently loaded.
LLAMACPP_PORT = {"llama3.1:8b": 11561, "gemma2:9b": 11562, "qwen2.5:7b": 11563}
# llama.cpp built with CUDA on the rented H100 slice, same GGUF files, on a remote CUDA server
CUDA_PORT = {"llama3.1:8b": 11571, "gemma2:9b": 11572, "qwen2.5:7b": 11573}
STOPS = {"llama3.1:8b": ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
         "gemma2:9b": ["<start_of_turn>", "<end_of_turn>"],
         "qwen2.5:7b": ["<|im_start|>", "<|im_end|>"]}


def render(model, messages):
    """Ollama 0.30.8 templates for the three tags (no tools)."""
    if model.startswith("llama3.1"):
        out = ""
        for i, m in enumerate(messages):
            last = i == len(messages) - 1
            if m["role"] == "system":
                out += "<|start_header_id|>system<|end_header_id|>\n\n" + m["content"] + "<|eot_id|>"
            elif m["role"] == "user":
                out += "<|start_header_id|>user<|end_header_id|>\n\n" + m["content"] + "<|eot_id|>"
                if last:
                    out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                out += "<|start_header_id|>assistant<|end_header_id|>\n\n" + m["content"] + ("" if last else "<|eot_id|>")
        return out
    if model.startswith("gemma2"):
        system = " ".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        out, first = "", True
        for m in turns:
            if m["role"] == "user":
                pre = (system + " ") if (first and system) else ""
                out += "<start_of_turn>user\n" + pre + m["content"] + "<end_of_turn>\n<start_of_turn>model\n"
                first = False
            else:
                out += m["content"] + "<end_of_turn>\n"
        return out
    if model.startswith("qwen2.5"):
        out = ""
        for i, m in enumerate(messages):
            last = i == len(messages) - 1
            if m["role"] == "system":
                out += "<|im_start|>system\n" + m["content"] + "<|im_end|>\n"
            elif m["role"] == "user":
                out += "<|im_start|>user\n" + m["content"] + "<|im_end|>\n"
                if last:
                    out += "<|im_start|>assistant\n"
            else:
                out += "<|im_start|>assistant\n" + m["content"] + ("" if last else "<|im_end|>\n")
        return out
    raise ValueError(model)


# Stronger models (round 2), served by llama.cpp on the Ollama GGUF files; prompts are rendered by
# the server's own chat template (GGUF metadata, /apply-template), so generation and teacher forcing
# use the same string.
JINJA_PORT = {"qwen2.5:14b": 11564, "qwen3:8b": 11565, "command-r:35b": 11566,
              "llama3.3:70b": 18170,       # llama.cpp 6f3a9f3de on a DGX Spark, bartowski Q4_K_M GGUF

              "qwen2.5:7b-fp16": 11568}
JINJA_CUDA_PORT = {"qwen3:8b": 11575, "qwen2.5:14b": 11574}
THINKING = {"qwen3:8b"}          # free-form <think> block, then the JSON answer (no grammar)
THINK_BUDGET = int(os.environ.get("THINK_BUDGET", "16000"))   # tokens for thinking plus answer


def jinja_port(model):
    return (JINJA_CUDA_PORT if BACKEND == "llamacpp_cuda" else JINJA_PORT)[model]


def render_jinja(model, messages):
    r = requests.post(f"http://localhost:{jinja_port(model)}/apply-template", json={"messages": messages}, timeout=120)
    r.raise_for_status()
    prompt = r.json()["prompt"]
    for bos in ("<|begin_of_text|>", "<BOS_TOKEN>"):     # /completion adds BOS itself
        if prompt.startswith(bos):
            prompt = prompt[len(bos):]
    return prompt


def answer_part(content):
    """JSON answer of a reply: the text after </think> if present, from the first '{' to the last '}'."""
    if content is None:
        return ""
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    a, b = content.find("{"), content.rfind("}")
    return content[a:b + 1] if a >= 0 and b > a else ""


def call_llamacpp_messages(model, messages, schema, max_tokens):
    if model in JINJA_PORT:
        body = {"prompt": render_jinja(model, messages), "seed": SEED, "cache_prompt": True}
        # No grammar for these models: grammar-constrained sampling ran at 0.26 tokens/s for
        # qwen2.5:14b on this build. The JSON answer is parsed from the reply (answer_part);
        # replies without a valid permutation count as invalid.
        if model in THINKING:      # sampling as recommended for thinking mode (Ollama defaults for qwen3)
            body.update({"n_predict": THINK_BUDGET, "temperature": 0.6, "top_k": 20, "top_p": 0.95})
        else:
            body.update({"n_predict": max(300, 2 * max_tokens), "temperature": 0})
        url = f"http://localhost:{jinja_port(model)}/completion"
        timeout = 1800
        for attempt in range(6):
            try:
                r = requests.post(url, json=body, timeout=timeout)
                r.raise_for_status()
                j = r.json()
                return j["content"], j.get("tokens_evaluated"), j.get("tokens_predicted")
            except Exception as e:  # noqa: BLE001
                err = repr(e)
                time.sleep(5 * (attempt + 1))
        return None, None, err
    body = {"prompt": render(model, messages), "n_predict": max_tokens, "temperature": 0, "seed": SEED,
            "json_schema": schema, "stop": STOPS[model], "cache_prompt": True}
    port = (CUDA_PORT if BACKEND == "llamacpp_cuda" else LLAMACPP_PORT)[model]
    url = f"http://localhost:{port}/completion"
    timeout = 240 if BACKEND == "llamacpp_cuda" else 900
    for attempt in range(6):
        try:
            r = requests.post(url, json=body, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            return j["content"], j.get("tokens_evaluated"), j.get("tokens_predicted")
        except Exception as e:  # noqa: BLE001
            err = repr(e)
            time.sleep(5 * (attempt + 1))
    return None, None, err


def call_llamacpp(model, prompt, schema, max_tokens):
    return call_llamacpp_messages(model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                                  schema, max_tokens)


def raw_path(model, dataset, cell_name):
    d = os.path.join(ROOT, "results", "raw", model.replace(":", "_").replace("/", "_"), dataset)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{cell_name}.jsonl")


def load_done(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            rec = json.loads(line)
            out[rec["sid"]] = rec
    return out


def run_cell(model, dataset, cell_name, n, caller, workers, tag=""):
    cell = CELLS[cell_name]
    slates = json.load(open(os.path.join(ROOT, "data", f"slates_{dataset}.json")))[:n]
    path = raw_path(model, dataset, tag + cell_name)
    done = load_done(path)
    base_done = load_done(raw_path(model, dataset, AUDIT_SOURCE[cell_name])) if cell["prompt"] == "audit" else None
    jobs = []
    for s in slates:
        if s["sid"] in done:
            continue
        cands = present(s, dataset, cell)
        if len({c["group"] for c in cands}) < 2:
            continue                                   # constraint undefined with one group
        if cell["prompt"] == "audit":
            b = base_done.get(s["sid"])
            if b is None or not b.get("valid"):
                continue
            prompt, fl = build_audit_prompt(cands, cell, b["ranking"])
            jobs.append((s["sid"], cands, prompt, fl, AUDIT_SCHEMA, 700, b["ranking"]))
        else:
            if cell["prompt"] == "count" and [c["group"] for c in cands].count(count_spec(cands)["count_group"]) < COUNT_MIN:
                continue                               # count constraint infeasible
            if cell["prompt"] in ("reason", "paraphrase", "minbudget", "history", "countm"):
                if cell["prompt"] == "history" and dataset not in HISTORY:
                    HISTORY[dataset] = json.load(open(os.path.join(ROOT, "data", f"history_{dataset}.json")))
                prompt, fl = build_prompt_v2(cands, cell, s["sid"], dataset)
                schema = REASON_SCHEMA if cell["prompt"] == "reason" else RANK_SCHEMA
                budget = 1200 if cell["prompt"] == "reason" else 60 + 8 * cell["k"]
                jobs.append((s["sid"], cands, prompt, fl, schema, budget, None))
                continue
            prompt, fl = build_prompt(cands, cell)
            jobs.append((s["sid"], cands, prompt, fl, RANK_SCHEMA, 60 + 8 * cell["k"], None))

    def work(job):
        sid, cands, prompt, fl, schema, mx, given = job
        t = time.time()
        content, ptok, otok = caller(model, prompt, schema, mx)
        rec = {"sid": sid, "model": model, "backend": BACKEND, "dataset": dataset, "cell": cell_name, **cell,
               "cands": cands, "floors": fl, "raw": content, "prompt_tokens": ptok,
               "output_tokens": otok, "seconds": round(time.time() - t, 2)}
        if content is None:
            rec["error"] = otok
            return rec
        try:
            obj = json.loads(answer_part(content) if model in JINJA_PORT else content)
        except Exception:  # noqa: BLE001
            obj = {}
        if not isinstance(obj, dict):
            obj = {}
        ranking = given if given is not None else obj.get("ranking")
        ids = sorted(c["cid"] for c in cands)
        rec["valid"] = isinstance(ranking, list) and sorted(map(str, ranking)) == ids
        rec["ranking"] = ranking
        rec["claim"] = obj.get("satisfied")
        if "work" in obj:
            rec["work"] = obj["work"]
        rec["confidence"] = obj.get("confidence")
        return rec

    with cf.ThreadPoolExecutor(workers) as ex, open(path, "a") as f:
        futures = [ex.submit(work, j) for j in jobs]           # write each result as soon as it is ready
        for i, fut in enumerate(cf.as_completed(futures)):
            rec = fut.result()
            if "error" in rec:
                print("ERROR", rec["sid"], rec["error"], flush=True)
                continue
            f.write(json.dumps(rec) + "\n"); f.flush()
            if i % 50 == 0:
                print(time.strftime("%H:%M:%S"), model, dataset, cell_name, i, len(jobs), rec["seconds"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cells", default="base")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--backend", default="ollama", choices=["ollama", "vllm", "llamacpp", "llamacpp_cuda", "llamacpp_dgx"])
    ap.add_argument("--tag", default="", help="suffix for the output file (fidelity checks)")
    a = ap.parse_args()
    BACKEND = a.backend
    caller = {"ollama": call_ollama, "vllm": call_vllm, "llamacpp": call_llamacpp,
              "llamacpp_cuda": call_llamacpp, "llamacpp_dgx": call_llamacpp}[a.backend]
    for c in (list(CELLS) if a.cells == "all" else a.cells.split(",")):
        run_cell(a.model, a.dataset, c, a.n, caller, a.workers, a.tag)
