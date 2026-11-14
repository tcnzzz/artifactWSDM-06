"""Two prompt-level repair baselines on the base cell (both need the numeric check).

retry  : per-slate numeric feedback. After a violating output the model is shown the
         measured exposure shares and asked again, up to 3 retries.
memory : FACTER-style reimplementation adapted to an exact numeric score. Slates are
         processed as a stream; each detected violation is pushed into a FIFO memory
         and the most recent ones are appended to later prompts as AVOID examples.
         The violating output itself is served unchanged, as in FACTER.
Usage: python llm_feedback.py --model qwen2.5:7b --dataset ml1m --mode retry
"""
import argparse
import collections
import json
import os
import time

from fairlib import satisfied, shares
from llm_rerank import CELLS, RANK_SCHEMA, ROOT, SYSTEM, SEED, OLLAMA, VLLM_URL, build_prompt, call_llamacpp_messages, load_done, present, raw_path

import requests

MAX_RETRY = 3
MEMORY_SIZE = 50      # FIFO size used by FACTER
MEMORY_SHOWN = 5      # most recent entries placed in the prompt (not specified by FACTER)


BACKEND = "ollama"


def chat(model, messages, max_tokens):
    if BACKEND in ("llamacpp", "llamacpp_cuda"):
        import llm_rerank
        llm_rerank.BACKEND = BACKEND
        return call_llamacpp_messages(model, messages, RANK_SCHEMA, max_tokens)[0]
    if BACKEND == "vllm":
        body = {"model": model, "temperature": 0, "seed": SEED, "max_tokens": max_tokens, "messages": messages,
                "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": RANK_SCHEMA}}}
        if "gemma" in model:      # no system role: fold it into the first user turn
            body["messages"] = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}] + messages[2:]
        url = VLLM_URL
    else:
        body = {"model": model, "stream": False, "format": RANK_SCHEMA, "messages": messages,
                "options": {"temperature": 0, "seed": SEED, "num_predict": max_tokens}}
        url = OLLAMA
    for attempt in range(4):
        try:
            r = requests.post(url, json=body, timeout=900)
            r.raise_for_status()
            j = r.json()
            return j["choices"][0]["message"]["content"] if BACKEND == "vllm" else j["message"]["content"]
        except Exception:  # noqa: BLE001
            time.sleep(5 * (attempt + 1))
    return None


def parse(content, cands):
    try:
        obj = json.loads(content)
    except Exception:  # noqa: BLE001
        obj = {}
    rk = obj.get("ranking")
    valid = isinstance(rk, list) and sorted(map(str, rk)) == sorted(c["cid"] for c in cands)
    return rk, valid, obj.get("satisfied"), obj.get("confidence")


def share_text(rk, cands, fl):
    by = {c["cid"]: c["group"] for c in cands}
    sh = shares([by[c] for c in rk])
    return "; ".join(f"group {g}: share {sh.get(g, 0):.3f} (required at least {f:.3f})" for g, f in fl.items())


def main(model, dataset, mode, n):
    cell = CELLS["base"]
    slates = json.load(open(os.path.join(ROOT, "data", f"slates_{dataset}.json")))[:n]
    path = raw_path(model, dataset, f"feedback_{mode}")
    done = load_done(path)
    memory = collections.deque(maxlen=MEMORY_SIZE)
    with open(path, "a") as f:
        for s in slates:
            cands = present(s, dataset, cell)
            if len({c["group"] for c in cands}) < 2:
                continue
            by = {c["cid"]: c["group"] for c in cands}
            prompt, fl = build_prompt(cands, cell)
            if s["sid"] in done:
                if mode == "memory" and done[s["sid"]].get("memory_push"):
                    memory.append(done[s["sid"]]["memory_push"])
                continue
            rec = {"sid": s["sid"], "model": model, "backend": BACKEND, "dataset": dataset, "cell": f"feedback_{mode}", **cell,
                   "cands": cands, "floors": fl, "rounds": []}
            if mode == "memory" and memory:
                prompt += ("\n\nAVOID repeating these earlier violations of the exposure constraint "
                           "(group pattern from position 1 downward, then the measured shares):\n" +
                           "\n".join(list(memory)[-MEMORY_SHOWN:]))
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            for rnd in range(1 + (MAX_RETRY if mode == "retry" else 0)):
                content = chat(model, msgs, 60 + 8 * cell["k"])
                rk, valid, claim, conf = parse(content, cands)
                viol = (not satisfied([by[c] for c in rk], fl)) if valid else None
                rec["rounds"].append({"raw": content, "ranking": rk, "valid": valid, "claim": claim,
                                      "confidence": conf, "violated": viol})
                if not valid or not viol:
                    break
                if mode == "memory":
                    rec["memory_push"] = "".join(by[c] for c in rk) + " -> " + share_text(rk, cands, fl)
                    memory.append(rec["memory_push"])
                    break
                msgs += [{"role": "assistant", "content": content},
                         {"role": "user", "content": "A numeric check found that this ranking violates the "
                          "exposure constraint. Measured: " + share_text(rk, cands, fl) +
                          ". Produce a corrected ranking in the same JSON format."}]
            last = rec["rounds"][-1]
            rec.update(valid=last["valid"], ranking=last["ranking"], claim=last["claim"], confidence=last["confidence"])
            f.write(json.dumps(rec) + "\n"); f.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="ml1m")
    ap.add_argument("--mode", required=True, choices=["retry", "memory"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--backend", default="ollama", choices=["ollama", "vllm", "llamacpp", "llamacpp_cuda"])
    a = ap.parse_args()
    BACKEND = a.backend
    main(a.model, a.dataset, a.mode, a.n)
