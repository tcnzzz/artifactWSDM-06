"""Backend fidelity: outputs from the shared Ollama server against the dedicated llama.cpp server
(same GGUF weights, Ollama chat template) on the same slates (default ML-1M cell, and the
three-group cell for Qwen, whose Ollama run was stopped part-way and redone on llama.cpp).
Writes results/backend_fidelity.json."""
import json
import os

from fairlib import satisfied

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = {}
# (model, file from backend 1, file from backend 2): Ollama vs local llama.cpp (Metal), and
# local llama.cpp vs llama.cpp built with CUDA on the rented H100 slice.
PAIRS = [("llama3.1_8b", "fidelity_ollama_base", "base", "ollama_vs_metal"),
         ("gemma2_9b", "fidelity_ollama_base", "base", "ollama_vs_metal"),
         ("qwen2.5_7b", "base", "fidelity_llamacpp_base", "ollama_vs_metal"),
         ("qwen2.5_7b", "fidelity_ollama_G3", "G3", "ollama_vs_cuda"),
         ("llama3.1_8b", "base", "fidelity_cuda_base", "metal_vs_cuda"),
         ("gemma2_9b", "base", "fidelity_cuda_base", "metal_vs_cuda"),
         ("llama3.1_8b", "fidelity_local_history", "history", "metal_vs_cuda"),
         ("qwen2.5_7b", "base", "fidelity_cuda_base", "ollama_vs_cuda")]
for m, ollama_file, llamacpp_file, kind in PAIRS:
    d = os.path.join(ROOT, "results", "raw", m, "ml1m")
    if not os.path.exists(os.path.join(d, llamacpp_file + ".jsonl")):
        continue
    o = {json.loads(line)["sid"]: json.loads(line) for line in open(os.path.join(d, ollama_file + ".jsonl"))}
    c = {json.loads(line)["sid"]: json.loads(line) for line in open(os.path.join(d, llamacpp_file + ".jsonl"))}
    common = [s for s in o if s in c]

    def viol(r):
        by = {x["cid"]: x["group"] for x in r["cands"]}
        return not satisfied([by[i] for i in r["ranking"]], r["floors"])
    common = [s for s in common if o[s].get("valid") and c[s].get("valid")]
    out[f"{m}/{llamacpp_file}"] = {"kind": kind, "slates": len(common),
              "same_ranking": sum(o[s]["ranking"] == c[s]["ranking"] for s in common),
              "same_violation": sum(viol(o[s]) == viol(c[s]) for s in common),
              "same_claim": sum(o[s]["claim"] == c[s]["claim"] for s in common),
              "same_confidence": sum(o[s]["confidence"] == c[s]["confidence"] for s in common),
              "violations_1": sum(viol(o[s]) for s in common), "violations_2": sum(viol(c[s]) for s in common)}
json.dump(out, open(os.path.join(ROOT, "results", "backend_fidelity.json"), "w"), indent=1)
print(out)
