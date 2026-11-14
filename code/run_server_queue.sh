#!/bin/bash
# Remaining Llama/Gemma cells on the CUDA llama.cpp servers of the rented H100 slice, reached
# on a remote CUDA server (ports 11571/11572). Same GGUF files; fidelity in results/backend_fidelity.json.
# Every cell listed here is produced entirely by this backend (field "backend": "llamacpp_cuda").
# Usage: ./run_server_queue.sh <model> <workers>
cd "$(dirname "$0")"
M=$1
W=${2:-4}
PY=../venv/bin/python
B="--backend llamacpp_cuda --workers $W"
while pgrep -f "llm_rerank.py --model $M " > /dev/null; do sleep 30; done
$PY llm_rerank.py --model $M --dataset ml1m --cells history,paraphrase,minbudget,countm --n 200 $B
$PY llm_rerank.py --model $M --dataset mind --cells base,rho050,rho070,rho080,rho095,befair --n 200 $B
$PY llm_rerank.py --model $M --dataset ml1m --cells reason --n 200 $B
$PY llm_rerank.py --model $M --dataset mind --cells history --n 200 $B
$PY llm_rerank.py --model $M --dataset ml1m --cells audit,count,G3,G5 --n 200 $B
$PY llm_rerank.py --model $M --dataset mind --cells audit,count --n 200 $B
$PY llm_rerank.py --model $M --dataset ml1m --cells k05,k20,order_random,order_reverse,audit_rho080 --n 200 $B
for MODE in retry memory; do $PY llm_feedback.py --model $M --dataset ml1m --mode $MODE --n 200 --backend llamacpp_cuda; done
$PY llm_rerank.py --model $M --dataset mind --cells reason --n 100 $B
echo SERVER_QUEUE_DONE $M
