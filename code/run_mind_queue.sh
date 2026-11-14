#!/bin/bash
# All MIND cells (final slates, data/slates_mind.json) for one model on the CUDA llama.cpp server.
# Usage: ./run_mind_queue.sh <model> <workers>
cd "$(dirname "$0")"
M=$1
W=${2:-4}
PY=../venv/bin/python
B="--backend llamacpp_cuda --workers $W"
$PY llm_rerank.py --model $M --dataset mind --cells base,rho095,rho080,befair,history,rho070,rho050 --n 200 $B
$PY llm_rerank.py --model $M --dataset mind --cells audit,count --n 200 $B
$PY llm_rerank.py --model $M --dataset mind --cells reason --n 100 $B
echo MIND_QUEUE_DONE $M
