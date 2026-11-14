#!/bin/bash
# Sweep for one model through its dedicated llama.cpp server (see LLAMACPP_PORT in llm_rerank.py).
# Same cells and slate counts as run_local.sh.
cd "$(dirname "$0")"
M=$1
W=${W:-3}
PY=../venv/bin/python
CORE=base,rho050,rho070,rho080,rho095,befair
B="--backend llamacpp --workers $W"
$PY llm_rerank.py --model $M --dataset ml1m --cells $CORE --n 300 $B
$PY llm_rerank.py --model $M --dataset mind --cells $CORE --n 200 $B
$PY llm_rerank.py --model $M --dataset ml1m --cells audit,count --n 200 $B
$PY llm_rerank.py --model $M --dataset mind --cells audit,count --n 200 $B
$PY llm_rerank.py --model $M --dataset ml1m --cells k05,k20,G3,G5,order_random,order_reverse --n 200 $B
for MODE in retry memory; do $PY llm_feedback.py --model $M --dataset ml1m --mode $MODE --n 200 --backend llamacpp; done
$PY llm_rerank.py --model $M --dataset ml1m --cells audit_rho080 --n 200 $B
echo SWEEP_DONE $M
