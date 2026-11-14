#!/bin/bash
# Local sweep on the shared Ollama server. One model at a time, all of its cells, so that the
# model stays loaded. Core tightness sweep on MovieLens-1M uses the first 300 sampled slates,
# every other cell the first 200.
cd "$(dirname "$0")"
W=${W:-3}
PY=../venv/bin/python
MODELS=${MODELS:-"qwen2.5:7b llama3.1:8b gemma2:9b"}
CORE=base,rho050,rho070,rho080,rho095,befair
while pgrep -f "llm_rerank.py" > /dev/null; do sleep 20; done      # let a running cell finish
for M in $MODELS; do
  $PY llm_rerank.py --model $M --dataset ml1m --cells $CORE --n 300 --workers $W
  $PY llm_rerank.py --model $M --dataset mind --cells $CORE --n 200 --workers $W
  $PY llm_rerank.py --model $M --dataset ml1m --cells audit,count --n 200 --workers $W
  $PY llm_rerank.py --model $M --dataset mind --cells audit,count --n 200 --workers $W
  $PY llm_rerank.py --model $M --dataset ml1m --cells k05,k20,G3,G5,order_random,order_reverse --n 200 --workers $W
  for MODE in retry memory; do $PY llm_feedback.py --model $M --dataset ml1m --mode $MODE --n 200; done
  $PY llm_rerank.py --model $M --dataset ml1m --cells audit_rho080 --n 200 --workers $W
done
