#!/bin/bash
# Prioritised sweep per model on its dedicated llama.cpp server (resumable).
# Cells and slate counts are trimmed for Llama and Gemma to fit the shared machine (see STATUS.md).
# Usage: ./run_queue.sh <model>
cd "$(dirname "$0")"
M=$1
W=${W:-3}
PY=../venv/bin/python
B="--backend llamacpp --workers $W"
while pgrep -f "llm_rerank.py --model $M " > /dev/null || pgrep -f "llm_feedback.py --model $M " > /dev/null; do sleep 30; done
if [ "$M" == "qwen2.5:7b" ]; then
  $PY llm_rerank.py --model $M --dataset ml1m --cells reason,history,paraphrase,minbudget,countm --n 200 $B
  $PY llm_rerank.py --model $M --dataset mind --cells base,rho050,rho070,rho080,rho095,befair --n 200 $B
  $PY llm_rerank.py --model $M --dataset mind --cells history --n 150 $B
  $PY llm_rerank.py --model $M --dataset ml1m --cells G3,G5 --n 200 $B
  $PY llm_rerank.py --model $M --dataset mind --cells audit,count --n 150 $B
  $PY llm_rerank.py --model $M --dataset mind --cells reason --n 100 $B
else
  $PY llm_rerank.py --model $M --dataset ml1m --cells base,rho050,rho070,rho080,rho095,befair --n 300 $B
  $PY llm_rerank.py --model $M --dataset ml1m --cells history,paraphrase,minbudget,countm --n 150 $B
  $PY llm_rerank.py --model $M --dataset ml1m --cells reason --n 100 $B
  $PY llm_rerank.py --model $M --dataset mind --cells base,rho095,befair --n 150 $B
  $PY llm_rerank.py --model $M --dataset ml1m --cells audit --n 150 $B
  $PY llm_rerank.py --model $M --dataset mind --cells rho080 --n 150 $B
  $PY llm_rerank.py --model $M --dataset ml1m --cells G5 --n 100 $B
  $PY llm_rerank.py --model $M --dataset mind --cells audit --n 100 $B
fi
echo QUEUE_DONE $M
