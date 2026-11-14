#!/bin/bash
# Llama-3.3-70B-Instruct on the DGX Spark llama.cpp endpoint (forwarded local port), available 09:00-20:55.
# Resumable: rerun after an interruption.
cd "$(dirname "$0")"
PY=../venv/bin/python
W=${W:-2}
$PY llm_rerank.py --model llama3.3:70b --dataset ml1m --cells base --n 150 --backend llamacpp_dgx --workers $W
$PY ptrue.py --model llama3.3:70b --workers $W --only base
$PY llm_rerank.py --model llama3.3:70b --dataset ml1m --cells G5 --n 60 --backend llamacpp_dgx --workers $W
$PY ptrue.py --model llama3.3:70b --workers $W --only G5
$PY llm_rerank.py --model llama3.3:70b --dataset ml1m --cells audit --n 60 --backend llamacpp_dgx --workers $W
$PY ptrue.py --model llama3.3:70b --workers $W --only audit
echo STRONG_DONE llama70
