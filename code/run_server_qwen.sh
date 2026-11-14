#!/bin/bash
# Qwen cells that were not run locally, on the CUDA llama.cpp server (port 11573).
cd "$(dirname "$0")"
PY=../venv/bin/python
B="--backend llamacpp_cuda --workers 4"
$PY llm_rerank.py --model qwen2.5:7b --dataset ml1m --cells base --n 100 $B --tag fidelity_cuda_
$PY llm_rerank.py --model qwen2.5:7b --dataset mind --cells history --n 200 $B
$PY llm_rerank.py --model qwen2.5:7b --dataset ml1m --cells G3,G5 --n 200 $B
$PY llm_rerank.py --model qwen2.5:7b --dataset mind --cells audit,count --n 200 $B
$PY llm_rerank.py --model qwen2.5:7b --dataset mind --cells reason --n 100 $B
echo SERVER_QUEUE_DONE qwen
