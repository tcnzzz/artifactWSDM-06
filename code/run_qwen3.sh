#!/bin/bash
# Qwen3-8B in thinking mode (server-side client next to the CUDA server; 16k-token budget).
cd "$(dirname "$0")"
PY=${PY:-python3}
$PY llm_rerank.py --model qwen3:8b --dataset ml1m --cells base --n 160 --backend llamacpp_cuda --workers 3
$PY llm_rerank.py --model qwen3:8b --dataset ml1m --cells G5 --n 60 --backend llamacpp_cuda --workers 3
echo STRONG_DONE qwen3
