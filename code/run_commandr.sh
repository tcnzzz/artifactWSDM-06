#!/bin/bash
# Command-R-35B (local llama.cpp, weights locked in memory), default and five-group cells.
cd "$(dirname "$0")"
PY=../venv/bin/python
$PY llm_rerank.py --model command-r:35b --dataset ml1m --cells base --n 200 --backend llamacpp --workers 2
$PY llm_rerank.py --model command-r:35b --dataset ml1m --cells G5 --n 60 --backend llamacpp --workers 2
echo STRONG_DONE command-r
