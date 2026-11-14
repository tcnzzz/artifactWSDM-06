#!/bin/bash
# After the Qwen3 cells: token probabilities on the same CUDA server (thinking traces are long).
cd "$(dirname "$0")"
while pgrep -f "llm_rerank.py --model qwen3" > /dev/null; do sleep 30; done
python3 ptrue.py --model qwen3:8b --workers 3 --backend llamacpp_cuda --only base,G5
echo PTRUE_DONE qwen3
