#!/bin/bash
# Round 2: stronger models on ML-1M (default cell, infeasible five-group cell, audit).
# Usage: ./run_strong.sh <model> <backend> <workers> [n]
cd "$(dirname "$0")"
M=$1; BK=$2; W=${3:-3}; N=${4:-300}
PY=../venv/bin/python
$PY llm_rerank.py --model $M --dataset ml1m --cells base --n $N --backend $BK --workers $W
$PY llm_rerank.py --model $M --dataset ml1m --cells G5 --n $(( N < 200 ? N : 200 )) --backend $BK --workers $W
$PY llm_rerank.py --model $M --dataset ml1m --cells audit --n $(( N < 200 ? N : 200 )) --backend $BK --workers $W
echo STRONG_DONE $M
