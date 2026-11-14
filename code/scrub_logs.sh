#!/bin/bash
# Remove local paths and the GPU host address from run logs before the artifact is shared.
set -eu
cd "$(dirname "$0")/.."
find . -type f \( -name '*.log' -o -name '*.time' \) -not -path './.venv/*' -print0 \
  | xargs -0 perl -pi -e 's{/[U]sers/[^\s"\x27]+}{~}g; s{/[h]ome/[^\s"\x27]+}{~}g; s{103[.]73[.]\d+[.]\d+}{<gpu-host>}g'
remaining=$(grep -rlaE '/[U]sers/|/[h]ome/|103[.]73[.]' --exclude-dir=.venv . | wc -l)
echo "files still containing local paths or the GPU host: $remaining"
test "$remaining" -eq 0
