#!/usr/bin/env bash
# Rank runs by mean CV AUC (highest first), to pick the best params.
# Usage: bash scripts/rank.sh [dir]   (default: runs/sweep)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
base="${1:-runs/sweep}"

for d in "$base"/*/; do
  m=$(python3 -c "import json;print(json.load(open('$d/metrics.json'))['mean'])" 2>/dev/null)
  printf '%-26s %s\n' "$(basename "$d")" "${m:-NA}"
done | sort -k2 -rn
