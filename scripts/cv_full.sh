#!/usr/bin/env bash
# Full 5x20 CV (the reportable protocol) on each model's best sweep config.
# Run after the reduced-repeat sweep, before train_eval.sh. Then re-rank with:
#   bash scripts/rank.sh runs/cv_full
# and pick the overall best for the final submission.
#
# Configs below are the sweep winners; edit if you change the grids.
# Neural models get --model.device cuda; batched ones get --model.batch-size 256.
# gbm is CPU. Each config writes to runs/cv_full/<model>/ (results + run.log).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TQDM_DISABLE=1
echo "writing results + logs under: $(pwd)/runs/cv_full"
R=20

# cv <model> [flags...]  ->  full 5x20 CV into runs/cv_full/<model>/
cv() {
  local model=$1 dir="runs/cv_full/$1"; shift
  mkdir -p "$dir"
  bidbot cv "$model" --cv.n-repeats $R "$@" --out "$dir" 2>&1 | tee "$dir/run.log"
}

cv gbm    --model.learning-rate 0.03 --model.max-depth 4
cv tffm   --model.device cuda --model.batch-size 256 --model.channels 256
cv hybrid --model.device cuda --model.batch-size 256 --model.hidden 256
cv gnn    --model.device cuda --model.hidden 256
cv resnet --model.device cuda --model.batch-size 256 --model.lr 3e-4

echo "done. rank with: bash scripts/rank.sh runs/cv_full"
