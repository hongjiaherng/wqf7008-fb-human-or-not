#!/usr/bin/env bash
# Learning curves for the 4 neural models (tffm, hybrid, gnn, resnet): train each
# with its best params on a held-out val split, early stopping disabled, for many
# epochs. Logs train and val (auc and loss) to TensorBoard so you can see where each
# model plateaus / overfits.
#
#   bash scripts/learning_curves.sh
#   tensorboard --logdir runs/curves --port 6006
#
# In TensorBoard each metric (auc, loss) is one chart with train and val overlaid.
# This is a diagnostic: it uses a val split (not all data) and saves no checkpoint,
# so it is not the submission fit.
#
# gbm is excluded on purpose: it is gradient boosting, so its progression is over
# boosting iterations (~499 steps), not epochs, and is not comparable on this axis.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TQDM_DISABLE=1
echo "writing curves under: $(pwd)/runs/curves"

# run_curve <model> [flags...]  ->  train with val split, output to runs/curves/<model>/
run_curve() {
  local model=$1 dir="runs/curves/$1"; shift
  mkdir -p "$dir"
  bidbot train "$model" "$@" --val-fraction 0.2 --out "$dir" 2>&1 | tee "$dir/run.log"
}

# best params per model (same as train_eval.sh); high epochs + patience so early
# stopping does not cut the curve short.
run_curve tffm   --model.device cuda --model.batch-size 256 --model.channels 256 --model.epochs 100 --model.early-stop-patience 100
run_curve hybrid --model.device cuda --model.batch-size 256 --model.hidden 256 --model.epochs 100 --model.early-stop-patience 100
run_curve gnn    --model.device cuda --model.hidden 256 --model.epochs 100 --model.early-stop-patience 100
run_curve resnet --model.device cuda --model.batch-size 256 --model.lr 3e-4 --model.epochs 100 --model.early-stop-patience 100

echo "done. view with: tensorboard --logdir runs/curves --port 6006"
