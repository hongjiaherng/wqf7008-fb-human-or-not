#!/usr/bin/env bash
# Train the best model (tffm) on ALL labelled data with its best params, then eval
# to a submission.csv. Upload runs/final/tffm/submission.csv to Kaggle.
#
# tffm won the full 5x20 CV (mean AUC 0.9179); epochs=20 from the learning curve.
# Neural models get --model.device cuda and --model.batch-size 256.
# eval takes the device from the checkpoint.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TQDM_DISABLE=1   # no tqdm progress bars (they flood the log when piped to tee)
echo "writing final train+eval under: $(pwd)/runs/final"

dir="runs/final/tffm"
mkdir -p "$dir"
{
  bidbot train tffm --model.device cuda --model.batch-size 256 --model.channels 256 \
    --model.epochs 20 --val-fraction 0 --save-model --out "$dir"
  bidbot eval tffm --ckpt "$dir/ckpt.pt" --out "$dir/submission.csv"
} 2>&1 | tee "$dir/run.log"

echo "done. upload $dir/submission.csv to Kaggle."
