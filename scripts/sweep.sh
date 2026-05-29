#!/usr/bin/env bash
# Hyperparameter grids as plain bidbot commands.
# Neural models get --model.device cuda; batched ones get --model.batch-size 256.
# gbm is sklearn (CPU, no device/batch flags). gnn is full-graph (no batch flag).
# Each config writes to runs/sweep/<name>/ (results + run.log). R = CV repeats per config.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TQDM_DISABLE=1   # no tqdm progress bars (they flood the log when piped to tee)
echo "writing results + logs under: $(pwd)/runs/sweep"
R=5

# sweep <name> <model> [flags...]  ->  bidbot cv into runs/sweep/<name>/ (log: run.log)
sweep() {
  local name=$1 model=$2 dir="runs/sweep/$1"; shift 2
  mkdir -p "$dir"
  bidbot cv "$model" --cv.n-repeats $R "$@" --out "$dir" 2>&1 | tee "$dir/run.log"
}

# ---- gbm ----
sweep gbm_base         gbm
sweep gbm_lr0.03_d4    gbm --model.learning-rate 0.03 --model.max-depth 4
sweep gbm_lr0.1_d2     gbm --model.learning-rate 0.1 --model.max-depth 2
sweep gbm_n800_d4      gbm --model.n-estimators 800 --model.max-depth 4 --model.learning-rate 0.03
sweep gbm_ss0.7_leaf10 gbm --model.subsample 0.7 --model.min-samples-leaf 10

# ---- tffm ----
sweep tffm_base           tffm --model.device cuda --model.batch-size 256
sweep tffm_lr3e4          tffm --model.device cuda --model.batch-size 256 --model.lr 3e-4
sweep tffm_lr5e4          tffm --model.device cuda --model.batch-size 256 --model.lr 5e-4
sweep tffm_ch256          tffm --model.device cuda --model.batch-size 256 --model.channels 256
sweep tffm_l4             tffm --model.device cuda --model.batch-size 256 --model.num-layers 4
sweep tffm_ch256_l4_lr3e4 tffm --model.device cuda --model.batch-size 256 --model.channels 256 --model.num-layers 4 --model.lr 3e-4

# ---- hybrid ----
sweep hybrid_base         hybrid --model.device cuda --model.batch-size 256
sweep hybrid_lr5e4        hybrid --model.device cuda --model.batch-size 256 --model.lr 5e-4
sweep hybrid_h256         hybrid --model.device cuda --model.batch-size 256 --model.hidden 256
sweep hybrid_drop0.5      hybrid --model.device cuda --model.batch-size 256 --model.dropout 0.5
sweep hybrid_h256_drop0.4 hybrid --model.device cuda --model.batch-size 256 --model.hidden 256 --model.dropout 0.4

# ---- gnn ----
sweep gnn_base       gnn --model.device cuda
sweep gnn_lr5e4      gnn --model.device cuda --model.lr 5e-4
sweep gnn_h256       gnn --model.device cuda --model.hidden 256
sweep gnn_drop0.5    gnn --model.device cuda --model.dropout 0.5
sweep gnn_h256_lr5e4 gnn --model.device cuda --model.hidden 256 --model.lr 5e-4

# ---- resnet ----
sweep resnet_base       resnet --model.device cuda --model.batch-size 256
sweep resnet_lr3e4      resnet --model.device cuda --model.batch-size 256 --model.lr 3e-4
sweep resnet_h256_bh512 resnet --model.device cuda --model.batch-size 256 --model.hidden 256 --model.block-hidden 512
sweep resnet_l6         resnet --model.device cuda --model.batch-size 256 --model.n-layers 6
sweep resnet_drop0.3    resnet --model.device cuda --model.batch-size 256 --model.dropout 0.3

# After picking the best config, run the final full CV / train / eval, e.g.:
#   bidbot cv tffm --model.device cuda --model.batch-size 256 <best flags> --out runs/cv/tffm_full
#   bidbot train tffm --val-fraction 0 --save-model --model.device cuda --model.batch-size 256 <best flags> --out runs/train/tffm_full
#   bidbot eval tffm --ckpt runs/train/tffm_full/ckpt.pt --out runs/submit/tffm/submission.csv
