# Tuning recipe

How we ran the experiment: define a hyperparameter grid, sweep it at reduced repeats,
rank to pick the best params per model, confirm with full 5x20 CV, study the epoch
dynamics with learning curves, then train the chosen model on all data and submit.

Run every script with `bash scripts/<name>.sh`, never with `source` or `.`
(they set `cd` and shell options that would leak into your shell).

Scripts:

- `sweep.sh`           grid-search all models at reduced repeats (5x5)
- `rank.sh`            rank runs by mean CV AUC; takes an optional dir (default runs/sweep)
- `cv_full.sh`         full 5x20 CV on each model's best config
- `learning_curves.sh` train/val curves for the 4 neural models (TensorBoard)
- `train_eval.sh`      train the chosen model on all data, write the submission

## 0. Environment (once)

Python 3.11 or newer. On a CUDA VM install with the cu124 extra:

```bash
uv pip install --system -e ".[cu124]"
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

## 1. Data and feature cache (once)

Put the Kaggle files at data/facebook-recruiting-iv-human-or-bot/{bids,train,test}.csv, then:

```bash
bidbot features build
```

## 2. Define the grid and sweep

Edit the grids in `sweep.sh`: each line is one config (model name + flags), and `R`
at the top sets the CV repeats for the search (5). Then run it:

```bash
bash scripts/sweep.sh
```

Writes runs/sweep/<config>/ (results + run.log) for every grid point. Reduced repeats
keep the search fast; the full protocol comes in step 4.

## 3. Rank and pick the best params per model

```bash
bash scripts/rank.sh
```

Lists every config by mean CV AUC. Take the top config for each model. Best from this
run:

| model  | best config flags                                 |
| ------ | ------------------------------------------------- |
| tffm   | --model.channels 256                              |
| gbm    | --model.learning-rate 0.03 --model.max-depth 4    |
| hybrid | --model.hidden 256                                |
| gnn    | --model.hidden 256                                |
| resnet | --model.lr 3e-4                                   |

Put these into `cv_full.sh` (and later into `train_eval.sh` for the winner).

## 4. Full 5x20 CV on the best params

```bash
bash scripts/cv_full.sh
bash scripts/rank.sh runs/cv_full
```

Runs each model's best config at the reportable 5x20 = 100-fold protocol. Confirm the
overall winner here, since the reduced-repeat ranking can shuffle for close configs.
Report this AUC, not step 2's. Result of this run:

| model  | mean AUC |
| ------ | -------- |
| tffm   | 0.9179   |
| gbm    | 0.8922   |
| hybrid | 0.8636   |
| gnn    | 0.8195   |
| resnet | 0.7332   |

Winner: tffm.

## 5. Learning curves: understand the epoch dynamics, pick the epoch

```bash
bash scripts/learning_curves.sh
tensorboard --logdir runs/curves --port 6006
```

Trains the 4 neural models with their best params on a held-out val split, early
stopping off, for many epochs. Each metric (auc, loss) is one TensorBoard chart with
train and val overlaid, so you can see where the model plateaus or starts overfitting.
Read the winner's val curve and choose an epoch cap for the final fit (we picked 20 for
tffm). Diagnostic only (val split, no checkpoint). gbm is excluded: its progression is
over boosting iterations, not epochs.

## 6. Update train_eval.sh and run the final train

Set the winner's model name, best flags, and chosen epochs in `train_eval.sh` (this run:
tffm, `--model.channels 256 --model.epochs 20`). Then:

```bash
bash scripts/train_eval.sh
```

Trains the winner on all labelled data (--val-fraction 0), evals it, and writes
runs/final/tffm/submission.csv. Upload that to Kaggle.

## Notes

- Device: neural models (tffm, hybrid, gnn, resnet) train on cuda; gbm is CPU-only.
  The scripts set the right flags per model. eval takes the device from the checkpoint.
- Logs: each run tees to run.log inside its own output folder. Tail with
  `tail -f runs/sweep/<config>/run.log`.
- Long runs: use tmux or nohup so they survive an SSH drop.
- The full-fit run (--val-fraction 0) logs only train curves to TensorBoard: val would
  alias train there and be misleading. The reportable number is the step 4 full-CV AUC
  (tffm: 0.9179), not the training-fit AUC printed by the final train.
