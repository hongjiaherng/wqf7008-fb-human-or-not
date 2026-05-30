# WQF7008 Practical Deep Learning Project

Facebook Recruiting IV: Human or Robot?

This repository trains and evaluates bidder-classification models for the Kaggle
competition dataset. It includes feature engineering, repeated cross-validation,
final model fitting, checkpoint saving, and Kaggle-style prediction generation.

## Results

For this Kaggle-style task, ROC-AUC is the main selection metric. The selected
final model is therefore `tffm` (FT-Transformer).

Summary of the full 5-fold x 20-repeat CV runs:

| Model | Mean CV AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| `tffm` | `0.917933` | `0.256560` | `0.854369` | `0.394619` |
| `gbm` | `0.892172` | `0.462810` | `0.543689` | `0.500000` |
| `hybrid` | `0.863629` | `0.170132` | `0.873786` | `0.284810` |
| `gnn` | `0.819506` | `0.182448` | `0.766990` | `0.294776` |
| `resnet` | `0.733182` | `0.068122` | `1.000000` | `0.127554` |

## Setup

### 1. Install `uv`

This project uses [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
to create the Python environment and install dependencies.

### 2. Create the Environment

The project requires Python 3.12 or newer. Use one of the following commands
from the repository root.

CPU environment:

```bash
uv sync --dev --extra cpu
```

CUDA 13 environment:

```bash
uv sync --dev --extra cu130
```

### 3. Activate the Environment

Windows bash:

```bash
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

After activation, the CLI command is:

```bash
bidbot --help
```

### 4. Data Setup

Refer to the [data folder](data/README.md).

## Example Usage

The pipeline runs in order: build features, smoke test (optional), run cv to select
a model, fit the final model on all labelled data, then generate predictions.

### 1. Build Features

Only applicable to `gbm`, `tffm`, `hybrid`, `resnet` models. Run this once after placing the dataset:

```bash
bidbot features build
```

This creates cached tabular feature files under:

```text
runs/cache/
```

To force a rebuild:

```bash
bidbot features build --force
```

### 2. Runtime Check

Before running long experiments, use quick CV to confirm the environment,
dataset, and GPU/CPU setup work:

```bash
bidbot cv tffm --cv.quick
```

This uses a small 2-fold CV with 1 repeat and trains for 1 epoch (except `gbm`).

### 3. Cross-Validation Evaluation

The default evaluation protocol is:

```text
Repeated stratified 5-fold cross-validation, repeated 20 times
```

This produces 100 validation evaluations per model, i.e., split data into 5 folds (using a random seed), train on 4 folds and validate on the remaining 1 fold, then repeat this process 20 times with different random splits.

Run all model evaluations:

```bash
bidbot cv gbm    --out runs/cv/gbm_full
bidbot cv tffm   --out runs/cv/tffm_full
bidbot cv hybrid --out runs/cv/hybrid_full
bidbot cv gnn    --out runs/cv/gnn_full
bidbot cv resnet --out runs/cv/resnet_full
```

See [CV Output Files](#cv-output-files) for what each run writes.

### 4. Train and Monitor Learning Curves

`bidbot train <model>` fits on an 80/20 stratified split by default
(`--val-fraction 0.2`) and logs training curves to TensorBoard under `<out>/tb`.

```bash
# Train with the default 80 train / 20 val split (writes scalars under <out>/tb).
bidbot train tffm --out runs/train/tffm_dev

# In a second terminal, launch TensorBoard and open the printed URL (default :6006).
tensorboard --logdir runs/train
```

### 5. Final Model Fit

After selecting the model using CV/OOF results, retrain the chosen model on all
labelled training data and save a checkpoint.

For the selected TFFM model:

```bash
bidbot train tffm --val-fraction 0 --save-model --out runs/train/tffm_full_fit
```

Important reporting note:

- `--val-fraction 0` trains using all labelled data.
- The AUC written for this run is a training/self-validation AUC.
- The full-fit AUC DO NOT represent generalization performance.
- Report CV/OOF metrics from the `bidbot cv ...` runs instead.

### 6. Generate Predictions

Use the saved checkpoint to create `submission.csv`:

```bash
bidbot eval tffm --ckpt runs/train/tffm_full_fit/ckpt.pt --out runs/submit/tffm_final/submission.csv
```

The output CSV has the Kaggle submission format:

```text
bidder_id,prediction
...
```

## CLI Reference

### Command Overview

The project exposes one CLI command, `bidbot`, with four top-level subcommands:

```text
bidbot features build
bidbot train {gbm|hybrid|gnn|tffm|resnet}
bidbot cv    {gbm|hybrid|gnn|tffm|resnet}
bidbot eval  {gbm|hybrid|gnn|tffm|resnet}
```

Use `--help` to inspect options:

```bash
bidbot --help
bidbot cv tffm --help
bidbot train tffm --help
```

Common flag namespaces:

| Namespace | Purpose | Examples |
| --- | --- | --- |
| `--model.<field>` | Model and training parameters | `--model.lr`, `--model.epochs`, `--model.device` |
| `--cv.<field>` | Cross-validation protocol | `--cv.n-splits`, `--cv.n-repeats`, `--cv.quick` |
| `--data.<field>` | Data/cache paths | `--data.data-dir`, `--data.cache-dir` |
| Flat flags | Run-level options | `--out`, `--val-fraction`, `--quick`, `--save-model` |

### Model Architectures

The available model names are:

| CLI name | Model |
| --- | --- |
| `gbm` | Gradient Boosting Machine baseline |
| `tffm` | FT-Transformer tabular model |
| `hybrid` | Tabular plus bid-sequence BiLSTM hybrid |
| `gnn` | Heterogeneous graph neural network |
| `resnet` | Tabular ResNet baseline |

### CV Output Files

Each `bidbot cv ...` run writes:

| File | Meaning |
| --- | --- |
| `metrics.json` | Mean/std/q25/q10 AUC plus OOF threshold metrics |
| `folds.csv` | Per-fold AUC values |
| `oof_predictions.csv` | One row per validation appearance |
| `oof_by_bidder.csv` | Bidder-level OOF probabilities averaged across repeats |
| `roc.png` | ROC plot |

The threshold metrics are computed from bidder-averaged OOF probabilities using
a threshold of `0.5`. They include Precision, Recall, F1, and the confusion
matrix.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `bidbot` is not recognized | Activate `.venv`, or use `.\.venv\Scripts\bidbot.exe` on Windows |
| Dataset file not found | Check that `bids.csv`, `train.csv`, and `test.csv` are under `data/facebook-recruiting-iv-human-or-bot/` |
| CUDA is not detected | Run `uv sync --dev --extra cu130`, then check `torch.cuda.is_available()` |
| Matplotlib/backend error | Set `$env:MPLBACKEND = "Agg"` before running |
| Full CV takes a long time | Use `--cv.quick` first; full 5-fold x 20-repeat CV is intentionally more expensive |

## Development

Run these before submitting code changes:

```bash
ruff format src/
ruff check src/
ty check src/
```

## References

### Competition

- Kaggle: [Facebook Recruiting IV: Human or Robot?](https://kaggle.com/competitions/facebook-recruiting-iv-human-or-bot)

```bibtex
@misc{facebook-recruiting-iv-human-or-bot,
    author = {Jim Dullaghan and John Costella and John_W and Meghan O'Connell and Rafael and Ruchi and RuchiVarshney and Sergey and Sofus Macskassy and Wendy Kan},
    title  = {Facebook Recruiting IV: Human or Robot?},
    year   = {2015},
    howpublished = {\url{https://kaggle.com/competitions/facebook-recruiting-iv-human-or-bot}},
    note   = {Kaggle}
}
```

### Top Solution References

Scores are ROC-AUC values from the Kaggle leaderboard.

| Rank | Score, private / public | Write-up |
| --- | --- | --- |
| 1st | `0.94254 / 0.91946` | [Forum comment by the winner](https://www.kaggle.com/competitions/facebook-recruiting-iv-human-or-bot/writeups/small-yellow-duck-share-your-secret-sauce#81331) |
| 2nd | `0.94167 / 0.93277` | [small-yellow-duck: Share your secret sauce](https://www.kaggle.com/competitions/facebook-recruiting-iv-human-or-bot/writeups/small-yellow-duck-share-your-secret-sauce) |
| 3rd | `0.94113 / 0.93321` | [Forum comment by mechatroner](https://www.kaggle.com/competitions/facebook-recruiting-iv-human-or-bot/writeups/small-yellow-duck-share-your-secret-sauce#81396) |
