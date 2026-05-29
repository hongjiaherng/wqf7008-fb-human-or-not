# WQF7008 Practical Deep Learning Project

Facebook Recruiting IV: Human or Robot?

This repository trains and evaluates bidder-classification models for the Kaggle
competition dataset. It includes feature engineering, repeated cross-validation,
final model fitting, checkpoint saving, and Kaggle-style prediction generation.

## Quick Start

### 1. Install `uv`

This project uses [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
to create the Python environment and install dependencies.

### 2. Create the Environment

The project requires Python 3.12 or newer. Use one of the following commands
from the repository root.

CPU environment:

```powershell
uv sync --dev --extra cpu
```

CUDA 13 environment:

```powershell
uv sync --dev --extra cu130
```

### 3. Activate the Environment

Windows PowerShell:

```powershell
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

## Data Setup

Download the Kaggle competition files and place them here:

```text
data/facebook-recruiting-iv-human-or-bot/
  bids.csv
  train.csv
  test.csv
```

The data files are not committed to Git because they come from Kaggle.

## Command Overview

The project exposes one CLI command, `bidbot`, with four top-level subcommands:

```text
bidbot features build
bidbot train {gbm|hybrid|gnn|tffm|resnet}
bidbot cv    {gbm|hybrid|gnn|tffm|resnet}
bidbot eval  {gbm|hybrid|gnn|tffm|resnet}
```

Use `--help` to inspect options:

```powershell
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

## Build Features

Run this once after placing the dataset:

```powershell
bidbot features build
```

This creates cached tabular feature files under:

```text
runs/cache/
```

To force a rebuild:

```powershell
bidbot features build --force
```

## Model Architectures

The available model names are:

| CLI name | Model |
| --- | --- |
| `gbm` | Gradient Boosting Machine baseline |
| `tffm` | FT-Transformer tabular model |
| `hybrid` | Tabular plus bid-sequence BiLSTM hybrid |
| `gnn` | Heterogeneous graph neural network |
| `resnet` | Tabular ResNet baseline |

## Smoke Test

Before running long experiments, use quick CV to confirm the environment,
dataset, and GPU/CPU setup work:

```powershell
bidbot cv tffm --cv.quick
```

Quick CV uses a small 2-fold smoke-test protocol and, for neural models,
reduces training to one epoch. It is only a runtime check, not a final metric.

## Cross-Validation Evaluation

The default evaluation protocol is:

```text
Repeated stratified 5-fold cross-validation, repeated 20 times
```

This produces 100 validation evaluations per model. It should be described as
`5 folds x 20 repeats`, not as plain `100-fold CV`.

Run all model evaluations:

```powershell
bidbot cv gbm    --out runs/cv/gbm_full
bidbot cv tffm   --out runs/cv/tffm_full
bidbot cv hybrid --out runs/cv/hybrid_full
bidbot cv gnn    --out runs/cv/gnn_full
bidbot cv resnet --out runs/cv/resnet_full
```

Each CV run writes:

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

## Current Experiment Results

Summary of the full 5-fold x 20-repeat runs:

| Model | Mean CV AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| `tffm` | `0.915548` | `0.370968` | `0.669903` | `0.477509` |
| `gbm` | `0.886548` | `0.443548` | `0.533981` | `0.484581` |
| `gnn` | `0.814611` | `0.198953` | `0.737864` | `0.313402` |
| `hybrid` | `0.812057` | `0.251938` | `0.631068` | `0.360111` |
| `resnet` | `0.727332` | `0.089552` | `0.990291` | `0.164251` |

For this Kaggle-style task, ROC-AUC is the main selection metric. The selected
final model is therefore `tffm`.

## Final Model Fit

After selecting the model using CV/OOF results, retrain the chosen model on all
labelled training data and save a checkpoint.

For the selected TFFM model:

```powershell
bidbot train tffm --val-fraction 0 --save-model --out runs/train/tffm_full_fit
```

Important reporting note:

- `--val-fraction 0` trains using all labelled data.
- The AUC written for this run is a training/self-validation AUC.
- Do not report the full-fit AUC as generalization performance.
- Report CV/OOF metrics from the `bidbot cv ...` runs instead.

## Generate Predictions

Use the saved checkpoint to create `submission.csv`:

```powershell
bidbot eval tffm --ckpt runs/train/tffm_full_fit/ckpt.pt --out runs/submit/tffm_final/submission.csv
```

The output CSV has the Kaggle submission format:

```text
bidder_id,prediction
...
```

## Typical End-to-End Workflow

```powershell
# 1. Install dependencies. Choose one:
uv sync --dev --extra cpu
uv sync --dev --extra cu130

# 2. Activate the environment.
.\.venv\Scripts\Activate.ps1

# 3. Build feature cache.
bidbot features build

# 4. Smoke test.
bidbot cv tffm --cv.quick

# 5. Evaluate candidate models.
bidbot cv gbm    --out runs/cv/gbm_full
bidbot cv tffm   --out runs/cv/tffm_full
bidbot cv hybrid --out runs/cv/hybrid_full
bidbot cv gnn    --out runs/cv/gnn_full
bidbot cv resnet --out runs/cv/resnet_full

# 6. Fit the selected model on all labelled data.
bidbot train tffm --val-fraction 0 --save-model --out runs/train/tffm_full_fit

# 7. Generate predictions.
bidbot eval tffm --ckpt runs/train/tffm_full_fit/ckpt.pt --out runs/submit/tffm_final/submission.csv
```

## Output Directory Layout

```text
runs/
  cache/
    tabular_train.parquet
    tabular_test.parquet

  cv/
    <run_name>/
      metrics.json
      folds.csv
      oof_predictions.csv
      oof_by_bidder.csv
      roc.png

  train/
    <run_name>/
      metrics.json
      roc.png
      tb/
      ckpt.pt          # only when --save-model is used
      submission.csv   # only after bidbot eval if no --out is supplied

  submit/
    tffm_final/
      submission.csv
      ckpt.pt
      metrics.json
      roc.png
```

## Code Quality

Run these before submitting code changes:

```powershell
ruff format src/
ruff check src/
ty check src/
```

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `bidbot` is not recognized | Activate `.venv`, or use `.\.venv\Scripts\bidbot.exe` on Windows |
| Dataset file not found | Check that `bids.csv`, `train.csv`, and `test.csv` are under `data/facebook-recruiting-iv-human-or-bot/` |
| CUDA is not detected | Run `uv sync --dev --extra cu130`, then check `torch.cuda.is_available()` |
| Matplotlib/backend error | Set `$env:MPLBACKEND = "Agg"` before running |
| Full CV takes a long time | Use `--cv.quick` first; full 5-fold x 20-repeat CV is intentionally more expensive |

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
