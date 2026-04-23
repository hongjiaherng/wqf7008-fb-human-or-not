# WQF7008 Practical Deep Learning - Project

Kaggle's "Facebook Recruiting IV: Human or Robot?"

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

## Setup

Install the dev dependencies:

```bash
uv sync --dev --extra cpu
# If you have a CUDA-enabled GPU, you can install the GPU version:
# uv sync --dev --extra cu130
```

Add extra packages if needed:

```bash
uv add <package-name>
```

## Development

Use jupyter lab/notebook for development:

```bash
# Do this
.venv/Scripts/activate  # On Windows
source .venv/bin/activate  # On macOS/Linux
jupyter lab # jupyter notebook

# Or this
uv run jupyter lab  # uv run jupyter notebook
```

Or use any IDE/text editor.

## Training and Evaluation

The `bidbot` CLI has three top-level subcommands:

```bash
bidbot
├── features build                  # cache the per-bidder tabular feature matrix
├── train {gbm|hybrid|gnn|tffm}     # single train/val split, TensorBoard on, can save ckpt
└── cv    {gbm|hybrid|gnn|tffm}     # repeated stratified K-fold (100 folds), no TB, no save
```

`train` is for **tuning hyperparameters** (watch learning curves) and for the
**final submission fit** (pass `--val-fraction 0` to train on all data).
`cv` is for **reporting a generalization number** — K folds × N repeats, no TB.

> **Activate the venv once and call `bidbot` directly** (avoids `uv run --extra …`
> friction in every command):
>
> ```powershell
> # Windows PowerShell
> .\.venv\Scripts\Activate.ps1
> ```
>
> ```bash
> # macOS / Linux
> source .venv/bin/activate
> ```
>
> All examples below assume the venv is active. If you haven't run `uv sync`
> with the right extra yet, do it once: `uv sync --extra cpu` (CPU) or
> `uv sync --extra cu130` (CUDA 13).

### Data

Put the Kaggle competition files under `data/facebook-recruiting-iv-human-or-bot/`:

```bash
data/facebook-recruiting-iv-human-or-bot/
  bids.csv
  train.csv
  test.csv
```

### Features

```bash
bidbot features build            # uses parquet cache if present
bidbot features build --force    # rebuild
```

### Flag namespaces (auto-derived from the pydantic configs)

- `--model.<field>` — training + architecture: `--model.lr`, `--model.epochs`, `--model.hidden`,
  `--model.device`, `--model.seed`, `--model.ssl-pretrain` (hybrid only), etc.
- `--cv.<field>` — CV protocol (cv subcommand only): `--cv.n-splits`, `--cv.n-repeats`, `--cv.quick`
- `--data.<field>` — paths: `--data.data-dir`, `--data.runs-dir`, `--data.cache-dir`
- Flat flags on the subcommand: `--out`, `--val-fraction`, `--quick`, `--save-model`

Run `bidbot train hybrid --help` etc. for the full list.

### Tuning workflow (what `train` is for)

1. **Iterate on hyperparameters** while watching learning curves:

   ```bash
   bidbot train hybrid --model.lr 1e-3 --model.hidden 128
   bidbot train hybrid --model.lr 5e-4 --model.dropout 0.2
   tensorboard --logdir runs/train
   ```

   The default `--val-fraction 0.2` gives you `auc/train` vs `auc/val` plots
   so overfit / underfit are visible. **Always keep val > 0 during tuning.**
2. **Smoke-test** a config in ~30 s with `--quick` (1 epoch).

### Evaluation (what `cv` is for)

```bash
bidbot cv hybrid --model.lr 1e-3 --model.hidden 128
```

Runs 5×20 = 100 folds by default, writes per-fold AUC + mean/std/q25/q10 summary.
Add `--cv.quick` for a 2×1 smoke.

### SSL pretraining (hybrid only)

The hybrid model can be initialised from a masked-bid self-supervised encoder.
Turn it on with `--model.ssl-pretrain`:

```bash
bidbot cv    hybrid --model.ssl-pretrain --cv.quick
bidbot train hybrid --model.ssl-pretrain
```

Resolution order for the checkpoint (see `_maybe_pretrain` in `cli.py`):

1. `--model.pretrain-ckpt <path>` — use an existing checkpoint, must exist.
2. Default `runs/ssl/pretrain_ckpt.pt` if present — reuse.
3. Otherwise run pretraining (all train + test bidder sequences, no labels) and
   write to the default path.

Within one `cv hybrid --model.ssl-pretrain` run, pretraining happens **once**;
every fold's fine-tune loads the same checkpoint. When changing architecture
(`--model.hidden`, `--model.max-len`, …) delete the cached ckpt so pretraining
reruns against the new shape — there's no architecture hash in the filename.

Only the sequence encoder (embeddings + `input_norm` + LSTM) is transferred;
`head` and `tab_mlp` always reinit from scratch (`tabular_dim` differs between
pretrain=0 and fine-tune>0).

### Final submission fit

Once the config is locked, fit on 100% of the training data and save a checkpoint:

```bash
bidbot train hybrid --val-fraction 0 --save-model \
    --model.lr 1e-3 --model.hidden 128 --model.seed 7
```

`--val-fraction 0` disables the split — no val metrics, no early stopping,
trains for the full `--model.epochs`. Pair with `--save-model` to write
`ckpt.pt`. Runs with `--val-fraction 0` land in a folder suffixed `_full` (e.g.
`runs/train/20260424T034231_gbm_full/`) so they're visually distinct from
tuning runs.

### Generate a Kaggle submission

```bash
bidbot predict {gbm|hybrid|gnn|tffm} --ckpt <path-to-ckpt.pt> [--out <csv>]
```

With no `--out`, `submission.csv` lands next to the checkpoint. Example:

```bash
bidbot predict gbm --ckpt runs/train/20260424T034231_gbm_full/ckpt.pt
# → writes runs/train/20260424T034231_gbm_full/submission.csv
```

Each `predict_<model>` loader rebuilds the per-model scaffolding (sequence
store for hybrid, hetero graph for gnn, col_stats for tffm) from the current
`bids.csv` / `train.csv` / `test.csv` and runs one forward pass.

### End-to-end: from zero to submission.csv

Activate the venv once, then paste the blocks below. `--model.device` defaults
to `auto` → picks `cuda` when available, so no GPU flag needed.

```powershell
# venv activate
.\.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
```

```bash
# 1. build the tabular feature cache (once)
bidbot features build

# 2. compare models via repeated stratified K-fold CV (5 × 20 = 100 folds).
#    Add `--cv.quick` for a 2×1 × 1-epoch smoke run.
bidbot cv gbm
bidbot cv hybrid
bidbot cv hybrid --model.ssl-pretrain
bidbot cv gnn
bidbot cv tffm

# 3. pick a winner (check runs/cv/<ts>_<model>/metrics.json → mean / q25),
#    full-fit on 100% of train, save ckpt.pt. Pick ONE of these:
bidbot train gbm    --val-fraction 0 --save-model
bidbot train hybrid --val-fraction 0 --save-model
bidbot train hybrid --model.ssl-pretrain --val-fraction 0 --save-model
bidbot train gnn    --val-fraction 0 --save-model
bidbot train tffm   --val-fraction 0 --save-model

# 4. generate submission.csv (lands next to the ckpt by default)
bidbot predict gbm    --ckpt runs/train/<ts>_gbm_full/ckpt.pt
# (match predict <model> to the model you trained in step 3)
```

Optional — tune hyperparameters between steps 2 and 3 while watching learning
curves in TensorBoard:

```bash
bidbot train hybrid --val-fraction 0.2 --model.lr 1e-3 --model.hidden 128
bidbot train hybrid --val-fraction 0.2 --model.lr 5e-4 --model.dropout 0.2
tensorboard --logdir runs/train
```

### Runs directory layout

```bash
runs/
├── cache/                               # feature parquet cache
├── ssl/                                 # SSL pretrain checkpoint (hybrid only)
│
├── train/
│   ├── {timestamp}_{tag}/                # tuning run (val_fraction > 0)
│   │   ├── metrics.json                  # single val AUC
│   │   ├── roc.png
│   │   ├── tb/events.out.tfevents…       # TensorBoard scalars
│   │   └── ckpt.pt                       # only with --save-model
│   │
│   └── {timestamp}_{tag}_full/           # full-fit run (val_fraction == 0)
│       ├── metrics.json                  # self-val train AUC
│       ├── tb/events.out.tfevents…
│       ├── ckpt.pt                       # only with --save-model
│       └── submission.csv                # only after `bidbot predict`
│
└── cv/
    └── {timestamp}_{tag}/
        ├── metrics.json                  # per-fold AUC + mean/std/q25/q10
        ├── folds.csv
        └── roc.png                       # (no tb/, no ckpt)
```

`{tag}` is normally just the model name (`gbm`, `hybrid`, `gnn`, `tffm`).
Hybrid runs with `--model.ssl-pretrain` tag as `hybrid_ssl`. Runs with
`--val-fraction 0` get an additional `_full` suffix.

TB scalar tags:

- DL models (hybrid / gnn / tffm): `loss/{train,val}`, `auc/{train,val}`
- GBM: `deviance/train`, `auc/{train,val}` (one point per boosting iter)

### Notebook

`notebooks/02_train_compare.ipynb` drives the same `run_cv` / `run_train` code
as the CLI but lets you render plots inline and script the final refit /
submission step.

### Code quality

```bash
ruff format src/
ruff check src/
ty check src/
```

## References

### Competition

- Kaggle: [Facebook Recruiting IV: Human or Robot?](https://kaggle.com/competitions/facebook-recruiting-iv-human-or-bot) (2015)

```bibtex
@misc{facebook-recruiting-iv-human-or-bot,
    author = {Jim Dullaghan and John Costella and John_W and Meghan O'Connell and Rafael and Ruchi and RuchiVarshney and Sergey and Sofus Macskassy and Wendy Kan},
    title  = {Facebook Recruiting IV: Human or Robot?},
    year   = {2015},
    howpublished = {\url{https://kaggle.com/competitions/facebook-recruiting-iv-human-or-bot}},
    note   = {Kaggle}
}
```

### Top solutions

Scores reported as ROC-AUC (private / public leaderboard).

| Rank | Score (private / public) | Writeup |
| ---- | ------------------------ | ------- |
| 1st  | 0.94254 / 0.91946        | [Forum comment by the winner](https://www.kaggle.com/competitions/facebook-recruiting-iv-human-or-bot/writeups/small-yellow-duck-share-your-secret-sauce#81331) |
| 2nd  | 0.94167 / 0.93277        | [small-yellow-duck: "Share your secret sauce"](https://www.kaggle.com/competitions/facebook-recruiting-iv-human-or-bot/writeups/small-yellow-duck-share-your-secret-sauce), [blog post](http://small-yellow-duck.github.io/auction.html) |
| 3rd  | 0.94113 / 0.93321        | [Forum comment by mechatroner](https://www.kaggle.com/competitions/facebook-recruiting-iv-human-or-bot/writeups/small-yellow-duck-share-your-secret-sauce#81396) |
