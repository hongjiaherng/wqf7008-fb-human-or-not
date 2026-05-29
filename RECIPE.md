# Training Recipe

A runbook for reproducing the full pipeline on a (Linux) GPU cloud VM:
build features -> cross-validate all 5 models -> pick the best -> full-fit -> submit.

Commands are bash. The 4 neural models take `--model.device cuda`; **`gbm` has no
`device` field (sklearn, CPU-only), so never pass `--model.device` to it.**

---

## 0. Provision the VM

```bash
git clone <your-repo-url> && cd wqf7008-fb-human-or-bot

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

# GPU env (CUDA 13). Use --extra cpu instead if the VM has no GPU.
uv sync --dev --extra cu130

# Activate the env, then call `bidbot` directly. Do NOT use `uv run bidbot`: it
# re-resolves without the cu130/cpu extra and drops torch from the environment.
source .venv/bin/activate

# headless VM: make matplotlib write PNGs without a display
export MPLBACKEND=Agg

# sanity check the GPU is visible
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

All commands below assume the env is active. Re-run `source .venv/bin/activate` in
each new shell or tmux window (the binary is `bidbot`, or `bidbot.exe` on Windows).

## 1. Data

Place the Kaggle files:

```text
data/facebook-recruiting-iv-human-or-bot/{bids.csv,train.csv,test.csv}
```

## 2. Build the feature cache (once)

```bash
bidbot features build
```

Writes `runs/cache/tabular_{train,test}.parquet`. Every later command reuses these.

## 3. Smoke test (confirm env + GPU before the long runs)

```bash
bidbot cv gbm  --cv.quick
bidbot cv tffm --cv.quick --model.device cuda
```

`--cv.quick` = 2 folds x 1 repeat, 1 epoch. Runtime check only, not a real metric.

## 4. Full cross-validation (all 5 models)

Default protocol: **5 folds x 20 repeats = 100 evaluations per model.** This is the
expensive step, so run it in `tmux` (or `nohup`) so it survives an SSH drop, and
`tee` the console to a log.

```bash
tmux new -s cv          # detach with Ctrl-b d, reattach with: tmux attach -t cv
source .venv/bin/activate   # activate inside the tmux shell

bidbot cv gbm    --out runs/cv/gbm                       2>&1 | tee runs/cv/gbm.log
bidbot cv resnet --model.device cuda --out runs/cv/resnet 2>&1 | tee runs/cv/resnet.log
bidbot cv tffm   --model.device cuda --out runs/cv/tffm   2>&1 | tee runs/cv/tffm.log
bidbot cv hybrid --model.device cuda --out runs/cv/hybrid 2>&1 | tee runs/cv/hybrid.log
bidbot cv gnn    --model.device cuda --out runs/cv/gnn    2>&1 | tee runs/cv/gnn.log
```

Each writes to `runs/cv/<model>/`: `metrics.json`, `folds.csv`,
`oof_predictions.csv`, `oof_by_bidder.csv`, `roc.png`.

Tuning before a run: pass `--model.<field>` (e.g. `--model.lr 5e-4 --model.epochs 40`),
or edit the defaults in `src/wqf7008_fb_human_or_bot/models/registry.py`. Inspect
available knobs with `bidbot cv <model> --help`.

## 5. Compare and select

```bash
for m in gbm resnet tffm hybrid gnn; do
  python -c "import json; d=json.load(open('runs/cv/$m/metrics.json')); \
print(f\"{'$m':8} mean_auc={d['mean']:.4f} f1={d.get('oof_f1',0):.4f}\")"
done
```

ROC-AUC is the selection metric for this competition. Pick the highest `mean_auc`.

## 6. Full-fit the selected model (example: tffm)

Retrain on **all** labelled data and save a checkpoint. `--val-fraction 0` means no
held-out split, so the printed AUC is a training-fit number; report the CV metrics
from step 5, not this one.

```bash
bidbot train tffm --val-fraction 0 --save-model \
  --model.device cuda --out runs/train/tffm_full
```

Writes `runs/train/tffm_full/{ckpt.pt,metrics.json,roc.png,tb/}`.

### Monitor with TensorBoard

`train` logs per-epoch scalars (train/val loss and AUC) to `<out>/tb/`. Start
TensorBoard in a separate tmux window pointed at the whole `runs/train` tree so it
picks up every run live (activate the env first in that window):

```bash
source .venv/bin/activate
tensorboard --logdir runs/train --port 6006
```

From your laptop, tunnel the port over SSH, then open http://localhost:6006:

```bash
ssh -L 6006:localhost:6006 <user>@<vm-ip>
```

(Alternatively `--host 0.0.0.0` and open `<vm-ip>:6006` if the firewall allows it.)
Note: only `train` emits TensorBoard scalars; `cv` does not.

## 7. Generate the Kaggle submission

```bash
bidbot eval tffm \
  --ckpt runs/train/tffm_full/ckpt.pt \
  --out runs/submit/tffm/submission.csv
```

`eval` reuses the device saved in the checkpoint (so a cuda-trained ckpt evaluates
on cuda on this same VM). The CSV is `bidder_id,prediction`, ready to upload.

## 8. (Optional) full-fit + submit every model

```bash
for m in resnet tffm hybrid gnn; do
  bidbot train $m --val-fraction 0 --save-model \
    --model.device cuda --out runs/train/${m}_full
  bidbot eval $m --ckpt runs/train/${m}_full/ckpt.pt \
    --out runs/submit/$m/submission.csv
done
# gbm is CPU-only (omit --model.device):
bidbot train gbm --val-fraction 0 --save-model --out runs/train/gbm_full
bidbot eval  gbm --ckpt runs/train/gbm_full/ckpt.pt --out runs/submit/gbm/submission.csv
```

---

## Output layout

```text
runs/
  cache/   tabular_{train,test}.parquet
  cv/<model>/      metrics.json folds.csv oof_predictions.csv oof_by_bidder.csv roc.png
  train/<model>_full/  ckpt.pt metrics.json roc.png tb/
  submit/<model>/  submission.csv
```

## Notes

- **Device:** `--model.device cuda` for `resnet|tffm|hybrid|gnn`; never for `gbm`.
  `cv`/`train` accept `--model.*`; `eval` takes the device from the checkpoint.
- **Long runs:** use `tmux`; `tee` to a log so you can monitor with `tail -f`.
- **Re-running:** re-running a command with the same `--out` overwrites that dir.
  Delete `runs/cache/*.parquet` (or `bidbot features build --force`) only if the
  feature engineering changed.
- **Quick first:** validate any new hyperparameters with `--cv.quick` before the
  full 5x20 run.
