"""Per-fold fit engine, single-split `run_train`, and the `train` command.

`run_cv` (cv.py) reuses the engine here; CV is just repeated training.
"""

import random
from pathlib import Path

import numpy as np
import polars as pl
import torch
from pydantic import BaseModel
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.tensorboard.writer import SummaryWriter

from wqf7008_fb_human_or_bot import utils
from wqf7008_fb_human_or_bot.datasets import load_data
from wqf7008_fb_human_or_bot.metrics import CVResult, roc_points
from wqf7008_fb_human_or_bot.models.base import BidderClassifier, MakeModel, Split
from wqf7008_fb_human_or_bot.models.registry import apply_quick, get_model
from wqf7008_fb_human_or_bot.paths import PathConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _as_xy(
    X: pl.DataFrame | np.ndarray, y: np.ndarray, bidder_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(X, pl.DataFrame):
        cols = [c for c in X.columns if c != "bidder_id"]
        X_np = X.select(cols).to_numpy().astype(np.float32)
    else:
        X_np = np.asarray(X, dtype=np.float32)
    return X_np, np.asarray(y, dtype=np.int64), np.asarray(bidder_ids)


def _fit_one_fold(
    make_model: MakeModel,
    train: Split,
    val: Split,
    writer: SummaryWriter | None = None,
) -> tuple[BidderClassifier, float, tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Shared core: build the model for this fold, fit, predict on val, score."""
    try:
        model = make_model(train)
        model.fit(train, val, writer=writer)
        probs = np.asarray(model.predict_proba(val)).reshape(-1)
    finally:
        if writer is not None:
            writer.close()
    auc, rp = roc_points(val.y, probs)
    return model, auc, rp, probs


def run_train(
    make_model: MakeModel,
    X: pl.DataFrame | np.ndarray,
    y: np.ndarray,
    bidder_ids: np.ndarray,
    val_fraction: float,
    seed: int,
    model_name: str,
    tb_dir: Path | None = None,
) -> tuple[BidderClassifier, CVResult]:
    """Single-pass training.

    - `val_fraction > 0`: stratified held-out split; reported AUC is on the val
      split. Label: `"val"`.
    - `val_fraction == 0`: fit on all data (final-submission mode); `val` aliases
      `train`, so the reported AUC is the training-fit AUC. Label: `"train"`.
    """
    set_seed(seed)
    X_np, y_np, ids = _as_xy(X, y, bidder_ids)

    if val_fraction > 0:
        tr_i, val_i = next(
            iter(
                StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed).split(
                    X_np, y_np
                )
            )
        )
        train = Split(X=X_np[tr_i], y=y_np[tr_i], ids=ids[tr_i])
        val = Split(X=X_np[val_i], y=y_np[val_i], ids=ids[val_i])
        label = "val"
    else:
        # Self-val: val mirrors train. AUC is the training fit; early stop can't fire.
        train = Split(X=X_np, y=y_np, ids=ids)
        val = Split(X=X_np, y=y_np, ids=ids)
        label = "train"

    writer = SummaryWriter(tb_dir) if tb_dir else None
    model, auc, rp, _ = _fit_one_fold(make_model, train, val, writer=writer)
    print(f"  [{model_name}] {label} AUC={auc:.4f}")
    return model, CVResult(
        model_name=model_name, per_fold_auc=[auc], roc_points=[rp], labels=[label]
    )


def train_command(
    model_name: str,
    model_cfg: BaseModel,
    data_cfg: PathConfig,
    out: Path | None,
    val_fraction: float,
    quick: bool,
    save_model: bool,
) -> None:
    """`bidbot train <model>`: fit on a train/val split, log to TensorBoard, optionally checkpoint."""
    model_cls = get_model(model_name)
    cfg = apply_quick(model_cfg, quick)
    data = load_data(data_cfg)
    # val_fraction==0 -> full-fit submission run; flag it in the folder name.
    dir_tag = f"{model_name}_full" if val_fraction == 0 else model_name
    out_dir = utils.run_dir(out, "train", dir_tag)
    utils.header(
        f"train {model_name}",
        out_dir,
        model=utils.fmt(cfg),
        data=utils.fmt(data_cfg),
        run=f"val_fraction={val_fraction} quick={quick} save_model={save_model}",
    )
    make_model = model_cls.from_data(cfg, data)
    clf, result = run_train(
        make_model,
        data.Xtr,
        data.ytr,
        data.ids_tr,
        val_fraction=val_fraction,
        seed=cfg.model_dump()["seed"],
        model_name=model_name,
        tb_dir=out_dir / "tb",
    )
    utils.write_summary(result, out_dir)
    if save_model:
        clf.save(out_dir / "ckpt.pt")
        print(f"  saved checkpoint to {utils.rel(out_dir / 'ckpt.pt')}")
