"""Repeated stratified K-fold CV: config, protocol, and the `cv` command.

Torch-free at import (cli imports `CVConfig`); the training engine is imported
lazily inside `run_cv`.
"""

from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection import RepeatedStratifiedKFold

from wqf7008_fb_human_or_bot import utils
from wqf7008_fb_human_or_bot.datasets import load_data
from wqf7008_fb_human_or_bot.metrics import CVResult
from wqf7008_fb_human_or_bot.models.base import MakeModel, Split
from wqf7008_fb_human_or_bot.models.registry import apply_quick, get_model
from wqf7008_fb_human_or_bot.paths import PathConfig


class CVConfig(BaseModel):
    """Repeated stratified K-fold protocol. Seed comes from the model config."""

    model_config = ConfigDict(extra="forbid")

    n_splits: int = 5
    n_repeats: int = 20  # 100-fold repeated CV, matches the Kaggle winner protocol
    quick: bool = False  # shrink to 2x1 folds + 1 epoch for smoke runs


def resolve_quick_cv(cv: CVConfig) -> CVConfig:
    """Shrink the CV protocol to a 2x1 smoke test when `cv.quick` is set."""
    return cv.model_copy(update={"n_splits": 2, "n_repeats": 1}) if cv.quick else cv


def run_cv(
    make_model: MakeModel,
    X: pl.DataFrame | np.ndarray,
    y: np.ndarray,
    bidder_ids: np.ndarray,
    cv: CVConfig,
    seed: int,
    model_name: str,
) -> CVResult:
    """Repeated stratified K-fold CV. No TensorBoard, no model checkpoint."""
    from wqf7008_fb_human_or_bot.train import _as_xy, _fit_one_fold, set_seed

    set_seed(seed)
    X_np, y_np, ids = _as_xy(X, y, bidder_ids)
    splitter = RepeatedStratifiedKFold(
        n_splits=cv.n_splits, n_repeats=cv.n_repeats, random_state=seed
    )
    total = cv.n_splits * cv.n_repeats

    per_fold_auc: list[float] = []
    roc_pts: list[tuple[np.ndarray, np.ndarray]] = []
    oof_parts: list[pl.DataFrame] = []

    for fold_i, (tr_i, val_i) in enumerate(splitter.split(X_np, y_np)):
        train = Split(X=X_np[tr_i], y=y_np[tr_i], ids=ids[tr_i])
        val = Split(X=X_np[val_i], y=y_np[val_i], ids=ids[val_i])
        _, auc, rp, probs = _fit_one_fold(make_model, train, val, writer=None)
        per_fold_auc.append(auc)
        roc_pts.append(rp)
        oof_parts.append(
            pl.DataFrame(
                {
                    "fold": np.full(len(val.ids), fold_i, dtype=np.int64),
                    "bidder_id": [str(b) for b in val.ids],
                    "y_true": np.asarray(val.y, dtype=np.int64),
                    "y_prob": np.asarray(probs, dtype=np.float64),
                }
            )
        )
        print(f"  [{model_name}] fold {fold_i + 1}/{total}: AUC={auc:.4f}")

    return CVResult(
        model_name=model_name,
        per_fold_auc=per_fold_auc,
        roc_points=roc_pts,
        oof_predictions=pl.concat(oof_parts) if oof_parts else None,
    )


def cv_command(
    model_name: str,
    model_cfg: BaseModel,
    cv_cfg: CVConfig,
    data_cfg: PathConfig,
    out: Path | None,
) -> None:
    """`bidbot cv <model>`: evaluate via repeated stratified K-fold CV."""
    model_cls = get_model(model_name)
    cfg = apply_quick(model_cfg, cv_cfg.quick)
    cv = resolve_quick_cv(cv_cfg)
    data = load_data(data_cfg)
    out_dir = utils.run_dir(out, "cv", model_name)
    utils.header(
        f"cv {model_name}",
        out_dir,
        cv=utils.fmt(cv),
        model=utils.fmt(cfg),
        data=utils.fmt(data_cfg),
    )
    make_model = model_cls.from_data(cfg, data)
    result = run_cv(
        make_model,
        data.Xtr,
        data.ytr,
        data.ids_tr,
        cv=cv,
        seed=cfg.model_dump()["seed"],
        model_name=model_name,
    )
    utils.write_summary(result, out_dir)
