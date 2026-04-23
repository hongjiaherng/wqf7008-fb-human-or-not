import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from torch.utils.tensorboard.writer import SummaryWriter

from wqf7008_fb_human_or_bot.configs import CVConfig
from wqf7008_fb_human_or_bot.models.base import BidderClassifier


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_device(device: str) -> torch.device:
    """Turn a config device string (including 'auto') into a `torch.device`."""
    return torch.device(default_device() if device == "auto" else device)


@dataclass
class FoldData:
    Xtr: np.ndarray
    Xval: np.ndarray
    ytr: np.ndarray
    yval: np.ndarray | None
    ids_tr: np.ndarray
    ids_val: np.ndarray
    train_idx: np.ndarray | None = None  # GNN only
    val_idx: np.ndarray | None = None  # GNN only


@dataclass
class CVResult:
    """Result container for both `run_train` (1 entry) and `run_cv` (K×N entries).

    `labels[i]` is the legend / mode tag for the i-th entry:
      - run_cv:    "fold 0", "fold 1", ...
      - run_train with val:  "val"
      - run_train val=0:     "train"  (self-val AUC == train-fit AUC)
    """

    model_name: str
    per_fold_auc: list[float]
    roc_points: list[tuple[np.ndarray, np.ndarray]]
    labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.labels:
            self.labels = [f"fold {i}" for i in range(len(self.per_fold_auc))]

    @property
    def is_single(self) -> bool:
        return len(self.per_fold_auc) == 1

    @property
    def mean_auc(self) -> float:
        return float(np.mean(self.per_fold_auc))

    @property
    def std_auc(self) -> float:
        return float(np.std(self.per_fold_auc))

    @property
    def q10_auc(self) -> float:
        return float(np.percentile(self.per_fold_auc, 10))

    @property
    def q25_auc(self) -> float:
        return float(np.percentile(self.per_fold_auc, 25))

    def summary_str(self) -> str:
        if self.is_single:
            return f"{self.model_name}: {self.labels[0]}_auc={self.per_fold_auc[0]:.4f}"
        return (
            f"{self.model_name}: mean={self.mean_auc:.4f}, std={self.std_auc:.4f}, "
            f"q25={self.q25_auc:.4f}, q10={self.q10_auc:.4f} "
            f"over {len(self.per_fold_auc)} folds"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_quick_cv(cv: CVConfig) -> CVConfig:
    return cv.model_copy(update={"n_splits": 2, "n_repeats": 1}) if cv.quick else cv


def _apply_quick(model_cfg, quick: bool):
    if not quick:
        return model_cfg
    overrides: dict = {}
    names = set(model_cfg.model_fields)
    if "epochs" in names:
        overrides["epochs"] = 1
    if "early_stop_patience" in names:
        overrides["early_stop_patience"] = 1
    return model_cfg.model_copy(update=overrides) if overrides else model_cfg


def train_torch_loop(
    model: torch.nn.Module,
    model_cfg,
    *,
    model_name: str,
    train_epoch: Callable[[], dict[str, float]],
    eval_fn: Callable[[], dict[str, float]],
    writer: SummaryWriter | None = None,
) -> None:
    """Per-epoch loop with best-state tracking + early stopping.

    `model_cfg` must expose `epochs` and `early_stop_patience`.
    """
    best_auc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0

    for epoch in range(model_cfg.epochs):
        model.train()
        train_metrics = train_epoch()
        val_metrics = eval_fn()

        if writer is not None:
            for k, v in train_metrics.items():
                writer.add_scalar(f"{k}/train", v, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"{k}/val", v, epoch)

        val_auc = val_metrics.get("auc", 0.0)
        marker = ""
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            marker = " *"
        else:
            patience += 1
        train_loss = train_metrics.get("loss", float("nan"))
        print(
            f"    [{model_name}] epoch {epoch + 1:02d}/{model_cfg.epochs}: "
            f"train_loss={train_loss:.4f} val_auc={val_auc:.4f} "
            f"best={best_auc:.4f} patience={patience}/{model_cfg.early_stop_patience}{marker}"
        )
        if patience >= model_cfg.early_stop_patience:
            print(f"    [{model_name}] early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)


def _roc_points(yval, probs) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    has_both = len(np.unique(yval)) > 1
    if has_both:
        fpr, tpr, _ = roc_curve(yval, probs)
        return float(roc_auc_score(yval, probs)), (fpr, tpr)
    return 0.5, (np.array([0.0, 1.0]), np.array([0.0, 1.0]))


def _as_xy(X: pl.DataFrame | np.ndarray, y: np.ndarray, bidder_ids: np.ndarray):
    if isinstance(X, pl.DataFrame):
        cols = [c for c in X.columns if c != "bidder_id"]
        X_np = X.select(cols).to_numpy().astype(np.float32)
    else:
        X_np = np.asarray(X, dtype=np.float32)
    return X_np, np.asarray(y, dtype=np.int64), np.asarray(bidder_ids)


def _attach_graph_idx(fold: FoldData, bidder_to_graph_idx: dict[str, int] | None) -> None:
    if bidder_to_graph_idx is None:
        return
    fold.train_idx = np.array([bidder_to_graph_idx[b] for b in fold.ids_tr], dtype=np.int64)
    fold.val_idx = np.array([bidder_to_graph_idx[b] for b in fold.ids_val], dtype=np.int64)


def _fit_one_fold(
    factory: Callable[[], BidderClassifier],
    fold: FoldData,
    model_cfg,
    writer: SummaryWriter | None = None,
) -> tuple[BidderClassifier, float, tuple[np.ndarray, np.ndarray]]:
    """The shared core: create classifier, fit, predict, compute (auc, roc)."""
    try:
        model = factory()
        model.fit(fold, model_cfg, writer=writer)
        probs = np.asarray(model.predict_proba(fold)).reshape(-1)
    finally:
        if writer is not None:
            writer.close()
    auc, rp = _roc_points(fold.yval, probs)
    return model, auc, rp


def run_cv(
    factory: Callable[[], BidderClassifier],
    X: pl.DataFrame | np.ndarray,
    y: np.ndarray,
    bidder_ids: np.ndarray,
    *,
    cv: CVConfig,
    model_cfg,
    model_name: str,
    bidder_to_graph_idx: dict[str, int] | None = None,
) -> CVResult:
    """Repeated stratified K-fold CV. No TensorBoard, no model checkpoint."""
    cv = _resolve_quick_cv(cv)
    model_cfg = _apply_quick(model_cfg, cv.quick)
    set_seed(model_cfg.seed)

    X_np, y_np, ids = _as_xy(X, y, bidder_ids)
    splitter = RepeatedStratifiedKFold(
        n_splits=cv.n_splits, n_repeats=cv.n_repeats, random_state=model_cfg.seed
    )
    total = cv.n_splits * cv.n_repeats

    per_fold_auc: list[float] = []
    roc_points: list[tuple[np.ndarray, np.ndarray]] = []

    for fold_i, (tr_i, val_i) in enumerate(splitter.split(X_np, y_np)):
        fold = FoldData(
            Xtr=X_np[tr_i],
            Xval=X_np[val_i],
            ytr=y_np[tr_i],
            yval=y_np[val_i],
            ids_tr=ids[tr_i],
            ids_val=ids[val_i],
        )
        _attach_graph_idx(fold, bidder_to_graph_idx)
        _, auc, rp = _fit_one_fold(factory, fold, model_cfg, writer=None)
        per_fold_auc.append(auc)
        roc_points.append(rp)
        print(f"  [{model_name}] fold {fold_i + 1}/{total}: AUC={auc:.4f}")

    return CVResult(model_name=model_name, per_fold_auc=per_fold_auc, roc_points=roc_points)


def run_train(
    factory: Callable[[], BidderClassifier],
    X: pl.DataFrame | np.ndarray,
    y: np.ndarray,
    bidder_ids: np.ndarray,
    *,
    val_fraction: float,
    quick: bool,
    model_cfg,
    model_name: str,
    tb_dir: Path | None = None,
    bidder_to_graph_idx: dict[str, int] | None = None,
) -> tuple[BidderClassifier, CVResult]:
    """Single-pass training.

    - `val_fraction > 0`: stratified held-out split; reported AUC is on the val
      split. Label: `"val"`.
    - `val_fraction == 0`: fit on all data (final-submission mode). Reported AUC
      is the training-fit AUC (self-val). Label: `"train"`.
    """
    model_cfg = _apply_quick(model_cfg, quick)
    set_seed(model_cfg.seed)
    X_np, y_np, ids = _as_xy(X, y, bidder_ids)

    if val_fraction > 0:
        tr_i, val_i = next(
            iter(
                StratifiedShuffleSplit(
                    n_splits=1, test_size=val_fraction, random_state=model_cfg.seed
                ).split(X_np, y_np)
            )
        )
        fold = FoldData(
            Xtr=X_np[tr_i],
            Xval=X_np[val_i],
            ytr=y_np[tr_i],
            yval=y_np[val_i],
            ids_tr=ids[tr_i],
            ids_val=ids[val_i],
        )
        label = "val"
    else:
        # Self-val: Xval = Xtr. AUC is the training fit; early stop can't fire.
        fold = FoldData(
            Xtr=X_np,
            Xval=X_np,
            ytr=y_np,
            yval=y_np,
            ids_tr=ids,
            ids_val=ids,
        )
        label = "train"

    _attach_graph_idx(fold, bidder_to_graph_idx)
    writer = SummaryWriter(tb_dir) if tb_dir else None
    model, auc, rp = _fit_one_fold(factory, fold, model_cfg, writer=writer)
    print(f"  [{model_name}] {label} AUC={auc:.4f}")
    return model, CVResult(
        model_name=model_name, per_fold_auc=[auc], roc_points=[rp], labels=[label]
    )
