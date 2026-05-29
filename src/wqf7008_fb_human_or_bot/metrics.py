import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, roc_curve

OOF_THRESHOLD = 0.5


@dataclass
class CVResult:
    """Result container for both `run_train` (1 entry) and `run_cv` (KxN entries).

    `labels[i]` is the legend / mode tag for the i-th entry:
      - run_cv:    "fold 0", "fold 1", ...
      - run_train with val:  "val"
      - run_train val=0:     "train"  (self-val AUC == train-fit AUC)
    """

    model_name: str
    per_fold_auc: list[float]
    roc_points: list[tuple[np.ndarray, np.ndarray]]
    labels: list[str] = field(default_factory=list)
    oof_predictions: pl.DataFrame | None = None

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


def roc_points(y_true, probs) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    """Return (auc, (fpr, tpr)); degenerate diagonal if only one class is present."""
    has_both = len(np.unique(y_true)) > 1
    if has_both:
        fpr, tpr, _ = roc_curve(y_true, probs)
        return float(roc_auc_score(y_true, probs)), (fpr, tpr)
    return 0.5, (np.array([0.0, 1.0]), np.array([0.0, 1.0]))


def _threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _write_oof_outputs(
    oof_predictions: pl.DataFrame,
    out_dir: Path,
    *,
    threshold: float = OOF_THRESHOLD,
) -> dict:
    """Write repeated and bidder-averaged OOF predictions, then return metrics.

    Repeated CV validates each bidder more than once, so threshold metrics are
    computed after averaging each bidder's repeated out-of-fold probabilities.
    """
    oof_predictions.write_csv(out_dir / "oof_predictions.csv")
    by_bidder = (
        oof_predictions.group_by("bidder_id")
        .agg(
            pl.col("y_true").first(),
            pl.col("y_prob").mean().alias("y_prob_mean"),
            pl.len().alias("n_predictions"),
        )
        .with_columns((pl.col("y_prob_mean") >= threshold).cast(pl.Int64).alias("y_pred"))
        .sort("bidder_id")
    )
    by_bidder.write_csv(out_dir / "oof_by_bidder.csv")

    y_true = by_bidder["y_true"].to_numpy()
    y_pred = by_bidder["y_pred"].to_numpy()
    metrics = _threshold_metrics(y_true, y_pred)
    return {
        "oof_threshold": float(threshold),
        "oof_n_predictions": int(oof_predictions.height),
        "oof_n_bidders": int(by_bidder.height),
        "oof_precision": metrics["precision"],
        "oof_recall": metrics["recall"],
        "oof_f1": metrics["f1"],
        "oof_confusion_matrix": metrics["confusion_matrix"],
    }


def save_cv_summary(result: CVResult, out_dir: str | Path) -> Path:
    """Write `metrics.json` (and `folds.csv` if multi-fold)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if result.is_single:
        payload: dict = {
            "model": result.model_name,
            "split": result.labels[0],  # "val" or "train"
            "auc": result.per_fold_auc[0],
        }
    else:
        payload = {
            "model": result.model_name,
            "per_fold_auc": result.per_fold_auc,
            "mean": result.mean_auc,
            "std": result.std_auc,
            "q25": result.q25_auc,
            "q10": result.q10_auc,
            "n_folds": len(result.per_fold_auc),
        }
        pl.DataFrame(
            {"fold": list(range(len(result.per_fold_auc))), "auc": result.per_fold_auc}
        ).write_csv(out / "folds.csv")
        if result.oof_predictions is not None:
            payload.update(_write_oof_outputs(result.oof_predictions, out))
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    return out / "metrics.json"


def _finish_fig(fig, out_path) -> None:
    """Save + close, or leave open for matplotlib-inline to auto-display.

    Always returns None: if we returned the figure in a notebook, Jupyter's
    repr would display it in addition to `%matplotlib inline`'s post-run-cell
    auto-show, giving two copies of the same plot.
    """
    import matplotlib.pyplot as plt

    fig.tight_layout()
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=120)
        plt.close(fig)


def plot_roc(result: CVResult, out_path: str | Path | None = None, *, title: str | None = None):
    """ROC curves. Uses `result.labels` for legend (fold N for CV, val/train for single)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, ((fpr, tpr), label) in enumerate(zip(result.roc_points, result.labels, strict=False)):
        ax.plot(fpr, tpr, linewidth=0.8, alpha=0.8, label=label if i < 3 else None)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8, color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    if title is None:
        title = (
            f"{result.model_name}  {result.labels[0]}_auc={result.per_fold_auc[0]:.4f}"
            if result.is_single
            else f"{result.model_name}  mean AUC={result.mean_auc:.4f}"
        )
    ax.set_title(title)
    ax.legend(loc="lower right")
    return _finish_fig(fig, out_path)


def plot_cv_boxplot(
    results: list[CVResult], out_path: str | Path | None = None, *, title: str | None = None
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(
        [r.per_fold_auc for r in results],
        tick_labels=[r.model_name for r in results],
        showmeans=True,
    )
    ax.set_ylabel("AUC")
    ax.set_title(title or "Per-fold AUC by model")
    return _finish_fig(fig, out_path)


def write_submission(bidder_ids: np.ndarray, predictions: np.ndarray, out_path: str | Path) -> Path:
    preds = np.clip(np.asarray(predictions, dtype=np.float64), 0.0, 1.0)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"bidder_id": np.asarray(bidder_ids), "prediction": preds}).write_csv(p)
    return p


def compare_summary(results: list[CVResult]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "model": r.model_name,
                "mean_auc": r.mean_auc,
                "std_auc": r.std_auc,
                "q25_auc": r.q25_auc,
                "q10_auc": r.q10_auc,
                "n_folds": len(r.per_fold_auc),
            }
            for r in results
        ]
    )
