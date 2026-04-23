import json
from pathlib import Path

import numpy as np
import polars as pl

from wqf7008_fb_human_or_bot.train import CVResult


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
