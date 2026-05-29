"""Load a checkpoint, score the test set, write the Kaggle submission CSV."""

from pathlib import Path

import numpy as np

from wqf7008_fb_human_or_bot import utils
from wqf7008_fb_human_or_bot.datasets import Data, feature_cols, load_data
from wqf7008_fb_human_or_bot.metrics import write_submission
from wqf7008_fb_human_or_bot.models.base import BidderClassifier, Split
from wqf7008_fb_human_or_bot.models.registry import get_model
from wqf7008_fb_human_or_bot.paths import PathConfig


def run_eval(model_cls: type[BidderClassifier], ckpt: Path, data: Data) -> np.ndarray:
    """Return test-set probabilities aligned to `data.ids_te`."""
    clf = model_cls.load(ckpt, data)
    feat = feature_cols(data.Xtr)
    Xte = data.Xte.select(feat).to_numpy().astype(np.float32)
    return clf.predict_proba(Split(X=Xte, y=None, ids=data.ids_te))


def eval_command(
    model_name: str,
    ckpt: Path,
    data_cfg: PathConfig,
    out: Path | None,
) -> None:
    """`bidbot eval <model>`: load a checkpoint and write a Kaggle submission CSV."""
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    out_path = out if out is not None else ckpt.parent / "submission.csv"
    model_cls = get_model(model_name)
    data = load_data(data_cfg)
    utils.header("eval", out_path, ckpt=utils.rel(ckpt), data=utils.fmt(data_cfg))
    preds = run_eval(model_cls, ckpt, data)
    path = write_submission(data.ids_te, preds, out_path)
    print(f"  wrote {utils.rel(path)}  ({len(preds)} rows)")
