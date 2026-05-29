import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Self

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from wqf7008_fb_human_or_bot.datasets import Data
from wqf7008_fb_human_or_bot.models.base import BidderClassifier, Split
from wqf7008_fb_human_or_bot.models.registry import GBMConfig, pos_weight_from_labels


class GBMBidderClassifier(BidderClassifier[GBMConfig]):
    def __init__(self, cfg: GBMConfig):
        self.cfg = cfg
        self.clf = GradientBoostingClassifier(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            subsample=cfg.subsample,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.seed,
        )

    @classmethod
    def from_data(cls, cfg: GBMConfig, data: Data) -> Callable[[Split], Self]:
        return lambda _train: cls(cfg)

    def fit(self, train: Split, val: Split, writer=None) -> None:
        ytr = train.y
        assert ytr is not None  # labels are always present during fit
        sample_weight = np.where(ytr == 1, pos_weight_from_labels(ytr), 1.0)
        self.clf.fit(train.X, ytr, sample_weight=sample_weight)

        if writer is None:
            return
        # Per-boosting-iter scalars. `train_score_` is the in-sample deviance
        # (free, computed during fit); AUC needs a forward pass per iter.
        for i, score in enumerate(self.clf.train_score_):
            writer.add_scalar("deviance/train", float(score), i)
        if len(np.unique(ytr)) > 1:
            for i, p in enumerate(self.clf.staged_predict_proba(train.X)):
                writer.add_scalar("auc/train", float(roc_auc_score(ytr, p[:, 1])), i)
        if val.y is not None and len(np.unique(val.y)) > 1:
            for i, p in enumerate(self.clf.staged_predict_proba(val.X)):
                writer.add_scalar("auc/val", float(roc_auc_score(val.y, p[:, 1])), i)

    def predict_proba(self, split: Split) -> np.ndarray:
        return self.clf.predict_proba(split.X)[:, 1]

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as f:
            pickle.dump({"cfg": self.cfg, "clf": self.clf}, f)

    @classmethod
    def load(cls, ckpt: Path, data: Data) -> Self:
        with Path(ckpt).open("rb") as f:
            payload = pickle.load(f)
        inst = cls(payload["cfg"])
        inst.clf = payload["clf"]
        return inst
