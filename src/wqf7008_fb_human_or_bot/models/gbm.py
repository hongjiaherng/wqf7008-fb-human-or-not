import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from wqf7008_fb_human_or_bot.configs import GBMConfig
from wqf7008_fb_human_or_bot.models.base import pos_weight_from_labels


class GBMBidderClassifier:
    def __init__(self, *, cfg: GBMConfig | None = None):
        self.cfg = cfg or GBMConfig()
        self.clf: GradientBoostingClassifier | None = None

    def fit(self, fold, cfg: GBMConfig, *, writer=None) -> None:
        Xtr, ytr = fold.Xtr, fold.ytr
        sample_weight = np.where(ytr == 1, pos_weight_from_labels(ytr), 1.0)
        self.clf = GradientBoostingClassifier(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            subsample=cfg.subsample,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.seed,
        )
        self.clf.fit(Xtr, ytr, sample_weight=sample_weight)

        if writer is None:
            return
        # Per-boosting-iter scalars. `train_score_` is the in-sample deviance
        # (free — computed during fit); AUC needs a forward pass per iter.
        for i, score in enumerate(self.clf.train_score_):
            writer.add_scalar("deviance/train", float(score), i)
        if len(np.unique(ytr)) > 1:
            for i, p in enumerate(self.clf.staged_predict_proba(Xtr)):
                writer.add_scalar("auc/train", float(roc_auc_score(ytr, p[:, 1])), i)
        if fold.yval is not None and len(np.unique(fold.yval)) > 1:
            for i, p in enumerate(self.clf.staged_predict_proba(fold.Xval)):
                writer.add_scalar("auc/val", float(roc_auc_score(fold.yval, p[:, 1])), i)

    def predict_proba(self, fold_or_X) -> np.ndarray:
        assert self.clf is not None, "call fit() first"
        X = getattr(fold_or_X, "Xval", fold_or_X)
        return self.clf.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as f:
            pickle.dump({"cfg": self.cfg, "clf": self.clf}, f)

    @classmethod
    def load(cls, path: str | Path) -> "GBMBidderClassifier":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        inst = cls(cfg=payload["cfg"])
        inst.clf = payload["clf"]
        return inst
