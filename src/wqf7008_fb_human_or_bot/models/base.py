from typing import Protocol

import numpy as np


class BidderClassifier(Protocol):
    def fit(self, fold, cfg, /, *, writer=None) -> None: ...

    def predict_proba(self, fold_batch, /) -> np.ndarray: ...


def pos_weight_from_labels(y: np.ndarray) -> float:
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos
