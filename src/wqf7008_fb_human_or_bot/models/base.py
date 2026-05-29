from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Self, TypeVar

import numpy as np
from pydantic import BaseModel

from wqf7008_fb_human_or_bot.datasets import Data


@dataclass
class Split:
    """One slice of data: features, optional labels (None at predict time), and
    the bidder ids those rows belong to."""

    X: np.ndarray
    y: np.ndarray | None
    ids: np.ndarray


ConfigT = TypeVar("ConfigT", bound=BaseModel)


class BidderClassifier(ABC, Generic[ConfigT]):
    """Shared contract: build in `__init__`, `fit` trains, `predict_proba` scores.

    `from_data`/`save`/`load` keep per-model wiring in the model's own file.
    """

    cfg: ConfigT

    @classmethod
    @abstractmethod
    def from_data(cls, cfg: ConfigT, data: Data) -> Callable[[Split], Self]:
        """Build heavy shared state once; return a per-fold `make_model(train)`."""
        ...

    @abstractmethod
    def fit(self, train: Split, val: Split, writer=None) -> None:
        """Train on `train`, using `val` only for early stopping / monitoring."""
        ...

    @abstractmethod
    def predict_proba(self, split: Split) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, ckpt: Path, data: Data) -> Self:
        """Reconstruct a fitted classifier from a checkpoint (+ data for shapes)."""
        ...


# A model factory: given a fold's training split, build a fresh classifier (model
# already constructed, ready to fit). Heavy shared state is captured by the closure.
MakeModel = Callable[[Split], "BidderClassifier"]
