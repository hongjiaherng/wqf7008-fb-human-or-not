"""Tabular / sequence / graph views over the raw bids.

`load_data` reads the CSVs and bundles bids + tabular features into `Data`;
sequence and graph views are built on demand from `Data.bids`. Torch-free, so
importing this never pulls torch.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from wqf7008_fb_human_or_bot.datasets.tabular import (
    FEATURE_COLS,
    UNIT_PER_SEC,
    build_tabular,
    feature_cols,
)
from wqf7008_fb_human_or_bot.paths import PathConfig

__all__ = [
    "FEATURE_COLS",
    "UNIT_PER_SEC",
    "Data",
    "PathConfig",
    "build_tabular",
    "feature_cols",
    "load_data",
]


@dataclass
class Data:
    bids: pl.DataFrame
    Xtr: pl.DataFrame
    ytr: np.ndarray
    ids_tr: np.ndarray
    Xte: pl.DataFrame
    ids_te: np.ndarray


def load_data(path_cfg: PathConfig) -> Data:
    bids = pl.read_csv(path_cfg.data_dir / "bids.csv")
    train = pl.read_csv(path_cfg.data_dir / "train.csv")
    test = pl.read_csv(path_cfg.data_dir / "test.csv")
    Xtr, ids_tr, ytr = build_tabular(bids, train, path_cfg)
    Xte, ids_te, _ = build_tabular(bids, test, path_cfg)
    assert ytr is not None
    return Data(bids=bids, Xtr=Xtr, ytr=ytr, ids_tr=ids_tr, Xte=Xte, ids_te=ids_te)
