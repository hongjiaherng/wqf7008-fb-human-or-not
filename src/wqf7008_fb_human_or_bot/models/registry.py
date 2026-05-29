"""Per-model configs + name->class lookup.

`get_model` lazy-imports so the CLI never pulls torch. To tune a model, edit its
config here and its training loop in `models/<name>.py`.
"""

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from wqf7008_fb_human_or_bot.models.base import BidderClassifier

MODEL_NAMES = ("gbm", "hybrid", "gnn", "tffm", "resnet")


# ---- Per-model configs: training + architecture + reproducibility. ----


class GBMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 3
    subsample: float = 0.8
    min_samples_leaf: int = 5


class HybridConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    device: str = "cpu"  # 'cpu' | 'cuda'
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 10
    hidden: int = 128
    dropout: float = 0.3
    max_len: int = 512


class GNNConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    device: str = "cpu"
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 10
    hidden: int = 128
    dropout: float = 0.3


class TFFMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    device: str = "cpu"
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    early_stop_patience: int = 10
    channels: int = 128
    num_layers: int = 3


class ResNetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    device: str = "cpu"
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 10
    hidden: int = 128
    block_hidden: int = 256
    n_layers: int = 4
    dropout: float = 0.2


CONFIGS: dict[str, type[BaseModel]] = {
    "gbm": GBMConfig,
    "hybrid": HybridConfig,
    "gnn": GNNConfig,
    "tffm": TFFMConfig,
    "resnet": ResNetConfig,
}


def pos_weight_from_labels(y: np.ndarray) -> float:
    """BCEWithLogitsLoss positive-class weight = #neg / #pos (1.0 if no positives)."""
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def apply_quick(model_cfg: BaseModel, quick: bool) -> BaseModel:
    """Shrink training to 1 epoch / patience 1 for smoke runs (no-op for GBM)."""
    if not quick:
        return model_cfg
    overrides: dict = {}
    names = set(model_cfg.model_fields)
    if "epochs" in names:
        overrides["epochs"] = 1
    if "early_stop_patience" in names:
        overrides["early_stop_patience"] = 1
    return model_cfg.model_copy(update=overrides) if overrides else model_cfg


def get_model(name: str) -> "type[BidderClassifier]":
    match name:
        case "gbm":
            from wqf7008_fb_human_or_bot.models.gbm import GBMBidderClassifier as M
        case "hybrid":
            from wqf7008_fb_human_or_bot.models.hybrid import HybridBidderClassifier as M
        case "gnn":
            from wqf7008_fb_human_or_bot.models.gnn import GNNBidderClassifier as M
        case "tffm":
            from wqf7008_fb_human_or_bot.models.tffm import FTTransformerBidderClassifier as M
        case "resnet":
            from wqf7008_fb_human_or_bot.models.resnet import ResNetBidderClassifier as M
        case _:
            raise ValueError(f"unknown model: {name!r} (expected one of {MODEL_NAMES})")
    return M
