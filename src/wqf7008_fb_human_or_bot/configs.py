from pathlib import Path

from pydantic import BaseModel, ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "facebook-recruiting-iv-human-or-bot"
RUNS_DIR = PROJECT_ROOT / "runs"
CACHE_DIR = RUNS_DIR / "cache"
SSL_CKPT = RUNS_DIR / "ssl" / "pretrain_ckpt.pt"

# 1 second of real time ≈ 52,631,578 raw time units (recovered from EDA).
UNIT_PER_SEC = 52_631_578


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path = DATA_DIR
    runs_dir: Path = RUNS_DIR
    cache_dir: Path = CACHE_DIR


class CVConfig(BaseModel):
    """Repeated stratified K-fold protocol. Seed comes from the model config."""

    model_config = ConfigDict(extra="forbid")

    n_splits: int = 5
    n_repeats: int = 20  # 100-fold repeated CV — matches the Kaggle winner protocol
    quick: bool = False  # shrink to 2×1 folds + 1 epoch for smoke runs


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
    device: str = "auto"  # 'auto' | 'cpu' | 'cuda'
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 10
    hidden: int = 128
    dropout: float = 0.3
    max_len: int = 512
    # SSL pretraining init (hybrid only).
    ssl_pretrain: bool = False
    pretrain_epochs: int = 10
    # Path to an existing SSL checkpoint. If set, skip pretraining and load this file.
    # If None and ssl_pretrain=True, reuse runs/ssl/pretrain_ckpt.pt or run pretrain.
    pretrain_ckpt: Path | None = None


class GNNConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    device: str = "auto"
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 10
    hidden: int = 128
    dropout: float = 0.3


class TFFMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    device: str = "auto"
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    early_stop_patience: int = 10
    channels: int = 128
    num_layers: int = 3
