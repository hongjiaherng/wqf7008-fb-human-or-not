"""bidbot CLI: a thin pydantic-settings argument layer.

Each subcommand (features / cv / train / eval) collects args and delegates to the
matching `*_command` in features.py / cv.py / train.py / eval.py; no logic here.
"""

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, CliApp, CliSubCommand, SettingsConfigDict

from wqf7008_fb_human_or_bot.cv import CVConfig
from wqf7008_fb_human_or_bot.models.registry import (
    GBMConfig,
    GNNConfig,
    HybridConfig,
    ResNetConfig,
    TFFMConfig,
)
from wqf7008_fb_human_or_bot.paths import PathConfig

# ---------- shared field descriptions ----------

_VAL_DESC = "fraction of train held out for val (0 = fit on all data, for final submission)"
_OUT_DESC = (
    "output dir (default: runs/{train|cv}/{timestamp}_{tag}; train runs with "
    "val_fraction=0 additionally get a `_full` suffix)"
)
_CKPT_DESC = "path to a ckpt.pt written by `bidbot train ... --save-model`"
_SUB_DESC = "output submission CSV (default: <ckpt_dir>/submission.csv)"


# ==========================================================================
# features build
# ==========================================================================


class FeaturesBuild(BaseModel):
    """Build & cache the per-bidder tabular feature matrices."""

    data: PathConfig = Field(default_factory=PathConfig)
    force: bool = Field(default=False, description="clear existing parquet cache first")

    def cli_cmd(self) -> None:
        from wqf7008_fb_human_or_bot.features import features_command

        features_command(self.data, self.force)


class Features(BaseModel):
    """Feature engineering commands."""

    build: CliSubCommand[FeaturesBuild]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


# ==========================================================================
# Generic verb bases. Subclasses only set `_model_name` (+ a typed `model:`).
# ==========================================================================


class TrainCmd(BaseModel):
    """Train one model on a train/val split. TB scalars; optional checkpoint."""

    _model_name: ClassVar[str]

    model: BaseModel = Field(default_factory=BaseModel)
    data: PathConfig = Field(default_factory=PathConfig)
    out: Path | None = Field(default=None, description=_OUT_DESC)
    val_fraction: float = Field(default=0.2, description=_VAL_DESC)
    quick: bool = Field(default=False, description="1-epoch smoke run")
    save_model: bool = Field(default=False, description="save ckpt.pt after fit")

    def cli_cmd(self) -> None:
        from wqf7008_fb_human_or_bot.train import train_command

        train_command(
            self._model_name,
            self.model,
            self.data,
            self.out,
            self.val_fraction,
            self.quick,
            self.save_model,
        )


class CVCmd(BaseModel):
    """Evaluate one model via repeated stratified K-fold CV."""

    _model_name: ClassVar[str]

    cv: CVConfig = Field(default_factory=CVConfig)
    model: BaseModel = Field(default_factory=BaseModel)
    data: PathConfig = Field(default_factory=PathConfig)
    out: Path | None = Field(default=None, description=_OUT_DESC)

    def cli_cmd(self) -> None:
        from wqf7008_fb_human_or_bot.cv import cv_command

        cv_command(self._model_name, self.model, self.cv, self.data, self.out)


class EvalCmd(BaseModel):
    """Load a checkpoint and write a Kaggle submission CSV."""

    _model_name: ClassVar[str]

    ckpt: Path = Field(description=_CKPT_DESC)
    data: PathConfig = Field(default_factory=PathConfig)
    out: Path | None = Field(default=None, description=_SUB_DESC)

    def cli_cmd(self) -> None:
        from wqf7008_fb_human_or_bot.eval import eval_command

        eval_command(self._model_name, self.ckpt, self.data, self.out)


# ==========================================================================
# Per-model subcommands: 2 lines each.
# ==========================================================================


class TrainGBM(TrainCmd):
    """Train a Gradient Boosting Machine baseline."""

    _model_name: ClassVar[str] = "gbm"
    model: GBMConfig = Field(default_factory=GBMConfig)


class TrainHybrid(TrainCmd):
    """Train the tabular + bid-sequence BiLSTM hybrid."""

    _model_name: ClassVar[str] = "hybrid"
    model: HybridConfig = Field(default_factory=HybridConfig)


class TrainGNN(TrainCmd):
    """Train the heterogeneous graph neural network."""

    _model_name: ClassVar[str] = "gnn"
    model: GNNConfig = Field(default_factory=GNNConfig)


class TrainTFFM(TrainCmd):
    """Train the FT-Transformer tabular model."""

    _model_name: ClassVar[str] = "tffm"
    model: TFFMConfig = Field(default_factory=TFFMConfig)


class TrainResNet(TrainCmd):
    """Train the tabular ResNet baseline."""

    _model_name: ClassVar[str] = "resnet"
    model: ResNetConfig = Field(default_factory=ResNetConfig)


class Train(BaseModel):
    """Train a single model on a train/val split."""

    gbm: CliSubCommand[TrainGBM]
    hybrid: CliSubCommand[TrainHybrid]
    gnn: CliSubCommand[TrainGNN]
    tffm: CliSubCommand[TrainTFFM]
    resnet: CliSubCommand[TrainResNet]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


class CVGBM(CVCmd):
    """Evaluate the GBM baseline via repeated stratified K-fold CV."""

    _model_name: ClassVar[str] = "gbm"
    model: GBMConfig = Field(default_factory=GBMConfig)


class CVHybrid(CVCmd):
    """Evaluate the hybrid BiLSTM via repeated stratified K-fold CV."""

    _model_name: ClassVar[str] = "hybrid"
    model: HybridConfig = Field(default_factory=HybridConfig)


class CVGNN(CVCmd):
    """Evaluate the hetero-graph GNN via repeated stratified K-fold CV."""

    _model_name: ClassVar[str] = "gnn"
    model: GNNConfig = Field(default_factory=GNNConfig)


class CVTFFM(CVCmd):
    """Evaluate the FT-Transformer via repeated stratified K-fold CV."""

    _model_name: ClassVar[str] = "tffm"
    model: TFFMConfig = Field(default_factory=TFFMConfig)


class CVResNet(CVCmd):
    """Evaluate the tabular ResNet via repeated stratified K-fold CV."""

    _model_name: ClassVar[str] = "resnet"
    model: ResNetConfig = Field(default_factory=ResNetConfig)


class CV(BaseModel):
    """Evaluate a model via repeated stratified K-fold CV."""

    gbm: CliSubCommand[CVGBM]
    hybrid: CliSubCommand[CVHybrid]
    gnn: CliSubCommand[CVGNN]
    tffm: CliSubCommand[CVTFFM]
    resnet: CliSubCommand[CVResNet]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


class EvalGBM(EvalCmd):
    """Load a GBM checkpoint and write a Kaggle submission CSV."""

    _model_name: ClassVar[str] = "gbm"


class EvalHybrid(EvalCmd):
    """Load a hybrid checkpoint and write a Kaggle submission CSV."""

    _model_name: ClassVar[str] = "hybrid"


class EvalGNN(EvalCmd):
    """Load a GNN checkpoint and write a Kaggle submission CSV."""

    _model_name: ClassVar[str] = "gnn"


class EvalTFFM(EvalCmd):
    """Load an FT-Transformer checkpoint and write a Kaggle submission CSV."""

    _model_name: ClassVar[str] = "tffm"


class EvalResNet(EvalCmd):
    """Load a ResNet checkpoint and write a Kaggle submission CSV."""

    _model_name: ClassVar[str] = "resnet"


class Eval(BaseModel):
    """Load a checkpoint and write a Kaggle submission.csv."""

    gbm: CliSubCommand[EvalGBM]
    hybrid: CliSubCommand[EvalHybrid]
    gnn: CliSubCommand[EvalGNN]
    tffm: CliSubCommand[EvalTFFM]
    resnet: CliSubCommand[EvalResNet]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


class BidBot(BaseSettings):
    """bidbot: train and evaluate bidder-fraud models."""

    model_config = SettingsConfigDict(cli_implicit_flags=True, cli_kebab_case=True)

    features: CliSubCommand[Features]
    train: CliSubCommand[Train]
    cv: CliSubCommand[CV]
    eval: CliSubCommand[Eval]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    CliApp.run(BidBot)
