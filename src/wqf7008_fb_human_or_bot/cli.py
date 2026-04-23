"""bidbot CLI (pydantic-settings).

Top-level subcommands:
  features build   — cache the per-bidder tabular feature matrix
  train   {model}  — single train/val split, TensorBoard on, save_model allowed
  cv      {model}  — repeated stratified K-fold, no TensorBoard, no save
  predict {model}  — load a saved ckpt and write a Kaggle submission CSV

Flag namespaces:
  --model.<field>  — training + architecture (lr, epochs, hidden, seed, device, ...)
  --cv.<field>     — CV protocol (n_splits, n_repeats, quick)  [cv subcommand only]
  --data.<field>   — filesystem paths
  <flat>           — run-scope knobs on the subcommand (out, val_fraction, quick, ...)
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, CliApp, CliSubCommand, SettingsConfigDict

from wqf7008_fb_human_or_bot.configs import (
    PROJECT_ROOT,
    RUNS_DIR,
    SSL_CKPT,
    CVConfig,
    DataConfig,
    GBMConfig,
    GNNConfig,
    HybridConfig,
    TFFMConfig,
)

# ---------- small helpers ----------


def _rel(p) -> str:
    """Path as `./<rel>` under PROJECT_ROOT, otherwise the absolute path."""
    try:
        return f"./{Path(p).resolve().relative_to(PROJECT_ROOT)}".replace("\\", "/")
    except ValueError:
        return str(p)


def _fmt(cfg) -> str:
    """One-line `k=v k=v …` view of a pydantic config; paths get relativised."""
    return " ".join(
        f"{k}={_rel(v) if isinstance(v, Path) else v}"
        for k, v in cfg.model_dump().items()
        if not isinstance(v, dict)
    )


def _header(title: str, out: Path, **sections: str) -> None:
    print(f"\n== {title} ==")
    print(f"  out:   {_rel(out)}")
    for name, value in sections.items():
        print(f"  {name:6} {value}")
    print()


def _resolve_device(device: str) -> str:
    """Concrete device string for display in the run header. Models also
    resolve internally via `train.resolve_device`, but we materialise here so
    the header shows `cuda`/`cpu` rather than `auto`."""
    from wqf7008_fb_human_or_bot.train import default_device

    return default_device() if device == "auto" else device


def _run_dir(explicit: Path | None, mode: str, tag: str) -> Path:
    if explicit is not None:
        return explicit
    return RUNS_DIR / mode / f"{datetime.now():%Y%m%dT%H%M%S}_{tag}"


# ---------- data loading ----------


@dataclass
class Data:
    bids: pl.DataFrame
    Xtr: pl.DataFrame
    ytr: np.ndarray
    ids_tr: np.ndarray
    Xte: pl.DataFrame
    ids_te: np.ndarray


def _load(data_cfg: DataConfig) -> Data:
    from wqf7008_fb_human_or_bot.features import build_tabular

    bids = pl.read_csv(data_cfg.data_dir / "bids.csv")
    train = pl.read_csv(data_cfg.data_dir / "train.csv")
    test = pl.read_csv(data_cfg.data_dir / "test.csv")
    Xtr, ids_tr, ytr = build_tabular(bids, train, data_cfg)
    Xte, ids_te, _ = build_tabular(bids, test, data_cfg)
    assert ytr is not None
    return Data(bids=bids, Xtr=Xtr, ytr=ytr, ids_tr=ids_tr, Xte=Xte, ids_te=ids_te)


def _feature_cols(X) -> list[str]:
    return [c for c in X.columns if c != "bidder_id"]


def _write_summary(result, out: Path) -> None:
    from wqf7008_fb_human_or_bot.metrics import plot_roc, save_cv_summary

    save_cv_summary(result, out)
    plot_roc(result, out / "roc.png")
    print(result.summary_str())


# ---------- SSL pretrain ----------


def _maybe_pretrain(data: Data, cfg: HybridConfig, quick: bool) -> Path:
    """Resolve SSL checkpoint: user path → default reuse → pretrain+write default."""
    if cfg.pretrain_ckpt is not None:
        ckpt = Path(cfg.pretrain_ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"pretrain_ckpt not found: {ckpt}")
        print(f"  [pretrain] using user-supplied checkpoint at {_rel(ckpt)}")
        return ckpt
    if SSL_CKPT.exists():
        print(f"  [pretrain] reusing existing checkpoint at {_rel(SSL_CKPT)}")
        return SSL_CKPT
    from wqf7008_fb_human_or_bot.datasets import build_sequence_store
    from wqf7008_fb_human_or_bot.pretrain import PretrainConfig, pretrain_masked_bid

    store = build_sequence_store(data.bids, max_len=cfg.max_len)
    all_ids = np.concatenate([data.ids_tr, data.ids_te])
    pcfg = PretrainConfig(
        epochs=1 if quick else cfg.pretrain_epochs,
        batch_size=cfg.batch_size,
        device=cfg.device,
        seed=cfg.seed,
        hidden=cfg.hidden,
        dropout=cfg.dropout,
    )
    return pretrain_masked_bid(store, all_ids, pcfg, SSL_CKPT)


# ---------- per-model factories (lazy imports keep --help fast) ----------


def _gbm_factory(cfg: GBMConfig):
    from wqf7008_fb_human_or_bot.models.gbm import GBMBidderClassifier

    return lambda: GBMBidderClassifier(cfg=cfg)


def _hybrid_factory(cfg: HybridConfig, data: Data, ssl_ckpt: Path | None):
    from wqf7008_fb_human_or_bot.datasets import build_sequence_store
    from wqf7008_fb_human_or_bot.models.hybrid import HybridBidderClassifier

    store = build_sequence_store(data.bids, max_len=cfg.max_len)
    tabular_dim = len(_feature_cols(data.Xtr))
    return lambda: HybridBidderClassifier(
        store=store, tabular_dim=tabular_dim, pretrained_ckpt=ssl_ckpt
    )


def _gnn_graph(data: Data):
    from wqf7008_fb_human_or_bot.datasets import build_hetero_graph

    feat_cols = _feature_cols(data.Xtr)
    X_union = np.vstack(
        [data.Xtr.select(feat_cols).to_numpy(), data.Xte.select(feat_cols).to_numpy()]
    ).astype(np.float32)
    all_ids = np.concatenate([data.ids_tr, data.ids_te])
    return build_hetero_graph(data.bids, all_ids, X_union)


def _gnn_factory(bundle):
    from wqf7008_fb_human_or_bot.models.gnn import GNNBidderClassifier

    return lambda: GNNBidderClassifier(bundle=bundle)


def _tffm_factory(feature_names: list[str]):
    from wqf7008_fb_human_or_bot.models.tffm import FTTransformerBidderClassifier

    return lambda: FTTransformerBidderClassifier(feature_names=feature_names)


def _save_torch_ckpt(clf, cfg, out_path: Path) -> None:
    import torch

    inner = getattr(clf, "model", None)
    assert inner is not None, "classifier has no .model attribute to save"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": inner.state_dict(), "cfg": cfg.model_dump(mode="json")}, out_path)
    print(f"  saved checkpoint to {_rel(out_path)}")


# ---------- shared train/cv runners ----------


# `build` is a closure that can use `self` for SSL pretrain / GNN graph etc.
# It returns (factory, bidder_to_graph_idx_or_None) — both fed into run_train/run_cv.
Build = Callable[[Data], tuple[Callable, dict | None]]


def _do_train(
    *,
    tag: str,
    model_name: str,
    model_cfg,
    data_cfg: DataConfig,
    out: Path | None,
    val_fraction: float,
    quick: bool,
    save_model: bool,
    build: Build,
) -> None:
    from wqf7008_fb_human_or_bot.train import run_train

    data = _load(data_cfg)
    # val_fraction==0 → full-fit submission run; flag it in the folder name.
    dir_tag = f"{tag}_full" if val_fraction == 0 else tag
    out_dir = _run_dir(out, "train", dir_tag)
    _header(
        f"train {tag}",
        out_dir,
        model=_fmt(model_cfg),
        data=_fmt(data_cfg),
        run=f"val_fraction={val_fraction} quick={quick} save_model={save_model}",
    )
    factory, bidder_to_graph_idx = build(data)
    clf, result = run_train(
        factory=factory,
        X=data.Xtr,
        y=data.ytr,
        bidder_ids=data.ids_tr,
        val_fraction=val_fraction,
        quick=quick,
        model_cfg=model_cfg,
        model_name=model_name,
        tb_dir=out_dir / "tb",
        bidder_to_graph_idx=bidder_to_graph_idx,
    )
    _write_summary(result, out_dir)
    if save_model:
        from wqf7008_fb_human_or_bot.models.gbm import GBMBidderClassifier

        if isinstance(clf, GBMBidderClassifier):
            clf.save(out_dir / "ckpt.pt")
        else:
            _save_torch_ckpt(clf, model_cfg, out_dir / "ckpt.pt")


def _do_cv(
    *,
    tag: str,
    model_name: str,
    cv_cfg: CVConfig,
    model_cfg,
    data_cfg: DataConfig,
    out: Path | None,
    build: Build,
) -> None:
    from wqf7008_fb_human_or_bot.train import run_cv

    data = _load(data_cfg)
    out_dir = _run_dir(out, "cv", tag)
    _header(
        f"cv {tag}",
        out_dir,
        cv=_fmt(cv_cfg),
        model=_fmt(model_cfg),
        data=_fmt(data_cfg),
    )
    factory, bidder_to_graph_idx = build(data)
    result = run_cv(
        factory=factory,
        X=data.Xtr,
        y=data.ytr,
        bidder_ids=data.ids_tr,
        cv=cv_cfg,
        model_cfg=model_cfg,
        model_name=model_name,
        bidder_to_graph_idx=bidder_to_graph_idx,
    )
    _write_summary(result, out_dir)


# ==========================================================================
# CLI subcommand classes. Class docstrings become --help descriptions.
# ==========================================================================


class FeaturesBuild(BaseModel):
    """Build & cache the per-bidder tabular feature matrices."""

    data: DataConfig = Field(default_factory=DataConfig)
    force: bool = Field(default=False, description="clear existing parquet cache first")

    def cli_cmd(self) -> None:
        if self.force:
            for p in self.data.cache_dir.glob("tabular_*.parquet"):
                p.unlink()
        d = _load(self.data)
        print(f"train: {d.Xtr.shape}, test: {d.Xte.shape}")


class Features(BaseModel):
    """Feature engineering commands."""

    build: CliSubCommand[FeaturesBuild]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


# ---------- train {model} ----------

_VAL_DESC = "fraction of train held out for val (0 = fit on all data, for final submission)"
_OUT_DESC = (
    "output dir (default: runs/{train|cv}/{timestamp}_{tag}; train runs with "
    "val_fraction=0 additionally get a `_full` suffix)"
)


class TrainGBM(BaseModel):
    """Train one GBM (no CV). TB scalars; optionally saves the fitted model."""

    model: GBMConfig = Field(default_factory=GBMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    out: Path | None = Field(default=None, description=_OUT_DESC)
    val_fraction: float = Field(default=0.2, description=_VAL_DESC)
    quick: bool = Field(default=False, description="1-epoch smoke run")
    save_model: bool = Field(default=False, description="save ckpt.pt after fit")

    def cli_cmd(self) -> None:
        _do_train(
            tag="gbm",
            model_name="gbm",
            model_cfg=self.model,
            data_cfg=self.data,
            out=self.out,
            val_fraction=self.val_fraction,
            quick=self.quick,
            save_model=self.save_model,
            build=lambda _d: (_gbm_factory(self.model), None),
        )


class TrainHybrid(BaseModel):
    """Train one bi-LSTM hybrid (no CV). TB scalars + optional checkpoint."""

    model: HybridConfig = Field(default_factory=HybridConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    out: Path | None = Field(default=None, description=_OUT_DESC)
    val_fraction: float = Field(default=0.2, description=_VAL_DESC)
    quick: bool = Field(default=False, description="1-epoch smoke run")
    save_model: bool = Field(default=False, description="save ckpt.pt after fit")

    def cli_cmd(self) -> None:
        self.model.device = _resolve_device(self.model.device)
        tag = "hybrid_ssl" if self.model.ssl_pretrain else "hybrid"

        def build(data: Data):
            ssl_ckpt = (
                _maybe_pretrain(data, self.model, self.quick) if self.model.ssl_pretrain else None
            )
            return _hybrid_factory(self.model, data, ssl_ckpt), None

        _do_train(
            tag=tag,
            model_name="hybrid",
            model_cfg=self.model,
            data_cfg=self.data,
            out=self.out,
            val_fraction=self.val_fraction,
            quick=self.quick,
            save_model=self.save_model,
            build=build,
        )


class TrainGNN(BaseModel):
    """Train one hetero-graph GNN (no CV). TB scalars + optional checkpoint."""

    model: GNNConfig = Field(default_factory=GNNConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    out: Path | None = Field(default=None, description=_OUT_DESC)
    val_fraction: float = Field(default=0.2, description=_VAL_DESC)
    quick: bool = Field(default=False, description="1-epoch smoke run")
    save_model: bool = Field(default=False, description="save ckpt.pt after fit")

    def cli_cmd(self) -> None:
        self.model.device = _resolve_device(self.model.device)

        def build(data: Data):
            bundle = _gnn_graph(data)
            return _gnn_factory(bundle), bundle.bidder_index

        _do_train(
            tag="gnn",
            model_name="gnn",
            model_cfg=self.model,
            data_cfg=self.data,
            out=self.out,
            val_fraction=self.val_fraction,
            quick=self.quick,
            save_model=self.save_model,
            build=build,
        )


class TrainTFFM(BaseModel):
    """Train one FT-Transformer (no CV). TB scalars + optional checkpoint."""

    model: TFFMConfig = Field(default_factory=TFFMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    out: Path | None = Field(default=None, description=_OUT_DESC)
    val_fraction: float = Field(default=0.2, description=_VAL_DESC)
    quick: bool = Field(default=False, description="1-epoch smoke run")
    save_model: bool = Field(default=False, description="save ckpt.pt after fit")

    def cli_cmd(self) -> None:
        self.model.device = _resolve_device(self.model.device)
        _do_train(
            tag="tffm",
            model_name="tffm",
            model_cfg=self.model,
            data_cfg=self.data,
            out=self.out,
            val_fraction=self.val_fraction,
            quick=self.quick,
            save_model=self.save_model,
            build=lambda d: (_tffm_factory(_feature_cols(d.Xtr)), None),
        )


class Train(BaseModel):
    """Train a single model on a train/val split."""

    gbm: CliSubCommand[TrainGBM]
    hybrid: CliSubCommand[TrainHybrid]
    gnn: CliSubCommand[TrainGNN]
    tffm: CliSubCommand[TrainTFFM]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


# ---------- cv {model} ----------


class CVGBM(BaseModel):
    """Evaluate GBM via repeated stratified K-fold CV."""

    cv: CVConfig = Field(default_factory=CVConfig)
    model: GBMConfig = Field(default_factory=GBMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    out: Path | None = Field(default=None, description=_OUT_DESC)

    def cli_cmd(self) -> None:
        _do_cv(
            tag="gbm",
            model_name="gbm",
            cv_cfg=self.cv,
            model_cfg=self.model,
            data_cfg=self.data,
            out=self.out,
            build=lambda _d: (_gbm_factory(self.model), None),
        )


class CVHybrid(BaseModel):
    """Evaluate hybrid bi-LSTM via repeated stratified K-fold CV."""

    cv: CVConfig = Field(default_factory=CVConfig)
    model: HybridConfig = Field(default_factory=HybridConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    out: Path | None = Field(default=None, description=_OUT_DESC)

    def cli_cmd(self) -> None:
        self.model.device = _resolve_device(self.model.device)
        tag = "hybrid_ssl" if self.model.ssl_pretrain else "hybrid"

        def build(data: Data):
            ssl_ckpt = (
                _maybe_pretrain(data, self.model, self.cv.quick)
                if self.model.ssl_pretrain
                else None
            )
            return _hybrid_factory(self.model, data, ssl_ckpt), None

        _do_cv(
            tag=tag,
            model_name="hybrid",
            cv_cfg=self.cv,
            model_cfg=self.model,
            data_cfg=self.data,
            out=self.out,
            build=build,
        )


class CVGNN(BaseModel):
    """Evaluate hetero-graph GNN via repeated stratified K-fold CV."""

    cv: CVConfig = Field(default_factory=CVConfig)
    model: GNNConfig = Field(default_factory=GNNConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    out: Path | None = Field(default=None, description=_OUT_DESC)

    def cli_cmd(self) -> None:
        self.model.device = _resolve_device(self.model.device)

        def build(data: Data):
            bundle = _gnn_graph(data)
            return _gnn_factory(bundle), bundle.bidder_index

        _do_cv(
            tag="gnn",
            model_name="gnn",
            cv_cfg=self.cv,
            model_cfg=self.model,
            data_cfg=self.data,
            out=self.out,
            build=build,
        )


class CVTFFM(BaseModel):
    """Evaluate FT-Transformer via repeated stratified K-fold CV."""

    cv: CVConfig = Field(default_factory=CVConfig)
    model: TFFMConfig = Field(default_factory=TFFMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    out: Path | None = Field(default=None, description=_OUT_DESC)

    def cli_cmd(self) -> None:
        self.model.device = _resolve_device(self.model.device)
        _do_cv(
            tag="tffm",
            model_name="tffm",
            cv_cfg=self.cv,
            model_cfg=self.model,
            data_cfg=self.data,
            out=self.out,
            build=lambda d: (_tffm_factory(_feature_cols(d.Xtr)), None),
        )


class CV(BaseModel):
    """Evaluate a model via repeated stratified K-fold CV."""

    gbm: CliSubCommand[CVGBM]
    hybrid: CliSubCommand[CVHybrid]
    gnn: CliSubCommand[CVGNN]
    tffm: CliSubCommand[CVTFFM]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


# ---------- predict {model} ----------


def _do_predict(predict_fn, ckpt: Path, data_cfg: DataConfig, out: Path | None) -> None:
    from wqf7008_fb_human_or_bot.metrics import write_submission

    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    # Default: drop submission.csv next to the checkpoint.
    out = out if out is not None else ckpt.parent / "submission.csv"
    data = _load(data_cfg)
    _header("predict", out, ckpt=_rel(ckpt), data=_fmt(data_cfg))
    preds = predict_fn(ckpt, data)
    path = write_submission(data.ids_te, preds, out)
    print(f"  wrote {_rel(path)}  ({len(preds)} rows)")


_CKPT_DESC = "path to a ckpt.pt written by `bidbot train ... --save-model`"
_SUB_DESC = "output submission CSV (default: <ckpt_dir>/submission.csv)"


class PredictGBM(BaseModel):
    """Load a GBM checkpoint and write a Kaggle submission CSV."""

    ckpt: Path = Field(description=_CKPT_DESC)
    data: DataConfig = Field(default_factory=DataConfig)
    out: Path | None = Field(default=None, description=_SUB_DESC)

    def cli_cmd(self) -> None:
        from wqf7008_fb_human_or_bot.predict import predict_gbm

        _do_predict(predict_gbm, self.ckpt, self.data, self.out)


class PredictHybrid(BaseModel):
    """Load a hybrid checkpoint and write a Kaggle submission CSV."""

    ckpt: Path = Field(description=_CKPT_DESC)
    data: DataConfig = Field(default_factory=DataConfig)
    out: Path | None = Field(default=None, description=_SUB_DESC)

    def cli_cmd(self) -> None:
        from wqf7008_fb_human_or_bot.predict import predict_hybrid

        _do_predict(predict_hybrid, self.ckpt, self.data, self.out)


class PredictGNN(BaseModel):
    """Load a GNN checkpoint and write a Kaggle submission CSV."""

    ckpt: Path = Field(description=_CKPT_DESC)
    data: DataConfig = Field(default_factory=DataConfig)
    out: Path | None = Field(default=None, description=_SUB_DESC)

    def cli_cmd(self) -> None:
        from wqf7008_fb_human_or_bot.predict import predict_gnn

        _do_predict(predict_gnn, self.ckpt, self.data, self.out)


class PredictTFFM(BaseModel):
    """Load an FT-Transformer checkpoint and write a Kaggle submission CSV."""

    ckpt: Path = Field(description=_CKPT_DESC)
    data: DataConfig = Field(default_factory=DataConfig)
    out: Path | None = Field(default=None, description=_SUB_DESC)

    def cli_cmd(self) -> None:
        from wqf7008_fb_human_or_bot.predict import predict_tffm

        _do_predict(predict_tffm, self.ckpt, self.data, self.out)


class Predict(BaseModel):
    """Load a checkpoint and write a Kaggle submission.csv."""

    gbm: CliSubCommand[PredictGBM]
    hybrid: CliSubCommand[PredictHybrid]
    gnn: CliSubCommand[PredictGNN]
    tffm: CliSubCommand[PredictTFFM]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


class BidBot(BaseSettings):
    """bidbot — train & evaluate bidder-fraud models."""

    model_config = SettingsConfigDict(cli_implicit_flags=True, cli_kebab_case=True)

    features: CliSubCommand[Features]
    train: CliSubCommand[Train]
    cv: CliSubCommand[CV]
    predict: CliSubCommand[Predict]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    CliApp.run(BidBot)
