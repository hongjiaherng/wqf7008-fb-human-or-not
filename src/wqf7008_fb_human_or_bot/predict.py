"""Load a saved checkpoint and produce test-set probabilities.

Each `predict_<model>(ckpt, data)` returns a 1D numpy array aligned to
`data.ids_te`. The CLI (`bidbot predict <model>`) wraps these and writes a
Kaggle submission CSV.
"""

from pathlib import Path

import numpy as np

from wqf7008_fb_human_or_bot.configs import GNNConfig, HybridConfig, TFFMConfig
from wqf7008_fb_human_or_bot.train import FoldData, resolve_device


def _feat_cols(X) -> list[str]:
    return [c for c in X.columns if c != "bidder_id"]


def _test_fold(data, feat: list[str]) -> FoldData:
    """FoldData scaffold for test-set inference. `Xtr` / `ytr` / `ytr_*` are
    unused by the models' `predict_proba`, but `FoldData` requires them."""
    Xte_np = data.Xte.select(feat).to_numpy().astype(np.float32)
    n = len(data.ids_te)
    return FoldData(
        Xtr=Xte_np,
        Xval=Xte_np,
        ytr=np.zeros(n, dtype=np.int64),
        yval=None,
        ids_tr=data.ids_te,
        ids_val=data.ids_te,
    )


def predict_gbm(ckpt: Path, data) -> np.ndarray:
    from wqf7008_fb_human_or_bot.models.gbm import GBMBidderClassifier

    clf = GBMBidderClassifier.load(ckpt)
    feat = _feat_cols(data.Xtr)
    Xte_np = data.Xte.select(feat).to_numpy().astype(np.float32)
    return clf.predict_proba(Xte_np)


def predict_hybrid(ckpt: Path, data) -> np.ndarray:
    import torch

    from wqf7008_fb_human_or_bot.datasets import build_sequence_store
    from wqf7008_fb_human_or_bot.models.hybrid import (
        HybridBidderClassifier,
        HybridBidderModel,
    )

    payload = torch.load(ckpt, weights_only=False, map_location="cpu")
    cfg = HybridConfig.model_validate(payload["cfg"])
    feat = _feat_cols(data.Xtr)
    store = build_sequence_store(data.bids, max_len=cfg.max_len)

    clf = HybridBidderClassifier(store=store, tabular_dim=len(feat))
    clf.device = resolve_device(cfg.device)
    clf.model = HybridBidderModel(
        vocab_sizes=store.vocab_sizes,
        tabular_dim=len(feat),
        hidden=cfg.hidden,
        dropout=cfg.dropout,
    ).to(clf.device)
    clf.model.load_state_dict(payload["model_state"])
    return clf.predict_proba(_test_fold(data, feat))


def predict_gnn(ckpt: Path, data) -> np.ndarray:
    import torch

    from wqf7008_fb_human_or_bot.datasets import build_hetero_graph
    from wqf7008_fb_human_or_bot.models.gnn import GNNBidderClassifier, HeteroBidderGNN

    payload = torch.load(ckpt, weights_only=False, map_location="cpu")
    cfg = GNNConfig.model_validate(payload["cfg"])
    feat = _feat_cols(data.Xtr)
    X_union = np.vstack(
        [data.Xtr.select(feat).to_numpy(), data.Xte.select(feat).to_numpy()]
    ).astype(np.float32)
    all_ids = np.concatenate([data.ids_tr, data.ids_te])
    bundle = build_hetero_graph(data.bids, all_ids, X_union)

    clf = GNNBidderClassifier(bundle=bundle)
    clf.device = resolve_device(cfg.device)
    clf.model = HeteroBidderGNN(bundle, hidden=cfg.hidden, dropout=cfg.dropout).to(clf.device)
    # SAGEConv has lazy-init params; one forward pass materialises them before load.
    clf.model(bundle.data.to(clf.device))
    clf.model.load_state_dict(payload["model_state"])

    fold = _test_fold(data, feat)
    fold.val_idx = np.array([bundle.bidder_index[b] for b in data.ids_te], dtype=np.int64)
    return clf.predict_proba(fold)


def predict_tffm(ckpt: Path, data) -> np.ndarray:
    import torch
    from torch_frame.nn.models import FTTransformer as TFFTTransformer

    from wqf7008_fb_human_or_bot.models.tffm import FTTransformerBidderClassifier

    payload = torch.load(ckpt, weights_only=False, map_location="cpu")
    cfg = TFFMConfig.model_validate(payload["cfg"])
    feat = _feat_cols(data.Xtr)
    Xtr_np = data.Xtr.select(feat).to_numpy().astype(np.float32)

    clf = FTTransformerBidderClassifier(feature_names=feat)
    clf.device = resolve_device(cfg.device)
    # FT-Transformer needs col_stats from a materialised training tensor frame
    # to instantiate; those stats aren't serialised in the ckpt.
    tr_ds = clf._to_tf(Xtr_np, data.ytr)
    clf.model = TFFTTransformer(
        channels=cfg.channels,
        out_channels=1,
        num_layers=cfg.num_layers,
        col_stats=tr_ds.col_stats,
        col_names_dict=tr_ds.tensor_frame.col_names_dict,
    ).to(clf.device)
    clf.model.load_state_dict(payload["model_state"])
    return clf.predict_proba(_test_fold(data, feat))
