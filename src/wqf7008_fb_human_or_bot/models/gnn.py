"""Heterogeneous GNN classifier using SAGEConv + to_hetero."""

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch_geometric.nn import SAGEConv, to_hetero

from wqf7008_fb_human_or_bot.datasets import Data
from wqf7008_fb_human_or_bot.datasets.graph import GraphBundle
from wqf7008_fb_human_or_bot.models.base import BidderClassifier, Split
from wqf7008_fb_human_or_bot.models.registry import GNNConfig, pos_weight_from_labels


class _HomoSAGE(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden)
        self.conv2 = SAGEConv((-1, -1), hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        return self.conv2(x, edge_index).relu()


class HeteroBidderGNN(nn.Module):
    def __init__(self, bundle: GraphBundle, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        data = bundle.data
        self.auction_emb = nn.Embedding(int(data["auction"].num_nodes), hidden)
        self.device_emb = nn.Embedding(int(data["device"].num_nodes), hidden)
        self.ip_emb = nn.Embedding(int(data["ip"].num_nodes), hidden)
        # LayerNorm on raw bidder features (counts aren't normalized).
        in_dim = data["bidder"].x.size(-1)
        self.bidder_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden))
        self.hetero = to_hetero(_HomoSAGE(hidden, dropout), data.metadata(), aggr="sum")
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data, node_idx: torch.Tensor | None = None) -> torch.Tensor:
        x_dict = {
            "bidder": self.bidder_proj(data["bidder"].x),
            "auction": self.auction_emb.weight,
            "device": self.device_emb.weight,
            "ip": self.ip_emb.weight,
        }
        h = self.hetero(x_dict, data.edge_index_dict)["bidder"]
        if node_idx is not None:
            h = h[node_idx]
        return self.head(h).squeeze(-1)


class GNNBidderClassifier(BidderClassifier[GNNConfig]):
    name: ClassVar[str] = "gnn"

    model: nn.Module
    device: torch.device
    _loss_fn: nn.Module
    _opt: torch.optim.Optimizer

    def __init__(self, cfg: GNNConfig, bundle: GraphBundle):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.bundle = bundle
        self.model = HeteroBidderGNN(bundle, hidden=cfg.hidden, dropout=cfg.dropout).to(self.device)
        # SAGEConv has lazy-init params; one forward pass materialises them.
        self.model(bundle.data.to(self.device))

    @classmethod
    def from_data(cls, cfg: GNNConfig, data: Data) -> Callable[[Split], Self]:
        bundle = cls._build_bundle(data)
        return lambda _train: cls(cfg, bundle)

    @staticmethod
    def _build_bundle(data: Data) -> GraphBundle:
        from wqf7008_fb_human_or_bot.datasets import feature_cols
        from wqf7008_fb_human_or_bot.datasets.graph import build_hetero_graph

        feat = feature_cols(data.Xtr)
        X_union = np.vstack(
            [data.Xtr.select(feat).to_numpy(), data.Xte.select(feat).to_numpy()]
        ).astype(np.float32)
        all_ids = np.concatenate([data.ids_tr, data.ids_te])
        return build_hetero_graph(data.bids, all_ids, X_union)

    def _node_idx(self, ids: np.ndarray) -> torch.Tensor:
        idx = np.array([self.bundle.bidder_index[b] for b in ids], dtype=np.int64)
        return torch.as_tensor(idx, dtype=torch.long, device=self.device)

    def _setup(self, train: Split, val: Split) -> None:
        self._data = self.bundle.data.to(self.device)
        self._train_idx = self._node_idx(train.ids)
        self._val_idx = self._node_idx(val.ids)
        self._ytr = torch.as_tensor(np.asarray(train.y), dtype=torch.float32, device=self.device)
        self._yval = torch.as_tensor(np.asarray(val.y), dtype=torch.float32, device=self.device)

    def _train_epoch(self) -> dict[str, float]:
        logits = self.model(self._data, self._train_idx)
        loss = self._loss_fn(logits, self._ytr)
        self._opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self._opt.step()
        ys = self._ytr.detach().cpu().numpy()
        probs = torch.sigmoid(logits.detach()).cpu().numpy()
        auc = float(roc_auc_score(ys, probs)) if len(np.unique(ys)) > 1 else 0.5
        return {"loss": float(loss.detach()), "auc": auc}

    @torch.no_grad()
    def _evaluate(self) -> dict[str, float]:
        self.model.eval()
        logits = self.model(self._data, self._val_idx)
        loss = self._loss_fn(logits, self._yval)
        ys = self._yval.cpu().numpy()
        auc = (
            float(roc_auc_score(ys, torch.sigmoid(logits).cpu().numpy()))
            if len(np.unique(ys)) > 1
            else 0.5
        )
        return {"loss": float(loss.detach()), "auc": auc}

    def fit(self, train: Split, val: Split, writer=None) -> None:
        assert train.y is not None and val.y is not None  # labels present during fit
        pos_w = pos_weight_from_labels(train.y)
        self._loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=self.device))
        self._opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self._setup(train, val)

        best_auc = -1.0
        best_state: dict[str, torch.Tensor] | None = None
        patience = 0
        for epoch in range(self.cfg.epochs):
            self.model.train()
            train_metrics = self._train_epoch()
            val_metrics = self._evaluate()

            if writer is not None:
                for k in train_metrics:
                    writer.add_scalars(k, {"train": train_metrics[k], "val": val_metrics[k]}, epoch)

            val_auc = val_metrics.get("auc", 0.0)
            marker = ""
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience = 0
                marker = " *"
            else:
                patience += 1
            train_loss = train_metrics.get("loss", float("nan"))
            print(
                f"    [{self.name}] epoch {epoch + 1:02d}/{self.cfg.epochs}: "
                f"train_loss={train_loss:.4f} val_auc={val_auc:.4f} "
                f"best={best_auc:.4f} patience={patience}/{self.cfg.early_stop_patience}{marker}"
            )
            if patience >= self.cfg.early_stop_patience:
                print(f"    [{self.name}] early stop at epoch {epoch + 1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    @torch.no_grad()
    def predict_proba(self, split: Split) -> np.ndarray:
        self.model.eval()
        data = self.bundle.data.to(self.device)
        idx = self._node_idx(split.ids)
        return torch.sigmoid(self.model(data, idx)).cpu().numpy()

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": self.model.state_dict(), "cfg": self.cfg.model_dump(mode="json")},
            Path(path),
        )

    @classmethod
    def load(cls, ckpt: Path, data: Data) -> Self:
        payload = torch.load(Path(ckpt), weights_only=False, map_location="cpu")
        cfg = GNNConfig.model_validate(payload["cfg"])
        bundle = cls._build_bundle(data)
        inst = cls(cfg, bundle)  # __init__ materialises lazy params
        inst.model.load_state_dict(payload["model_state"])
        return inst
