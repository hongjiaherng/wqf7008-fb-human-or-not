"""Heterogeneous GNN classifier using SAGEConv + to_hetero."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch_geometric.nn import SAGEConv, to_hetero

from wqf7008_fb_human_or_bot.configs import GNNConfig
from wqf7008_fb_human_or_bot.datasets.bidder_graph import GraphBundle
from wqf7008_fb_human_or_bot.models.base import pos_weight_from_labels
from wqf7008_fb_human_or_bot.train import resolve_device, train_torch_loop


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


class GNNBidderClassifier:
    def __init__(self, bundle: GraphBundle):
        self.bundle = bundle
        self.model: HeteroBidderGNN | None = None
        self.device: torch.device = torch.device("cpu")

    def fit(self, fold, cfg: GNNConfig, *, writer=None) -> None:
        self.device = resolve_device(cfg.device)
        data = self.bundle.data.to(self.device)
        self.model = HeteroBidderGNN(self.bundle, hidden=cfg.hidden, dropout=cfg.dropout).to(
            self.device
        )
        # SAGEConv lazy init needs one dummy forward to materialise params.
        self.model(data)

        train_idx = torch.as_tensor(fold.train_idx, dtype=torch.long, device=self.device)
        val_idx = torch.as_tensor(fold.val_idx, dtype=torch.long, device=self.device)
        ytr = torch.as_tensor(fold.ytr, dtype=torch.float32, device=self.device)
        yval = torch.as_tensor(fold.yval, dtype=torch.float32, device=self.device)

        pos_w = pos_weight_from_labels(fold.ytr)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=self.device))
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        def train_epoch() -> dict[str, float]:
            assert self.model is not None
            logits = self.model(data, train_idx)
            loss = loss_fn(logits, ytr)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            ys = ytr.detach().cpu().numpy()
            probs = torch.sigmoid(logits.detach()).cpu().numpy()
            auc = float(roc_auc_score(ys, probs)) if len(np.unique(ys)) > 1 else 0.5
            return {"loss": float(loss.detach()), "auc": auc}

        @torch.no_grad()
        def eval_fn() -> dict[str, float]:
            assert self.model is not None
            self.model.eval()
            logits = self.model(data, val_idx)
            loss = loss_fn(logits, yval)
            ys = yval.cpu().numpy()
            auc = (
                float(roc_auc_score(ys, torch.sigmoid(logits).cpu().numpy()))
                if len(np.unique(ys)) > 1
                else 0.5
            )
            return {"loss": float(loss.detach()), "auc": auc}

        train_torch_loop(
            self.model,
            cfg,
            model_name="gnn",
            train_epoch=train_epoch,
            eval_fn=eval_fn,
            writer=writer,
        )

    @torch.no_grad()
    def predict_proba(self, fold_predict) -> np.ndarray:
        assert self.model is not None, "call fit() first"
        self.model.eval()
        data = self.bundle.data.to(self.device)
        idx = torch.as_tensor(fold_predict.val_idx, dtype=torch.long, device=self.device)
        return torch.sigmoid(self.model(data, idx)).cpu().numpy()
