"""ResNet-style tabular classifier from Gorishniy et al. (2021)."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from wqf7008_fb_human_or_bot.configs import ResNetConfig
from wqf7008_fb_human_or_bot.models.base import pos_weight_from_labels
from wqf7008_fb_human_or_bot.train import resolve_device, train_torch_loop


class ResNetBlock(nn.Module):
    def __init__(self, hidden: int, block_hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.BatchNorm1d(hidden)
        self.linear1 = nn.Linear(hidden, block_hidden)
        self.linear2 = nn.Linear(block_hidden, hidden)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return residual + x


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden: int = 128,
        block_hidden: int = 256,
        n_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList(
            ResNetBlock(hidden=hidden, block_hidden=block_hidden, dropout=dropout)
            for _ in range(n_layers)
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


class ResNetBidderClassifier:
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model: TabularResNet | None = None
        self.device: torch.device = torch.device("cpu")

    def _loader(
        self, X: np.ndarray, y: np.ndarray, *, batch_size: int, shuffle: bool
    ) -> DataLoader:
        ds = TensorDataset(
            torch.as_tensor(np.asarray(X, dtype=np.float32)),
            torch.as_tensor(np.asarray(y, dtype=np.float32)),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def fit(self, fold, cfg: ResNetConfig, *, writer=None) -> None:
        self.device = resolve_device(cfg.device)
        self.model = TabularResNet(
            input_dim=self.input_dim,
            hidden=cfg.hidden,
            block_hidden=cfg.block_hidden,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        ).to(self.device)

        tr_loader = self._loader(fold.Xtr, fold.ytr, batch_size=cfg.batch_size, shuffle=True)
        Xval = torch.as_tensor(np.asarray(fold.Xval, dtype=np.float32), device=self.device)
        yval = torch.as_tensor(np.asarray(fold.yval, dtype=np.float32), device=self.device)

        pos_w = pos_weight_from_labels(fold.ytr)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=self.device))
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        def train_epoch() -> dict[str, float]:
            assert self.model is not None
            loss_sum, n_batches = 0.0, 0
            probs_all, y_all = [], []
            pbar = tqdm(tr_loader, desc="[resnet] epoch", leave=False, dynamic_ncols=True)
            for X, y in pbar:
                X = X.to(self.device)
                y = y.to(self.device)
                logits = self.model(X)
                loss = loss_fn(logits, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                loss_sum += float(loss.detach())
                n_batches += 1
                probs_all.append(torch.sigmoid(logits.detach()).cpu().numpy())
                y_all.append(y.detach().cpu().numpy())
                pbar.set_postfix(loss=f"{loss_sum / n_batches:.4f}")
            probs = np.concatenate(probs_all)
            ys = np.concatenate(y_all)
            auc = float(roc_auc_score(ys, probs)) if len(np.unique(ys)) > 1 else 0.5
            return {"loss": loss_sum / max(n_batches, 1), "auc": auc}

        @torch.no_grad()
        def eval_fn() -> dict[str, float]:
            assert self.model is not None
            self.model.eval()
            logits = self.model(Xval)
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
            model_name="resnet",
            train_epoch=train_epoch,
            eval_fn=eval_fn,
            writer=writer,
        )

    @torch.no_grad()
    def predict_proba(self, fold_predict) -> np.ndarray:
        assert self.model is not None, "call fit() first"
        self.model.eval()
        X = torch.as_tensor(np.asarray(fold_predict.Xval, dtype=np.float32), device=self.device)
        return torch.sigmoid(self.model(X)).cpu().numpy()
