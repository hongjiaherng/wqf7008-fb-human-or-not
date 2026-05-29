"""ResNet-style tabular classifier from Gorishniy et al. (2021)."""

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from wqf7008_fb_human_or_bot.datasets import Data
from wqf7008_fb_human_or_bot.models.base import BidderClassifier, Split
from wqf7008_fb_human_or_bot.models.registry import ResNetConfig, pos_weight_from_labels


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


class ResNetBidderClassifier(BidderClassifier[ResNetConfig]):
    name: ClassVar[str] = "resnet"

    model: nn.Module
    device: torch.device
    _loss_fn: nn.Module
    _opt: torch.optim.Optimizer

    def __init__(self, cfg: ResNetConfig, input_dim: int):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = TabularResNet(
            input_dim=input_dim,
            hidden=cfg.hidden,
            block_hidden=cfg.block_hidden,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        ).to(self.device)

    @classmethod
    def from_data(cls, cfg: ResNetConfig, data: Data) -> Callable[[Split], Self]:
        from wqf7008_fb_human_or_bot.datasets import feature_cols

        input_dim = len(feature_cols(data.Xtr))
        return lambda _train: cls(cfg, input_dim)

    def _loader(
        self, X: np.ndarray, y: np.ndarray, *, batch_size: int, shuffle: bool
    ) -> DataLoader:
        ds = TensorDataset(
            torch.as_tensor(np.asarray(X, dtype=np.float32)),
            torch.as_tensor(np.asarray(y, dtype=np.float32)),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def _setup(self, train: Split, val: Split) -> None:
        assert train.y is not None and val.y is not None
        self._tr_loader = self._loader(
            train.X, train.y, batch_size=self.cfg.batch_size, shuffle=True
        )
        self._Xval = torch.as_tensor(np.asarray(val.X, dtype=np.float32), device=self.device)
        self._yval = torch.as_tensor(np.asarray(val.y, dtype=np.float32), device=self.device)

    def _train_epoch(self) -> dict[str, float]:
        loss_sum, n_batches = 0.0, 0
        probs_all, y_all = [], []
        pbar = tqdm(self._tr_loader, desc="[resnet] epoch", leave=False, dynamic_ncols=True)
        for X, y in pbar:
            X = X.to(self.device)
            y = y.to(self.device)
            logits = self.model(X)
            loss = self._loss_fn(logits, y)
            self._opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self._opt.step()
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
    def _evaluate(self) -> dict[str, float]:
        self.model.eval()
        logits = self.model(self._Xval)
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
                for k, v in train_metrics.items():
                    writer.add_scalar(f"{k}/train", v, epoch)
                for k, v in val_metrics.items():
                    writer.add_scalar(f"{k}/val", v, epoch)

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
        X = torch.as_tensor(np.asarray(split.X, dtype=np.float32), device=self.device)
        return torch.sigmoid(self.model(X)).cpu().numpy()

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": self.model.state_dict(), "cfg": self.cfg.model_dump(mode="json")},
            Path(path),
        )

    @classmethod
    def load(cls, ckpt: Path, data: Data) -> Self:
        from wqf7008_fb_human_or_bot.datasets import feature_cols

        payload = torch.load(Path(ckpt), weights_only=False, map_location="cpu")
        cfg = ResNetConfig.model_validate(payload["cfg"])
        inst = cls(cfg, len(feature_cols(data.Xtr)))
        inst.model.load_state_dict(payload["model_state"])
        return inst
