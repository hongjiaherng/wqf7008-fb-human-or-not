"""Hybrid bidder model: bi-LSTM over the bid stream + MLP on engineered tabular features."""

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from wqf7008_fb_human_or_bot.datasets import Data
from wqf7008_fb_human_or_bot.datasets.sequence import (
    CAT_COLS,
    NUM_COLS,
    BidderSequenceDataset,
    SequenceStore,
    collate_pack,
)
from wqf7008_fb_human_or_bot.models.base import BidderClassifier, Split
from wqf7008_fb_human_or_bot.models.registry import HybridConfig, pos_weight_from_labels

EMB_DIM = 16  # shared embedding dim across all 6 categorical fields
TAB_DIM = 32  # output dim of the tabular tower


class HybridBidderModel(nn.Module):
    """Bi-LSTM over bid sequences, concatenated with an MLP on the tabular features."""

    def __init__(
        self,
        vocab_sizes: dict[str, int],
        tabular_dim: int,
        hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.tabular_dim = tabular_dim
        self.hidden = hidden

        self.embs = nn.ModuleDict(
            {name: nn.Embedding(vocab_sizes[name], EMB_DIM, padding_idx=0) for name in CAT_COLS}
        )
        for emb in self.embs.values():
            assert isinstance(emb, nn.Embedding)
            emb.weight.data.normal_(0.0, 0.1)
            emb.weight.data[0].zero_()

        per_step_dim = len(NUM_COLS) + EMB_DIM * len(CAT_COLS)
        self.input_norm = nn.LayerNorm(per_step_dim)
        self.lstm = nn.LSTM(
            input_size=per_step_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.tab_mlp = (
            nn.Sequential(
                nn.LayerNorm(tabular_dim),
                nn.Linear(tabular_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, TAB_DIM),
                nn.GELU(),
            )
            if tabular_dim > 0
            else None
        )

        head_in = 2 * hidden + (TAB_DIM if self.tab_mlp is not None else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def encode_sequence(
        self, numeric: torch.Tensor, cat: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb_parts = [self.embs[name](cat[..., i]) for i, name in enumerate(CAT_COLS)]
        x = self.input_norm(torch.cat([numeric, *emb_parts], dim=-1))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        mask = (
            torch.arange(out.size(1), device=out.device).unsqueeze(0) < lengths.unsqueeze(1)
        ).unsqueeze(-1)
        return out, mask

    def forward(
        self,
        numeric: torch.Tensor,
        cat: torch.Tensor,
        lengths: torch.Tensor,
        tabular: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out, mask = self.encode_sequence(numeric, cat, lengths)
        denom = lengths.clamp(min=1).unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / denom  # masked mean-pool
        if self.tab_mlp is not None and tabular is not None and tabular.numel() > 0:
            pooled = torch.cat([pooled, self.tab_mlp(tabular)], dim=-1)
        return self.head(pooled).squeeze(-1)


def _move_batch(batch, device):
    numeric, cat, lengths, tab, y = batch
    return (
        numeric.to(device),
        cat.to(device),
        lengths.to(device),
        tab.to(device) if tab.numel() > 0 else None,
        y.to(device),
    )


class HybridBidderClassifier(BidderClassifier[HybridConfig]):
    name: ClassVar[str] = "hybrid"

    model: nn.Module
    device: torch.device
    _loss_fn: nn.Module
    _opt: torch.optim.Optimizer

    def __init__(self, cfg: HybridConfig, store: SequenceStore, tabular_dim: int):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.store = store
        self.tabular_dim = tabular_dim
        self.model = HybridBidderModel(
            vocab_sizes=store.vocab_sizes,
            tabular_dim=tabular_dim,
            hidden=cfg.hidden,
            dropout=cfg.dropout,
        ).to(self.device)

    @classmethod
    def from_data(cls, cfg: HybridConfig, data: Data) -> Callable[[Split], Self]:
        from wqf7008_fb_human_or_bot.datasets import feature_cols
        from wqf7008_fb_human_or_bot.datasets.sequence import build_sequence_store

        store = build_sequence_store(data.bids, max_len=cfg.max_len)
        tabular_dim = len(feature_cols(data.Xtr))
        return lambda _train: cls(cfg, store, tabular_dim)

    def _make_loader(self, ids, X, y, *, shuffle: bool, batch_size: int) -> DataLoader:
        ds = BidderSequenceDataset(self.store, ids, y, X)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_pack, num_workers=0
        )

    def _setup(self, train: Split, val: Split) -> None:
        self._tr_loader = self._make_loader(
            train.ids, train.X, train.y, shuffle=True, batch_size=self.cfg.batch_size
        )
        self._val_loader = self._make_loader(
            val.ids, val.X, val.y, shuffle=False, batch_size=self.cfg.batch_size
        )

    def _train_epoch(self) -> dict[str, float]:
        loss_sum, n_batches = 0.0, 0
        probs_all, y_all = [], []
        pbar = tqdm(self._tr_loader, desc="[hybrid] epoch", leave=False, dynamic_ncols=True)
        for batch in pbar:
            numeric, cat, lengths, tab, y = _move_batch(batch, self.device)
            logits = self.model(numeric, cat, lengths, tab)
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
        loss_sum, n_batches = 0.0, 0
        probs_all, y_all = [], []
        for batch in self._val_loader:
            numeric, cat, lengths, tab, y = _move_batch(batch, self.device)
            logits = self.model(numeric, cat, lengths, tab)
            loss_sum += float(self._loss_fn(logits, y).detach())
            n_batches += 1
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
            y_all.append(y.cpu().numpy())
        probs = np.concatenate(probs_all)
        ys = np.concatenate(y_all)
        auc = float(roc_auc_score(ys, probs)) if len(np.unique(ys)) > 1 else 0.5
        return {"loss": loss_sum / max(n_batches, 1), "auc": auc}

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
        ids = split.ids
        y = split.y if split.y is not None else np.zeros(len(ids))
        loader = self._make_loader(ids, split.X, y, shuffle=False, batch_size=256)
        out = []
        for batch in loader:
            numeric, cat, lengths, tab, _ = _move_batch(batch, self.device)
            logits = self.model(numeric, cat, lengths, tab)
            out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": self.model.state_dict(), "cfg": self.cfg.model_dump(mode="json")},
            Path(path),
        )

    @classmethod
    def load(cls, ckpt: Path, data: Data) -> Self:
        from wqf7008_fb_human_or_bot.datasets import feature_cols
        from wqf7008_fb_human_or_bot.datasets.sequence import build_sequence_store

        payload = torch.load(Path(ckpt), weights_only=False, map_location="cpu")
        cfg = HybridConfig.model_validate(payload["cfg"])
        store = build_sequence_store(data.bids, max_len=cfg.max_len)
        inst = cls(cfg, store, len(feature_cols(data.Xtr)))
        inst.model.load_state_dict(payload["model_state"])
        return inst
