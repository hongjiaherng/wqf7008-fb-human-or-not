"""Hybrid bidder model: bi-LSTM over the bid stream + MLP on engineered tabular features."""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from wqf7008_fb_human_or_bot.configs import HybridConfig
from wqf7008_fb_human_or_bot.datasets.bidder_sequence import (
    CAT_COLS,
    NUM_COLS,
    BidderSequenceDataset,
    SequenceStore,
    collate_pack,
)
from wqf7008_fb_human_or_bot.models.base import pos_weight_from_labels
from wqf7008_fb_human_or_bot.train import resolve_device, train_torch_loop

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

    def load_pretrained(self, ckpt_path: str | Path) -> None:
        """Load only the sequence encoder (embeddings + LSTM + numeric projection).

        Filters to encoder-prefix keys (head/tab_mlp shapes depend on
        `tabular_dim` which differs between pretrain and fine-tune), and also
        drops any key whose shape doesn't match the current model. The latter
        matters because `country`/`merchandise` vocab sizes depend on polars
        `Categorical` ordering, which is not stable across sessions: a stale
        ckpt will have different embedding rows than the live model.
        """
        import warnings

        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = sd.get("model_state", sd)
        own = self.state_dict()
        encoder_sd: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        for k, v in sd.items():
            if not k.startswith(("embs.", "input_norm.", "lstm.")):
                continue
            if k in own and own[k].shape == v.shape:
                encoder_sd[k] = v
            else:
                skipped.append(k)
        if skipped:
            warnings.warn(
                f"load_pretrained: skipped shape-mismatched keys {skipped}; "
                "consider deleting the cached ckpt and re-pretraining",
                stacklevel=2,
            )
        self.load_state_dict(encoder_sd, strict=False)


def _move_batch(batch, device):
    numeric, cat, lengths, tab, y = batch
    return (
        numeric.to(device),
        cat.to(device),
        lengths.to(device),
        tab.to(device) if tab.numel() > 0 else None,
        y.to(device),
    )


class HybridBidderClassifier:
    def __init__(
        self,
        store: SequenceStore,
        tabular_dim: int,
        pretrained_ckpt: str | Path | None = None,
    ):
        self.store = store
        self.tabular_dim = tabular_dim
        self.pretrained_ckpt = pretrained_ckpt
        self.model: HybridBidderModel | None = None
        self.device: torch.device = torch.device("cpu")

    def _make_loader(self, fold_ids, X, y, *, shuffle: bool, batch_size: int) -> DataLoader:
        ds = BidderSequenceDataset(self.store, fold_ids, y, X)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_pack, num_workers=0
        )

    def fit(self, fold, cfg: HybridConfig, *, writer=None) -> None:
        self.device = resolve_device(cfg.device)
        self.model = HybridBidderModel(
            vocab_sizes=self.store.vocab_sizes,
            tabular_dim=self.tabular_dim,
            hidden=cfg.hidden,
            dropout=cfg.dropout,
        ).to(self.device)
        if self.pretrained_ckpt is not None:
            self.model.load_pretrained(self.pretrained_ckpt)

        tr_loader = self._make_loader(
            fold.ids_tr, fold.Xtr, fold.ytr, shuffle=True, batch_size=cfg.batch_size
        )
        val_loader = self._make_loader(
            fold.ids_val, fold.Xval, fold.yval, shuffle=False, batch_size=cfg.batch_size
        )

        pos_w = pos_weight_from_labels(fold.ytr)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=self.device))
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        def train_epoch() -> dict[str, float]:
            assert self.model is not None
            loss_sum, n_batches = 0.0, 0
            probs_all, y_all = [], []
            pbar = tqdm(tr_loader, desc="[hybrid] epoch", leave=False, dynamic_ncols=True)
            for batch in pbar:
                numeric, cat, lengths, tab, y = _move_batch(batch, self.device)
                logits = self.model(numeric, cat, lengths, tab)
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
            return self._eval(val_loader, loss_fn)

        train_torch_loop(
            self.model,
            cfg,
            model_name="hybrid",
            train_epoch=train_epoch,
            eval_fn=eval_fn,
            writer=writer,
        )

    @torch.no_grad()
    def _eval(self, loader: DataLoader, loss_fn: nn.Module) -> dict[str, float]:
        assert self.model is not None
        self.model.eval()
        loss_sum, n_batches = 0.0, 0
        probs_all, y_all = [], []
        for batch in loader:
            numeric, cat, lengths, tab, y = _move_batch(batch, self.device)
            logits = self.model(numeric, cat, lengths, tab)
            loss_sum += float(loss_fn(logits, y).detach())
            n_batches += 1
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
            y_all.append(y.cpu().numpy())
        probs = np.concatenate(probs_all)
        ys = np.concatenate(y_all)
        auc = float(roc_auc_score(ys, probs)) if len(np.unique(ys)) > 1 else 0.5
        return {"loss": loss_sum / max(n_batches, 1), "auc": auc}

    @torch.no_grad()
    def predict_proba(self, fold_predict) -> np.ndarray:
        assert self.model is not None, "call fit() first"
        self.model.eval()
        ids = fold_predict.ids_val
        y = fold_predict.yval if fold_predict.yval is not None else np.zeros(len(ids))
        loader = self._make_loader(ids, fold_predict.Xval, y, shuffle=False, batch_size=256)
        out = []
        for batch in loader:
            numeric, cat, lengths, tab, _ = _move_batch(batch, self.device)
            logits = self.model(numeric, cat, lengths, tab)
            out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out)
