"""FT-Transformer classifier via the `pytorch-frame` library.

Note: pytorch-frame's `TFDataset` requires a pandas DataFrame, so pandas is imported
locally here (only place in the repo that still uses it).
"""

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import pandas as pd  # required by pytorch-frame
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch_frame import stype
from torch_frame.data import Dataset as TFDataset
from torch_frame.nn.models import FTTransformer as TFFTTransformer
from tqdm.auto import tqdm

from wqf7008_fb_human_or_bot.datasets import Data
from wqf7008_fb_human_or_bot.models.base import BidderClassifier, Split
from wqf7008_fb_human_or_bot.models.registry import TFFMConfig, pos_weight_from_labels


class FTTransformerBidderClassifier(BidderClassifier[TFFMConfig]):
    name: ClassVar[str] = "tffm"

    model: nn.Module
    device: torch.device
    _loss_fn: nn.Module
    _opt: torch.optim.Optimizer

    def __init__(self, cfg: TFFMConfig, feature_names: list[str], train: Split):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.feature_names = list(feature_names)
        # FT-Transformer needs col_stats from a materialised training frame to
        # instantiate, so the training split is required at construction time.
        self._tr_ds = self._to_tf(train.X, train.y)
        self.model = TFFTTransformer(
            channels=cfg.channels,
            out_channels=1,
            num_layers=cfg.num_layers,
            col_stats=self._tr_ds.col_stats,
            col_names_dict=self._tr_ds.tensor_frame.col_names_dict,
        ).to(self.device)

    @classmethod
    def from_data(cls, cfg: TFFMConfig, data: Data) -> Callable[[Split], Self]:
        from wqf7008_fb_human_or_bot.datasets import feature_cols

        feat = feature_cols(data.Xtr)
        return lambda train: cls(cfg, feat, train)

    def _to_tf(self, X: np.ndarray, y: np.ndarray | None):
        df = pd.DataFrame(np.asarray(X, dtype=np.float32), columns=self.feature_names)
        col_to_stype = {c: stype.numerical for c in self.feature_names}
        if y is not None:
            df["_target"] = np.asarray(y, dtype=np.int64)
            col_to_stype["_target"] = stype.categorical
        ds = TFDataset(
            df, col_to_stype=col_to_stype, target_col="_target" if y is not None else None
        )
        ds.materialize()
        return ds

    def _setup(self, train: Split, val: Split) -> None:
        self._tf_tr = self._tr_ds.tensor_frame.to(self.device)
        self._tf_val = self._to_tf(val.X, val.y).tensor_frame.to(self.device)
        self._N, self._B = self._tf_tr.num_rows, self.cfg.batch_size

    def _train_epoch(self) -> dict[str, float]:
        perm = torch.randperm(self._N, device=self.device)
        loss_sum, n_batches = 0.0, 0
        probs_all, y_all = [], []
        pbar = tqdm(
            range(0, self._N, self._B), desc="[tffm] epoch", leave=False, dynamic_ncols=True
        )
        for i in pbar:
            batch = self._tf_tr[perm[i : i + self._B]]
            y = batch.y.float()
            logits = self.model(batch).squeeze(-1)
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
        y = self._tf_val.y.float()
        logits = self.model(self._tf_val).squeeze(-1)
        loss = self._loss_fn(logits, y)
        ys = y.cpu().numpy()
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
        y = split.y if split.y is not None else np.zeros(len(split.X), dtype=np.int64)
        tf = self._to_tf(split.X, y).tensor_frame.to(self.device)
        return torch.sigmoid(self.model(tf).squeeze(-1)).cpu().numpy()

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
        cfg = TFFMConfig.model_validate(payload["cfg"])
        feat = feature_cols(data.Xtr)
        # col_stats come from the full training frame (as in the original eval path).
        Xtr_np = data.Xtr.select(feat).to_numpy().astype(np.float32)
        train = Split(X=Xtr_np, y=data.ytr, ids=data.ids_tr)
        inst = cls(cfg, feat, train)
        inst.model.load_state_dict(payload["model_state"])
        return inst
