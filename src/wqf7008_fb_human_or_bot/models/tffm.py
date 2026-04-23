"""FT-Transformer classifier via the `pytorch-frame` library.

Note: pytorch-frame's `TFDataset` requires a pandas DataFrame, so pandas is imported
locally here (only place in the repo that still uses it).
"""

import numpy as np
import pandas as pd  # required by pytorch-frame
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch_frame import stype
from torch_frame.data import Dataset as TFDataset
from torch_frame.nn.models import FTTransformer as TFFTTransformer
from tqdm.auto import tqdm

from wqf7008_fb_human_or_bot.configs import TFFMConfig
from wqf7008_fb_human_or_bot.models.base import pos_weight_from_labels
from wqf7008_fb_human_or_bot.train import resolve_device, train_torch_loop


class FTTransformerBidderClassifier:
    def __init__(self, feature_names: list[str]):
        self.feature_names = list(feature_names)
        self.model: TFFTTransformer | None = None
        self.device: torch.device = torch.device("cpu")

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

    def fit(self, fold, cfg: TFFMConfig, *, writer=None) -> None:
        self.device = resolve_device(cfg.device)
        tr_ds = self._to_tf(fold.Xtr, fold.ytr)
        val_ds = self._to_tf(fold.Xval, fold.yval)

        self.model = TFFTTransformer(
            channels=cfg.channels,
            out_channels=1,
            num_layers=cfg.num_layers,
            col_stats=tr_ds.col_stats,
            col_names_dict=tr_ds.tensor_frame.col_names_dict,
        ).to(self.device)

        tf_tr = tr_ds.tensor_frame.to(self.device)
        tf_val = val_ds.tensor_frame.to(self.device)

        pos_w = pos_weight_from_labels(fold.ytr)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=self.device))
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        N, B = tf_tr.num_rows, cfg.batch_size

        def train_epoch() -> dict[str, float]:
            assert self.model is not None
            perm = torch.randperm(N, device=self.device)
            loss_sum, n_batches = 0.0, 0
            probs_all, y_all = [], []
            pbar = tqdm(range(0, N, B), desc="[tffm] epoch", leave=False, dynamic_ncols=True)
            for i in pbar:
                batch = tf_tr[perm[i : i + B]]
                y = batch.y.float()
                logits = self.model(batch).squeeze(-1)
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
            y = tf_val.y.float()
            logits = self.model(tf_val).squeeze(-1)
            loss = loss_fn(logits, y)
            ys = y.cpu().numpy()
            auc = (
                float(roc_auc_score(ys, torch.sigmoid(logits).cpu().numpy()))
                if len(np.unique(ys)) > 1
                else 0.5
            )
            return {"loss": float(loss.detach()), "auc": auc}

        train_torch_loop(
            self.model,
            cfg,
            model_name="tffm",
            train_epoch=train_epoch,
            eval_fn=eval_fn,
            writer=writer,
        )

    @torch.no_grad()
    def predict_proba(self, fold_predict) -> np.ndarray:
        assert self.model is not None, "call fit() first"
        self.model.eval()
        y = (
            fold_predict.yval
            if fold_predict.yval is not None
            else np.zeros(len(fold_predict.Xval), dtype=np.int64)
        )
        tf = self._to_tf(fold_predict.Xval, y).tensor_frame.to(self.device)
        return torch.sigmoid(self.model(tf).squeeze(-1)).cpu().numpy()
