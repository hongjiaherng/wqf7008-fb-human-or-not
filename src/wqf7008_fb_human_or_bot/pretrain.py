"""Masked-bid SSL pretraining for the Hybrid sequence encoder.

Runs on ALL bid sequences (train + test bidders) using only bid events, not test
labels. Masks 15% of per-step categorical fields and predicts them from the bi-LSTM
hidden states. The resulting checkpoint contains the embedding tables and LSTM
state dict, which `HybridBidderModel.load_pretrained` consumes at fine-tune time.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from wqf7008_fb_human_or_bot.datasets.bidder_sequence import (
    CAT_COLS,
    BidderSequenceDataset,
    SequenceStore,
    collate_pack,
)
from wqf7008_fb_human_or_bot.models.hybrid import HybridBidderModel


@dataclass
class PretrainConfig:
    epochs: int = 10
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-4
    mask_prob: float = 0.15
    device: str = "auto"
    seed: int = 42
    # Architecture - must match the downstream HybridConfig so the pretrained
    # weights load cleanly into the fine-tune model (load_pretrained uses
    # strict=False, so a mismatch would silently skip most params).
    hidden: int = 128
    dropout: float = 0.3


class _MaskedBidDecoder(nn.Module):
    """Small per-category decoder: Linear(2*hidden -> vocab_size) per categorical field."""

    def __init__(self, vocab_sizes: dict[str, int], hidden: int):
        super().__init__()
        self.heads = nn.ModuleDict(
            {name: nn.Linear(2 * hidden, vocab_sizes[name]) for name in CAT_COLS}
        )

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: self.heads[name](h) for name in CAT_COLS}


def pretrain_masked_bid(
    store: SequenceStore,
    all_bidder_ids: np.ndarray,
    pcfg: PretrainConfig,
    ckpt_path: str | Path,
) -> Path:
    """Pretrain a HybridBidderModel-compatible encoder + embeddings."""
    from wqf7008_fb_human_or_bot.train import resolve_device

    device = resolve_device(pcfg.device)
    torch.manual_seed(pcfg.seed)
    np.random.seed(pcfg.seed)

    model = HybridBidderModel(
        vocab_sizes=store.vocab_sizes,
        tabular_dim=0,
        hidden=pcfg.hidden,
        dropout=pcfg.dropout,
    ).to(device)
    decoder = _MaskedBidDecoder(store.vocab_sizes, hidden=model.hidden).to(device)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(decoder.parameters()),
        lr=pcfg.lr,
        weight_decay=pcfg.weight_decay,
    )

    ds = BidderSequenceDataset(store, all_bidder_ids, y=np.zeros(len(all_bidder_ids)))
    loader = DataLoader(
        ds,
        batch_size=pcfg.batch_size,
        shuffle=True,
        collate_fn=collate_pack,
        num_workers=0,
    )

    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(pcfg.epochs):
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(
            loader,
            desc=f"[pretrain] epoch {epoch + 1}/{pcfg.epochs}",
            leave=False,
            dynamic_ncols=True,
        )
        for numeric, cat, lengths, _tab, _y in pbar:
            numeric = numeric.to(device)
            cat = cat.to(device)
            lengths = lengths.to(device)

            # Mask over valid positions (15% of non-padded steps).
            valid_mask = torch.arange(cat.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(
                1
            )
            rnd = torch.rand(cat.shape[:2], device=device)
            token_mask = valid_mask & (rnd < pcfg.mask_prob)  # (B, T)

            # Replace masked tokens with padding_idx (0) in the encoder input;
            # keep originals as targets.
            targets = cat.clone()
            mask_exp = token_mask.unsqueeze(-1)  # (B, T, 1) broadcast over fields
            cat_masked = torch.where(mask_exp, torch.zeros_like(cat), cat)

            out, _mask = model.encode_sequence(numeric, cat_masked, lengths)
            # out: (B, T, 2*hidden). Gather only the masked positions before the
            # per-field heads — turning (B, T, V) logits into (N_masked, V).
            h_masked = out[token_mask]  # (N_masked, 2*hidden)
            tgt_masked = targets[token_mask]  # (N_masked, n_cat_fields)

            if h_masked.numel() == 0:
                continue

            loss = torch.tensor(0.0, device=device)
            for f_idx, name in enumerate(CAT_COLS):
                logits = decoder.heads[name](h_masked)  # (N_masked, V_name)
                loss = loss + loss_fn(logits, tgt_masked[:, f_idx])

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(decoder.parameters()), 1.0
            )
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

        print(
            f"  [pretrain] epoch {epoch + 1}/{pcfg.epochs} avg_loss={total_loss / max(n_batches, 1):.4f}"
        )

    out = Path(ckpt_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_sizes": store.vocab_sizes,
        },
        out,
    )
    return out
