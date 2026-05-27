"""Per-bidder bid-sequence dataset for the Hybrid model and SSL pretraining.

Implementation detail: the full bids table (~7.6M rows) is processed end-to-end in
polars for speed. Only the final per-bidder numpy arrays are materialised — at
that point sizes are small and numpy is what the DataLoader collate path needs.

Column order in the per-bidder arrays:

    numeric cols: log1p(dt_self)/20, log1p(dt_others)/20, hour_of_day/24, day_of_campaign/8
    cat     cols: country, merchandise, auction, device, ip, url
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from wqf7008_fb_human_or_bot.configs import UNIT_PER_SEC

NUM_COLS = ("log1p_dt_self", "log1p_dt_others", "hour_of_day", "day_of_campaign")
CAT_COLS = ("country", "merchandise", "auction", "device", "ip", "url")

VOCAB_HASH = {
    "auction": 15_000,
    "device": 4_096,
    "ip": 16_384,
    "url": 8_192,
}


@dataclass
class SequenceStore:
    """Per-bidder sequences and vocab sizes needed for embedding tables."""

    by_id: dict[str, tuple[np.ndarray, np.ndarray]]
    vocab_sizes: dict[str, int]


def _hash_bucket_pl(col: str, buckets: int, seed: int = 0) -> pl.Expr:
    """Deterministic polars hash -> bucket expression. Reserves 0 for padding."""
    return ((pl.col(col).hash(seed=seed) % (buckets - 1)) + 1).cast(pl.Int32)


def build_sequence_store(
    bids: pl.DataFrame,
    *,
    max_len: int = 512,
) -> SequenceStore:
    """Encode every bidder's bid stream into numeric + categorical arrays (polars-first)."""
    df = bids

    # dt_self (diff over bidder_id), dt_others (diff over auction, nonzero only when the previous
    # bid was a different bidder). We compute both in one pass by sorting twice via two .with_columns.
    df = df.sort(["bidder_id", "time"]).with_columns(
        pl.col("time").diff().over("bidder_id").fill_null(0).alias("dt_self"),
    )

    df = (
        df.sort(["auction", "time"])
        .with_columns(
            (pl.col("time") - pl.col("time").shift(1).over("auction"))
            .fill_null(0)
            .alias("_dt_any"),
            (pl.col("bidder_id") != pl.col("bidder_id").shift(1).over("auction"))
            .fill_null(True)
            .alias("_is_react"),
        )
        .with_columns(
            pl.when(pl.col("_is_react")).then(pl.col("_dt_any")).otherwise(0).alias("dt_others")
        )
    )

    t_min = int(cast(int, df["time"].min()))
    df = df.with_columns(
        ((pl.col("time") - t_min) / UNIT_PER_SEC / 3600.0).alias("_hours"),
    ).with_columns(
        ((pl.col("_hours") % 24.0) / 24.0).alias("hour_of_day"),
        (pl.min_horizontal(pl.col("_hours") / 24.0, pl.lit(8.0)) / 8.0).alias("day_of_campaign"),
    )

    # Categorical encoding.
    # Low-cardinality: take a dictionary (rank) so we get contiguous ids.
    # High-cardinality: hash into bucket.
    df = df.with_columns(
        pl.col("country")
        .fill_null("__na__")
        .cast(pl.Categorical)
        .to_physical()
        .cast(pl.Int32)
        .alias("country_idx"),
        pl.col("merchandise")
        .fill_null("__na__")
        .cast(pl.Categorical)
        .to_physical()
        .cast(pl.Int32)
        .alias("merch_idx"),
        _hash_bucket_pl("auction", VOCAB_HASH["auction"]).alias("auction_idx"),
        _hash_bucket_pl("device", VOCAB_HASH["device"]).alias("device_idx"),
        _hash_bucket_pl("ip", VOCAB_HASH["ip"]).alias("ip_idx"),
        _hash_bucket_pl("url", VOCAB_HASH["url"]).alias("url_idx"),
    )

    # Re-sort so per-bidder rows are contiguous (needed for the slicing trick below).
    df = df.sort(["bidder_id", "time"])

    # Extract the numeric and categorical matrices as numpy arrays.
    n_rows = df.height
    numeric = np.empty((n_rows, len(NUM_COLS)), dtype=np.float32)
    numeric[:, 0] = np.log1p(df["dt_self"].clip(lower_bound=0).to_numpy()) / 20.0
    numeric[:, 1] = np.log1p(df["dt_others"].clip(lower_bound=0).to_numpy()) / 20.0
    numeric[:, 2] = df["hour_of_day"].to_numpy().astype(np.float32)
    numeric[:, 3] = df["day_of_campaign"].to_numpy().astype(np.float32)

    # country: +1 so 0 stays as padding
    cat = np.stack(
        [
            df["country_idx"].to_numpy().astype(np.int32) + 1,
            df["merch_idx"].to_numpy().astype(np.int32) + 1,
            df["auction_idx"].to_numpy().astype(np.int32),
            df["device_idx"].to_numpy().astype(np.int32),
            df["ip_idx"].to_numpy().astype(np.int32),
            df["url_idx"].to_numpy().astype(np.int32),
        ],
        axis=1,
    )

    # Slice into per-bidder sub-arrays by finding group boundaries on the sorted bidder_id column.
    bidder_arr = df["bidder_id"].to_numpy()
    change = np.nonzero(bidder_arr[1:] != bidder_arr[:-1])[0] + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n_rows]])
    ids = bidder_arr[starts]

    store: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for bid, s, e in zip(ids, starts, ends, strict=False):
        if e - s > max_len:
            s = e - max_len  # keep the most recent max_len bids
        store[bid] = (numeric[s:e].copy(), cat[s:e].copy())

    n_country = int(cast(int, df["country_idx"].max()) or 0) + 2
    n_merch = int(cast(int, df["merch_idx"].max()) or 0) + 2

    vocab_sizes = {
        "country": n_country,
        "merchandise": n_merch,
        "auction": VOCAB_HASH["auction"] + 1,
        "device": VOCAB_HASH["device"] + 1,
        "ip": VOCAB_HASH["ip"] + 1,
        "url": VOCAB_HASH["url"] + 1,
    }

    return SequenceStore(by_id=store, vocab_sizes=vocab_sizes)


class BidderSequenceDataset(Dataset):
    """Wraps a SequenceStore and a list of bidder_ids with optional labels + tabular row.

    If `tabular` is provided (shape (N, F)), each __getitem__ also returns the matching
    tabular feature vector so the Hybrid model can consume both in lockstep.
    """

    _EMPTY_NUM = np.zeros((1, len(NUM_COLS)), dtype=np.float32)
    _EMPTY_CAT = np.zeros((1, len(CAT_COLS)), dtype=np.int32)

    def __init__(
        self,
        store: SequenceStore,
        bidder_ids: np.ndarray,
        y: np.ndarray | None = None,
        tabular: np.ndarray | None = None,
    ):
        self.store = store
        self.bidder_ids = np.asarray(bidder_ids)
        self.y = None if y is None else np.asarray(y, dtype=np.float32)
        self.tabular = None if tabular is None else np.asarray(tabular, dtype=np.float32)
        self.tabular_dim = 0 if self.tabular is None else self.tabular.shape[1]

    def __len__(self) -> int:
        return len(self.bidder_ids)

    def __getitem__(self, index: int):
        bid = self.bidder_ids[index]
        entry = self.store.by_id.get(bid)
        if entry is None:
            numeric, cat = self._EMPTY_NUM, self._EMPTY_CAT
        else:
            numeric, cat = entry
        y = float(self.y[index]) if self.y is not None else 0.0
        tab = self.tabular[index] if self.tabular is not None else np.zeros(0, dtype=np.float32)
        return numeric, cat, tab, y


def collate_pack(batch):
    """Pad a batch of variable-length bid sequences.

    Returns:
        numeric : (B, T_max, N_NUM) float32
        cat     : (B, T_max, N_CAT) int64
        lengths : (B,) int64
        tabular : (B, F) float32 (F may be 0 if no tabular was passed)
        y       : (B,) float32
    """
    nums, cats, tabs, ys = zip(*batch, strict=False)
    lengths = np.array([n.shape[0] for n in nums], dtype=np.int64)
    T_max = int(lengths.max())
    B = len(nums)
    N_NUM = nums[0].shape[1]
    N_CAT = cats[0].shape[1]

    num_padded = np.zeros((B, T_max, N_NUM), dtype=np.float32)
    cat_padded = np.zeros((B, T_max, N_CAT), dtype=np.int64)
    for i, (n, c) in enumerate(zip(nums, cats, strict=False)):
        L = n.shape[0]
        num_padded[i, :L] = n
        cat_padded[i, :L] = c

    F = tabs[0].shape[0] if len(tabs[0].shape) > 0 else 0
    tab_stack = np.zeros((B, F), dtype=np.float32)
    if F > 0:
        for i, t in enumerate(tabs):
            tab_stack[i] = t

    return (
        torch.from_numpy(num_padded),
        torch.from_numpy(cat_padded),
        torch.from_numpy(lengths),
        torch.from_numpy(tab_stack),
        torch.tensor(ys, dtype=torch.float32),
    )
