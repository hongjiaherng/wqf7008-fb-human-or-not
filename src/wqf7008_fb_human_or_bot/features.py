"""Per-bidder tabular feature engineering (polars-only).

Columns produced, in order:

    total_bids, mean_time_diff, min_time_diff, bids_per_auction,
    unique_ips, unique_devices, unique_urls, unique_countries,
    ip_to_bid_ratio, auction_to_merch_ratio, simultaneous_bids,
    has_simultaneous, log1p_n_bids, device_to_bid_ratio, url_to_bid_ratio,
    hour_of_day_mean

The output is a polars DataFrame keyed by `bidder_id` (first column), reordered
to match the input `bidders` order so rows line up with `y`.
"""

from typing import cast

import numpy as np
import polars as pl

from wqf7008_fb_human_or_bot.configs import UNIT_PER_SEC, DataConfig

FEATURE_COLS = [
    "total_bids",
    "mean_time_diff",
    "min_time_diff",
    "bids_per_auction",
    "unique_ips",
    "unique_devices",
    "unique_urls",
    "unique_countries",
    "ip_to_bid_ratio",
    "auction_to_merch_ratio",
    "simultaneous_bids",
    "has_simultaneous",
    "log1p_n_bids",
    "device_to_bid_ratio",
    "url_to_bid_ratio",
    "hour_of_day_mean",
]


def _engineer(bids: pl.DataFrame) -> pl.DataFrame:
    """Return a per-bidder feature frame keyed by `bidder_id`."""
    t_min = int(cast(int, bids["time"].min()))
    enriched = bids.sort(["bidder_id", "time"]).with_columns(
        pl.col("time").diff().over("bidder_id").alias("time_diff"),
        (((pl.col("time") - t_min) / UNIT_PER_SEC / 3600.0) % 24.0).alias("hour_of_day"),
    )

    # Bidders with multiple auctions active at the same `time` → simultaneous bids count.
    simult = (
        bids.group_by(["bidder_id", "time"])
        .agg(pl.col("auction").n_unique().alias("n_auc"))
        .filter(pl.col("n_auc") > 1)
        .group_by("bidder_id")
        .agg(pl.len().alias("simultaneous_bids"))
    )

    agg = enriched.group_by("bidder_id").agg(
        pl.len().alias("total_bids"),
        pl.col("time_diff").mean().fill_null(0).alias("mean_time_diff"),
        pl.col("time_diff").min().fill_null(0).alias("min_time_diff"),
        pl.col("auction").n_unique().alias("_n_auctions"),
        pl.col("merchandise").n_unique().alias("_n_merchandise"),
        pl.col("ip").n_unique().alias("unique_ips"),
        pl.col("device").n_unique().alias("unique_devices"),
        pl.col("url").n_unique().alias("unique_urls"),
        pl.col("country").n_unique().alias("unique_countries"),
        pl.col("hour_of_day").mean().alias("hour_of_day_mean"),
    )

    features = (
        agg.join(simult, on="bidder_id", how="left")
        .with_columns(
            pl.col("simultaneous_bids").fill_null(0),
        )
        .with_columns(
            (pl.col("total_bids") / pl.col("_n_auctions")).alias("bids_per_auction"),
            (pl.col("unique_ips") / pl.col("total_bids")).alias("ip_to_bid_ratio"),
            (pl.col("_n_auctions") / pl.col("_n_merchandise")).alias("auction_to_merch_ratio"),
            (pl.col("simultaneous_bids") > 0).cast(pl.Int8).alias("has_simultaneous"),
            pl.col("total_bids").log1p().alias("log1p_n_bids"),
            (pl.col("unique_devices") / pl.col("total_bids")).alias("device_to_bid_ratio"),
            (pl.col("unique_urls") / pl.col("total_bids")).alias("url_to_bid_ratio"),
        )
        .drop(["_n_auctions", "_n_merchandise"])
    )
    return features.select(["bidder_id", *FEATURE_COLS])


def build_tabular(
    bids: pl.DataFrame,
    bidders: pl.DataFrame,
    data_cfg: DataConfig | None = None,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray | None]:
    """Return (X, bidder_ids, y). Cached to `data_cfg.cache_dir` as parquet."""
    cfg = data_cfg or DataConfig()
    bidder_ids = bidders["bidder_id"].to_numpy()
    has_labels = "outcome" in bidders.columns
    y = bidders["outcome"].cast(pl.Int64).to_numpy() if has_labels else None

    cache_key = "train" if has_labels else "test"
    cache_path = cfg.cache_dir / f"tabular_{cache_key}.parquet"

    if cache_path.exists():
        X_full = pl.read_parquet(cache_path)
    else:
        subset = bids.filter(pl.col("bidder_id").is_in(bidders["bidder_id"]))
        X_full = _engineer(subset)
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        X_full.write_parquet(cache_path)

    # Reorder to match `bidders` order; bidders with no bids get zeros.
    order = pl.DataFrame({"bidder_id": bidder_ids})
    X = (
        order.join(X_full, on="bidder_id", how="left")
        .with_columns([pl.col(c).fill_null(0.0) for c in FEATURE_COLS])
        .select(["bidder_id", *FEATURE_COLS])
    )
    return X, bidder_ids, y
