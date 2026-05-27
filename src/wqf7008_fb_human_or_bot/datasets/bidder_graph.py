"""Heterogeneous graph builder for the GNN model (polars-first).

Nodes: `bidder`, `auction`, `device` (hashed), `ip` (hashed).
Edges: one deduped pair per (bidder, entity). Multi-edges are collapsed so the
graph stays small (~1M edges vs ~7.6M raw bids). SAGEConv ignores edge weights,
so we don't carry them.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch_geometric.data import HeteroData

from wqf7008_fb_human_or_bot.datasets.bidder_sequence import VOCAB_HASH, _hash_bucket_pl

AUCTION_BUCKETS = VOCAB_HASH["auction"]
DEVICE_BUCKETS = VOCAB_HASH["device"]
IP_BUCKETS = VOCAB_HASH["ip"]


@dataclass
class GraphBundle:
    data: HeteroData
    bidder_index: dict[str, int]


def _edge_index(df: pl.DataFrame, right_col: str) -> np.ndarray:
    pairs = df.select(["bidder_idx", right_col]).unique()
    src = pairs["bidder_idx"].to_numpy().astype(np.int64)
    dst = pairs[right_col].to_numpy().astype(np.int64)
    return np.stack([src, dst], axis=0)


def build_hetero_graph(
    bids: pl.DataFrame,
    bidder_ids: np.ndarray,
    bidder_features: np.ndarray,
) -> GraphBundle:
    """Build a HeteroData graph keyed by the given bidder_ids order."""
    bidder_index = {b: i for i, b in enumerate(bidder_ids)}

    # Keep only bids for bidders in our index, and attach the integer indices.
    bidder_index_df = pl.DataFrame(
        {
            "bidder_id": list(bidder_index.keys()),
            "bidder_idx": list(bidder_index.values()),
        },
        schema={"bidder_id": pl.Utf8, "bidder_idx": pl.Int64},
    )
    df = bids.join(bidder_index_df, on="bidder_id", how="inner").with_columns(
        (_hash_bucket_pl("auction", AUCTION_BUCKETS) - 1).alias("auction_idx"),
        (_hash_bucket_pl("device", DEVICE_BUCKETS) - 1).alias("device_idx"),
        (_hash_bucket_pl("ip", IP_BUCKETS) - 1).alias("ip_idx"),
    )

    data = HeteroData()
    data["bidder"].x = torch.as_tensor(bidder_features, dtype=torch.float32)
    data["auction"].num_nodes = AUCTION_BUCKETS
    data["device"].num_nodes = DEVICE_BUCKETS
    data["ip"].num_nodes = IP_BUCKETS

    for (src_type, rel, dst_type), right_col in [
        (("bidder", "bids_in", "auction"), "auction_idx"),
        (("bidder", "uses", "device"), "device_idx"),
        (("bidder", "from_ip", "ip"), "ip_idx"),
    ]:
        edges = _edge_index(df, right_col)
        data[src_type, rel, dst_type].edge_index = torch.from_numpy(edges).long()
        rev = np.stack([edges[1], edges[0]], axis=0)
        data[dst_type, f"rev_{rel}", src_type].edge_index = torch.from_numpy(rev).long()

    return GraphBundle(data=data, bidder_index=bidder_index)
