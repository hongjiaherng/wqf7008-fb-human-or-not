import numpy as np
import pandas as pd


def load_data():
    bids = pd.read_csv("dataset/bids.csv")
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")
    return bids, train, test


def engineer_features(df):
    df = df.sort_values(["bidder_id", "time"]).reset_index(drop=True)

    # Time diff per bidder — first row per bidder gets NaN, filled later
    df["time_diff"] = df.groupby("bidder_id")["time"].diff()

    # Simultaneous bids: time slots where a bidder has bids in >1 auction
    simult = (
        df.groupby(["bidder_id", "time"])["auction"]
        .nunique()
        .gt(1)
        .groupby("bidder_id")
        .sum()
        .rename("simultaneous_bids")
    )

    agg = (
        df.groupby("bidder_id")
        .agg(
            total_bids=("bid_id", "count"),
            mean_time_diff=("time_diff", "mean"),
            min_time_diff=("time_diff", "min"),
            bids_per_auction=("auction", lambda s: len(s) / s.nunique()),
            unique_ips=("ip", "nunique"),
            unique_devices=("device", "nunique"),
            unique_urls=("url", "nunique"),
            unique_countries=("country", "nunique"),
            unique_auctions=("auction", "nunique"),
            unique_merchandise=("merchandise", "nunique"),
        )
    )

    agg["ip_to_bid_ratio"] = agg["unique_ips"] / agg["total_bids"]
    agg["auction_to_merch_ratio"] = agg["unique_auctions"] / agg["unique_merchandise"]

    # Single-bid users have NaN time diffs → fill with 0
    agg["mean_time_diff"] = agg["mean_time_diff"].fillna(0)
    agg["min_time_diff"] = agg["min_time_diff"].fillna(0)

    agg = agg.drop(columns=["unique_auctions", "unique_merchandise"])
    agg = agg.join(simult).astype({"simultaneous_bids": np.int64})

    return agg


def main():
    bids, train, test = load_data()

    train_df = bids.merge(train, on="bidder_id", how="inner")
    test_df = bids.merge(test, on="bidder_id", how="inner")

    train_features = engineer_features(train_df)
    test_features = engineer_features(test_df)

    # Attach outcome label
    train_features = train_features.join(
        train.drop_duplicates("bidder_id").set_index("bidder_id")[["outcome"]]
    )

    print(f"Train features shape: {train_features.shape}")
    print(train_features.head())

    train_features.to_csv("dataset/train_features.csv")
    test_features.to_csv("dataset/test_features.csv")


if __name__ == "__main__":
    main()
