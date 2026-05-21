from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import polars as pl

class BiddingDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    @classmethod
    def from_polars(cls, df: pl.DataFrame, feature_cols: list[str]):
        """Factory method for Polars DataFrames (already log1p + standardized)."""
        features = torch.tensor(
            df.select(feature_cols).to_numpy().astype(np.float32),
            dtype=torch.float32
        )
        labels = torch.tensor(
            df["outcome"].to_numpy(),
            dtype=torch.float32
        )
        return cls(features, labels)