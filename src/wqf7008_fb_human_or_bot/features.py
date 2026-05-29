"""Build & cache the tabular feature matrices (`bidbot features build`)."""

from wqf7008_fb_human_or_bot.datasets import load_data
from wqf7008_fb_human_or_bot.paths import PathConfig


def features_command(data_cfg: PathConfig, force: bool) -> None:
    """`bidbot features build`: build & cache the tabular feature matrices."""
    if force:
        for p in data_cfg.cache_dir.glob("tabular_*.parquet"):
            p.unlink()
    data = load_data(data_cfg)
    print(f"train: {data.Xtr.shape}, test: {data.Xte.shape}")
