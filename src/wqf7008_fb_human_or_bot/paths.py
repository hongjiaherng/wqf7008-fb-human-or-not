"""Filesystem paths and the path config (no internal deps)."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "facebook-recruiting-iv-human-or-bot"
RUNS_DIR = PROJECT_ROOT / "runs"
CACHE_DIR = RUNS_DIR / "cache"


class PathConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path = DATA_DIR
    runs_dir: Path = RUNS_DIR
    cache_dir: Path = CACHE_DIR
