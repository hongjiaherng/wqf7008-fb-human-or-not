"""Console + run-directory presentation shared by the train/cv/eval commands.

Pure formatting and output wiring; no model or data logic lives here.
"""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from wqf7008_fb_human_or_bot.metrics import CVResult, plot_roc, save_cv_summary
from wqf7008_fb_human_or_bot.paths import PROJECT_ROOT, RUNS_DIR


def rel(p: Path | str) -> str:
    """Path as `./<rel>` under PROJECT_ROOT, otherwise the absolute path."""
    try:
        return f"./{Path(p).resolve().relative_to(PROJECT_ROOT)}".replace("\\", "/")
    except ValueError:
        return str(p)


def fmt(cfg: BaseModel) -> str:
    """One-line `k=v k=v ...` view of a pydantic config; paths get relativised."""
    return " ".join(
        f"{k}={rel(v) if isinstance(v, Path) else v}"
        for k, v in cfg.model_dump().items()
        if not isinstance(v, dict)
    )


def header(title: str, out: Path, **sections: str) -> None:
    print(f"\n== {title} ==")
    print(f"  out:   {rel(out)}")
    for name, value in sections.items():
        print(f"  {name:6} {value}")
    print()


def run_dir(explicit: Path | None, mode: str, tag: str) -> Path:
    if explicit is not None:
        return explicit
    return RUNS_DIR / mode / f"{datetime.now():%Y%m%dT%H%M%S}_{tag}"


def write_summary(result: CVResult, out: Path) -> None:
    save_cv_summary(result, out)
    plot_roc(result, out / "roc.png")
    print(result.summary_str())
