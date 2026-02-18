from __future__ import annotations

from pathlib import Path
import re
from textwrap import shorten

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import ProjectConfig


def apply_notebook_style() -> None:
    """Apply shared style across all notebooks."""
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    plt.rcParams.update(
        {
            "figure.figsize": (11, 6.5),
            "figure.dpi": 130,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def ensure_eda_dirs(config: ProjectConfig) -> tuple[Path, Path]:
    config.eda_figures_dir.mkdir(parents=True, exist_ok=True)
    config.eda_tables_dir.mkdir(parents=True, exist_ok=True)
    return config.eda_figures_dir, config.eda_tables_dir


def build_eda_output_paths(project_root: Path) -> tuple[Path, Path]:
    """Return standardized figure/table directories for notebooks."""
    figures_dir = project_root / "output" / "eda" / "figures"
    tables_dir = project_root / "src" / "eda" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


def build_artifact_path(
    base_dir: Path,
    notebook_id: str,
    section_id: str,
    slug: str,
    extension: str,
) -> Path:
    """Create standardized artifact paths: {notebook}_S{section}_{slug}.{ext}."""
    safe_slug = re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")
    safe_section = re.sub(r"[^0-9a-z]+", "", section_id.lower())
    safe_notebook = re.sub(r"[^0-9a-z]+", "", notebook_id.lower())
    safe_ext = extension.lstrip(".").lower()
    return base_dir / f"{safe_notebook}_S{safe_section}_{safe_slug}.{safe_ext}"


def save_figure(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def save_table_snapshot(
    df: pd.DataFrame,
    path: Path,
    title: str,
    max_rows: int = 20,
    index: bool = True,
    float_precision: int = 3,
) -> Path:
    """Save a dataframe as a styled, presentation-ready PNG table."""
    path.parent.mkdir(parents=True, exist_ok=True)

    preview = df.head(max_rows).copy()
    if index:
        preview = preview.reset_index()

    for col in preview.columns:
        if pd.api.types.is_float_dtype(preview[col]):
            preview[col] = preview[col].map(lambda v: f"{v:.{float_precision}f}")
        else:
            preview[col] = preview[col].astype(str).map(lambda v: shorten(v, width=34, placeholder="..."))

    n_rows, n_cols = preview.shape
    fig_w = max(10, min(22, 1.6 * n_cols))
    fig_h = max(2.8, min(14, 0.45 * (n_rows + 2)))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)

    table = ax.table(
        cellText=preview.values,
        colLabels=preview.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2f5597")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f7fb")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#d9d9d9")

    save_figure(fig, path, dpi=240)
    plt.close(fig)
    return path
