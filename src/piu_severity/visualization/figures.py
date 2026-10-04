"""Save report figures to disk."""

from pathlib import Path

from matplotlib.figure import Figure


def save_figure(fig: Figure, name: str, figures_dir: Path, dpi: int = 300) -> Path:
    """Save ``fig`` as a PNG file inside ``figures_dir`` and return its path."""
    if not name or Path(name).name != name:
        raise ValueError("name must be a non-empty simple file name")
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{Path(name).stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
