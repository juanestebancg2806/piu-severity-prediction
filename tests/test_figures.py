"""Tests for saving report figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from piu_severity.visualization.figures import save_figure

plt.switch_backend("Agg")


def test_save_figure_writes_png(tmp_path: Path) -> None:
    """A simple file name is written as PNG inside the given directory."""
    figures_dir = tmp_path / "figures"
    fig = plt.figure()
    try:
        path = save_figure(fig, "check.svg", figures_dir)
    finally:
        plt.close(fig)

    assert path == figures_dir / "check.png"
    assert path.is_file()


def test_save_figure_rejects_parent_segment(tmp_path: Path) -> None:
    """A file name with a parent directory component is rejected."""
    fig = plt.figure()
    try:
        with pytest.raises(ValueError):
            save_figure(fig, "../outside", tmp_path)
    finally:
        plt.close(fig)
