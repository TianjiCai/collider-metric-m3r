"""Diagnostic heatmaps shared by the pairwise experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def upper_triangle_heatmap(values: np.ndarray, title: str, colorbar_label: str, output: Path,
                           vmin: float = 0.5, vmax: float = 1.0, labels: list[str] | None = None,
                           fmt: str = "{:.3f}") -> Path:
    n = values.shape[0]
    mask = np.triu(np.ones_like(values, dtype=bool), k=1)
    data = np.where(mask & np.isfinite(values), values, np.nan)
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color=(1, 1, 1, 0))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ticks = labels or [f"{i + 1}" for i in range(n)]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(ticks, fontsize=8)
    ax.set_title(title, fontsize=10)
    cbar = plt.colorbar(im, ax=ax); cbar.set_label(colorbar_label)
    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(data[i, j]):
                ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center", fontsize=7, color="w")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output
