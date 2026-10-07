"""House style for the paper figures: mplhep CMS base, Times/STIX fonts, shared palettes.

Colour tables are the ones used in the published figures (ColorBrewer RdBu-derived,
checked for colour-vision-deficiency separation); keep them in sync across sections.
"""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from m3r.classes import CLASS_NAMES, METRIC_LABELS, TASKS


def use_paper_style(
    *,
    base: float = 15,
    title: float = 24,
    label: float = 19,
    tick: float = 15.5,
    legend: float = 17.5,
    extra: dict[str, Any] | None = None,
) -> None:
    """Apply the CMS mplhep style with serif (Times) fonts and the given font sizes."""
    try:
        import mplhep as hep

        plt.style.use(hep.style.CMS)
    except Exception:  # pragma: no cover - mplhep optional at runtime
        pass
    params = {
        "font.size": base,
        "axes.titlesize": title,
        "axes.labelsize": label,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "legend.fontsize": legend,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "xtick.top": False,
        "ytick.right": True,
        "figure.dpi": 110,
        "savefig.dpi": 300,
    }
    if extra:
        params.update(extra)
    plt.rcParams.update(params)


def use_plain_serif_style(**rc: Any) -> None:
    """Minimal serif style (no mplhep) used by the Sec. 5 curve figure."""
    params = {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
    params.update(rc)
    plt.rcParams.update(params)


def savefig(fig: matplotlib.figure.Figure, path: str | Path, **kwargs: Any) -> Path:
    """Save a figure without creation/modification dates so re-runs are byte-stable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("bbox_inches", "tight")
    if path.suffix.lower() == ".pdf":
        metadata = dict(kwargs.pop("metadata", {}) or {})
        metadata.setdefault("CreationDate", None)
        metadata.setdefault("ModDate", None)
        kwargs["metadata"] = metadata
    fig.savefig(path, **kwargs)
    return path


# --------------------------------------------------------------------------- #
# Palettes
# --------------------------------------------------------------------------- #

#: Six-task colour scheme (Figs. 6-8), keyed by task slug.
TASK_COLORS: dict[str, str] = dict(zip(
    (t.slug for t in TASKS),
    ("#333333", "#076fb8", "#43aedf", "#133c8b", "#b2182b", "#e27641"),
))
#: Markers carry task identity redundantly with colour.
TASK_MARKERS: dict[str, str] = dict(zip((t.slug for t in TASKS), ("o", "s", "^", "P", "X", "D")))
#: Matplotlib sizes markers by bounding box; these factors equalise the apparent area.
MARKER_SIZE_SCALE: dict[str, float] = {"o": 1.00, "s": 0.90, "^": 1.12, "D": 0.95, "P": 1.18, "X": 1.10}
ABS_MARKER_GAIN = 1.55


def scaled_marker_size(marker: str, base_size: float) -> float:
    return base_size * MARKER_SIZE_SCALE.get(marker, 1.0)


#: Six representative classes of Fig. 3 (prong-ordered: neutral, cool ramp, warm pole).
REPRESENTATIVE_CLASS_COLORS: dict[str, str] = {
    "QCD": "#333333",
    "HToBB": "#076fb8",
    "WToQQ": "#43aedf",
    "ZToQQ": "#133c8b",
    "TTBar": "#b2182b",
    "HToWW4Q": "#e27641",
}
CLASS_MARKERS: dict[str, str] = {
    "QCD": "o", "HToBB": "s", "HToCC": "^", "HToGG": "v", "HToWW4Q": "P",
    "TTBar": "D", "WToQQ": "X", "ZToQQ": "*", "HToWW2Q1L": "<", "TTBarLep": ">",
}
#: Prong-ordered class sequence used to spread the 8-class viridis ramp.
PRONG_ORDER: tuple[str, ...] = ("QCD", "HToBB", "HToCC", "HToGG", "WToQQ", "ZToQQ", "TTBar", "HToWW4Q")


def vivid_viridis_color(value: float) -> tuple[float, float, float, float]:
    r, g, b, alpha = plt.get_cmap("viridis")(value)
    hue, sat, val = colorsys.rgb_to_hsv(r, g, b)
    sat = min(1.0, sat * 1.65 + 0.08)
    val = min(1.0, val * 1.08)
    return (*colorsys.hsv_to_rgb(hue, sat, val), alpha)


#: Saturation-boosted viridis colours for the eight-class variants of Fig. 3.
JET_COLORS: dict[str, tuple[float, float, float, float]] = {
    name: vivid_viridis_color(v) for name, v in zip(PRONG_ORDER, np.linspace(0.05, 0.95, len(PRONG_ORDER)))
}

#: Metric line styles for the per-class grids.
METRIC_STYLE: dict[str, dict[str, Any]] = {
    "w2": dict(color="#0072B2", marker="o", label=r"$W_2$"),
    "ps": dict(color="#D55E00", marker="s", label="PS"),
    "sw": dict(color="#009E73", marker="D", label=r"$\mathrm{SW}_2$"),
}

#: Fig. 4 metric-pair bars.
PAIR_COLORS: dict[tuple[str, str], str] = {
    ("w2", "sw"): "#b2182b",
    ("w2", "ps"): "#133c8b",
    ("sw", "ps"): "#43aedf",
}
#: Fig. 5 directed-transfer bars (source-metric families) and self-transfer greys.
TRANSFER_COLORS: dict[tuple[str, str], str] = {
    ("sw", "w2"): "#b2182b",
    ("sw", "ps"): "#d6604d",
    ("w2", "sw"): "#2e7d32",
    ("w2", "ps"): "#74c476",
    ("ps", "w2"): "#1f78b4",
    ("ps", "sw"): "#6cc3ea",
}
SELF_TRANSFER_COLORS: dict[str, str] = {"w2": "0.36", "sw": "0.50", "ps": "0.80"}

__all__ = [
    "ABS_MARKER_GAIN", "CLASS_MARKERS", "JET_COLORS", "MARKER_SIZE_SCALE", "METRIC_LABELS", "METRIC_STYLE",
    "PAIR_COLORS", "PRONG_ORDER", "REPRESENTATIVE_CLASS_COLORS", "SELF_TRANSFER_COLORS", "TASK_COLORS",
    "TASK_MARKERS", "TRANSFER_COLORS", "CLASS_NAMES", "savefig", "scaled_marker_size", "use_paper_style",
    "use_plain_serif_style", "vivid_viridis_color",
]
