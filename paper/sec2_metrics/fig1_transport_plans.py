#!/usr/bin/env python
"""Fig. 1: optimal transport plans between the same two QCD jets under W2 and SW2.

Left panel: constituents in the (y, phi) plane with the W2 coupling drawn as lines.
Right panel: the SW2 coupling between the two spectral functions with their marginals.

By default the plotted arrays are read from results/paper/sec2/qcd_transport_data.npz;
``--recompute`` rebuilds them from data/jets/QCD_10k.npz (jets 5862 and 2168 of the
paper, see configs/paper.yaml fig1) and refreshes the metadata JSON next to it.
Writes figures/w2_plan.pdf and figures/sw2_plan.pdf.

Definitions: unit-normalised constituent pT in both metrics; W2 ground cost is squared
rapidity separation plus squared shortest periodic azimuthal separation; the spectral
function uses Euclidean pair separations (beta = 1), mass 2 z_i z_j on each unordered pair
and the self-pair mass at omega = 0; both plans come from POT's exact solvers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import add_common_args, describe, load_cfg  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullLocator  # noqa: E402

from m3r.plotting.style import savefig  # noqa: E402
from m3r.results_io import write_provenance  # noqa: E402

COLORS = ("#b2182b", "#076fb8")
BINS = 64
BASE_FONT_SIZE, AXIS_LABEL_FONT_SIZE, TITLE_FONT_SIZE = 16, 22, 22
TICK_LABEL_FONT_SIZE, LEGEND_FONT_SIZE, DISTANCE_FONT_SIZE = 17, 18, 21
MARGINAL_LABEL_FONT_SIZE, MARGINAL_TICK_FONT_SIZE = 16, 14
COLORBAR_TITLE_FONT_SIZE, COLORBAR_TICK_FONT_SIZE = 15, 14
MARKER_AREA_SCALE, MARKER_AREA_FLOOR = 520, 14
FIGSIZE = (6.8, 6.8)


def truncated_cmap(name, low, high):
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(f"{name}_{low}_{high}", base(np.linspace(low, high, 256)))


SPECTRAL_CMAP = truncated_cmap("magma", 0.0, 0.85)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--recompute", action="store_true", help="Recompute the plans from data/jets/QCD_10k.npz.")
    parser.add_argument("--jets", nargs=2, type=int, default=None, help="Zero-based QCD jet indices (implies --recompute).")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


# ----------------------------------------------------------------------------- computation
def constituents(jet):
    active = np.asarray(jet[jet[:, 0] > 0, :3], dtype=np.float64)
    return active[:, 0] / active[:, 0].sum(), active[:, 1:3]


def spectral_measure(weights, coords):
    """beta=1 spectrum: sum(z_i^2) at 0 and 2 z_i z_j at |x_i-x_j|."""
    i, j = np.triu_indices(len(weights), k=1)
    omega = np.r_[0.0, np.linalg.norm(coords[i] - coords[j], axis=1)]
    mass = np.r_[weights @ weights, 2 * weights[i] * weights[j]]
    order = np.argsort(omega)
    return omega[order], mass[order] / mass.sum()


def compute(jet_1, jet_2, num_iter_max: int = 100_000) -> dict:
    import ot
    from scipy.ndimage import label

    w1, c1 = constituents(jet_1)
    w2, c2 = constituents(jet_2)
    dphi = (c1[:, None, 1] - c2[None, :, 1] + np.pi) % (2 * np.pi) - np.pi
    cost = (c1[:, None, 0] - c2[None, :, 0]) ** 2 + dphi ** 2
    gamma = ot.emd(w1, w2, cost, numItermax=num_iter_max)
    omega1, mass1 = spectral_measure(w1, c1)
    omega2, mass2 = spectral_measure(w2, c2)
    plan = ot.emd_1d(omega1, omega2, mass1, mass2, metric="sqeuclidean", dense=False).tocoo()
    edges = np.linspace(0, 1.04 * max(omega1[-1], omega2[-1]), BINS + 1)
    matrix = np.histogram2d(omega1[plan.row], omega2[plan.col], bins=(edges, edges), weights=plan.data)[0]
    hist1 = np.histogram(omega1, bins=edges, weights=mass1)[0]
    hist2 = np.histogram(omega2, bins=edges, weights=mass2)[0]
    np.testing.assert_allclose(gamma.sum(1), w1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(gamma.sum(0), w2, atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.asarray(plan.sum(1)).ravel(), mass1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.asarray(plan.sum(0)).ravel(), mass2, atol=1e-12, rtol=0)
    np.testing.assert_allclose(matrix.sum(1), hist1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(matrix.sum(0), hist2, atol=1e-12, rtol=0)
    np.testing.assert_allclose(matrix.sum(), 1.0, atol=1e-12, rtol=0)
    distance_w2 = float(np.sqrt(np.sum(gamma * cost)))
    distance_sw2 = float(np.sqrt(np.sum(plan.data * (omega1[plan.row] - omega2[plan.col]) ** 2)))
    return dict(w1=w1, c1=c1, w2=w2, c2=c2, gamma=gamma, omega1=omega1, mass1=mass1, omega2=omega2, mass2=mass2,
                sw_row=plan.row, sw_col=plan.col, sw_flow=plan.data, matrix=matrix, edges=edges, hist1=hist1,
                hist2=hist2, distance_w2=distance_w2, distance_sw2=distance_sw2,
                components=int(label(matrix > 0, structure=np.ones((3, 3)))[1]),
                components_4=int(label(matrix > 0)[1]))


# ----------------------------------------------------------------------------- plotting
def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.unicode_minus": False, "font.size": BASE_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_FONT_SIZE, "axes.titlesize": TITLE_FONT_SIZE, "axes.titlepad": 12,
        "xtick.labelsize": TICK_LABEL_FONT_SIZE, "ytick.labelsize": TICK_LABEL_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE, "axes.linewidth": 1.1,
        "xtick.major.size": 6, "ytick.major.size": 6, "xtick.minor.size": 3, "ytick.minor.size": 3,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0, "xtick.minor.width": .8, "ytick.minor.width": .8,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 150, "savefig.dpi": 300,
    })


def marker_handles():
    return [Line2D([], [], linestyle="none", marker=m, markerfacecolor=c, markeredgecolor="white",
                   markeredgewidth=.5, markersize=12, label=f"QCD Jet {i}")
            for i, (c, m) in enumerate(zip(COLORS, ("o", "^")), 1)]


def line_handles():
    return [Line2D([], [], color=c, linewidth=2.4, label=f"QCD Jet {i}") for i, c in enumerate(COLORS, 1)]


def plot_w2(p, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=.165, right=.965, bottom=.135, top=.9)
    ij = np.argwhere(p["gamma"] > 0)
    flow = p["gamma"][ij[:, 0], ij[:, 1]]
    order = np.argsort(flow); ij, flow = ij[order], flow[order]
    segments = np.stack((p["c1"][ij[:, 0]], p["c2"][ij[:, 1]]), axis=1)
    if np.any(np.abs(segments[:, 0, 1] - segments[:, 1, 1]) > np.pi):
        raise ValueError("This pair requires wrapped transport-line drawing.")
    strength = np.sqrt(flow / flow.max())
    rgba = np.zeros((len(flow), 4)); rgba[:, :3] = .22; rgba[:, 3] = .14 + .66 * strength
    ax.add_collection(LineCollection(segments, colors=rgba, linewidths=.3 + 2.8 * strength, zorder=1))
    max_weight = max(p["w1"].max(), p["w2"].max())
    for n, marker, color in zip((1, 2), ("o", "^"), COLORS):
        area_correction = 1.0 if marker == "o" else np.pi / 2
        area = MARKER_AREA_FLOOR + MARKER_AREA_SCALE * p[f"w{n}"] / max_weight
        ax.scatter(*p[f"c{n}"].T, s=area * area_correction, marker=marker, color=color, alpha=.9,
                   edgecolor="white", linewidth=.5, zorder=3 + n)
    for method in (ax.axhline, ax.axvline):
        method(0, color=".78", linestyle=(0, (4, 3)), linewidth=.9, zorder=0)
    extent = max(np.abs(p["c1"]).max(), np.abs(p["c2"]).max())
    limit = max(.4, np.ceil((extent + .075) * 10) / 10)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
    ax.xaxis.set_major_locator(MultipleLocator(.2)); ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.set_xlabel(r"$y$", loc="center"); ax.set_ylabel(r"$\phi$", loc="center")
    ax.set_title("Energy-Flow 2-Wasserstein")
    ax.legend(handles=marker_handles(), loc="upper left", frameon=False, borderpad=.3, labelspacing=.4, handletextpad=.4)
    ax.text(.965, .96, rf"$\mathrm{{W}}_2 = {float(p['distance_w2']):.4f}$", transform=ax.transAxes, ha="right", va="top",
            fontsize=DISTANCE_FONT_SIZE)
    savefig(fig, out, bbox_inches=None); plt.close(fig)
    return out


def plot_sw2(p, out: Path) -> Path:
    fig = plt.figure(figsize=FIGSIZE)
    left, bottom, width, marginal = .165, .135, .6, .165
    ax = fig.add_axes([left, bottom, width, width])
    top = fig.add_axes([left, bottom + width, width, marginal], sharex=ax)
    right = fig.add_axes([left + width, bottom, marginal, width], sharey=ax)
    edges, matrix = p["edges"], p["matrix"]
    positive = matrix[matrix > 0]
    norm = LogNorm(vmin=10 ** np.floor(np.log10(positive.min())), vmax=10 ** np.ceil(np.log10(positive.max())))
    mesh = ax.pcolormesh(edges, edges, np.ma.masked_equal(matrix.T, 0), norm=norm, cmap=SPECTRAL_CMAP, shading="flat",
                         edgecolors="face", linewidth=.4, antialiased=False, rasterized=False)
    ax.set(xlim=(0, edges[-1]), ylim=(0, edges[-1]), aspect="equal")
    ax.set_xlabel(r"$\omega$ of QCD Jet 1", loc="center"); ax.set_ylabel(r"$\omega$ of QCD Jet 2", loc="center")
    ax.xaxis.set_major_locator(MultipleLocator(.2)); ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.tick_params(which="both", top=False, right=False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: "" if np.isclose(v, 0) else f"{v:.1f}"))
    ax.legend(handles=line_handles(), loc="upper left", frameon=False, borderpad=.3, labelspacing=.4, handletextpad=.6, handlelength=1.8)
    ax.text(.04, .77, rf"$\mathrm{{SW}}_2 = {float(p['distance_sw2']):.4f}$", transform=ax.transAxes, va="top", fontsize=DISTANCE_FONT_SIZE)
    densities = [p["hist1"] / np.diff(edges), p["hist2"] / np.diff(edges)]
    positive_density = np.concatenate([d[d > 0] for d in densities])
    floor = 10 ** np.floor(np.log10(positive_density.min())); ceiling = 10 ** np.ceil(np.log10(positive_density.max()))
    top.stairs(np.where(densities[0] > 0, densities[0], np.nan), edges, baseline=None, color=COLORS[0], linewidth=2.0)
    right.stairs(np.where(densities[1] > 0, densities[1], np.nan), edges, baseline=None, orientation="horizontal",
                 color=COLORS[1], linewidth=2.0)
    top.set_yscale("log"); right.set_xscale("log")
    top.set_ylim(floor, ceiling); right.set_xlim(floor, ceiling)
    top.set_ylabel("Density", fontsize=MARGINAL_LABEL_FONT_SIZE, loc="center", labelpad=6)
    right.set_xlabel("Density", fontsize=MARGINAL_LABEL_FONT_SIZE, loc="center", labelpad=7)
    top.set_title("Spectral 2-Wasserstein", x=(width + marginal) / (2 * width))
    top.tick_params(axis="x", which="both", labelbottom=False, top=False)
    top.tick_params(axis="y", which="both", labelsize=MARGINAL_TICK_FONT_SIZE, right=False)
    right.tick_params(axis="y", which="both", labelleft=False, right=False)
    right.tick_params(axis="x", which="both", labelsize=MARGINAL_TICK_FONT_SIZE, top=False)
    ticks = [10.0 ** k for k in range(int(np.log10(floor)), int(np.log10(ceiling)) + 1, 2)]
    top.set_yticks(ticks); right.set_xticks(ticks)
    top.yaxis.set_minor_locator(NullLocator()); right.xaxis.set_minor_locator(NullLocator())
    cax = fig.add_axes([left + width + .045, bottom + width + .03, .03, marginal - .04])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.ax.set_title("Mass / Bin", fontsize=COLORBAR_TITLE_FONT_SIZE, pad=7, loc="left")
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE, length=3.5, pad=3)
    cbar.locator = LogLocator(base=10, numticks=3); cbar.update_ticks()
    cbar.ax.yaxis.set_minor_locator(NullLocator())
    savefig(fig, out, bbox_inches=None); plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)
    out_dir = args.out_dir or cfg.path("figures")
    result = cfg.result_path("fig1_transport")
    jets_idx = tuple(args.jets) if args.jets else tuple(int(i) for i in cfg["fig1"]["jets"])
    if args.recompute or args.jets or args.force or not result.exists():
        data_path = cfg.jets_path("QCD")
        with np.load(data_path) as data:
            jets = data["x"]
        if len(set(jets_idx)) != 2 or any(i < 0 or i >= len(jets) for i in jets_idx):
            raise ValueError("Specify two distinct, valid zero-based QCD jet indices.")
        p = compute(jets[jets_idx[0]], jets[jets_idx[1]], int(cfg["fig1"].get("num_iter_max", 100_000)))
        if jets_idx == tuple(int(i) for i in cfg["fig1"]["jets"]):
            assert p["components"] == 1 and p["components_4"] == 1, "Default spectral plan must have contiguous support."
        result.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(result, jet_indices=np.array(jets_idx), **p)
        metadata = {
            "dataset": describe(data_path, cfg), "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "jet_indices_zero_based": list(jets_idx), "constituent_counts": [len(p["w1"]), len(p["w2"])],
            "W2": p["distance_w2"], "SW2": p["distance_sw2"], "spectral_bins": BINS,
            "spectral_axis_limit": float(p["edges"][-1]),
            "spectral_support_connected_components_8_neighbor": p["components"],
            "spectral_support_connected_components_4_neighbor": p["components_4"],
        }
        result.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        write_provenance(result, config_hash=cfg.hash(), stage="paper/sec2_metrics/fig1_transport_plans.py", jets=list(jets_idx))
        print(f"wrote {describe(result, cfg)}")
    with np.load(result) as payload:
        p = {k: payload[k] for k in payload.files}
    style()
    names = cfg["figures"]["fig1"]
    w2_out = plot_w2(p, out_dir / names[0])
    sw2_out = plot_sw2(p, out_dir / names[1])
    print(f"jets {tuple(int(i) for i in p['jet_indices'])}: W2 = {float(p['distance_w2']):.6f}, SW2 = {float(p['distance_sw2']):.6f}")
    print(f"wrote {describe(w2_out, cfg)} and {describe(sw2_out, cfg)}")


if __name__ == "__main__":
    main()
