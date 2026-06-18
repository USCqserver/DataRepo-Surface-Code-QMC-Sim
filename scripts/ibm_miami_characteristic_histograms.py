#!/usr/bin/env python3
"""Plot IBM Miami ZZ-coupling and detuning histograms."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

_cache_root = Path(tempfile.gettempdir()) / "surfacecodeqmc-mpl-cache"
(_cache_root / "fontconfig").mkdir(parents=True, exist_ok=True)
(_cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root.resolve()))
os.environ.setdefault("MPLCONFIGDIR", str((_cache_root / "matplotlib").resolve()))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path("data")
PLOTS_DIR = Path("plots")
COUPLINGS_PATH = DATA_DIR / "20260113-121729_ibm_miami_couplings_hz.json"
DETUNINGS_PATH = DATA_DIR / "20260113-121729_ibm_miami_detunings_hz.json"


def load_values_khz(path: Path) -> np.ndarray:
    with path.open() as f:
        values_hz = json.load(f).values()
    return np.asarray(list(values_hz), dtype=float) / 1_000.0


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "font.family": "serif",
            "font.serif": ["Computer Modern", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 15,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    configure_style()
    PLOTS_DIR.mkdir(exist_ok=True)

    couplings = load_values_khz(COUPLINGS_PATH)
    detunings = load_values_khz(DETUNINGS_PATH)

    fig = plt.figure(
        figsize=(7.0, 3.0),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(1, 3, width_ratios=(2.65, 0.62, 3.05), wspace=0.08)
    ax_coupling = fig.add_subplot(gs[0, 0])
    ax_coupling_tail = fig.add_subplot(gs[0, 1], sharey=ax_coupling)
    ax_detuning = fig.add_subplot(gs[0, 2])

    coupling_color = "#3f7f93"
    detuning_color = "#b65d3b"
    edge_color = "#202020"

    coupling_main = couplings[couplings <= 25.0]
    coupling_bins = np.arange(0.0, 25.0 + 1.0, 1.0)
    hist_kwargs = {
        "color": coupling_color,
        "edgecolor": edge_color,
        "linewidth": 0.5,
        "alpha": 0.88,
    }
    ax_coupling.hist(coupling_main, bins=coupling_bins, **hist_kwargs)
    ax_coupling_tail.hist(couplings, bins=np.arange(118.0, 126.0 + 1.0, 1.0), **hist_kwargs)
    #ax_coupling.set_title(r"ZZ coupling")
    ax_coupling.set_xlabel(r"$J_{ij}/2\pi$ (kHz)")
    ax_coupling.set_ylabel("Count")
    ax_coupling.set_xlim(0.0, 25.0)
    ax_coupling_tail.set_xlim(118.0, 126.0)
    ax_coupling_tail.set_xlabel("")
    ax_coupling_tail.tick_params(labelleft=False, left=False)
    ax_coupling_tail.set_xticks([120, 125])
    ax_coupling.spines["right"].set_visible(False)
    ax_coupling_tail.spines["left"].set_visible(False)
    ax_coupling_tail.spines["right"].set_visible(False)
    ax_coupling_tail.spines["top"].set_visible(False)
    ax_coupling_tail.yaxis.set_visible(False)
    ax_coupling_tail.grid(axis="y", color="#d7d7d7", linewidth=0.5, alpha=0.8)
    ax_coupling_tail.set_axisbelow(True)
    break_kwargs = {"color": "black", "clip_on": False, "linewidth": 1.0}
    dx, dy = 0.008, 0.008
    ax_coupling.plot(
        (1.0 - dx, 1.0 + dx),
        (-dy, dy),
        transform=ax_coupling.transAxes,
        **break_kwargs,
    )
    ax_coupling_tail.plot(
        (-dx, dx),
        (-dy, dy),
        transform=ax_coupling_tail.transAxes,
        **break_kwargs,
    )
    ax_coupling.text(
        0.45,
        0.94,
        rf"$n={len(couplings)}$",
        transform=ax_coupling.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )

    detuning_bins = np.arange(-55.0, 55.0 + 5.0, 5.0)
    ax_detuning.hist(
        detunings,
        bins=detuning_bins,
        color=detuning_color,
        edgecolor=edge_color,
        linewidth=0.5,
        alpha=0.88,
    )
    #ax_detuning.set_title("Detuning")
    ax_detuning.set_xlabel(r"$\Delta_g/2\pi$ (kHz)")
    ax_detuning.set_ylabel("Count")
    ax_detuning.set_xlim(-55.0, 55.0)
    ax_detuning.text(
        0.97,
        0.94,
        rf"$n={len(detunings)}$",
        transform=ax_detuning.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )

    for ax in (ax_coupling, ax_detuning):
        ax.grid(axis="y", color="#d7d7d7", linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    png_path = PLOTS_DIR / "ibm_miami_characteristic_histograms.png"
    pdf_path = PLOTS_DIR / "ibm_miami_characteristic_histograms.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
