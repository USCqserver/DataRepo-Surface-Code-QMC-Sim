#!/usr/bin/env python3
"""Build the composite surface-code figure for a two-column paper."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import json
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

import plot_heatmap_d7_minimal as ph


# --- Figure knobs ---------------------------------------------------------
OUT_PDF = "surface_code_composite.pdf"
OUT_PNG = "surface_code_composite.png"

FIGSIZE = (7.8, 5.55)
WIDTH_RATIOS = (0.85, 1.0)
DPI = 300

TOP_BOTTOM_HEIGHT_RATIOS = (4.1, 1.08)
TOP_BOTTOM_HSPACE = 0.34

WSPACE = 0.02
RIGHT_HSPACE = 0.2
PLAQUETTE_WSPACE = 0.2

CIRCUIT_SCALE = 1.5
PLAQUETTE_DATA_RADIUS = 0.18
PLAQUETTE_ANCILLA_RADIUS = 0.23

PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
PANEL_LABEL_FONTSIZE = 13
PANEL_LABEL_X = -0.1
PANEL_LABEL_Y = 1.0

COLORBAR_TICK_FONTSIZE = 11
COLORBAR_LABEL_FONTSIZE = 11

PULSE_JSON_PATH = Path("./circuits/pulses.json")
PULSE_LINEWIDTH = 1.5
PULSE_LABEL_FONTSIZE = 11
PULSE_TICK_FONTSIZE = 10
PULSE_XLIM = (0, 300)

BACKGROUND = "none"


def configure_matplotlib():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Computer Modern"
    plt.rcParams["text.usetex"] = True


def enlarge_gate_boxes(ax, sx=1.25, sy=1.20):
    for p in ax.patches:
        if isinstance(p, patches.Rectangle):
            w, h = p.get_width(), p.get_height()
            if 0.2 < abs(w) < 1.5 and 0.2 < abs(h) < 1.5:
                x, y = p.get_xy()
                cx, cy = x + w / 2, y + h / 2
                nw, nh = w * sx, h * sy
                p.set_width(nw)
                p.set_height(nh)
                p.set_xy((cx - nw / 2, cy - nh / 2))


def load_segments(json_path=PULSE_JSON_PATH):
    with open(json_path, "r") as f:
        return json.load(f)


def plot_segmented_pulse_train(ax, segments):
    for i, seg in enumerate(segments):
        color = "C0" if i in (0, 2, 5, 7) else "C1"
        ax.plot(seg["times"], seg["amps"], linewidth=PULSE_LINEWIDTH, color=color)

    ax.set_xlabel("Time (ns)", fontsize=PULSE_LABEL_FONTSIZE)
    ax.set_ylabel(r"$a_g(t)$", fontsize=PULSE_LABEL_FONTSIZE)
    ax.set_xlim(*PULSE_XLIM)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", which="major", labelsize=PULSE_TICK_FONTSIZE)
    ax.tick_params(axis="both", which="minor", length=2.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def make_figure():
    configure_matplotlib()

    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_alpha(0.0)

    outer_gs = fig.add_gridspec(
        2,
        1,
        height_ratios=TOP_BOTTOM_HEIGHT_RATIOS,
        hspace=TOP_BOTTOM_HSPACE,
    )

    top_gs = outer_gs[0, 0].subgridspec(
        1,
        2,
        width_ratios=WIDTH_RATIOS,
        wspace=WSPACE,
    )

    ax_patch = fig.add_subplot(top_gs[0, 0])
    ph.plot_surface_code(ax=ax_patch)

    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    cbar_ax.set_xlabel("Crosstalk strength (kHz)", fontsize=COLORBAR_LABEL_FONTSIZE)

    right_gs = top_gs[0, 1].subgridspec(
        3,
        1,
        height_ratios=(0.58, 1.0, 1.0),
        hspace=RIGHT_HSPACE,
    )

    plaquette_gs = right_gs[0, 0].subgridspec(1, 2, wspace=PLAQUETTE_WSPACE)
    ax_x_plaquette = fig.add_subplot(plaquette_gs[0, 0])
    ax_z_plaquette = fig.add_subplot(plaquette_gs[0, 1])
    ph.plot_x_plaquette(
        ax=ax_x_plaquette,
        label_data=True,
        data_radius=PLAQUETTE_DATA_RADIUS,
        ancilla_radius=PLAQUETTE_ANCILLA_RADIUS,
    )
    ph.plot_z_plaquette(
        ax=ax_z_plaquette,
        label_data=True,
        data_radius=PLAQUETTE_DATA_RADIUS,
        ancilla_radius=PLAQUETTE_ANCILLA_RADIUS,
    )

    circuit_style = {"backgroundcolor": BACKGROUND, "fontsize": 18, "subfontsize": 14}
    ax_x_circuit = fig.add_subplot(right_gs[1, 0])
    ax_z_circuit = fig.add_subplot(right_gs[2, 0])
    ph.plot_x_plaquette_circuit(ax=ax_x_circuit, scale=CIRCUIT_SCALE, style=circuit_style)
    enlarge_gate_boxes(ax_x_circuit, sx=1.35, sy=1.30)
    ph.plot_z_plaquette_circuit(ax=ax_z_circuit, scale=CIRCUIT_SCALE, style=circuit_style)
    enlarge_gate_boxes(ax_z_circuit, sx=1.35, sy=1.30)

    ax_pulse = fig.add_subplot(outer_gs[1, 0])
    plot_segmented_pulse_train(ax_pulse, load_segments())

    panel_axes = [
        ax_patch,
        ax_x_plaquette,
        ax_z_plaquette,
        ax_x_circuit,
        ax_z_circuit,
        ax_pulse,
    ]
    label_positions = {
        "(a)": (-0.005, 0.9),
        "(b)": (-0.025, 0.9),
        "(c)": (-0.025, 0.9),
        "(d)": (-0.025, 0.8),
        "(e)": (-0.025, 0.8),
        "(f)": (-0.055, 1.05),
    }
    for label, ax in zip(PANEL_LABELS, panel_axes):
        lx, ly = label_positions.get(label, (PANEL_LABEL_X, PANEL_LABEL_Y))
        ax.text(
            lx,
            ly,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=PANEL_LABEL_FONTSIZE,
            va="bottom",
            ha="right",
        )

    for ax in fig.axes:
        ax.patch.set_alpha(0.0)

    # Qiskit's mpl drawer may leave a full-axes background rectangle behind.
    # Remove only huge background patches, leaving gates and markers untouched.
    for ax in fig.axes:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xspan = abs(x1 - x0)
        yspan = abs(y1 - y0)
        for patch in ax.patches:
            if hasattr(patch, "get_width") and hasattr(patch, "get_height"):
                if abs(patch.get_width()) > 0.75 * xspan and abs(patch.get_height()) > 0.75 * yspan:
                    patch.set_alpha(0.0)

    return fig


def main():
    fig = make_figure()
    fig.savefig(OUT_PDF, bbox_inches="tight", transparent=True)
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
