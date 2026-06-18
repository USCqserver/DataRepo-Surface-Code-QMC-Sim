#!/usr/bin/env python3
"""Draft only panel (a) from the composite surface-code figure.

The layout is intentionally constructed with the same figure size, gridspec,
subplot slot, plotting call, colorbar formatting, and panel-label placement as
make_surface_code_composite.py. The final save crops to panel (a).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.transforms import Bbox

import scripts.surface_code_composite as composite
import plot_heatmap_d7_minimal as ph


OUTDIR = Path("figures")
OUT_BASE = OUTDIR / "draft_surface_code_panel_a"
DPI = composite.DPI
CROP_PAD_INCHES = 0.015
P_ANNOTATION_COLOR = "#D99A1E"
Z_CHECK_EDGE_COLOR = P_ANNOTATION_COLOR
Z_CHECK_EDGE_LINEWIDTH = 2.6
GAMMA_COLOR = "#2F83C9"
P_PATCH_COLOR = P_ANNOTATION_COLOR


def _z_row_patches(ancillas):
    z_by_y = {}
    for ancilla in ancillas:
        if ancilla["type"] != "Z":
            continue
        y = round(float(ancilla["coord"][1]), 6)
        z_by_y.setdefault(y, []).append(ancilla)

    return [
        (y, sorted(z_by_y[y], key=lambda a: float(a["coord"][0])))
        for y in sorted(z_by_y.keys(), reverse=True)
    ]


def colorize_z_check_ancilla_edges(ax):
    """Overlay colored outlines on all Z-type measurement ancillas."""
    _, ancillas = ph.build_surface_code_geometry(ph.DISTANCE)
    for ancilla in ancillas:
        if ancilla["type"] != "Z":
            continue
        x, y = map(float, ancilla["coord"])
        ax.add_patch(
            Circle(
                (x, y),
                radius=ph.ANCILLA_RADIUS,
                facecolor="none",
                edgecolor=Z_CHECK_EDGE_COLOR,
                linewidth=Z_CHECK_EDGE_LINEWIDTH,
                zorder=6.5,
            )
        )


def draw_gamma_row_guides(ax):
    x_min = -0.62
    x_max = ph.DISTANCE - 0.66
    for r, y in enumerate(reversed(range(ph.DISTANCE)), start=1):
        ax.add_patch(
            Rectangle(
                (x_min, y - 0.18),
                x_max - x_min,
                0.36,
                facecolor=GAMMA_COLOR,
                edgecolor="none",
                alpha=0.34,
                zorder=2.35,
            )
        )
        ax.add_patch(
            FancyArrowPatch(
                (x_max + 0.07, y),
                (x_max + 0.70, y),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.25,
                color=GAMMA_COLOR,
                alpha=0.95,
                zorder=7,
                clip_on=False,
            )
        )
        ax.text(
            x_max + 0.84,
            y,
            rf"$\gamma_{{{r}}}$",
            ha="left",
            va="center",
            fontsize=11,
            color=GAMMA_COLOR,
            clip_on=False,
            zorder=8,
        )


def draw_p_patch_rails(ax):
    _, ancillas = ph.build_surface_code_geometry(ph.DISTANCE)
    rows = _z_row_patches(ancillas)
    if len(rows) != ph.DISTANCE - 1:
        raise AssertionError(f"Expected {ph.DISTANCE - 1} Z-row patches, got {len(rows)}")

    label_x = -1.58
    arrow_head_x = -1.28

    for r, (y, row) in enumerate(rows, start=1):
        if len(row) != 4:
            raise AssertionError(f"Expected 4 Z ancillas in P_{r}, got {len(row)}")

        xs = [float(a["coord"][0]) for a in row]
        rail_end = max(xs) + 0.42
        ax.add_patch(
            FancyArrowPatch(
                (rail_end, y),
                (arrow_head_x, y),
                arrowstyle="-|>",
                mutation_scale=15,
                shrinkA=0,
                shrinkB=0,
                linewidth=1.55,
                color=P_PATCH_COLOR,
                alpha=0.95,
                zorder=5.65,
                clip_on=False,
            )
        )
        ax.text(
            label_x,
            y,
            rf"$P_{{{r}}}$",
            ha="right",
            va="center",
            fontsize=11,
            color=P_PATCH_COLOR,
            clip_on=False,
            zorder=8,
        )


def annotate_panel_a(ax):
    draw_gamma_row_guides(ax)
    draw_p_patch_rails(ax)
    colorize_z_check_ancilla_edges(ax)


def _panel_a_bbox_inches(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = Bbox.union([ax.get_tightbbox(renderer) for ax in axes])
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    return bbox.padded(CROP_PAD_INCHES)


def make_panel_a():
    composite.configure_matplotlib()

    fig = plt.figure(figsize=composite.FIGSIZE)
    fig.patch.set_alpha(0.0)

    outer_gs = fig.add_gridspec(
        2,
        1,
        height_ratios=composite.TOP_BOTTOM_HEIGHT_RATIOS,
        hspace=composite.TOP_BOTTOM_HSPACE,
    )
    top_gs = outer_gs[0, 0].subgridspec(
        1,
        2,
        width_ratios=composite.WIDTH_RATIOS,
        wspace=composite.WSPACE,
    )

    ax_patch = fig.add_subplot(top_gs[0, 0])
    ph.plot_surface_code(ax=ax_patch)
    annotate_panel_a(ax_patch)

    cbar_ax = fig.axes[-1]
    cbar_ax.remove()

    ax_patch.set_xlim(-1.78, ph.DISTANCE + 0.28)
    ax_patch.set_ylim(-0.90, ph.DISTANCE + 0.10)

    for ax in fig.axes:
        ax.patch.set_alpha(0.0)

    return fig, (ax_patch,)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, crop_axes = make_panel_a()
    crop = _panel_a_bbox_inches(fig, crop_axes)
    fig.savefig(f"{OUT_BASE}.pdf", bbox_inches=crop, transparent=True)
    fig.savefig(f"{OUT_BASE}.svg", bbox_inches=crop, transparent=True)
    fig.savefig(f"{OUT_BASE}.png", dpi=DPI, bbox_inches=crop, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
