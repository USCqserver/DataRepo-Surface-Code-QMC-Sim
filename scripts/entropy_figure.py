#!/usr/bin/env python3
"""Create the patch-syndrome entropy/cross-entropy figure."""

from __future__ import annotations

import argparse
import csv
import os
import re
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


ROW_LABELS = [f"row_{i}" for i in range(1, 7)]
GAMMA_LABELS = [f"gamma_{i}" for i in range(1, 8)]
ROW_TICK_LABELS = [rf"$P_{i}$" for i in range(1, 7)]
GAMMA_TICK_LABELS = [rf"$\gamma_{i}$" for i in range(1, 8)]
DETUNING_ORDER = ["-50", "-25", "0", "+25", "+50"]

# Panel (a) heatmap typography knobs.
HEATMAP_ENTRY_FONTSIZE = 10.0
HEATMAP_GAMMA_TICK_FONTSIZE = 15.0
HEATMAP_ROW_TICK_FONTSIZE = 12.0
HEATMAP_COLORBAR_TICK_FONTSIZE = 12.0
HEATMAP_COLORBAR_LABEL_FONTSIZE = 12.0
HEATMAP_FIGSIZE = (7.8, 3.55)
HEATMAP_WSPACE = 0.16
HEATMAP_COLORBAR_Y_OFFSET = 0.135
HEATMAP_COLORBAR_HEIGHT = 0.018


def default_input_dir() -> Path:
    for candidate in (Path("examples/stim-sim"), Path("data")):
        if (candidate / "mi_heatmap_ME_no_detuning.csv").exists():
            return candidate
    return Path("examples/stim-sim")


def read_heatmap(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        columns = header[1:]
        rows: list[str] = []
        values: list[list[float]] = []
        for row in reader:
            if not row:
                continue
            rows.append(row[0])
            values.append([abs(float(x)) for x in row[1:]])
    return rows, columns, np.asarray(values, dtype=float)


def detuning_label(path: Path) -> str | None:
    stem = path.stem
    if stem in {"entropy_reduction_report", "entropy_reduction_report_no_detuning"}:
        return "0"
    match = re.search(r"_detuning_([mp])(\d+)$", stem)
    if not match:
        return None
    sign = "-" if match.group(1) == "m" else "+"
    return f"{sign}{int(match.group(2))}"


def strip_md_cell(cell: str) -> str:
    cell = cell.strip()
    cell = cell.replace("`", "").replace("*", "")
    return cell.strip()


def parse_cross_entropy_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    text = path.read_text()
    section_match = re.search(
        r"##\s*2\.\s*ME-Reference Cross Entropy and KL Gap(?P<body>.*?)(?:\n##\s+|\Z)",
        text,
        flags=re.S,
    )
    if not section_match:
        raise ValueError(f"Could not find cross-entropy section in {path}")

    rows: list[tuple[str, float, float]] = []
    for line in section_match.group("body").splitlines():
        if not line.startswith("|"):
            continue
        cells = [strip_md_cell(cell) for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] in {"Patch", "---"} or "Average" in cells[0]:
            continue
        if cells[0] not in ROW_LABELS:
            continue
        ce_stim = float(cells[1])
        me_cond = float(cells[2])
        rows.append((cells[0], ce_stim, me_cond))

    rows_by_label = {label: (ce, me) for label, ce, me in rows}
    missing = [label for label in ROW_LABELS if label not in rows_by_label]
    if missing:
        raise ValueError(f"Missing rows in {path}: {', '.join(missing)}")

    ce_values = np.asarray([rows_by_label[label][0] for label in ROW_LABELS], dtype=float)
    me_values = np.asarray([rows_by_label[label][1] for label in ROW_LABELS], dtype=float)
    return me_values, ce_values


def collect_detuning_data(input_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reports = sorted(input_dir.glob("entropy_reduction_report*.md"))
    reports = [path for path in reports if "summary" not in path.stem.lower()]
    if not reports:
        raise FileNotFoundError(f"No entropy_reduction_report*.md files found in {input_dir}")

    by_label: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for report in reports:
        label = detuning_label(report)
        if label is None:
            continue
        by_label[label] = parse_cross_entropy_table(report)

    labels = [label for label in DETUNING_ORDER if label in by_label]
    if not labels:
        raise ValueError(f"No recognized detuning report filenames found in {input_dir}")

    me_rows = np.vstack([by_label[label][0] for label in labels])
    ce_rows = np.vstack([by_label[label][1] for label in labels])
    return (
        labels,
        me_rows.mean(axis=1),
        me_rows.std(axis=1, ddof=1),
        ce_rows.mean(axis=1),
        ce_rows.std(axis=1, ddof=1),
    )


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "font.family": "serif",
            "font.serif":'Computer Modern',
            "text.usetex": True,
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 10,
            "ytick.major.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def annotate_heatmap(ax: plt.Axes, values: np.ndarray, norm: mpl.colors.Normalize) -> None:
    cmap = mpl.colormaps["viridis"]
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            rgba = cmap(norm(values[i, j]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "black" if luminance > 0.56 else "white"
            ax.text(
                j,
                i,
                f"{values[i, j]:.3f}",
                ha="center",
                va="center",
                color=color,
                fontsize=HEATMAP_ENTRY_FONTSIZE,
            )


def load_heatmaps(input_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    me_rows, me_cols, me_values = read_heatmap(input_dir / "mi_heatmap_ME_no_detuning.csv")
    stim_rows, stim_cols, stim_values = read_heatmap(input_dir / "mi_heatmap_Stim_zz_symmetric.csv")
    if me_rows != ROW_LABELS or stim_rows != ROW_LABELS:
        raise ValueError("Heatmap row labels do not match row_1 ... row_6")
    if me_cols != GAMMA_LABELS or stim_cols != GAMMA_LABELS:
        raise ValueError("Heatmap column labels do not match gamma_1 ... gamma_7")
    return me_values, stim_values


def draw_heatmap_figure(input_dir: Path, output_dir: Path, basename: str) -> tuple[Path, Path]:
    me_values, stim_values = load_heatmaps(input_dir)
    configure_style()
    fig = plt.figure(figsize=HEATMAP_FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, wspace=HEATMAP_WSPACE)
    ax_me = fig.add_subplot(gs[0, 0])
    ax_stim = fig.add_subplot(gs[0, 1], sharex=ax_me, sharey=ax_me)

    norm = mpl.colors.Normalize(vmin=0.0, vmax=0.15)
    heatmap_kwargs = {"cmap": "viridis", "norm": norm, "aspect": "equal", "interpolation": "nearest"}

    im = ax_me.imshow(me_values, **heatmap_kwargs)
    im_stim = ax_stim.imshow(stim_values, **heatmap_kwargs)
    for ax, title, values in ((ax_me, "ME (QMC)", me_values), (ax_stim, "Clifford (Stim)", stim_values)):
        #ax.set_title(title, pad=3)
        ax.set_yticks(np.arange(len(ROW_LABELS)), ROW_TICK_LABELS)
        ax.set_xlim(-0.5, len(GAMMA_LABELS) - 0.5)
        ax.set_ylim(len(ROW_LABELS) - 0.5, -0.5)
        ax.tick_params(axis="both", length=0)
        ax.tick_params(axis="y", labelsize=HEATMAP_ROW_TICK_FONTSIZE)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(len(GAMMA_LABELS)), GAMMA_TICK_LABELS)
        annotate_heatmap(ax, values, norm)
    ax_stim.tick_params(labelleft=False)
    for ax in (ax_me, ax_stim):
        for label in ax.get_xticklabels():
            label.set_rotation(35)
            label.set_ha("right")
            label.set_rotation_mode("anchor")
            label.set_fontsize(HEATMAP_GAMMA_TICK_FONTSIZE)

    #fig.text(
    #    0.015,
    #    0.985,
    #    r"\textbf{(a)}",
    #    ha="left",
    #    va="top",
    #)

    fig.canvas.draw()
    caxes = []
    for ax, image in ((ax_me, im), (ax_stim, im_stim)):
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0, pos.y0 - HEATMAP_COLORBAR_Y_OFFSET, pos.width, HEATMAP_COLORBAR_HEIGHT])
        cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
        cbar.set_ticks([0.0, 0.05, 0.10, 0.15])
        cbar.set_label(r"$I\left(\mathbf{Y_r} ; L_c^{(X)}\right)$ [bits]", labelpad=2, fontsize=HEATMAP_COLORBAR_LABEL_FONTSIZE)
        cbar.ax.tick_params(length=2, pad=1, labelsize=HEATMAP_COLORBAR_TICK_FONTSIZE)
        caxes.append(cax)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{basename}_a_heatmaps.png"
    pdf_path = output_dir / f"{basename}_a_heatmaps.pdf"
    fig.savefig(png_path, bbox_inches="tight", bbox_extra_artists=caxes)
    fig.savefig(pdf_path, bbox_inches="tight", bbox_extra_artists=caxes)
    plt.close(fig)
    return png_path, pdf_path


def draw_bar_figure(input_dir: Path, output_dir: Path, basename: str) -> tuple[Path, Path]:
    labels, me_mean, me_sd, ce_mean, ce_sd = collect_detuning_data(input_dir)
    configure_style()
    fig, ax_bar = plt.subplots(figsize=(3.8, 3.05), constrained_layout=True)

    x = np.arange(len(labels))
    width = 0.36
    colors = ("#0072B2", "#D55E00")
    error_kw = {"elinewidth": 0.8, "capthick": 0.8, "capsize": 2.5}
    ax_bar.bar(
        x - width / 2,
        me_mean,
        width,
        yerr=me_sd,
        label=r"$H_{\rm ME}(L^{(X)}_c \mid \mathbf{Y_r})$",
        color=colors[0],
        edgecolor="black",
        linewidth=0.35,
        error_kw=error_kw,
    )
    ax_bar.bar(
        x + width / 2,
        ce_mean,
        width,
        yerr=ce_sd,
        label=r"$H_{\rm Clif|ME}(L^{(X)}_c\mid \mathbf{Y_r})$",
        color=colors[1],
        edgecolor="black",
        linewidth=0.35,
        error_kw=error_kw,
    )
    ax_bar.set_xticks(x, labels)
    ax_bar.set_xlabel(r"$\Delta_g/2\pi$ (kHz)")
    ax_bar.set_ylabel("Entropy [bits]")
    ax_bar.legend(frameon=False, loc="upper left")
    ax_bar.grid(axis="y", color="0.86", linewidth=0.6)
    ax_bar.set_axisbelow(True)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.set_ylim(0.1, 0.2)
    #ax_bar.text(
    #    -0.22,
    #    1.08,
    #    r"\textbf{(b)}",
    #    transform=ax_bar.transAxes,
    #    ha="left",
    #    va="bottom",
    #)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{basename}_b_cross_entropy.png"
    pdf_path = output_dir / f"{basename}_b_cross_entropy.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_input_dir())
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument("--basename", default="patch_entropy_cross_entropy")
    args = parser.parse_args()

    for path in draw_heatmap_figure(args.input_dir, args.output_dir, args.basename):
        print(f"Wrote {path}")
    for path in draw_bar_figure(args.input_dir, args.output_dir, args.basename):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
