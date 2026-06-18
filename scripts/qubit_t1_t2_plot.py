#!/usr/bin/env python3
"""Create the distance-7 per-qubit T1/T2 lattice plot."""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

_cache_root = Path(tempfile.gettempdir()) / "surfacecodeqmc-mpl-cache"
(_cache_root / "fontconfig").mkdir(parents=True, exist_ok=True)
(_cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root.resolve()))
os.environ.setdefault("MPLCONFIGDIR", str((_cache_root / "matplotlib").resolve()))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


DATA_PATH = Path("data") / "1Qparams_d7.csv"
PLOTS_DIR = Path("plots")
PNG_PATH = PLOTS_DIR / "qubit_t1_t2_d7.png"
PDF_PATH = PLOTS_DIR / "qubit_t1_t2_d7.pdf"

DISTANCE = 7
GRID_SIZE = 2 * DISTANCE + 1

# Figure/layout knobs.
FIGSIZE = (7.8, 4.15)
FIG_WSPACE = 0.05
DIAMOND_RADIUS = 1.0
AXIS_PAD = 0.5
X_LABEL_PAD = 9
Y_LABEL_PAD = 8
TITLE_PAD = 8
PANEL_LABEL_X = -0.07
PANEL_LABEL_Y = 1.03
RANGE_TEXT_Y = -0.18

# Typography knobs.
FONT_FAMILY = "serif"
FONT_SERIF = ["Computer Modern", "Times New Roman", "DejaVu Serif"]
FONT_SIZE = 11
TITLE_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 10
TICK_LENGTH = 7
TICK_WIDTH = 1.2
VALUE_FONTSIZE = 8.0
VALUE_FONTWEIGHT = "semibold"
PANEL_LABEL_FONTSIZE = 12
AXIS_LINEWIDTH = 1.2

# Colormap knobs.
CMAP_NAME = "viridis"
TEXT_LUMINANCE_THRESHOLD = 0.56


@dataclass(frozen=True)
class QubitValue:
    q: int
    t1: float
    t2: float


@dataclass(frozen=True)
class Cell:
    kind: str
    x: float
    y: float
    q: int
    t1: float
    t2: float
    stabilizer: str | None = None


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "font.family": FONT_FAMILY,
            "font.serif": FONT_SERIF,
            "mathtext.fontset": "cm",
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.labelsize": AXIS_LABEL_FONTSIZE,
            "xtick.labelsize": TICK_LABEL_FONTSIZE,
            "ytick.labelsize": TICK_LABEL_FONTSIZE,
            "axes.linewidth": AXIS_LINEWIDTH,
            "xtick.major.width": TICK_WIDTH,
            "ytick.major.width": TICK_WIDTH,
            "xtick.major.size": TICK_LENGTH,
            "ytick.major.size": TICK_LENGTH,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_qubit_params(csv_path: Path) -> list[QubitValue]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"q", "T1_us", "T2_us"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")

        rows = [
            QubitValue(q=int(row["q"]), t1=float(row["T1_us"]), t2=float(row["T2_us"]))
            for row in reader
            if row
        ]
    return sorted(rows, key=lambda row: row.q)


def make_layout(values: list[QubitValue]) -> list[Cell]:
    expected = 2 * DISTANCE * DISTANCE - 1
    if len(values) != expected:
        raise ValueError(f"Expected {expected} qubits for d={DISTANCE}, found {len(values)}.")

    cells: list[Cell] = []
    index = 0

    for x in (0.5, 2.5, 4.5):
        value = values[index]
        cells.append(Cell(kind="check", stabilizer="X", x=x, y=0.0, **value.__dict__))
        index += 1

    for data_row in range(DISTANCE):
        for data_col in range(DISTANCE):
            value = values[index]
            cells.append(Cell(kind="data", x=float(data_col), y=float(2 * data_row + 1), **value.__dict__))
            index += 1

        if data_row < DISTANCE - 1:
            start_x = -0.5 if data_row % 2 == 0 else 0.5
            for check_col in range(DISTANCE):
                value = values[index]
                cells.append(
                    Cell(
                        kind="check",
                        stabilizer="Z" if check_col % 2 == 0 else "X",
                        x=start_x + check_col,
                        y=float(2 * data_row + 2),
                        **value.__dict__,
                    )
                )
                index += 1

    for x in (1.5, 3.5, 5.5):
        value = values[index]
        cells.append(Cell(kind="check", stabilizer="X", x=x, y=14.0, **value.__dict__))
        index += 1

    return cells


def lattice_position(cell: Cell) -> tuple[float, float]:
    return 2 * cell.x + 2, cell.y + 1


def field_value(cell: Cell, field: str) -> float:
    return getattr(cell, field)


def luminance(rgba: tuple[float, float, float, float]) -> float:
    return 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]


def draw_panel(
    ax: plt.Axes,
    cells: list[Cell],
    *,
    field: str,
    title: str,
    panel_label: str,
) -> None:
    values = [field_value(cell, field) for cell in cells]
    vmin = min(values)
    vmax = max(values)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.colormaps[CMAP_NAME]

    for cell in cells:
        x, y = lattice_position(cell)
        value = field_value(cell, field)
        facecolor = cmap(norm(value))
        diamond = Polygon(
            [
                (x, y - DIAMOND_RADIUS),
                (x + DIAMOND_RADIUS, y),
                (x, y + DIAMOND_RADIUS),
                (x - DIAMOND_RADIUS, y),
            ],
            closed=True,
            facecolor=facecolor,
            edgecolor="none",
        )
        ax.add_patch(diamond)
        text_color = "black" if luminance(facecolor) > TEXT_LUMINANCE_THRESHOLD else "white"
        ax.text(
            x,
            y,
            f"{value:.1f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=VALUE_FONTSIZE,
            fontweight=VALUE_FONTWEIGHT,
        )

    ax.set_title(title, pad=TITLE_PAD)
    ax.set_aspect("equal")
    ax.set_xlim(0.5 - AXIS_PAD, GRID_SIZE + 0.5 + AXIS_PAD)
    ax.set_ylim(GRID_SIZE + 0.5 + AXIS_PAD, 0.5 - AXIS_PAD)
    ax.set_xticks(range(1, GRID_SIZE + 1))
    ax.set_yticks(range(1, GRID_SIZE + 1))
    ax.set_xlabel("column", labelpad=X_LABEL_PAD)
    ax.set_ylabel("row", labelpad=Y_LABEL_PAD)
    ax.tick_params(width=TICK_WIDTH, length=TICK_LENGTH, pad=3)
    #ax.text(
    #    PANEL_LABEL_X,
    #    PANEL_LABEL_Y,
    #    panel_label,
    #    transform=ax.transAxes,
    #    ha="left",
    #    va="bottom",
    #    fontsize=PANEL_LABEL_FONTSIZE,
    #    fontweight="bold",
    #)
    #ax.text(
    #    0.0,
    #    RANGE_TEXT_Y,
    #    f"range: {vmin:.1f}-{vmax:.1f} us",
    #    transform=ax.transAxes,
    #    ha="left",
    #    va="top",
    #    fontsize=RANGE_FONTSIZE,
    #    color="#444444",
    #)


def main() -> None:
    configure_style()
    PLOTS_DIR.mkdir(exist_ok=True)

    cells = make_layout(read_qubit_params(DATA_PATH))

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(wspace=FIG_WSPACE)
    draw_panel(axes[0], cells, field="t1", title=r"Qubit $T_1$ [us]", panel_label="a")
    draw_panel(axes[1], cells, field="t2", title=r"Qubit $T_\phi$ [us]", panel_label="b")

    fig.savefig(PNG_PATH, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
