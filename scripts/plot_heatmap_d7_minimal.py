import json
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable


DISTANCE = 7
BACKGROUND_COLOR = "#FAFAFA"
CROSSTALK_JSON_PATH = "./circuits/surface_code_d7_flat-ir.json"
CROSSTALK_VMIN = 10.0
CROSSTALK_VMAX = 100.0
PROBE_LINK_LINEWIDTH = 5.0
PROBE_LINK_ALPHA_MIN = 0.40
PROBE_LINK_ALPHA_MAX = 1.0

PLAQUETTE_STYLE = dict(
    z_facecolor="#FAFAFA",
    z_edgecolor="#EEEEEE",
    x_facecolor="#FAFAFA",
    x_edgecolor="#EEEEEE",
    alpha=0.85,
    linewidth=0.5,
)

DATA_QUBIT_STYLE = dict(
    radius=0.16,
    facecolor="#B5B3B3",
    edgecolor="#FAFAFA",
    linewidth=0.8,
    alpha=1.0,
)

ANCILLA_RADIUS = 0.28
ANCILLA_LABEL_STYLE = dict(ha="center", va="center", fontsize=10, color="black", zorder=6)
ANCILLA_OUTLINE_STYLE = dict(edgecolor="black", linewidth=0.8, linestyle="-", alpha=1.0)

PLAQUETTE_ORDER = ("NE", "NW", "SE", "SW")
QISKIT_DRAW_STYLE = dict(
    name="bw",
    backgroundcolor=BACKGROUND_COLOR,
    displaycolor=dict(
        h=(BACKGROUND_COLOR, "#202020"),
        cz=("#202020", BACKGROUND_COLOR),
        M=(BACKGROUND_COLOR, "#202020"),
    ),
)


def signed_diff_cmap():
    return LinearSegmentedColormap.from_list(
        "signed_diff", ["#1C2E8A", "#FAFAFA", "#C64A1C"], N=256
    )


def sequential_diff_cmap(vmin=0.0, vmax=1.0):
    colors = ["#1C2E8A", "#FAFAFA"] if float(vmax) <= 0.0 else ["#FAFAFA", "#C64A1C"]
    return LinearSegmentedColormap.from_list("sequential_diff", colors, N=256)


def probe_link_cmap():
    return LinearSegmentedColormap.from_list(
        "probe_link_teal", ["#7ED6DC", "#0F6A73"], N=256
    )


def load_static_zz_crosstalk_khz(json_path: str = CROSSTALK_JSON_PATH):
    """Return {(qid_a, qid_b): ZZ coupling in kHz} from a compiled-circuit JSON."""
    with open(json_path, "r") as f:
        blob = json.load(f)

    pair_to_khz = {}
    for term in blob.get("hamiltonian", {}).get("static_hamiltonian", []):
        op = term.get("operator", {})
        labels = op.get("labels", [])
        subs = op.get("subsystems", [])
        coeff = term.get("coefficients")

        if labels != ["Z", "Z"]:
            continue
        if not (isinstance(subs, list) and len(subs) == 2):
            continue
        if not isinstance(coeff, (int, float)):
            continue

        i, j = int(subs[0]), int(subs[1])
        if i != j:
            pair_to_khz[(min(i, j), max(i, j))] = float(coeff) / (2.0 * math.pi) * 1e6

    return pair_to_khz


def build_surface_code_geometry(distance: int = DISTANCE):
    """Build rotated surface-code data coordinates and globally numbered ancillas."""
    d = int(distance)
    if d < 3 or d % 2 == 0:
        raise ValueError("distance must be an odd integer >= 3")

    data_xy = [(x, y) for y in range(d) for x in range(d)]

    def stab_type_from_cell(cell_x: int, cell_y: int) -> str:
        return "X" if (cell_x + cell_y) % 2 == 0 else "Z"

    ancillas_raw = []

    for cy in range(d - 1):
        for cx in range(d - 1):
            corners = [(cx, cy), (cx + 1, cy), (cx + 1, cy + 1), (cx, cy + 1)]
            ancillas_raw.append((cx + 0.5, cy + 0.5, stab_type_from_cell(cx, cy), corners, 4))

    def add_boundary(cell_x: int, cell_y: int, neighbor_xy, keep_type: str):
        t = stab_type_from_cell(cell_x, cell_y)
        if t == keep_type:
            ancillas_raw.append((cell_x + 0.5, cell_y + 0.5, t, list(neighbor_xy), 2))

    for cx in range(d - 1):
        add_boundary(cx, -1, [(cx, 0), (cx + 1, 0)], "X")
        add_boundary(cx, d - 1, [(cx, d - 1), (cx + 1, d - 1)], "X")
    for cy in range(d - 1):
        add_boundary(-1, cy, [(0, cy), (0, cy + 1)], "Z")
        add_boundary(d - 1, cy, [(d - 1, cy), (d - 1, cy + 1)], "Z")

    expected_ancillas = d * d - 1
    if len(ancillas_raw) != expected_ancillas:
        raise AssertionError(f"Expected {expected_ancillas} ancillas, got {len(ancillas_raw)}")

    all_qubits = [(float(x), float(y), ("D", x, y)) for x, y in data_xy]
    all_qubits += [(float(ax), float(ay), ("A", ax, ay)) for ax, ay, _, _, _ in ancillas_raw]
    global_id_of_key = {
        key: idx
        for idx, (_, _, key) in enumerate(sorted(all_qubits, key=lambda t: (-t[1], t[0])), start=1)
    }

    data_coords = {}
    data_id_of_xy = {}
    for x, y in data_xy:
        qid = global_id_of_key[("D", x, y)]
        data_id_of_xy[(x, y)] = qid
        data_coords[qid] = np.array([x, y], dtype=float)

    anc_id_of_coord = {
        (ax, ay): global_id_of_key[("A", ax, ay)] for ax, ay, _, _, _ in ancillas_raw
    }

    def rel_pos(ax: float, ay: float, dx: int, dy: int) -> str:
        rx = dx - ax
        ry = dy - ay
        if rx > 0 and ry > 0:
            return "NE"
        if rx < 0 and ry > 0:
            return "NW"
        if rx > 0 and ry < 0:
            return "SE"
        if rx < 0 and ry < 0:
            return "SW"
        raise ValueError("Unexpected relative offset")

    ancillas = []
    for ax, ay, t, nbrs_xy, weight in ancillas_raw:
        neighbors = {
            rel_pos(ax, ay, x, y): data_id_of_xy[(x, y)]
            for x, y in nbrs_xy
        }
        ancillas.append(
            dict(
                id=anc_id_of_coord[(ax, ay)],
                type=t,
                coord=np.array([ax, ay], dtype=float),
                neighbors=neighbors,
                weight=weight,
            )
        )

    ancillas.sort(key=lambda a: (-float(a["coord"][1]), float(a["coord"][0])))
    for stab_index, ancilla in enumerate(ancillas):
        ancilla["stab_index"] = stab_index

    return data_coords, ancillas


def _parse_ancilla_expectations(values, ancillas):
    if values is None:
        return None
    if isinstance(values, dict):
        return {int(k): float(v) for k, v in values.items()}

    arr = np.asarray(values, dtype=float).ravel()
    n_anc = DISTANCE * DISTANCE - 1
    if arr.shape[0] != n_anc:
        raise ValueError(f"expectation vector must have length {n_anc}, got {arr.shape[0]}")
    return {a["id"]: float(arr[a["stab_index"]]) for a in ancillas}


def _plaquette_polygon_4body(ancilla, data_coords):
    pts = np.vstack([data_coords[q] for q in ancilla["neighbors"].values()])
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


def _plaquette_polygon_2body(ancilla, data_coords, thickness=0.5):
    ids = list(ancilla["neighbors"].values())
    p1 = data_coords[ids[0]]
    p2 = data_coords[ids[1]]
    mid = 0.5 * (p1 + p2)
    edge = p2 - p1
    edge_norm = np.linalg.norm(edge)
    if edge_norm < 1e-9:
        raise ValueError("Degenerate boundary edge")

    normal = np.array([-edge[1], edge[0]], dtype=float) / edge_norm
    if np.dot(ancilla["coord"] - mid, normal) < 0:
        normal = -normal
    return np.vstack([p1, p2, p2 + thickness * normal, p1 + thickness * normal])


def _main_norm(vmin, vmax, *, symmetric=True):
    if symmetric:
        vmax_abs = max(abs(float(vmin)), abs(float(vmax)))
        if math.isclose(vmax_abs, 0.0):
            vmax_abs = 1.0
        return TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs)

    if not float(vmin) < float(vmax):
        raise ValueError("vmin must be less than vmax for an asymmetric colorbar")
    return plt.Normalize(vmin=float(vmin), vmax=float(vmax))


def plot_surface_code(
    stab_expectations=None,
    *,
    ax=None,
    vmin=-1.0,
    vmax=1.0,
    cmap=None,
    crosstalk_cmap=None,
    crosstalk_json_path=CROSSTALK_JSON_PATH,
    symmetric_colorbar=True,
    colorbar_label=None,
):
    data_coords, ancillas = build_surface_code_geometry(DISTANCE)
    show_syndrome_colorbar = stab_expectations is not None
    if stab_expectations is None:
        exp_by_anc_id = {int(a["id"]): 0.0 for a in ancillas}
    else:
        exp_by_anc_id = _parse_ancilla_expectations(stab_expectations, ancillas)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor=BACKGROUND_COLOR)
    else:
        fig = ax.figure
        fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    norm = _main_norm(vmin, vmax, symmetric=symmetric_colorbar)
    if cmap is None:
        cmap_obj = signed_diff_cmap() if symmetric_colorbar else sequential_diff_cmap(vmin, vmax)
    else:
        cmap_obj = cmap
    xt_cmap_obj = probe_link_cmap() if crosstalk_cmap is None else crosstalk_cmap

    for ancilla in ancillas:
        if ancilla["weight"] == 4:
            poly = _plaquette_polygon_4body(ancilla, data_coords)
        else:
            poly = _plaquette_polygon_2body(ancilla, data_coords)

        prefix = "z" if ancilla["type"] == "Z" else "x"
        ax.add_patch(
            Polygon(
                poly,
                closed=True,
                facecolor=PLAQUETTE_STYLE[f"{prefix}_facecolor"],
                edgecolor=PLAQUETTE_STYLE[f"{prefix}_edgecolor"],
                alpha=PLAQUETTE_STYLE["alpha"],
                linewidth=PLAQUETTE_STYLE["linewidth"],
                zorder=0,
            )
        )

    crosstalk_pairs_khz = load_static_zz_crosstalk_khz(crosstalk_json_path)
    segments = []
    vals = []
    for ancilla in ancillas:
        ax_, ay_ = map(float, ancilla["coord"])
        for dq in ancilla["neighbors"].values():
            dx, dy = data_coords[dq]
            segments.append([[ax_, ay_], [float(dx), float(dy)]])
            key = (min(int(ancilla["id"]), int(dq)), max(int(ancilla["id"]), int(dq)))
            vals.append(float(crosstalk_pairs_khz.get(key, 0.0)))

    vals_arr = np.asarray(vals, dtype=float)
    xt_norm = plt.Normalize(vmin=CROSSTALK_VMIN, vmax=CROSSTALK_VMAX)
    strengths = np.clip(xt_norm(vals_arr), 0.0, 1.0)
    rgba = xt_cmap_obj(strengths)
    rgba[:, 3] = PROBE_LINK_ALPHA_MIN + (PROBE_LINK_ALPHA_MAX - PROBE_LINK_ALPHA_MIN) * strengths

    links = LineCollection(segments, zorder=2)
    links.set_linewidths(np.full(vals_arr.shape, PROBE_LINK_LINEWIDTH, dtype=float))
    links.set_color(rgba)
    ax.add_collection(links)

    for coord in data_coords.values():
        ax.add_patch(
            Circle(
                coord,
                radius=DATA_QUBIT_STYLE["radius"],
                facecolor=DATA_QUBIT_STYLE["facecolor"],
                edgecolor=DATA_QUBIT_STYLE["edgecolor"],
                linewidth=DATA_QUBIT_STYLE["linewidth"],
                alpha=DATA_QUBIT_STYLE["alpha"],
                zorder=3,
            )
        )

    for ancilla in ancillas:
        x, y = map(float, ancilla["coord"])
        facecolor = cmap_obj(norm(exp_by_anc_id.get(ancilla["id"], 0.0)))

        ax.add_patch(
            Circle((x, y), radius=ANCILLA_RADIUS, facecolor=facecolor, edgecolor="none", zorder=4)
        )
        ax.add_patch(
            Circle(
                (x, y),
                radius=ANCILLA_RADIUS,
                facecolor="none",
                edgecolor=ANCILLA_OUTLINE_STYLE["edgecolor"],
                linewidth=ANCILLA_OUTLINE_STYLE["linewidth"],
                linestyle=ANCILLA_OUTLINE_STYLE["linestyle"],
                alpha=ANCILLA_OUTLINE_STYLE["alpha"],
                zorder=5,
            )
        )
        ax.text(x, y, ancilla["type"], **ANCILLA_LABEL_STYLE)

    ax.set_aspect("equal", "box")
    ax.set_xlim(-0.85, DISTANCE - 0.15)
    ax.set_ylim(-0.85, DISTANCE - 0.15)
    ax.axis("off")

    divider = make_axes_locatable(ax)
    cax_bottom = divider.append_axes("bottom", size="4.5%", pad=0.12)
    if show_syndrome_colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        cbar_label = "Syndrome Extraction Bias" if colorbar_label is None else colorbar_label
    else:
        sm = plt.cm.ScalarMappable(norm=xt_norm, cmap=xt_cmap_obj)
        cbar_label = "Crosstalk strength (kHz)" if colorbar_label is None else colorbar_label
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax_bottom, orientation="horizontal")
    cax_bottom.xaxis.set_ticks_position("bottom")
    cax_bottom.xaxis.set_label_position("bottom")
    cbar.set_label(cbar_label, fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    return ax


def _setup_patch_axes(ax, xlim, ylim):
    if ax is None:
        _, ax = plt.subplots(1, 1, facecolor=BACKGROUND_COLOR)
    ax.figure.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.set_aspect("equal", "box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return ax


def _draw_circle_marker(
    ax,
    xy,
    *,
    radius,
    facecolor,
    edgecolor="#202020",
    linewidth=1.4,
    label=None,
    fontsize=11,
    zorder=3,
):
    ax.add_patch(
        Circle(
            xy,
            radius=radius,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )
    )
    if label is not None:
        ax.text(
            xy[0],
            xy[1],
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="#101010",
            zorder=zorder + 1,
        )


def _example_plaquette(kind):
    data_coords, ancillas = build_surface_code_geometry(DISTANCE)
    for ancilla in ancillas:
        if ancilla["type"] == kind and ancilla["weight"] == 4:
            return data_coords, ancilla
    raise ValueError(f"No interior {kind} plaquette found")


def _centered_plaquette_points(data_coords, ancilla):
    center = ancilla["coord"]
    points = {
        pos: np.asarray(data_coords[qid], dtype=float) - center
        for pos, qid in ancilla["neighbors"].items()
    }
    return points, np.array([0.0, 0.0])


def _draw_heatmap_style_plaquette(
    ax,
    kind,
    data_pos,
    center,
    *,
    label_data=True,
    data_radius=None,
    ancilla_radius=None,
):
    if data_radius is None:
        data_radius = DATA_QUBIT_STYLE["radius"]
    if ancilla_radius is None:
        ancilla_radius = ANCILLA_RADIUS

    polygon = np.vstack([data_pos[pos] for pos in ("SW", "SE", "NE", "NW")])
    prefix = "z" if kind == "Z" else "x"
    ax.add_patch(
        Polygon(
            polygon,
            closed=True,
            facecolor="none",
            edgecolor=PLAQUETTE_STYLE[f"{prefix}_edgecolor"],
            linewidth=max(PLAQUETTE_STYLE["linewidth"], 1.0),
            zorder=0,
        )
    )

    link_color = probe_link_cmap()(0.75)
    for pos in ("NW", "NE", "SW", "SE"):
        ax.plot(
            [center[0], data_pos[pos][0]],
            [center[1], data_pos[pos][1]],
            color=link_color,
            linewidth=PROBE_LINK_LINEWIDTH,
            solid_capstyle="round",
            zorder=1,
        )

    data_labels = {"NW": r"$q_0$", "NE": r"$q_1$", "SW": r"$q_2$", "SE": r"$q_3$"}
    label_offsets = {
        "NW": np.array([-0.09, 0.12]),
        "NE": np.array([0.09, 0.12]),
        "SW": np.array([-0.09, -0.12]),
        "SE": np.array([0.09, -0.12]),
    }
    label_align = {
        "NW": ("right", "bottom"),
        "NE": ("left", "bottom"),
        "SW": ("right", "top"),
        "SE": ("left", "top"),
    }
    for pos in ("NW", "NE", "SW", "SE"):
        _draw_circle_marker(
            ax,
            data_pos[pos],
            radius=data_radius,
            facecolor=DATA_QUBIT_STYLE["facecolor"],
            edgecolor=DATA_QUBIT_STYLE["edgecolor"],
            linewidth=DATA_QUBIT_STYLE["linewidth"],
            zorder=4,
        )
        if label_data:
            label_xy = data_pos[pos] + label_offsets[pos]
            ha, va = label_align[pos]
            ax.text(
                label_xy[0],
                label_xy[1],
                data_labels[pos],
                ha=ha,
                va=va,
                fontsize=9,
                color="#101010",
                zorder=6,
            )

    _draw_circle_marker(
        ax,
        center,
        radius=ancilla_radius,
        facecolor=BACKGROUND_COLOR,
        edgecolor=ANCILLA_OUTLINE_STYLE["edgecolor"],
        linewidth=1.4,
        label=kind,
        fontsize=8,
        zorder=5,
    )


def plot_plaquette(
    kind="X",
    *,
    ax=None,
    label_data=True,
    data_radius=None,
    ancilla_radius=None,
):
    """Draw a static heatmap-style interior plaquette."""
    kind = str(kind).upper()
    if kind not in ("X", "Z"):
        raise ValueError('kind must be "X" or "Z"')

    data_coords, ancilla = _example_plaquette(kind)
    data_pos, center = _centered_plaquette_points(data_coords, ancilla)
    ax = _setup_patch_axes(ax, (-0.90, 0.90), (-0.90, 0.90))
    _draw_heatmap_style_plaquette(
        ax,
        kind,
        data_pos,
        center,
        label_data=label_data,
        data_radius=data_radius,
        ancilla_radius=ancilla_radius,
    )
    return ax


def plot_x_plaquette(*, ax=None, label_data=True, data_radius=None, ancilla_radius=None):
    return plot_plaquette(
        "X",
        ax=ax,
        label_data=label_data,
        data_radius=data_radius,
        ancilla_radius=ancilla_radius,
    )


def plot_z_plaquette(*, ax=None, label_data=True, data_radius=None, ancilla_radius=None):
    return plot_plaquette(
        "Z",
        ax=ax,
        label_data=label_data,
        data_radius=data_radius,
        ancilla_radius=ancilla_radius,
    )


def build_plaquette_circuit(kind="X"):
    """Build the four-data-qubit extraction circuit using Qiskit."""
    kind = str(kind).upper()
    if kind not in ("X", "Z"):
        raise ValueError('kind must be "X" or "Z"')

    try:
        from qiskit import QuantumCircuit, QuantumRegister
        from qiskit.circuit import Gate
    except ImportError as exc:
        raise ImportError(
            'Qiskit is required for circuit plots. Install "qiskit[visualization]" '
            "in the notebook environment."
        ) from exc

    anc = QuantumRegister(1, kind)
    q = QuantumRegister(4, "q")
    qc = QuantumCircuit(anc, q)

    ancilla = anc[0]
    data = {"NW": q[0], "NE": q[1], "SW": q[2], "SE": q[3]}
    measurement_icon = Gate("M", 1, [])

    if kind == "X":
        qc.h(ancilla)
        for key in ("NW", "NE", "SW", "SE"):
            qc.h(data[key])
        qc.barrier()
        for key in PLAQUETTE_ORDER:
            qc.cz(ancilla, data[key])
        qc.barrier()
        for key in ("NW", "NE", "SW", "SE"):
            qc.h(data[key])
        qc.h(ancilla)
    else:
        qc.h(ancilla)
        qc.barrier()
        for key in PLAQUETTE_ORDER:
            qc.cz(ancilla, data[key])
        qc.barrier()
        qc.h(ancilla)

    qc.append(measurement_icon, [ancilla])
    return qc


def _replace_m_gate_with_measurement_symbol(fig):
    """Turn the display-only M gate into a measurement glyph without a classical wire."""
    for ax in fig.axes:
        for text in list(ax.texts):
            if text.get_text() != "M":
                continue
            x, y = text.get_position()
            text.set_text("")
            ax.add_patch(
                Arc(
                    (x - 0.01, y - 0.11),
                    width=0.44,
                    height=0.34,
                    angle=0.0,
                    theta1=12.0,
                    theta2=168.0,
                    color="#202020",
                    linewidth=0.8,
                    zorder=text.get_zorder() + 1,
                )
            )
            ax.add_patch(
                FancyArrowPatch(
                    (x - 0.14, y - 0.15),
                    (x + 0.28, y + 0.22),
                    arrowstyle="-",
                    mutation_scale=1,
                    linewidth=0.7,
                    color="#202020",
                    zorder=text.get_zorder() + 1,
                )
            )
    return fig


def plot_plaquette_circuit(kind="X", **draw_kwargs):
    """Draw one plaquette extraction circuit with Qiskit's Matplotlib drawer."""
    qc = build_plaquette_circuit(kind)
    style = dict(QISKIT_DRAW_STYLE)
    style["displaycolor"] = dict(QISKIT_DRAW_STYLE["displaycolor"])
    style.update(draw_kwargs.pop("style", {}) or {})
    draw_kwargs.setdefault("fold", -1)
    draw_kwargs.setdefault("scale", 0.9)
    draw_kwargs.setdefault("plot_barriers", False)
    target_ax = draw_kwargs.get("ax")
    fig = qc.draw(output="mpl", style=style, **draw_kwargs)
    if fig is None and target_ax is not None:
        fig = target_ax.figure
    return _replace_m_gate_with_measurement_symbol(fig)


def plot_x_plaquette_circuit(**draw_kwargs):
    return plot_plaquette_circuit("X", **draw_kwargs)


def plot_z_plaquette_circuit(**draw_kwargs):
    return plot_plaquette_circuit("Z", **draw_kwargs)


def plot_plaquette_circuits(**draw_kwargs):
    """Return separate Qiskit Matplotlib figures for the X and Z circuits."""
    return (
        plot_x_plaquette_circuit(**draw_kwargs),
        plot_z_plaquette_circuit(**draw_kwargs),
    )


__all__ = [
    "build_plaquette_circuit",
    "build_surface_code_geometry",
    "load_static_zz_crosstalk_khz",
    "plot_plaquette",
    "plot_plaquette_circuit",
    "plot_plaquette_circuits",
    "plot_surface_code",
    "plot_x_plaquette",
    "plot_x_plaquette_circuit",
    "plot_z_plaquette",
    "plot_z_plaquette_circuit",
    "probe_link_cmap",
    "sequential_diff_cmap",
    "signed_diff_cmap",
]
