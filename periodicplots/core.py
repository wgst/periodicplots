"""Vector periodic-table heatmaps in matplotlib.

The single entry point is :func:`periodic_table`.  It draws each element as a
matplotlib ``Rectangle`` + text (no rasterisation), so the result is fully
vector and composes into multi-panel figures via the ``ax=`` argument.

Default appearance: element symbol (bold) with the value beneath it, a viridis
heatmap, a slim colourbar and detached La-Lu / Ac-Lr f-block rows.  Atomic
number, atomic mass and full element name are optional (off by default).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize, TwoSlopeNorm, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ._elements import ELEMENTS, SYMBOL_TO_Z

# detached f-block layout (matches the reference figure): lanthanoids and
# actinoids sit in their own rows below the main table.
_LANTH_ROW, _ACT_ROW, _FCOL = 8.5, 9.5, 3


@dataclass
class PeriodicTablePlot:
    """Result of :func:`periodic_table`."""
    fig: Figure
    ax: Axes
    mappable: ScalarMappable

    def save(self, path: str, **kwargs) -> "PeriodicTablePlot":
        """Save the figure; format is inferred from the extension."""
        kwargs.setdefault("bbox_inches", "tight")
        self.fig.savefig(path, **kwargs)
        return self


# --------------------------------------------------------------------- helpers
def _cell_pos(Z: int, group: int, period: int):
    """(column, row) of an element in the drawn table (row 1 at the top)."""
    if 57 <= Z <= 71:
        return _FCOL + (Z - 57), _LANTH_ROW      # La-Lu
    if 89 <= Z <= 103:
        return _FCOL + (Z - 89), _ACT_ROW        # Ac-Lr
    return float(group), float(period)


def _to_Z(key: Union[int, str]) -> int:
    """Accept an atomic number or an element symbol (any case)."""
    if isinstance(key, str):
        s = key.strip()
        z = SYMBOL_TO_Z.get(s[:1].upper() + s[1:].lower())
        if z is None:
            raise KeyError(f"unknown element symbol: {key!r}")
        return z
    z = int(key)
    if z not in ELEMENTS:
        raise KeyError(f"atomic number out of range (1-118): {key!r}")
    return z


def _value_dict(data, values) -> dict:
    """Normalise the input to ``{Z: value}``.

    ``data`` may be a mapping ``{symbol|Z: value}`` (``values`` omitted), or a
    sequence of element keys paired with a ``values`` sequence.
    """
    if values is not None:
        return {_to_Z(k): float(v) for k, v in zip(data, values)}
    if isinstance(data, Mapping):
        return {_to_Z(k): float(v) for k, v in data.items()}
    if hasattr(data, "items"):                   # pandas Series etc.
        return {_to_Z(k): float(v) for k, v in data.items()}
    raise TypeError(
        "pass either a mapping {element: value} or two sequences "
        "periodic_table(elements, values)"
    )


def _resolve_norm(norm, vals):
    if isinstance(norm, Normalize):
        return norm
    if norm is None:
        return Normalize(vmin=min(vals), vmax=max(vals))
    if isinstance(norm, str) and norm == "diverging":
        m = max(abs(min(vals)), abs(max(vals))) or 1.0
        return TwoSlopeNorm(0.0, -m, m)
    if isinstance(norm, (tuple, list)) and len(norm) == 2:
        return Normalize(vmin=norm[0], vmax=norm[1])
    raise ValueError(
        "norm must be None, 'diverging', (vmin, vmax) or a matplotlib Normalize"
    )


def _shrink_to_fit(fig, ax, texts, max_frac: float = 0.94):
    """Reduce the font size of any text wider than ``max_frac`` of one cell so
    long element names (e.g. Praseodymium) stay inside their box.  Only shrinks,
    never grows; silently no-ops if a renderer is unavailable."""
    if not texts:
        return
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        x0 = ax.transData.transform((0.0, 0.0))[0]
        x1 = ax.transData.transform((1.0, 0.0))[0]
        cell_w = abs(x1 - x0) * max_frac
        for t in texts:
            w = t.get_window_extent(renderer).width
            if w > cell_w:
                t.set_fontsize(t.get_fontsize() * cell_w / w)
    except Exception:
        pass


def _auto_text_color(facecolor, has_value: bool) -> str:
    if not has_value:
        return "0.45"                            # greyed symbol on the empty cell
    r, g, b = to_rgb(facecolor)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < 0.5 else "black"


# ------------------------------------------------------------------- main entry
def periodic_table(
    data: Union[Mapping, Sequence],
    values: Optional[Sequence] = None,
    *,
    cmap: Union[str, Colormap] = "viridis",
    norm=None,
    value_fmt: str = "{:.2f}",
    mass_fmt: str = "{:.0f}",
    label: Optional[str] = None,
    ax: Optional[Axes] = None,
    # optional per-cell text (all off by default -> matches the reference look)
    show_symbol: bool = True,
    show_value: bool = True,
    show_number: bool = False,
    show_mass: bool = False,
    show_name: bool = False,
    # cell appearance
    missing_color="0.92",
    draw_missing: bool = True,
    edge_color="black",
    edge_width: float = 0.9,
    text_color: str = "auto",
    fblock_labels: bool = True,
    show_group_period: bool = False,
    # colourbar
    colorbar: bool = True,
    cbar_kw: Optional[dict] = None,
    # font sizes (points)
    symbol_fontsize: float = 8.6,
    value_fontsize: float = 6.0,
    number_fontsize: float = 5.0,
    mass_fontsize: float = 4.4,
    name_fontsize: float = 4.2,
    figsize=(9.0, 5.0),
    savepath: Optional[str] = None,
    **savefig_kw,
) -> PeriodicTablePlot:
    """Draw a periodic-table heatmap.

    Parameters
    ----------
    data, values :
        Either a mapping ``{element: value}`` (element = symbol or atomic
        number) with ``values`` omitted, or two parallel sequences
        ``periodic_table(elements, values)``.
    cmap :
        Matplotlib colormap name or instance.
    norm :
        ``None`` (auto min-max), ``'diverging'`` (symmetric ``TwoSlopeNorm``
        about 0), a ``(vmin, vmax)`` tuple, or a matplotlib ``Normalize``.
    value_fmt :
        Format string for the value printed in each cell.
    label :
        Colourbar label.
    ax :
        Draw into an existing axes (for composing multi-panel figures).  If
        ``None`` a new figure/axes is created.
    show_number, show_mass, show_name :
        Optionally print the atomic number, atomic mass and full element name
        in each cell (all off by default).
    show_group_period :
        Label the groups (1-18) across the top and periods (1-7) down the left.
    savepath :
        If given, save the figure there (format inferred from the extension);
        extra ``**savefig_kw`` are forwarded to ``savefig``.

    Returns
    -------
    PeriodicTablePlot
        Dataclass with ``.fig``, ``.ax`` and ``.mappable`` (add your own
        colourbar with ``fig.colorbar(result.mappable, ...)`` when composing).
    """
    vd = _value_dict(data, values)
    if not vd:
        raise ValueError("no values provided")
    n = _resolve_norm(norm, list(vd.values()))
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    mappable = ScalarMappable(norm=n, cmap=cmap_obj)
    mappable.set_array([])

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    name_texts = []                              # collected for auto-shrink-to-fit
    for Z, (sym, name, mass, grp, per) in ELEMENTS.items():
        has = Z in vd
        if not has and not draw_missing:
            continue
        c, r = _cell_pos(Z, grp, per)
        fc = mappable.to_rgba(vd[Z]) if has else missing_color
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=fc,
                               edgecolor=edge_color, linewidth=edge_width))
        tc = _auto_text_color(fc, has) if text_color == "auto" else text_color

        # Vertical layout inside the cell.  Atomic number / mass sit in the top
        # corners (narrow, so they don't reach the centre); the symbol keeps its
        # natural height, with the value below it and the name pinned to the
        # bottom when shown.
        want_value = show_value and has
        if show_name:
            sym_dy, value_dy = (-0.14, 0.16) if want_value else (-0.06, None)
        elif want_value:
            sym_dy, value_dy = -0.18, 0.27
        else:
            sym_dy, value_dy = 0.0, None

        if show_symbol:
            ax.text(c, r + sym_dy, sym, ha="center", va="center",
                    fontsize=symbol_fontsize, fontweight="bold", color=tc)
        if want_value:
            ax.text(c, r + value_dy, value_fmt.format(vd[Z]), ha="center",
                    va="center", fontsize=value_fontsize, color=tc)
        if show_number:
            ax.text(c - 0.44, r - 0.44, str(Z), ha="left", va="top",
                    fontsize=number_fontsize, color=tc)
        if show_mass:
            ax.text(c + 0.44, r - 0.44, mass_fmt.format(mass), ha="right", va="top",
                    fontsize=mass_fontsize, color=tc)
        if show_name:
            name_texts.append(ax.text(c, r + 0.45, name, ha="center", va="bottom",
                                      fontsize=name_fontsize, color=tc))

    if fblock_labels:
        for row, lab in ((_LANTH_ROW, "La-Lu"), (_ACT_ROW, "Ac-Lr")):
            ax.text(_FCOL - 1.15, row, lab, ha="center", va="center",
                    fontsize=6.5, color="0.4", style="italic")

    # optional group (1-18, across the top) and period (1-7, down the left) labels
    if show_group_period:
        for g in range(1, 19):
            ax.text(g, 0.02, str(g), ha="center", va="center",
                    fontsize=7.0, color="0.45")
        for p in range(1, 8):
            ax.text(0.05, p, str(p), ha="center", va="center",
                    fontsize=7.0, color="0.45")

    xlo = -0.75 if show_group_period else -0.3
    ytop = -0.35 if show_group_period else 0.3
    ax.set_xlim(xlo, 19)
    ax.set_ylim(10.4, ytop)                       # row 1 at the top
    ax.set_aspect("auto")
    ax.axis("off")

    if colorbar:
        kw = dict(fraction=0.020, pad=0.008, shrink=0.92)
        kw.update(cbar_kw or {})
        cb = fig.colorbar(mappable, ax=ax, **kw)
        if label:
            cb.set_label(label)

    _shrink_to_fit(fig, ax, name_texts, max_frac=0.94)

    result = PeriodicTablePlot(fig=fig, ax=ax, mappable=mappable)
    if savepath:
        result.save(savepath, **savefig_kw)
    return result
