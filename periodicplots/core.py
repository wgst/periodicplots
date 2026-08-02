"""Vector periodic-table heatmaps in matplotlib.

The single entry point is :func:`periodic_table`.  It draws each element as a
matplotlib patch (a rounded rectangle by default) + text, so the table is pure
vector and composes into multi-panel figures via the ``ax=`` argument.  The
optional ``tile_style="3d"`` finish is the one exception: its drop shadows and
gloss are small rasters, which PDF/SVG embed alongside the vector artwork.

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
from matplotlib.patches import FancyBboxPatch, Rectangle

from ._elements import ELEMENTS, SYMBOL_TO_Z

# detached f-block layout (matches the reference figure): lanthanoids and
# actinoids sit in their own rows below the main table.
_LANTH_ROW, _ACT_ROW, _FCOL = 8.5, 9.5, 3


@dataclass
class PeriodicTablePlot:
    """Result of every renderer (``periodic_table``, ``periodic_table_3d``).

    ``mappable`` drives the colourbar -- use it to place your own when
    composing.  It is ``None`` in exactly one case: the no-data
    family-coloured poster (``periodic_table_3d()`` without values), which
    has no value scale."""
    fig: Figure
    ax: Axes
    mappable: Optional[ScalarMappable]

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
        n_data = len(data) if hasattr(data, "__len__") else None
        n_vals = len(values) if hasattr(values, "__len__") else None
        if n_data is not None and n_vals is not None and n_data != n_vals:
            raise ValueError(
                f"data and values differ in length ({n_data} != {n_vals})")
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
        "cmap_norm must be None, 'diverging', (vmin, vmax) or a matplotlib "
        "Normalize"
    )


def _norm_frac(norm, value):
    """``norm(value)`` clipped to [0, 1], or ``None`` if it is not a number."""
    try:
        t = norm(float(value))
    except Exception:
        return None
    if bool(getattr(t, "mask", False)):          # masked, e.g. 0 on a log scale
        return None
    t = float(t)
    if t != t:                                   # NaN
        return None
    return min(max(t, 0.0), 1.0)


def _check_savefig_kw(savepath, savefig_kw):
    """Fail loudly on unrecognised keywords.

    Every entry point's ``**savefig_kw`` is forwarded to ``savefig``, which
    only runs when ``savepath`` is given -- so extra keywords without one can
    only be typos, or options that belong to a different renderer."""
    if savefig_kw and savepath is None:
        raise TypeError(
            "unexpected keyword argument(s): " + ", ".join(sorted(savefig_kw))
            + " -- extra keywords are forwarded to savefig (with savepath=); "
            "check for a typo, or for an option of the other tile_style")


def _shrink_to_fit(fig, ax, texts, max_frac: float = 0.85):
    """Scale ALL the given texts by one shared factor so the widest of them fits
    within ``max_frac`` of a cell.  A single factor keeps every element name the
    same size (mixed sizes look odd); only shrinks, never grows.  No-ops if a
    renderer is unavailable."""
    if not texts:
        return
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        x0 = ax.transData.transform((0.0, 0.0))[0]
        x1 = ax.transData.transform((1.0, 0.0))[0]
        cell_w = abs(x1 - x0) * max_frac
        widest = max(t.get_window_extent(renderer).width for t in texts)
        if widest > cell_w:
            scale = cell_w / widest
            for t in texts:
                t.set_fontsize(t.get_fontsize() * scale)
    except (AttributeError, RuntimeError):       # no renderer to measure with
        pass


# The empty block between Be and B, in (column, row) units -- where the "gap"
# colourbar lies.  Every renderer leaves this region free, so it is the one
# placement they can all share.
_GAP_BAND = (2.75, 11.85, 1.78)                  # x0, x1, centre row
# Bar thickness, in cell WIDTHS -- nothing foreshortens those, so the bar comes
# out the same weight whatever projection the renderer draws in.
_GAP_THICKNESS = 0.34
_CBAR_LOCS = ("gap", "right", "left", "top", "bottom")


# Corner radius of the colourbar frame, as a fraction of the bar's short side.
# Chosen so the bar's corners read like the tiles' own.
_CBAR_ROUNDING = {"round": 0.22, "square": 0.06}


# Colourbar text, as a fraction of one cell -- so the bar reads the same on a
# small panel and a full-page figure, whatever the renderer's default figsize.
_CBAR_TICK_FRAC, _CBAR_TITLE_FRAC = 0.34, 0.39


def _frame_colorbar(fig, ax, cb, cax, shape, font_scale=1.0):
    """Style the bar like the table it belongs to: an element-tile frame (same
    corner geometry, same thin tinted edge, gradient clipped to it, matplotlib's
    rectangular outline hidden), and text sized from the cell rather than in
    fixed points.  Needs the settled box, so it no-ops without a renderer."""
    try:
        fig.canvas.draw()
        pt = 72.0 / fig.dpi
        cell = abs(ax.transData.transform((1, 0))[0]
                   - ax.transData.transform((0, 0))[0]) * pt
        cb.ax.tick_params(labelsize=_CBAR_TICK_FRAC * cell * font_scale,
                          length=0.10 * cell * font_scale, color="0.25")
        for axis in (cb.ax.xaxis, cb.ax.yaxis):
            axis.label.set_size(_CBAR_TITLE_FRAC * cell * font_scale)
        bb = cax.get_window_extent()
        w, h = float(bb.width), float(bb.height)
        if not (w > 0 and h > 0):
            return
        r = _CBAR_ROUNDING[shape] * min(w, h)
        frame = FancyBboxPatch(
            (0, 0), 1, 1, transform=cax.transAxes,
            boxstyle=f"round,pad=0,rounding_size={r / w}",
            mutation_aspect=w / h,                # isotropic on a long thin bar
            facecolor="none", edgecolor="0.25", linewidth=0.5,
            clip_on=False, zorder=5)
        cax.add_patch(frame)
        if cb.solids is not None:
            cb.solids.set_clip_path(frame)
        cb.outline.set_visible(False)
    except (AttributeError, RuntimeError):       # no renderer to measure with
        pass


def _add_colorbar(fig, ax, mappable, *, loc="gap", label=None, cbar_kw=None,
                  project=None, shape="round", font_scale=1.0):
    """Attach the colourbar in one of :data:`_CBAR_LOCS` and return it.

    ``"gap"`` lays a horizontal bar inside the empty Be-B block, as an inset
    axes positioned in DATA coordinates -- so it follows whatever projection
    the caller draws in.  ``project`` maps a ``(column, row)`` of the plain
    table onto the data coordinates actually used (identity when omitted);
    Every other location is matplotlib's own ``fig.colorbar(location=...)``.
    Either way the bar is framed like an element tile, ``shape`` matching the
    table's ``tile_shape``, and its text is sized from the cell so every
    renderer's bar reads alike; ``font_scale`` scales that text.
    """
    if shape not in _CBAR_ROUNDING:
        raise ValueError(
            f"cbar_shape must be 'round' or 'square', got {shape!r}")
    if loc not in _CBAR_LOCS:
        raise ValueError(
            f"cbar_loc must be one of {_CBAR_LOCS}, got {loc!r}")
    kw = dict(cbar_kw or {})
    if loc == "gap":
        x0, x1, rc = _GAP_BAND
        P = project or (lambda x, y: (x, y))

        def tofrac(x, y):                        # data -> axes fraction
            return ax.transAxes.inverted().transform(ax.transData.transform(P(x, y)))

        # The band's ENDS follow the projection, so the bar sits in the gap and
        # leans with the table.  Its THICKNESS does not: a projection squashes
        # rows (by 1 - tilt, or by depth), which would leave the bar visibly
        # thinner in the tipped views than on the flat table.
        e0, e1 = tofrac(x0, rc), tofrac(x1, rc)  # the two ends, at mid-band
        half = _GAP_THICKNESS / 2.0
        try:
            fig.canvas.draw()
            cw = abs(ax.transData.transform((1, 0))[0]
                     - ax.transData.transform((0, 0))[0])
            h = _GAP_THICKNESS * cw / ax.get_window_extent().height
        except (AttributeError, RuntimeError):   # no renderer: fall back to rows
            h = abs(tofrac(x0, rc + half)[1] - tofrac(x0, rc - half)[1])
        cax = ax.inset_axes([min(e0[0], e1[0]), (e0[1] + e1[1]) / 2.0 - h / 2.0,
                             abs(e1[0] - e0[0]), h])
        kw.setdefault("orientation", "horizontal")
        cb = fig.colorbar(mappable, cax=cax, **kw)
        cax.xaxis.set_ticks_position("top")      # ticks, labels and title above
        cax.xaxis.set_label_position("top")
        if label:
            cb.set_label(label)
    else:
        cax = None
        kw.setdefault("fraction", 0.020)
        kw.setdefault("pad", 0.008)
        kw.setdefault("shrink", 0.92)
        kw["location"] = loc                     # the dedicated option wins:
        # cbar_kw is for the rest, and a "location" smuggled through it would
        # silently beat the validated cbar_loc argument
        cb = fig.colorbar(mappable, ax=ax, **kw)
        if label:
            cb.set_label(label)
    _frame_colorbar(fig, ax, cb, cb.ax, shape, font_scale)
    return cb


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
    cmap_norm=None,
    value_fmt: str = "{:.2f}",
    mass_fmt: str = "{:.0f}",
    label_cbar: Optional[str] = None,
    ax: Optional[Axes] = None,
    # optional per-cell text (all off by default -> matches the reference look)
    show_symbol: bool = True,
    show_value: bool = True,
    show_at_number: bool = False,
    show_at_mass: bool = False,
    show_name: bool = False,
    # cell appearance
    tile_style: str = "flat",
    tile_shape: str = "round",
    missing_color="0.92",
    draw_missing: bool = True,
    max_z: int = 118,
    edge_color="black",
    edge_width: float = 0.9,
    text_color: str = "auto",
    fblock_labels: bool = True,
    show_group_period: bool = False,
    # colourbar
    colorbar: bool = True,
    cbar_loc: str = "gap",
    cbar_shape: Optional[str] = None,
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
    cmap_norm :
        How data values map onto the colormap (and the colourbar's range):
        ``None`` (auto min-max), ``'diverging'`` (symmetric ``TwoSlopeNorm``
        about 0), a ``(vmin, vmax)`` tuple, or a matplotlib ``Normalize``.
        The automatic range spans every value supplied -- including elements
        that ``max_z`` drops from the drawing -- so panels built from one
        dataset share a scale.
    value_fmt :
        Format string for the value printed in each cell.
    label_cbar :
        Colourbar label.
    cbar_loc :
        Where the colourbar goes: ``"gap"`` (default) lays it horizontally in
        the empty block between Be and B; ``"right"``, ``"left"``, ``"top"``
        and ``"bottom"`` place it outside the table.  The bar is framed like an
        element cell; ``cbar_shape`` -- ``"round"`` or ``"square"`` --
        overrides the corner geometry, which otherwise follows ``tile_shape``.
        ``cbar_kw`` is passed on to ``fig.colorbar`` for the rest
        (``fraction``, ``pad``, ``shrink``).
    ax :
        Draw into an existing axes (for composing multi-panel figures).  If
        ``None`` a new figure/axes is created.
    show_at_number, show_at_mass, show_name :
        Optionally print the atomic number, atomic mass and full element name
        in each cell (all off by default).
    tile_style :
        How each cell is rendered.  ``"flat"`` (default) is the plain vector
        cell.  ``"3d"`` draws each cell as a physical-looking tile (soft drop
        shadow, lit face) -- purely cosmetic: tiles never rise or sink with
        the value (that is :func:`periodic_table_3d`'s job).  In 3d mode the
        tile outline is derived from the face colour, so ``edge_color`` and
        ``edge_width`` are ignored; everything else (text, fonts, colourbar,
        group/period labels) behaves exactly as in the flat look.  The tile
        shadows and gloss are small rasters, which PDF/SVG embed alongside
        the vector outlines and text -- any output format works, just pick a
        ``dpi`` you're happy with.
    tile_shape :
        Corner geometry, independent of ``tile_style``; ``"round"`` is the
        default for both styles.  ``"round"`` flat cells are rounded vector
        rectangles; ``"square"`` flat cells are plain sharp ``Rectangle``s;
        ``"square"`` 3d tiles take the photographic finish (near-square
        corners, matte grainy faces, chamfered edges).
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
    # public parameter names -> the internal shorthands used below
    norm, label = cmap_norm, label_cbar
    show_number, show_mass = show_at_number, show_at_mass

    _check_savefig_kw(savepath, savefig_kw)
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

    if tile_style not in ("flat", "3d"):
        raise ValueError(
            f"tile_style must be 'flat' or '3d', got {tile_style!r}")
    if tile_shape not in ("round", "square"):
        raise ValueError(
            f"tile_shape must be 'round' or 'square', got {tile_shape!r}")
    tiles3d = tile_style == "3d"
    shape = tile_shape
    if tiles3d:
        # Imported here, not at module level: style3d itself imports from core,
        # so a top-level import back would be circular.  Only the tile-drawing
        # helper and its shared rasters are borrowed -- every other aspect of
        # this function (text, fonts, colourbar, limits logic) stays its own.
        from .style3d import (_draw_tile, _gloss_image, _lip_image,
                              _matte_image, _noise_image, _shadow_image)
        square = shape == "square"
        rounding = 0.022 if square else 0.07
        shadow_img = (_shadow_image(radius=9, alpha=0.55) if square
                      else _shadow_image())
        face_img = _matte_image() if square else _gloss_image()
        lip_img = _lip_image() if square else None
        noise_img = _noise_image() if square else None

    name_texts = []                              # collected for auto-shrink-to-fit
    for Z, (sym, name, mass, grp, per) in ELEMENTS.items():
        if Z > max_z:                            # e.g. drop the superheavies
            continue
        has = Z in vd
        if not has and not draw_missing:
            continue
        c, r = _cell_pos(Z, grp, per)
        fc = mappable.to_rgba(vd[Z]) if has else missing_color
        if tiles3d:
            _draw_tile(ax, c, r, to_rgb(fc), shadow_img, face_img,
                       rounding=rounding, lip_img=lip_img, noise_img=noise_img,
                       bevel=square, nflip=(1 if Z % 2 else -1,
                                            1 if (Z >> 1) % 2 else -1))
        elif shape == "round":                   # rounded flat cell, still vector
            ax.add_patch(FancyBboxPatch(
                (c - 0.5, r - 0.5), 1, 1,
                boxstyle="round,pad=0,rounding_size=0.08", mutation_scale=1.0,
                facecolor=fc, edgecolor=edge_color, linewidth=edge_width))
        else:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=fc,
                                   edgecolor=edge_color, linewidth=edge_width))
        tc = _auto_text_color(fc, has) if text_color == "auto" else text_color

        # Layout inside the cell.  Atomic number / mass sit in the top corners
        # (narrow, so they don't reach the centre); the symbol keeps its
        # natural height, with the value below it and the name pinned to the
        # bottom when shown.
        #
        # A flat cell fills its 1x1 cell, but a 3d tile's face is smaller and
        # stands on a lip, so a tile takes the layout that the *identical*
        # tile gets in periodic_table_3d (see style3d): the symbol centred on
        # the face with the value beneath it, and the corner text pulled in
        # off the edge.  Both functions draw the same tile, so their text has
        # to sit the same way on it.
        want_value = show_value and has
        if tiles3d:
            sym_dy = 0.0 if (show_name or want_value) else 0.02
            value_dy = 0.31
            name_dy = 0.41 if want_value else 0.28
            corner_dx, corner_dy = 0.41, -0.40
        else:
            if show_name:
                sym_dy, value_dy = (-0.14, 0.16) if want_value else (-0.06, None)
            elif want_value:
                sym_dy, value_dy = -0.18, 0.27
            else:
                sym_dy, value_dy = 0.0, None
            name_dy = 0.45
            corner_dx, corner_dy = 0.44, -0.44

        if show_symbol:
            ax.text(c, r + sym_dy, sym, ha="center", va="center", zorder=6,
                    fontsize=symbol_fontsize, fontweight="bold", color=tc)
        if want_value:
            ax.text(c, r + value_dy, value_fmt.format(vd[Z]), ha="center",
                    va="center", zorder=6, fontsize=value_fontsize, color=tc)
        if show_number:
            ax.text(c - corner_dx, r + corner_dy, str(Z), ha="left", va="top", zorder=6,
                    fontsize=number_fontsize, color=tc)
        if show_mass:
            ax.text(c + corner_dx, r + corner_dy, mass_fmt.format(mass), ha="right",
                    va="top", zorder=6, fontsize=mass_fontsize, color=tc)
        if show_name:
            name_texts.append(ax.text(c, r + name_dy, name, ha="center",
                                      va="bottom", zorder=6,
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
    ybot = 10.78 if tiles3d else 10.4             # tiles cast a shadow below Ac-Lr
    ax.set_xlim(xlo, 19)
    ax.set_ylim(ybot, ytop)                       # row 1 at the top
    ax.set_aspect("auto")
    ax.axis("off")

    if colorbar:
        _add_colorbar(fig, ax, mappable, loc=cbar_loc, label=label,
                      cbar_kw=cbar_kw, shape=cbar_shape or tile_shape)

    _shrink_to_fit(fig, ax, name_texts)

    result = PeriodicTablePlot(fig=fig, ax=ax, mappable=mappable)
    if savepath:
        result.save(savepath, **savefig_kw)
    return result
