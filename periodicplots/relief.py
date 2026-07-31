"""Relief ("2.5-D") periodic tables — the value also lifts the cell off the page.

This is the renderer behind ``periodic_table_3d(tile_style="flat")`` -- the way
in; :func:`periodic_table_relief` itself is not exported.  It draws the same
table as :func:`periodicplots.periodic_table`, but every element becomes a
little block whose height grows with its value, so the largest values stand
furthest out towards the viewer.

The drawing is a hand-rolled oblique projection (not ``mplot3d``): each block is
two or three ``Polygon`` patches — the top face plus the walls below it — shaded
and painted back-to-front so near rows occlude far ones.  Everything therefore
stays vector, the text stays horizontal and crisp, and the figure still composes
into other axes via ``ax=``.

World coordinates are ``(x=column, y=row, z=height)`` and the projection is

    screen_x = x + shear * y
    screen_y = y * depth - z          (the y axis is inverted, so -z is "up")

``depth`` is the row pitch: it both squashes the table plane and decides how much
a row overlaps the one behind it.  ``depth=1`` gives a flat, un-overlapped table
with the cells merely lifted; smaller values tip the view further overhead.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch, Polygon
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

from ._elements import ELEMENTS
from .core import (
    _ACT_ROW,
    _FCOL,
    _LANTH_ROW,
    PeriodicTablePlot,
    _auto_text_color,
    _cell_pos,
    _resolve_norm,
    _shrink_to_fit,
    _value_dict,
)


# A wall gradient is painted as this many flat strips.  Vector output has no
# portable gradient primitive, so strips it is: enough of them to read as smooth,
# few enough to keep the PDF small.
_GRADIENT_STRIPS = 20

# How far below the fold the wall gradient runs, in cell widths.  Beyond this the
# wall keeps its settled colour — deeper down it is hidden by the row in front.
_FADE_LENGTH = 0.42


# --------------------------------------------------------------------- helpers
def _shade(color, factor: float):
    """Darken (``factor`` < 1) a colour, for the walls of a block."""
    if factor == 1.0:
        return color
    r, g, b = to_rgb(color)
    return (r * factor, g * factor, b * factor)


def _mix(c1, c2, t: float):
    """Blend two colours; ``t=0`` gives ``c1``, ``t=1`` gives ``c2``."""
    a, b = to_rgb(c1), to_rgb(c2)
    return tuple(u + (v - u) * t for u, v in zip(a, b))


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


def _data_per_point(fig, ax):
    """Data units per typographic point, or ``None`` if it can't be measured.

    Needed to lay text out in the plane of the table: glyph outlines come out of
    ``TextPath`` in points and have to be scaled into data coordinates.
    """
    try:
        fig.canvas.draw()
        x0 = ax.transData.transform((0.0, 0.0))[0]
        x1 = ax.transData.transform((1.0, 0.0))[0]
        px_per_data = abs(x1 - x0)
        return (fig.dpi / 72.0) / px_per_data if px_per_data else None
    except Exception:
        return None


def _surface_label(ax, text, X, Y, size, weight, color, ha, va, zorder,
                   scale, depth, shear):
    """Draw one label lying *in* the plane of the table rather than upright.

    The glyphs are converted to outlines and put through the same projection as
    the blocks, so the text reads as painted on the cell instead of standing on
    it.  Still vector — just paths instead of a text object.
    """
    tp = TextPath((0, 0), text, prop=FontProperties(weight=weight, size=size))
    bb = tp.get_extents()
    dx = {"center": -(bb.x0 + bb.x1) / 2, "left": -bb.x0, "right": -bb.x1}[ha]
    dy = {"center": -(bb.y0 + bb.y1) / 2, "top": -bb.y1, "bottom": -bb.y0}[va]
    # glyph (gx, gy) -> surface offset (gx, -gy) * scale -> projected
    m = Affine2D().translate(dx, dy) + Affine2D.from_values(
        scale, 0.0, -scale * shear, -scale * depth, X, Y)
    ax.add_patch(PathPatch(m.transform_path(tp), facecolor=color,
                           edgecolor="none", linewidth=0, zorder=zorder))
    return bb.width * scale                       # drawn width, in data units


def _draw_block(ax, x, y, z, half, depth, shear, top_color, wall_shade,
                side_shade, wall_fade, fade_color, pit_color, edge_color,
                edge_width, zorder, z_back=None, z_front=None, z_side_out=None,
                c_back=None, side_back_open=False):
    """Add the viewer-facing faces of one extruded cell.

    ``z`` may be negative, in which case the cell is sunk into the table.  The
    view looks from the front and from one side, so a wall shows exactly where the
    neighbour it faces is lower than this cell: the neighbours' levels
    (``z_back``, ``z_front``, ``z_side_out``; ``None`` where the table's layout has
    no cell) decide which walls exist and how tall they are.  Cells at the same
    level therefore run together with no wall between them.

    The face carries the mapped colour flat (so it still matches the colourbar);
    each wall is brightest at its top and shades towards ``fade_color`` going
    down.  A wall exposing another element takes that element's colour; one
    exposing the bare slab (no cell in the layout) is neutral ``pit_color``.

    Returns the projected centre of the face, i.e. where the text goes.
    """
    def P(px, py, pz=0.0):
        return (px + shear * py, py * depth - pz)

    x0, x1 = x - half, x + half
    y0, y1 = y - half, y + half

    # Which walls exist follows from where the camera is: it looks from the front
    # (+y) and from one side in x, so a face shows exactly where the neighbour it
    # faces is *lower* than the material behind the face.  Each wall is
    # (shade factor, its two world corners, its z range, the colour of the
    # material it exposes) — the cell's own colour when the cell is the higher
    # side, and the neutral slab where the table's layout has no cell at all.
    walls = []

    # behind us: the near face of the cell there, wherever it stands above us.
    # Only up to the surface — anything above that is that cell's own wall.
    hi = min(0.0, z_back) if z_back is not None else 0.0
    if hi > z:
        walls.append((wall_shade, (x0, y0), (x1, y0), z, hi,
                      c_back if (z_back is not None and z_back < 0 and c_back)
                      else pit_color, False))


    # in front of us, and on the side the view comes from: our own faces, wherever
    # the neighbour there is lower.  Above the surface that face is our own tile;
    # below it, it is the slab we are cut into.
    sides = [(wall_shade, z_front, (x0, y1), (x1, y1), False)]
    if shear:                                                # side faces are edge-on at 0
        sides.append((side_shade, z_side_out, *(((x1, y0), (x1, y1)) if shear < 0
                                                else ((x0, y0), (x0, y1))), True))
    # A side wall runs up to the surface where the trench's edge carries on behind
    # it (so the boundary of a sunken region is one continuous wall), but stops at
    # the floor behind where it does not — left at full height there it would stand
    # up in front of the row behind, which is where the stray wedges came from.
    cap = 0.0 if side_back_open or z_back is None else min(0.0, z_back)

    for factor, nb, a, b, sideways in sides:
        top = cap if sideways else 0.0
        if nb is None:                                       # no cell there at all
            if z > 0:                                        # our own side, above the slab
                walls.append((factor, a, b, 0.0, z, top_color, sideways))
            elif z < top and sideways:                       # trench edge: slab, cut open
                walls.append((factor, a, b, z, top, pit_color, sideways))
            # at the open front of the table there is nothing to cut through:
            # the tiles simply end, so no face is drawn there
        elif nb < z:                                         # we stand above that cell
            if z > 0:
                walls.append((factor, a, b, max(nb, 0.0), z, top_color, sideways))
                if nb < top:                                 # slab cut below our foot
                    walls.append((factor, a, b, nb, top, pit_color, sideways))
            else:                                            # both sunk: our tile's side
                walls.append((factor, a, b, nb, z, top_color, sideways))
        # A neighbour on the camera side that is higher hides this face entirely:
        # the step between the two floors points away from the camera, so there
        # is nothing of ours to draw there.

    layer = 0
    for factor, a, b, zbot, ztop, wall_base, sideways in walls:
        if ztop - zbot < 1e-9:                               # nothing exposed
            continue
        c_top = _shade(wall_base, factor)                    # at the top of the wall
        c_deep = _mix(c_top, fade_color, wall_fade)          # further down
        quad = lambda lo, hi: (P(*a, lo), P(*b, lo), P(*b, hi), P(*a, hi))

        if wall_fade <= 0:
            ax.add_patch(Polygon(quad(zbot, ztop), closed=True, facecolor=c_top,
                                 edgecolor=edge_color, linewidth=edge_width,
                                 joinstyle="round", zorder=zorder + layer * 1e-3))
            layer += 1
            continue
        # The ramp runs a fixed distance down from the top of the wall rather than
        # over its whole height: past the first row or so a wall is hidden by the
        # block in front, and a gradient spread over the full height would fall
        # in the hidden part.
        ramp = min(ztop - zbot, _FADE_LENGTH)
        if ztop - ramp > zbot:                               # settled colour below
            ax.add_patch(Polygon(quad(zbot, ztop - ramp), closed=True,
                                 facecolor=c_deep, edgecolor=c_deep,
                                 linewidth=0.3, zorder=zorder + layer * 1e-3))
            layer += 1
        steps = max(2, round(_GRADIENT_STRIPS * ramp / _FADE_LENGTH))
        for k in range(steps):                               # top -> settled
            f0, f1 = k / steps, (k + 1) / steps
            col = _mix(c_top, c_deep, (k + 0.5) / steps)
            # the strips share an edge colour so no hairline seams show through
            ax.add_patch(Polygon(quad(ztop - ramp * f1, ztop - ramp * f0),
                                 closed=True, facecolor=col, edgecolor=col,
                                 linewidth=0.3, zorder=zorder + layer * 1e-3))
            layer += 1
        ax.add_patch(Polygon(quad(zbot, ztop), closed=True, fill=False,  # silhouette
                             edgecolor=edge_color, linewidth=edge_width,
                             joinstyle="round", zorder=zorder + layer * 1e-3))
        layer += 1

    ax.add_patch(Polygon((P(x0, y0, z), P(x1, y0, z), P(x1, y1, z), P(x0, y1, z)),
                         closed=True, facecolor=top_color, edgecolor=edge_color,
                         linewidth=edge_width, joinstyle="round",
                         zorder=zorder + layer * 1e-3))
    return P(x, y, z)


# ------------------------------------------------------------------- main entry
def periodic_table_relief(
    data: Union[Mapping, Sequence],
    values: Optional[Sequence] = None,
    *,
    cmap: Union[str, Colormap] = "viridis",
    cmap_norm=None,
    value_fmt: str = "{:.2f}",
    mass_fmt: str = "{:.0f}",
    label_cbar: Optional[str] = None,
    ax: Optional[Axes] = None,
    # the relief
    relief_height: float = 1.0,
    base_height: float = 0.10,
    relief_signed: bool = False,
    depth: float = 0.72,
    shear: float = -0.11,
    gap: float = 0.0,
    wall_shade: float = 1.0,
    side_shade: float = 0.9,
    wall_fade: float = 0.45,
    fade_color="black",
    pit_color="0.75",
    # optional per-cell text (same switches as the flat table)
    show_symbol: bool = True,
    show_value: bool = True,
    show_at_number: bool = False,
    show_at_mass: bool = False,
    show_name: bool = False,
    # cell appearance
    missing_color="0.92",
    draw_missing: bool = True,
    max_z: int = 118,
    edge_color="black",
    edge_width: float = 0.6,
    text_color: str = "auto",
    text_on_surface: bool = True,
    fblock_labels: bool = True,
    show_group_period: bool = False,
    # colourbar
    colorbar: bool = True,
    cbar_kw: Optional[dict] = None,
    # font sizes (points)
    symbol_fontsize: float = 8.4,
    value_fontsize: float = 5.5,
    number_fontsize: float = 4.6,
    mass_fontsize: float = 4.2,
    name_fontsize: float = 4.0,
    figsize=(10.0, 5.0),
    savepath: Optional[str] = None,
    **savefig_kw,
) -> PeriodicTablePlot:
    """Draw a periodic table whose cells are extruded towards the viewer.

    Colour *and* block height encode the value, so the trend reads as relief:
    the largest values stand furthest out of the page.  Everything stays vector.

    Parameters
    ----------
    data, values, cmap, norm, label, ax, show_*, colorbar, savepath, ... :
        As in :func:`periodicplots.periodic_table`.
    relief_height :
        Height of the tallest block, in cell widths (with ``relief_signed``,
        of the value furthest from zero).
    base_height :
        Height of the *smallest* value, so low cells still read as blocks
        rather than flat patches (use 0 for a true zero baseline).  Ignored
        when ``relief_signed`` is set — there the baseline is zero itself.
    relief_signed :
        Measure the relief from zero instead of from the smallest value:
        positive values stand out of the table and negative ones sink into it,
        the deeper the more negative.  This is what you want for a diverging
        property; ``base_height`` no longer applies, elements without a value
        sit flat in the baseline plane, and zero requires a norm that spans it.
    depth :
        Row pitch, in cell widths: how far apart successive rows sit on screen.
        Smaller = the view tips further overhead and rows overlap more (1 = a
        flat table whose cells are merely lifted).
    shear :
        Horizontal slant per row: how far each row shifts sideways as it comes
        towards the viewer.  This swings the camera to one side and exposes a
        second wall on every block — negative looks from the right, positive
        from the left.  ``0`` keeps the table's columns vertical (a head-on
        skyline).
    gap :
        Gap between neighbouring blocks, in cell widths.
    wall_shade, side_shade :
        Brightness of the front and side walls where they meet the top face
        (1 = no fold shading).
    text_on_surface :
        Lay the cell labels *in* the plane of the table (as vector outlines) so
        they read as printed on each face rather than standing upright on it.
        Set ``False`` to keep upright, selectable text objects instead.
    wall_fade, fade_color :
        How far, and towards what, the walls shade on the way down to the
        floor: 0 gives flat walls, 1 reaches ``fade_color`` at the ground.
        ``fade_color="white"`` fades them out instead of darkening them.  The
        gradient is painted as flat strips, so it stays vector.
    pit_color :
        Colour of the plain slab a sunken cell (``relief_signed``) cuts through where
        the table's layout has no element — above the d-block, say, or beyond the
        table's edge.  A cut into another element exposes that element's own
        colour instead, and a raised block always shows its own.

    Returns
    -------
    PeriodicTablePlot
        Same dataclass as the flat table: ``.fig``, ``.ax``, ``.mappable``.
    """
    # public parameter names -> the internal shorthands used below
    norm, label = cmap_norm, label_cbar
    show_number, show_mass = show_at_number, show_at_mass
    height, signed = relief_height, relief_signed

    vd = _value_dict(data, values)
    if not vd:
        raise ValueError("no values provided")
    if not 0 < depth <= 1:
        raise ValueError("depth must be in (0, 1]")
    n = _resolve_norm(norm, list(vd.values()))
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    mappable = ScalarMappable(norm=n, cmap=cmap_obj)
    mappable.set_array([])

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    half = min(max(0.5 - gap / 2.0, 0.05), 0.5)

    if signed:
        # Where zero sits on the colour scale; the relief is measured from there,
        # scaled so the value furthest from it reaches +/- `relief_height`.
        t_zero = _norm_frac(n, 0.0)
        if t_zero is None:
            raise ValueError("signed=True needs a norm that can place 0 "
                             "(e.g. norm='diverging' or a (vmin, vmax) spanning 0)")
        reach = max(t_zero, 1.0 - t_zero) or 1.0

    # Collect the cells first: they must be painted back-to-front (painter's
    # algorithm) — far rows before near ones and, once sheared, the far side of
    # each row before the near side.
    cells = []
    for Z, (sym, name, mass, grp, per) in ELEMENTS.items():
        if Z > max_z:
            continue
        has = Z in vd
        if not has and not draw_missing:
            continue
        c, r = _cell_pos(Z, grp, per)
        if has:
            t = _norm_frac(n, vd[Z]) or 0.0
            z = (height * (t - t_zero) / reach if signed
                 else base_height + (height - base_height) * t)
            fc = mappable.to_rgba(vd[Z])
        else:
            # no data: flat in the baseline plane (signed), else level with the
            # lowest block so a tall neighbour cannot swallow it whole
            z, fc = (0.0 if signed else base_height), missing_color
        cells.append((r - shear * c, Z, sym, name, mass, c, r, z, fc, has))
    cells.sort(key=lambda t: t[0])

    # floor level of each cell, so a pit's inner walls can stop at its neighbours'
    # (missing neighbours count as the baseline plane)
    floors = {(t[5], t[6]): t[7] for t in cells}
    colours = {(t[5], t[6]): t[8] for t in cells}     # what a wall into it exposes
    side_dc = 1 if shear > 0 else -1                  # away from the viewing side

    labels = []            # (text, X, Y, size, weight, colour, ha, va, zorder, is_name)
    tops = []                                    # projected top faces, for the ylim
    for i, (_, Z, sym, name, mass, c, r, z, fc, has) in enumerate(cells):
        zo = 2.0 + i
        cx, cy = _draw_block(ax, c, r, z, half, depth, shear, fc, wall_shade,
                             side_shade, wall_fade, fade_color, pit_color,
                             edge_color, edge_width, zo,
                             z_back=floors.get((c, r - 1)),
                             z_front=floors.get((c, r + 1)),
                             z_side_out=floors.get((c - side_dc, r)),
                             c_back=colours.get((c, r - 1)),
                             side_back_open=(c, r - 1) in floors
                             and (c - side_dc, r - 1) not in floors)
        tops.append(cy)
        tc = _auto_text_color(fc, has) if text_color == "auto" else text_color
        tz = zo + 0.5                            # above its own block, behind the next

        # Offsets are given in the plane of the table and then projected, so the
        # labels sit on the face wherever the view is pointing.
        def surf(a, b):
            return cx + a + shear * b, cy + depth * b

        # Same layout as the flat table, but biased upwards: a taller block in
        # the row in front eats a cell from the bottom, so the text has to live
        # in the upper part of the face.
        want_value = show_value and has
        if show_name:
            sym_dy, value_dy = (-0.22, 0.06) if want_value else (-0.14, None)
        elif want_value:
            sym_dy, value_dy = -0.26, 0.14
        else:
            sym_dy, value_dy = -0.08, None

        if show_symbol:
            labels.append((sym, *surf(0.0, sym_dy), symbol_fontsize, "bold", tc,
                           "center", "center", tz, False))
        if want_value:
            labels.append((value_fmt.format(vd[Z]), *surf(0.0, value_dy),
                           value_fontsize, "normal", tc, "center", "center", tz, False))
        if show_number:
            labels.append((str(Z), *surf(-0.92 * half, -0.88 * half),
                           number_fontsize, "normal", tc, "left", "top", tz, False))
        if show_mass:
            labels.append((mass_fmt.format(mass), *surf(0.92 * half, -0.88 * half),
                           mass_fontsize, "normal", tc, "right", "top", tz, False))
        if show_name:
            labels.append((name, *surf(0.0, 0.9 * half), name_fontsize, "normal",
                           tc, "center", "bottom", tz, True))

    if fblock_labels:
        for row, lab in ((_LANTH_ROW, "La-Lu"), (_ACT_ROW, "Ac-Lr")):
            ax.text(_FCOL - 1.15 + shear * row, row * depth, lab, ha="center",
                    va="center", fontsize=6.5, color="0.4", style="italic", zorder=1)

    ytop = min(tops) - half * depth               # top of the tallest block
    ybot = max(_ACT_ROW * depth + half * depth,   # floor of the nearest row, or
               max(tops) + half * depth)          # the bottom of the deepest pit
    xlo = 1 - half + min(0.0, shear * _ACT_ROW)
    xhi = 18 + half + max(0.0, shear * _ACT_ROW)

    if show_group_period:
        for g in range(1, 19):                    # groups: clear above the blocks
            ax.text(g + shear, ytop - 0.32, str(g), ha="center", va="center",
                    fontsize=7.0, color="0.45", zorder=1)
        for p in range(1, 8):                     # periods: at each row's floor
            ax.text(xlo - 0.45 + shear * p, p * depth, str(p), ha="center",
                    va="center", fontsize=7.0, color="0.45", zorder=1)
        ytop -= 0.60
        xlo -= 0.75

    ax.set_xlim(xlo - 0.35, xhi + 0.35)
    ax.set_ylim(ybot + 0.30, ytop - 0.30)         # inverted: row 1 at the top
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    if colorbar:
        kw = dict(fraction=0.020, pad=0.008, shrink=0.92)
        kw.update(cbar_kw or {})
        cb = fig.colorbar(mappable, ax=ax, **kw)
        if label:
            cb.set_label(label)

    # The cell labels go on last: laying them in the plane of the table needs the
    # settled data->display scale, so the axes has to be finished first.
    scale = _data_per_point(fig, ax) if text_on_surface else None
    if scale is None:                            # upright text (or no renderer)
        texts = [ax.text(X, Y, s, ha=ha, va=va, fontsize=size, color=col,
                         fontweight=w, zorder=zo)
                 for s, X, Y, size, w, col, ha, va, zo, _ in labels]
        _shrink_to_fit(fig, ax, [t for t, spec in zip(texts, labels) if spec[-1]],
                       max_frac=0.85 * 2 * half)
    else:
        # one shared size for every element name, chosen so the widest one fits
        names = [spec for spec in labels if spec[-1]]
        name_scale = 1.0
        if names:
            widest = max(TextPath((0, 0), s, prop=FontProperties(size=size)).get_extents().width
                         for s, _, _, size, *_ in names) * scale
            name_scale = min(1.0, 0.85 * 2 * half / widest) if widest else 1.0
        for s, X, Y, size, w, col, ha, va, zo, is_name in labels:
            _surface_label(ax, s, X, Y, size * (name_scale if is_name else 1.0),
                           w, col, ha, va, zo, scale, depth, shear)

    result = PeriodicTablePlot(fig=fig, ax=ax, mappable=mappable)
    if savepath:
        result.save(savepath, **savefig_kw)
    return result
