"""A decorative "3D realistic" periodic-table style.

:func:`periodic_table_3d` draws each element as a physical-looking tile:
rounded corners, an extruded darker side face, a glossy top-light gradient
and a soft blurred drop shadow, on a pastel gradient background.

Two colouring modes:

* no data -> tiles are coloured by chemical family using a pastel
  coral / rose / sage / gold / blue palette (the "poster" look);
* ``data``/``values`` given -> the usual heatmap colouring, rendered with the
  same 3D tile treatment.  The default ``cmap`` is ``"poster"``, a sequential
  pastel ramp (pale blue -> greens -> brief khaki -> corals -> red) built in
  the family palette's language so heatmaps match the poster look; any
  matplotlib cmap works.

Unlike :func:`periodicplots.periodic_table` this style uses small rasters for
its shadows and sheen; PDF/SVG embed them alongside the vector outlines and
text, so any output format works.
"""
from __future__ import annotations

import colorsys
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (Colormap, LinearSegmentedColormap, Normalize,
                               to_rgb)
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

from ._elements import ELEMENTS
from .core import (PeriodicTablePlot, _ACT_ROW, _FCOL, _LANTH_ROW,
                   _add_colorbar, _cell_pos, _resolve_norm, _to_Z, _value_dict)

# ------------------------------------------------------------------ palette
# Colour families sampled from the reference poster: corals/pinks on the
# outer columns, dusty rose and sage green through the d-block, golden
# amber for the reactive nonmetals and pale blues for the f-block.
_FAMILY_COLORS = {
    "alkali":       "#ef6a67",
    "alkaline":     "#f5a09b",
    "early_tm":     "#e9b3ab",   # groups 3-7: dusty rose
    "late_tm":      "#a9cab4",   # groups 8-12: sage green
    "post_tm":      "#c3d9c8",
    "metalloid":    "#ead9b6",
    "nonmetal":     "#f0b64f",
    "halogen":      "#eda03f",
    "noble":        "#f2837b",
    "lanthanoid":   "#bcd6dd",
    "actinoid":     "#a7c6d2",
    "superheavy":   "#d9c6cd",
}


# Sequential colormap in the same pastel language as the family palette:
# cool pale blue -> greens -> (brief khaki) -> corals -> red.  Greens and
# reds carry most of the range; yellow is only a narrow transition band.
# Registered as "poster" (and "poster_r"), usable with any matplotlib plot.
_POSTER_RAMP = ["#aecfd8", "#a3c8b4", "#8fbf9d", "#a9c795", "#dccf9d",
                "#e9a04f", "#ee7e68", "#e6564f", "#d94a45"]
POSTER_CMAP = LinearSegmentedColormap.from_list("poster", _POSTER_RAMP)
try:
    mpl.colormaps.register(POSTER_CMAP, name="poster")
    mpl.colormaps.register(POSTER_CMAP.reversed(), name="poster_r")
except Exception:                                  # already registered
    pass


def _family(Z: int, group: int, period: int) -> str:
    if 57 <= Z <= 71:
        return "lanthanoid"
    if 89 <= Z <= 103:
        return "actinoid"
    if Z >= 104:
        return "superheavy"
    if Z == 1:
        return "alkali"                            # poster look: coral corner tile
    if Z in (5, 14, 32, 33, 51, 52):
        return "metalloid"
    if Z in (6, 7, 8, 15, 16, 34):
        return "nonmetal"
    if group == 18:
        return "noble"
    if group == 17:
        return "halogen"
    if group == 1:
        return "alkali"
    if group == 2:
        return "alkaline"
    if 3 <= group <= 7:
        return "early_tm"
    if 8 <= group <= 12:
        return "late_tm"
    return "post_tm"


def _shade(color, dl: float, ds: float = 0.0):
    """Move a colour's lightness (and optionally saturation) in HLS space."""
    h, l, s = colorsys.rgb_to_hls(*to_rgb(color))
    l = min(1.0, max(0.0, l + dl))
    s = min(1.0, max(0.0, s + ds))
    return colorsys.hls_to_rgb(h, l, s)


def _ink(color):
    """Dark, slightly hue-tinted text colour for a tile face."""
    h, l, s = colorsys.rgb_to_hls(*to_rgb(color))
    return colorsys.hls_to_rgb(h, 0.16, min(0.55, s * 0.8))


# ----------------------------------------------------------------- rasters
def _rounded_mask(n: int, corner: float) -> np.ndarray:
    """Anti-aliased rounded-rectangle mask on an n x n grid in [0, 1]^2."""
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    qx = np.maximum(np.abs(x - 0.5) - (0.5 - corner), 0.0)
    qy = np.maximum(np.abs(y - 0.5) - (0.5 - corner), 0.0)
    d = np.hypot(qx, qy) - corner
    aa = 1.5 / n
    return np.clip(0.5 - d / aa, 0.0, 1.0)


def _blur(a: np.ndarray, radius: int, passes: int = 3) -> np.ndarray:
    """Separable box blur repeated a few times (~ Gaussian), numpy-only."""
    k = np.ones(2 * radius + 1) / (2 * radius + 1)
    for _ in range(passes):
        a = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, a)
        a = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, a)
    return a


def _shadow_image(n: int = 48, corner: float = 0.16, radius: int = 8,
                  alpha: float = 0.50) -> np.ndarray:
    """Shared RGBA image of a blurred rounded-rect shadow: a wide soft halo
    plus a tighter, darker core (contact shadow).  Kept deliberately small:
    the raster is upsampled bilinearly at draw time, and PDF output embeds
    one copy of it per tile."""
    pad = 3 * radius
    m = np.zeros((n + 2 * pad, n + 2 * pad))
    m[pad:pad + n, pad:pad + n] = _rounded_mask(n, corner)
    soft = _blur(m, radius)
    core = _blur(m, max(2, radius // 3))
    rgba = np.zeros(m.shape + (4,))
    rgba[..., 3] = np.clip(soft * alpha + core * 0.18, 0, 1)   # black, soft alpha
    return rgba


def _vgrad_image(sheen_a: float, sheen_span: float, shade_a: float,
                 shade_start: float, n: int = 64) -> np.ndarray:
    """Vertical lighting overlay: white sheen fading down from the top edge,
    black shade rising towards the bottom.  Row 0 is the TOP of the face."""
    t = np.linspace(0.0, 1.0, n)[:, None]         # 0 at top -> 1 at bottom
    sheen = sheen_a * np.clip(1.0 - t / sheen_span, 0.0, 1.0) ** 1.3
    shade = shade_a * np.clip((t - shade_start) / (1 - shade_start),
                              0.0, 1.0) ** 1.3
    rgba = np.zeros((n, 1, 4))
    rgba[..., 0:3] = 1.0
    rgba[..., 3] = sheen
    dark = np.zeros((n, 1, 4))
    dark[..., 3] = shade
    # composite the two single-column layers into one image
    out = np.zeros((n, 1, 4))
    out[..., 3] = np.clip(sheen + shade, 0, 1)
    a = out[..., 3]
    with np.errstate(invalid="ignore"):
        out[..., 0:3] = np.where(a[..., None] > 0,
                                 (rgba[..., 0:3] * rgba[..., 3:4]
                                  + dark[..., 0:3] * dark[..., 3:4]) / a[..., None],
                                 0.0)
    return out


def _gloss_image() -> np.ndarray:
    """Subtle sheen for the straight-on view."""
    return _vgrad_image(sheen_a=0.14, sheen_span=0.65,
                        shade_a=0.12, shade_start=0.55)


def _matte_image() -> np.ndarray:
    """Flatter lighting for the photographic ``square`` style."""
    return _vgrad_image(sheen_a=0.09, sheen_span=0.55,
                        shade_a=0.15, shade_start=0.45)


def _lip_image() -> np.ndarray:
    """Lighting for the visible side lip: a hint of light at the crease
    under the face, falling to dark at the bottom edge."""
    return _vgrad_image(sheen_a=0.10, sheen_span=0.30,
                        shade_a=0.32, shade_start=0.12)


def _noise_image(n: int = 128, alpha: float = 0.055, seed: int = 7) -> np.ndarray:
    """Shared film-grain raster: bright/dark speckle with tiny alpha, laid
    over faces so the material reads as physical when zoomed in."""
    rng = np.random.default_rng(seed)
    g = _blur(rng.standard_normal((n, n)), 1, passes=1)
    g = (g - g.min()) / (np.ptp(g) or 1.0)
    rgba = np.zeros((n, n, 4))
    rgba[..., 0:3] = (g >= 0.5)[..., None] * 1.0   # speckle is white or black
    rgba[..., 3] = np.abs(g - 0.5) * 2.0 * alpha
    return rgba


def _wall_image() -> np.ndarray:
    """Stronger wall lighting for the tilted view: clearly lighter at the
    top (near the lit top surface) and darker towards the base."""
    return _vgrad_image(sheen_a=0.18, sheen_span=0.50,
                        shade_a=0.26, shade_start=0.22)


def _background_image(n: int = 256) -> np.ndarray:
    """Soft pastel backdrop: pale blue lower-left -> white -> pink upper-right."""
    y, x = np.mgrid[0:n, 0:n] / (n - 1)           # y: 0 top -> 1 bottom
    blue = np.array(to_rgb("#dcebf4"))
    white = np.array(to_rgb("#fbfbfc"))
    pink = np.array(to_rgb("#f6e6ea"))
    t = np.clip(0.5 * (x + (1 - y)), 0, 1)        # diagonal coordinate
    img = np.empty((n, n, 3))
    lo = np.clip(t * 2, 0, 1)[..., None]
    hi = np.clip(t * 2 - 1, 0, 1)[..., None]
    img = blue * (1 - lo) + white * lo
    img = img * (1 - hi) + pink * hi
    return img


_CAP_HEIGHT: dict = {}


def _face_text(ax, x: float, y: float, s: str, em: float, squash: float,
               color, weight: str = "normal", ha: str = "center",
               va: str = "center", zorder=5, skew: float = 0.0):
    """Text painted onto a foreshortened top face: rendered as a vector
    outline and squashed vertically by the same factor as the face.  ``em``
    is the em size in data units; the y axis is inverted, hence the negative
    y scale.  ``skew`` shears the glyphs (screen dx per dy) so text follows
    a sheared face under the table's sideways lean.

    Vertical centring uses the font's baseline-to-cap-height band, not the
    string's own tight box — otherwise descenders (Mg, Np) would push the
    string upward relative to descender-free neighbours (Mn, Se)."""
    tp = TextPath((0, 0), s, size=1.0, prop=FontProperties(weight=weight))
    bb = tp.get_extents()
    dx = {"center": -(bb.x0 + bb.x1) / 2, "left": -bb.x0, "right": -bb.x1}[ha]
    if va == "center":
        if weight not in _CAP_HEIGHT:
            _CAP_HEIGHT[weight] = TextPath(
                (0, 0), "H", size=1.0,
                prop=FontProperties(weight=weight)).get_extents().y1
        dy = -_CAP_HEIGHT[weight] / 2
    else:
        dy = {"top": -bb.y1, "bottom": -bb.y0}[va]
    tr = (Affine2D().translate(dx, dy).scale(em, -em * squash)
          + Affine2D.from_values(1, 0, skew, 1, 0, 0)
          + Affine2D().translate(x, y))
    ax.add_patch(PathPatch(tr.transform_path(tp), facecolor=color,
                           edgecolor="none", zorder=zorder))


def _block_silhouette(x0, w, ox0, ox1, yg0, yg1, zs, r) -> MplPath:
    """Complete outline of one standing block in screen coordinates
    (y grows downward).  Boundary: vertical left edge, rounded ground
    corners, ground edge receding along the table's lean on the right,
    vertical back-right edge, rounded top-back corners, then back along the
    top rim (which the top face later covers) and down the left rim to a
    small corner onto the left edge.  Filling this with the wall colour and
    painting the top face over it leaves exactly the front wall, the side
    wall and the rounded vertical-edge wedges visible — no seams."""
    gdir = np.array([ox0 - ox1, yg0 - yg1])
    gl = float(np.hypot(*gdir))
    gdir = gdir / gl                              # front->back along the ground
    r = max(0.0, min(r, 0.4 * gl, 0.4 * zs, 0.4 * w))
    fl = np.array([x0 + ox1, yg1])                # ground front-left
    fr = np.array([x0 + w + ox1, yg1])            # ground front-right
    br = np.array([x0 + w + ox0, yg0])            # ground back-right
    tbr = np.array([x0 + w + ox0, yg0 - zs])      # top back-right
    tbl = np.array([x0 + ox0, yg0 - zs])          # top back-left
    tfl = np.array([x0 + ox1, yg1 - zs])          # top front-left
    P = MplPath
    verts = [fl + (0, -r), fl, fl + (r, 0),
             fr + (-r, 0), fr, fr + gdir * r,
             br - gdir * r, br, br + (0, -r),
             tbr + (0, r), tbr, tbr + (-r, 0),
             tbl + (r, 0), tbl, tbl - gdir * r,
             tfl,                                  # sharp wall/rim corner
             fl + (0, -r)]
    codes = [P.MOVETO, P.CURVE3, P.CURVE3,
             P.LINETO, P.CURVE3, P.CURVE3,
             P.LINETO, P.CURVE3, P.CURVE3,
             P.LINETO, P.CURVE3, P.CURVE3,
             P.LINETO, P.CURVE3, P.CURVE3,
             P.LINETO,
             P.CLOSEPOLY]
    return P(np.asarray(verts, dtype=float), codes)


def _pit_silhouette(x0, w, ox0, ox1, yg0, yg1, p, rad) -> MplPath:
    """Outline of one SUNKEN tile -- a pit of depth ``p`` -- in screen
    coordinates (y grows downward).  The mirror of :func:`_block_silhouette`:
    the rim's back edge, down the far wall, along the sunken face's right and
    front edges, then up the near side and back along the rim's left edge.
    Filled with the wall colour and painted over with the face, this leaves
    exactly the far inner wall and the side inner wall the view looks in past,
    inside one unbroken rounded outline."""
    rbl = np.array([x0 + ox0, yg0])                # rim   back-left
    rbr = np.array([x0 + w + ox0, yg0])            # rim   back-right
    fbr = np.array([x0 + w + ox0, yg0 + p])        # face  back-right
    ffr = np.array([x0 + w + ox1, yg1 + p])        # face  front-right
    ffl = np.array([x0 + ox1, yg1 + p])            # face  front-left
    rfl = np.array([x0 + ox1, yg1])                # rim   front-left
    pts = [rbl, rbr, fbr, ffr, ffl, rfl]
    n = len(pts)
    edges = [float(np.hypot(*(pts[(i + 1) % n] - pts[i]))) for i in range(n)]
    rad = max(0.0, min(rad, 0.4 * min(e for e in edges if e > 1e-9)))
    P = MplPath
    verts, codes = [], []
    for i, v in enumerate(pts):                    # round every corner
        d_in = v - pts[i - 1]
        d_out = pts[(i + 1) % n] - v
        d_in = d_in / (np.hypot(*d_in) or 1.0)
        d_out = d_out / (np.hypot(*d_out) or 1.0)
        verts += [v - d_in * rad, v, v + d_out * rad]
        codes += [P.MOVETO if i == 0 else P.LINETO, P.CURVE3, P.CURVE3]
    verts.append(verts[0])
    codes.append(P.CLOSEPOLY)
    return P(np.asarray(verts, dtype=float), codes)


# --------------------------------------------------------------- tile drawing
def _draw_tile(ax, c: float, r: float, face, shadow_img, face_img, *,
               size: float = 0.94, depth: float = 0.09, rounding: float = 0.07,
               lift: float = 0.0, tilt: float = 0.0, side: float = 0.0,
               lip_img=None, noise_img=None, bevel: bool = False,
               nflip=(1, 1)):
    """One 3D tile centred on column ``c``, row ``r`` (y axis is inverted).

    With ``tilt == 0`` (straight-on view) the block is a face plus a
    constant front lip of height ``depth``.  With ``tilt > 0`` the whole
    table plane is tipped backwards and seen from the front-above: the
    ground is foreshortened by ``1 - tilt``, each block stands on its cell
    with height ``depth + lift``, showing a squashed lit top face and a
    standing front wall shaded light-at-top to dark-at-base.  Taller blocks
    rise higher and partly occlude the row behind (rows must therefore be
    drawn back to front).  ``face_img`` is the gradient overlay clipped onto
    the text-bearing face.

    The optional effect layers (used by the ``square`` style) push realism:
    ``lip_img`` is a lighting gradient clipped onto the visible side lip,
    ``noise_img`` is a shared film-grain raster (``nflip`` mirrors it per
    tile so the pattern doesn't repeat), and ``bevel`` strokes a light inner
    chamfer edge around the face.

    ``side`` adds an oblique horizontal view (tilted mode only): the top
    face shifts left by ``side * height`` and the block's right side wall
    becomes visible.

    Returns ``(x_anchor, y_anchor, y_scale, text_zorder)``: the centre of
    the text-bearing face, the offset scale for placing the cell text, and
    the zorder it must use.
    """
    w = h = size
    x0 = c - w / 2
    style = f"round,pad=0,rounding_size={rounding}"
    fx, fy = nflip

    def _grain(patch, X0, X1, Y0, Y1, zo, tr=None):
        if noise_img is None:
            return
        ex = (X0, X1) if fx > 0 else (X1, X0)
        ey = (Y0, Y1) if fy > 0 else (Y1, Y0)
        gi = ax.imshow(noise_img, extent=ex + ey, zorder=zo,
                       interpolation="bilinear", aspect="auto")
        if tr is not None:
            gi.set_transform(tr)
        gi.set_clip_path(patch)

    def _chamfer(X0, Y0, W, H, zo, tr=None):
        if not bevel:
            return
        inset = 0.022
        p = FancyBboxPatch(
            (X0 + inset, Y0 + inset), W - 2 * inset, H - 2 * inset,
            boxstyle=f"round,pad=0,rounding_size={max(rounding - inset, 0.005)}",
            mutation_scale=1.0, facecolor="none",
            edgecolor=(1, 1, 1, 0.30), linewidth=0.6, zorder=zo)
        if tr is not None:
            p.set_transform(tr)
        ax.add_patch(p)

    if not tilt:
        y0 = r - h / 2
        grow = 0.85 * w
        ax.imshow(shadow_img,
                  extent=(x0 - grow / 2, x0 + w + grow / 2,
                          y0 + h + 0.20 + grow / 2, y0 - grow / 2 + 0.20),
                  zorder=1, interpolation="bilinear", aspect="auto")
        body = FancyBboxPatch((x0, y0), w, h + depth,
                              boxstyle=style, mutation_scale=1.0,
                              facecolor=_shade(face, -0.20, -0.08),
                              edgecolor="none", zorder=2)
        ax.add_patch(body)
        if lip_img is not None:                    # lit crease -> dark bottom
            li = ax.imshow(lip_img, extent=(x0, x0 + w, y0 + h + depth, y0 + h),
                           zorder=2.2, interpolation="bilinear", aspect="auto")
            li.set_clip_path(body)
        front = FancyBboxPatch((x0, y0), w, h,
                               boxstyle=style, mutation_scale=1.0,
                               facecolor=face,
                               edgecolor=_shade(face, -0.22, -0.05),
                               linewidth=0.5, zorder=3)
        ax.add_patch(front)
        im = ax.imshow(face_img, extent=(x0, x0 + w, y0 + h, y0),
                       zorder=4, interpolation="bilinear", aspect="auto")
        im.set_clip_path(front)
        _grain(front, x0, x0 + w, y0 + h, y0, 4.2)
        _chamfer(x0, y0, w, h, 4.4)
        return c, r, 1.0, 5

    # ---- tilted-back view ----
    squash = 1.0 - tilt                           # ground foreshortening
    zs = depth + lift                             # face level: >0 stands, <0 sinks
    d = size - 0.06                               # footprint depth < width, so
    yg0 = (r - d / 2) * squash                    # a ground gap shows between
    yg1 = (r + d / 2) * squash                    # rows (back / front edges)
    # zorder: rows back-to-front, and within a row left-to-right, so each
    # block can layer above its LEFT neighbour (occlusion shadow) while
    # staying under everything of the row in front
    zb = 10 + 60 * r + 2.5 * c

    # rigid yaw of the whole table: the ground plane leans sideways with
    # depth, every point at row-depth rr shifting left by side * rr — so
    # rows slide horizontally as they come forward and the RIGHT side wall
    # of each block is exposed (fully on row-edge blocks, a sliver next to a
    # neighbour).  Vertical edges stay vertical: walls keep horizontal
    # top/bottom edges, top faces shear into parallelograms.
    ox0 = -side * (r - d / 2)                     # ground x-offset, back edge
    ox1 = -side * (r + d / 2)                     # ground x-offset, front edge
    oxm = -side * r

    if zs < 0:
        # ---- sunken tile: the value pulls the tile BELOW the table plane, and
        # we look into the pit it leaves.  A hole casts no shadow on the ground
        # and has no left-hand occlusion sliver, so both are skipped; what shows
        # instead is the far inner wall and, under the table's lean, the inner
        # wall on the side the view looks in past.
        p = -zs
        wall = _shade(face, -0.10, -0.03)          # inside a hole is shadowed
        body = PathPatch(_pit_silhouette(x0, w, ox0, ox1, yg0, yg1, p, rounding),
                         facecolor=wall,
                         edgecolor=_shade(face, -0.24, -0.05),
                         linewidth=0.5, zorder=zb + 1)
        ax.add_patch(body)
        if side:
            so = Polygon([(x0 + ox0, yg0), (x0 + ox1, yg1),
                          (x0 + ox1, yg1 + p), (x0 + ox0, yg0 + p)],
                         closed=True, facecolor=_shade(face, -0.17, -0.04),
                         edgecolor=(_shade(face, -0.24, -0.05) if bevel
                                    else "none"),
                         linewidth=0.5, zorder=zb + 1.2)
            so.set_clip_path(body)
            ax.add_patch(so)
        # far wall lighting: lit at the rim, darkening down into the hole --
        # the same top-lit profile a standing wall uses
        prof = face_img[::4, 0, :]
        base = np.asarray(wall)
        cols = base * (1 - prof[:, 3:4]) + prof[:, :3] * prof[:, 3:4]
        ny = len(cols)
        yy = np.linspace(yg0, yg0 + p, ny)[:, None]
        qm = ax.pcolormesh(
            np.tile([x0 + ox0, x0 + ox0 + w], (ny, 1)), np.tile(yy, (1, 2)),
            np.tile(np.linspace(0.0, 1.0, ny)[:, None], (1, 2)),
            shading="gouraud", cmap=LinearSegmentedColormap.from_list("w", cols),
            vmin=0.0, vmax=1.0, zorder=zb + 1.4)
        qm.set_clip_path(body)
        _grain(body, x0 + ox1, x0 + w + ox0, yg1 + p, yg0, zb + 1.6)

        shear = None                               # face follows the lean as usual
        if side:
            y_ref = yg0 + p
            shear = (Affine2D().translate(0, -y_ref)
                     + Affine2D.from_values(1, 0, -side / squash, 1, 0, 0)
                     + Affine2D().translate(ox0, y_ref) + ax.transData)
        top = FancyBboxPatch((x0, yg0 + p), w, d * squash,
                             boxstyle=style, mutation_scale=1.0,
                             facecolor=_shade(face, -0.02, -0.01),
                             edgecolor=_shade(face, -0.22, -0.05),
                             linewidth=0.5, zorder=zb + 3)
        if shear is not None:
            top.set_transform(shear)
        ax.add_patch(top)
        _grain(top, x0, x0 + w, yg0 + p + d * squash, yg0 + p, zb + 3.2, tr=shear)
        _chamfer(x0, yg0 + p, w, d * squash, zb + 3.4, tr=shear)
        # Text sits on the face exactly as it does on a standing block -- same
        # offsets, same corner for the atomic number -- so a sunken tile is
        # labelled identically to a raised one.  The row in front does crop the
        # bottom of a deep pit's face, so a deep enough tile loses part of its
        # value; the price of depth, left to the caller's `relief_height`.
        return c + oxm, r * squash - zs - 0.08 * squash, squash, zb + 4

    # soft contact shadow on the ground around the footprint
    grow = (0.55 + 0.45 * lift) * w
    off = 0.06 + 0.22 * zs
    ax.imshow(shadow_img,
              extent=(x0 + oxm - grow / 2, x0 + oxm + w + grow / 2,
                      yg1 + (off + grow / 2) * squash,
                      yg0 + (off - grow / 2) * squash),
              zorder=zb, interpolation="bilinear", aspect="auto")

    # soft occlusion along the left silhouette (rim + left edge): dims the
    # sliver of the LEFT neighbour's receding side wall glimpsed through
    # the inter-block gap, so the slit reads as shadow, not a broken edge.
    # Many thin nested layers approximate a smooth fade like the ground
    # shadows (two hard-edged bands would read as stripes).
    for i in range(6):
        wd = 0.085 * (1.0 - i / 6.0)
        ax.add_patch(Polygon([(x0 + ox0, yg0 - zs), (x0 + ox1, yg1 - zs),
                              (x0 + ox1, yg1), (x0 + ox1 - wd, yg1),
                              (x0 + ox1 - wd, yg1 - zs),
                              (x0 + ox0 - wd, yg0 - zs)],
                             closed=True, facecolor=(0, 0, 0, 0.026),
                             edgecolor="none", zorder=zb + 0.95))

    # block body: ONE continuous silhouette filled with the wall colour.
    # The top face painted over it leaves the front wall, the side wall and
    # the rounded vertical-edge wedges visible, with a single unbroken
    # rounded outline — no seams or corner tabs.
    body = PathPatch(_block_silhouette(x0, w, ox0, ox1, yg0, yg1, zs,
                                       rounding),
                     facecolor=_shade(face, -0.04, -0.02),
                     edgecolor=_shade(face, -0.24, -0.05),
                     linewidth=0.5, zorder=zb + 1)
    ax.add_patch(body)

    if side:
        # darker overlay on the receding side region, clipped by the body's
        # rounded outline.  The square finish strokes the inner boundary so
        # the vertical edge between front and side wall reads crisp; the
        # soft finish leaves it as a smooth tone change (rounded edge).
        so = Polygon([(x0 + w + ox1, yg1 + 0.02), (x0 + w + ox0, yg0),
                      (x0 + w + ox0, yg0 - zs),
                      (x0 + w + ox1, yg1 - zs - 0.02)],
                     closed=True, facecolor=_shade(face, -0.11, -0.03),
                     edgecolor=(_shade(face, -0.24, -0.05) if bevel
                                else "none"),
                     linewidth=0.5, zorder=zb + 1.2)
        so.set_clip_path(body)
        ax.add_patch(so)

    # wall lighting over the front region: the overlay profile is composited
    # onto the wall colour and drawn as a gouraud-shaded mesh — identical
    # pixels, but pure vector shading in PDF/SVG instead of one raster per
    # tile
    prof = face_img[::4, 0, :]                    # (16, 4) rows, top->bottom
    base = np.asarray(_shade(face, -0.04, -0.02))
    cols = base * (1 - prof[:, 3:4]) + prof[:, :3] * prof[:, 3:4]
    ny = len(cols)
    yy = np.linspace(yg1 - zs, yg1, ny)[:, None]
    qm = ax.pcolormesh(
        np.tile([x0 + ox1, x0 + ox1 + w], (ny, 1)), np.tile(yy, (1, 2)),
        np.tile(np.linspace(0.0, 1.0, ny)[:, None], (1, 2)),
        shading="gouraud", cmap=LinearSegmentedColormap.from_list("w", cols),
        vmin=0.0, vmax=1.0, zorder=zb + 1.4)
    qm.set_clip_path(body)
    _grain(body, x0 + ox1, x0 + w + ox0, yg1, yg0 - zs, zb + 1.6)

    # lit top face: sheared parallelogram between the back and front offsets
    shear = None
    if side:
        y_ref = yg0 - zs
        shear = (Affine2D().translate(0, -y_ref)
                 + Affine2D.from_values(1, 0, -side / squash, 1, 0, 0)
                 + Affine2D().translate(ox0, y_ref) + ax.transData)
    top = FancyBboxPatch((x0, yg0 - zs), w, d * squash,
                         boxstyle=style, mutation_scale=1.0,
                         facecolor=_shade(face, +0.06, -0.01),
                         edgecolor=_shade(face, -0.22, -0.05),
                         linewidth=0.5, zorder=zb + 3)
    if shear is not None:
        top.set_transform(shear)
    ax.add_patch(top)
    _grain(top, x0, x0 + w, yg0 - zs + d * squash, yg0 - zs, zb + 3.2, tr=shear)
    _chamfer(x0, yg0 - zs, w, d * squash, zb + 3.4, tr=shear)
    # anchor the text slightly towards the back edge: the face's front part
    # is what a taller block in the row ahead may occlude
    return c + oxm, r * squash - zs - 0.08 * squash, squash, zb + 4


# ------------------------------------------------------------------ main entry
def periodic_table_3d(
    data: Optional[Union[Mapping, Sequence]] = None,
    values: Optional[Sequence] = None,
    *,
    cmap: Union[str, Colormap] = "poster",
    cmap_norm=None,
    label_cbar: Optional[str] = None,
    ax: Optional[Axes] = None,
    # per-tile text
    show_symbol: bool = True,
    show_at_number: bool = True,
    show_name: bool = False,
    show_at_mass: bool = True,
    show_value: bool = False,
    value_fmt: str = "{:.2f}",
    mass_fmt: str = "{:.2f}",
    # appearance
    style: str = "soft",
    relief_height: float = 0.0,
    relief_norm=None,
    relief_signed: bool = False,
    font_scale: float = 1.0,
    tilt: float = 0.20,
    side_tilt: float = 0.15,
    max_z: int = 118,
    elements: Optional[Sequence] = None,
    missing_color: str = "#e7e3dc",
    draw_missing: bool = True,
    background: Union[bool, str] = False,
    fblock_labels: bool = False,
    colorbar: bool = True,
    cbar_loc: str = "gap",
    cbar_shape: Optional[str] = None,
    cbar_kw: Optional[dict] = None,
    figsize=(15.0, 8.6),
    savepath: Optional[str] = None,
    **savefig_kw,
) -> PeriodicTablePlot:
    """Draw a periodic table as glossy 3D tiles with soft shadows.

    With no ``data`` the tiles are coloured by chemical family (pastel
    poster palette).  With ``data``/``values`` the faces are coloured by the
    heatmap ``cmap`` exactly as in :func:`periodic_table`.

    ``style`` selects the tile finish.  ``"soft"`` (default) is the original
    look: rounded corners, glossy sheen.  ``"square"`` is closer to the
    reference photograph: near-square corners, matte faces with a film-grain
    texture, a light chamfered inner edge, a lit crease above a darkening
    side lip and larger, softer shadows.

    ``relief_height`` (value mode only) maps the value onto the physical height of
    each block, in cell heights, and tips the whole table plane backwards so
    the heights are visible: blocks stand on a foreshortened ground, showing
    a squashed lit top face and a standing front wall shaded light-at-top to
    dark-at-base; taller blocks rise higher, partly occlude the row behind
    and cast a wider shadow.  ``0`` (the default here) keeps the straight-on
    flat view.  Elements without a value stay flat either way.

    ``relief_signed`` measures the relief from ZERO instead of from the bottom of the
    scale: values above zero raise their tile out of the table as usual, values
    below it pull the tile DOWN and the view looks into the pit left behind
    (lit at the rim, darkening with depth).  Use it for a property with a
    meaningful zero; the value furthest from zero either way reaches
    ``relief_height``.
    The norm has to be able to place zero, so a log scale will not do.

    ``relief_norm`` drives the block heights from a normalisation of its own
    instead of the colour ``cmap_norm``.  Use it when the two should differ --
    e.g.
    a diverging colour scale about zero, but height standing for the
    *magnitude* of the property, so the strongest values are tallest whichever
    sign they carry.  Accepts anything ``cmap_norm`` does, or a plain callable
    mapping a value to [0, 1] (``lambda v: abs(v) / 0.8``).

    ``font_scale`` multiplies every per-tile text size at once (the symbol,
    value, number, name and mass keep their relative proportions).  The
    built-in sizes suit a full-width figure; scale them down when the table is
    one panel of a composed figure.

    ``tilt`` sets how far the table tips back when ``relief_height`` is nonzero: the
    ground plane is foreshortened to ``1 - tilt`` of its depth (0.20 by
    default; larger -> steeper look-down angle, more foreshortened faces).

    ``side_tilt`` yaws the whole table (nonzero ``relief_height`` only): the
    ground plane
    leans sideways with depth, each row shifting left by ``side_tilt`` cells
    per row as it comes forward, so the table reads as one rigid object
    seen from front-above-right and the RIGHT side wall of each block is
    exposed — fully on blocks at the right edge of their row, as a sliver
    next to a neighbour.  Top faces (and their text) shear to follow the
    lean.  ``0`` restores the straight-on oblique view.

    ``elements`` optionally restricts which elements are drawn (symbols or
    atomic numbers); ``max_z`` is a simpler trailing cutoff (e.g. ``103`` to
    omit the Rf-Og superheavies).

    ``cbar_loc`` places the colourbar: ``"gap"`` (default) lays it
    horizontally in the empty block between Be and B, following the table's
    tilt and lean; ``"right"``, ``"left"``, ``"top"`` and ``"bottom"`` put it
    outside the table.  The bar is framed like an element tile; ``cbar_shape``
    overrides the corner geometry, which otherwise follows the tile finish.

    ``background`` draws nothing by default (the figure facecolor shows
    through); pass ``"gradient"`` for the soft pastel backdrop, ``True`` for
    white, or any matplotlib colour for a flat fill.

    The shadows and sheen are small rasters; vector output embeds each source
    image once alongside the vector outlines and text, so PDF/SVG work as well
    as PNG -- just pick a ``dpi`` you're happy with.
    """
    # public parameter names -> the internal shorthands used below
    norm, label = cmap_norm, label_cbar
    show_number, show_mass = show_at_number, show_at_mass
    height, height_norm, signed = relief_height, relief_norm, relief_signed

    value_mode = data is not None
    mappable = None
    vd: dict = {}
    if value_mode:
        vd = _value_dict(data, values)
        if not vd:
            raise ValueError("no values provided")
        n = _resolve_norm(norm, list(vd.values()))
        if cmap == "poster":
            cmap_obj = POSTER_CMAP                 # works even if not registered
        else:
            cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        mappable = ScalarMappable(norm=n, cmap=cmap_obj)
        mappable.set_array([])

    lift_max = 0.0
    lift_norm = None
    if height:
        if not value_mode:
            raise ValueError("height/relief needs data (value mode)")
        lift_max = float(height)
        # heights follow the colour norm unless given one of their own
        if height_norm is None:
            lift_norm = mappable.norm
        elif callable(height_norm) and not isinstance(height_norm, Normalize):
            lift_norm = height_norm                # plain function: value -> [0, 1]
        else:
            lift_norm = _resolve_norm(height_norm, list(vd.values()))

    def _lift(value):
        """Signed height of one tile, in cell heights (negative sinks it)."""
        t = float(np.clip(lift_norm(value), 0.0, 1.0))
        if not signed:
            return lift_max * t
        return lift_max * (t - t_zero) / reach

    t_zero = reach = 0.0
    if lift_max and signed:
        try:
            t_zero = float(np.clip(lift_norm(0.0), 0.0, 1.0))
        except Exception:
            t_zero = float("nan")
        if t_zero != t_zero:                       # NaN: the norm cannot place 0
            raise ValueError("signed=True needs a norm that can place 0 "
                             "(e.g. 'diverging' or a (vmin, vmax) spanning it)")
        reach = max(t_zero, 1.0 - t_zero) or 1.0

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    keep = None
    if elements is not None:
        keep = {_to_Z(e) for e in elements}

    last_row = 9.5 if (keep is None or any(z >= 89 for z in keep)) else 7.0
    if lift_max:
        # Limits from what is actually drawn rather than a worst-case formula:
        # each block's footprint, leaned left by `side_tilt` per row of depth and
        # raised by its own value, plus room for the shadow halo.  A formula over
        # the whole grid wastes a cell of width on the left, because the deepest
        # row that reaches column 1 is period 7 -- the f-block rows behind it
        # start at column 3.
        squash = 1.0 - tilt
        HALF, D2, PAD = 0.47, 0.44, 0.34           # tile half-width, half-depth, halo
        placed = []
        for Z, (sym, name, mass, grp, per) in ELEMENTS.items():
            if Z > max_z or (keep is not None and Z not in keep):
                continue
            if value_mode and Z not in vd and not draw_missing:
                continue
            c, r = _cell_pos(Z, grp, per)
            placed.append((c, r, 0.09 + (_lift(vd[Z]) if Z in vd else 0.0)))
        x_lo = min(c - HALF - side_tilt * (r + D2) for c, r, _ in placed) - PAD
        x_hi = max(c + HALF - side_tilt * (r - D2) for c, r, _ in placed) + PAD
        if fblock_labels:                          # the "La-Lu" / "Ac-Lr" captions
            x_lo = min(x_lo, _FCOL - 1.75 - side_tilt * _ACT_ROW)
        # a standing block reaches up by zs and casts a shadow past its front
        # edge; a sunken one (zs < 0) instead drops its face that far below
        y_hi = min((r - D2) * squash - max(zs, 0.0) for c, r, zs in placed) - 0.22
        y_lo = max((r + D2) * squash - min(zs, 0.0)
                   + (0.34 + 0.22 * max(zs, 0.0)) * squash
                   for c, r, zs in placed) + 0.10
        xlim, ylim = (x_lo, x_hi), (y_lo, y_hi)
    else:
        xlim = (-0.35, 19.05)
        ylim = (10.75, 0.2)                        # row 1 at the top

    if background == "gradient":
        ax.imshow(_background_image(),
                  extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                  zorder=0, interpolation="bilinear", aspect="auto")
    elif background:
        ax.add_patch(Rectangle((xlim[0], ylim[1]),
                               xlim[1] - xlim[0], ylim[0] - ylim[1],
                               facecolor=("white" if background is True
                                          else background),
                               edgecolor="none", zorder=0))

    if style not in ("soft", "square"):
        raise ValueError(f"style must be 'soft' or 'square', got {style!r}")
    square = style == "square"
    rounding = 0.022 if square else 0.07
    shadow_img = (_shadow_image(radius=9, alpha=0.55) if square
                  else _shadow_image())
    if lift_max:
        face_img = _wall_image()
    else:
        face_img = _matte_image() if square else _gloss_image()
    lip_img = _lip_image() if square else None
    noise_img = _noise_image() if square else None

    # point-size -> data-unit conversion for text painted onto tilted faces
    pt2data = 1.0
    if lift_max:
        ax_pt = ax.get_position().width * fig.get_size_inches()[0] * 72.0
        pt2data = (xlim[1] - xlim[0]) / ax_pt

    for Z, (sym, name, mass, grp, per) in ELEMENTS.items():
        if Z > max_z or (keep is not None and Z not in keep):
            continue
        has = Z in vd
        if value_mode and not has and not draw_missing:
            continue
        c, r = _cell_pos(Z, grp, per)

        if value_mode:
            face = mappable.to_rgba(vd[Z])[:3] if has else to_rgb(missing_color)
        else:
            face = to_rgb(_FAMILY_COLORS[_family(Z, grp, per)])
        lift = _lift(vd[Z]) if (lift_max and has) else 0.0

        cx, ry, ys, tz = _draw_tile(ax, c, r, face, shadow_img, face_img,
                                    lift=lift, tilt=(tilt if lift_max else 0.0),
                                    side=(side_tilt if lift_max else 0.0),
                                    rounding=rounding, lip_img=lip_img,
                                    noise_img=noise_img, bevel=square,
                                    nflip=(1 if Z % 2 else -1,
                                           1 if (Z >> 1) % 2 else -1))
        # lower-half text offsets, tightened on foreshortened faces
        d_val, d_nm1, d_nm2, d_ms = ((0.31, 0.29, 0.19, 0.30) if lift_max
                                     else (0.35, 0.34, 0.22, 0.36))

        if value_mode:
            rl, gl, bl = face
            lum = 0.299 * rl + 0.587 * gl + 0.114 * bl
            ink = (1, 1, 1) if lum < 0.45 else _ink(face)
        else:
            ink = _ink(face)
        faint = ink + (0.72,)

        K = side_tilt if lift_max else 0.0         # table lean, cells per row

        def put(dx, dy, s, pt, color, weight="normal", ha="center", va="center"):
            # (dx, dy) are cell offsets on the face; depth offsets follow
            # the table's sideways lean
            pt = pt * font_scale
            x, y = cx + dx - K * dy, ry + dy * ys
            if lift_max:                           # painted onto the tilted face
                _face_text(ax, x, y, s, em=pt * pt2data, squash=ys,
                           skew=-K / ys, color=color, weight=weight,
                           ha=ha, va=va, zorder=tz)
            else:
                ax.text(x, y, s, ha=ha, va=va, fontsize=pt, fontweight=weight,
                        color=color, zorder=tz)

        if show_number:
            put(-0.36, -(0.28 if lift_max else 0.31), str(Z),
                6.2, faint, weight="bold", ha="left", va="top")
        if show_symbol:
            dy = 0.0 if (show_name or (show_value and has)) else 0.05
            put(0.0, dy, sym, 14.5, ink, weight="bold")
        if show_value and has:
            put(0.0, d_val, value_fmt.format(vd[Z]), 9.5, ink)
        if show_name:
            put(0.0, d_nm1 if (show_value and has) else d_nm2, name, 4.3, faint)
        if show_mass and not (show_value and has):
            put(0.0, d_ms, mass_fmt.format(mass), 3.9, faint)

    if fblock_labels:
        yscale = (1.0 - tilt) if lift_max else 1.0
        xlean = side_tilt if lift_max else 0.0
        for row, lab, lo, hi in ((_LANTH_ROW, "La–Lu", 57, 71),
                                 (_ACT_ROW, "Ac–Lr", 89, 103)):
            if max_z < lo or (keep is not None
                              and not any(lo <= z <= hi for z in keep)):
                continue
            ax.text(_FCOL - 1.2 - xlean * row,
                    row * yscale - (0.1 if lift_max else 0.0),
                    lab, ha="center", va="center",
                    fontsize=7.5, color="#9b9083", style="italic", zorder=5)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(0.92)                            # tiles slightly taller than wide
    ax.axis("off")

    if value_mode and colorbar:
        ysc = (1.0 - tilt) if lift_max else 1.0
        xsh = -side_tilt if lift_max else 0.0

        # the gap bar follows the relief squash and sideways lean
        cb = _add_colorbar(fig, ax, mappable, loc=cbar_loc, label=label,
                           cbar_kw=cbar_kw, font_scale=font_scale,
                           shape=cbar_shape or ("square" if square else "round"),
                           project=lambda x, y: (x + xsh * y, y * ysc))

    result = PeriodicTablePlot(fig=fig, ax=ax, mappable=mappable)
    if savepath:
        kwargs = dict(savefig_kw)
        kwargs.setdefault("dpi", 250)
        if str(savepath).lower().endswith((".pdf", ".svg")):
            # vector output: embed each small source raster once, letting
            # the viewer interpolate, instead of baking a dpi-resampled
            # copy of every shadow/gradient into the file; also use the
            # strongest stream compression
            for im_ in ax.images:
                im_.set_interpolation("none")
            try:
                with mpl.rc_context({"pdf.compression": 9}):
                    result.save(savepath, **kwargs)
            finally:
                for im_ in ax.images:
                    im_.set_interpolation("bilinear")
        else:
            result.save(savepath, **kwargs)
    return result
