"""periodicplots — periodic-table heatmaps in matplotlib.

    import periodicplots as pp
    pp.periodic_table({"Fe": 1.2, "O": 3.4, "Si": 1.1})   # 2D table
    pp.periodic_table(values, tile_style="3d")            # 2D, 3D-look tiles
    pp.periodic_table_3d(values)                          # 3D: value -> height
    pp.periodic_table_3d(values, tile_style="flat")       # 3D, flat-shaded

``periodic_table`` colours each element by its value; ``periodic_table_3d``
also maps the value onto the block's height.  Both take ``tile_style``
(``"flat"`` or ``"3d"``) and ``tile_shape`` (``"round"`` or ``"square"``) for
how a cell is drawn.  The flat-shaded renderers are pure matplotlib patches +
text; the "3d" tile style adds small rasters for its shadows and gloss, which
PDF/SVG embed alongside the vector outlines and text.
"""
from .core import PeriodicTablePlot, periodic_table
from .relief import periodic_table_relief as _blocks_3d
from .style3d import POSTER_CMAP
from .style3d import periodic_table_3d as _tiles_3d

__all__ = ["periodic_table", "periodic_table_3d", "PeriodicTablePlot",
           "POSTER_CMAP", "ELEMENTS", "SYMBOL_TO_Z"]
__version__ = "0.2.0"

from ._elements import ELEMENTS, SYMBOL_TO_Z


def periodic_table_3d(data=None, values=None, *, tile_style="3d",
                      tile_shape="round", **kwargs):
    """Draw a periodic table in relief: the value also sets each block's height.

    Relief is what makes this function "_3d" -- whenever data is given, blocks
    rise out of the table with their value (``relief_height`` sets the tallest
    one: 0.60 cell heights for ``tile_style="3d"``, 1.0 for ``"flat"``, whose
    shallower view needs a taller block to read the same; ``relief_signed=True``
    lets negative values sink into the table instead).  With no data the tiles
    are coloured by chemical family and stay flat (nothing to encode).

    tile_style :
        How each block is rendered -- the same property `periodic_table` has:
        ``"3d"`` (default) — physical-looking tiles: drop shadows, lit faces.
        ``"flat"`` — flat-shaded extruded blocks drawn from plain polygons, so
        the table itself contains no rasters; the labels are laid *in* the
        plane of the table.  Requires data (unlike ``"3d"``, it has no
        family-colour mode).
    tile_shape :
        Corner geometry of the ``"3d"`` tiles.  ``"round"`` (the default) is
        the glossy rounded-corner look; ``"square"`` the photographic finish:
        near-square corners, matte grainy faces, chamfered edges.  Ignored by
        ``tile_style="flat"``, whose blocks are sharp-cornered by construction.

    The remaining keyword arguments belong to the selected renderer:

    * ``"3d"`` -- ``tilt``, ``side_tilt``, ``font_scale``, ``background``,
      ``cbar_loc``, ``elements``: see
      :func:`periodicplots.style3d.periodic_table_3d`.
    * ``"flat"`` -- ``depth``, ``shear``, ``base_height``, ``gap``,
      ``wall_shade``, ``side_shade``, ``wall_fade``, ``fade_color``,
      ``pit_color``, ``text_on_surface``, ``edge_color``, ``edge_width``,
      ``text_color``, ``show_group_period`` and the per-role ``*_fontsize``
      options: see :func:`periodicplots.relief.periodic_table_relief`.
    """
    if tile_shape not in ("round", "square"):
        raise ValueError(
            f"tile_shape must be 'round' or 'square', got {tile_shape!r}")
    if tile_style not in ("3d", "flat"):
        raise ValueError(
            f"tile_style must be '3d' or 'flat', got {tile_style!r}")
    # Relief is the point of _3d, so a height is always set once there is data.
    # The two defaults differ because the projections do: "3d" tips the whole
    # table back (tilt/side_tilt), so a modest height already reads as relief,
    # while "flat" uses a shallower oblique view and needs a taller block for
    # the same apparent lift.
    if data is not None:
        kwargs.setdefault("relief_height", 0.60 if tile_style == "3d" else 1.0)
    if tile_style == "3d":
        return _tiles_3d(data, values,
                         style=("square" if tile_shape == "square" else "soft"),
                         **kwargs)
    if data is None:
        # "3d" has a no-data mode (chemical-family colours); "flat" has none,
        # and without this it fails deep inside with a confusing TypeError
        raise ValueError(
            'tile_style="flat" needs data: it draws blocks whose height comes '
            'from the value.  Call periodic_table_3d() with no data for the '
            'chemical-family poster, or pass a mapping {element: value}.')
    return _blocks_3d(data, values, **kwargs)
