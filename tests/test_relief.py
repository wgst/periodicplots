import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Polygon
import pytest

import periodicplots as pp
from periodicplots.relief import (_mix, _shade,
                                 periodic_table_relief)


def upright(values, **kw):
    """A table whose labels are plain Text objects, so positions are readable."""
    kw.setdefault("colorbar", False)
    return periodic_table_relief(values, text_on_surface=False, **kw)


def _text_y(result, symbol):
    """Screen y of an element's symbol (the axis is inverted, so smaller = higher)."""
    return next(t.get_position()[1] for t in result.ax.texts
                if t.get_text() == symbol)


def _blocks(result):
    return [p for p in result.ax.patches if isinstance(p, Polygon)]


def _labels(result):
    return [p for p in result.ax.patches if isinstance(p, PathPatch)]


def test_returns_axes_and_mappable():
    r = periodic_table_relief({"Fe": 1.0, "O": 2.0}, colorbar=False)
    assert r.ax is not None and r.mappable is not None
    plt.close(r.fig)


def test_higher_value_stands_further_out():
    # same period (Fe, Co, Ni) -> only the value can lift a cell
    r = upright({"Fe": 1.0, "Co": 2.0, "Ni": 3.0})
    assert _text_y(r, "Ni") < _text_y(r, "Co") < _text_y(r, "Fe")
    plt.close(r.fig)


def test_height_scales_the_relief():
    flat = upright({"Fe": 1.0, "Co": 2.0}, relief_height=0.2)
    tall = upright({"Fe": 1.0, "Co": 2.0}, relief_height=2.0)
    spread = lambda r: _text_y(r, "Fe") - _text_y(r, "Co")
    assert spread(tall) > spread(flat) > 0
    plt.close(flat.fig), plt.close(tall.fig)


def test_faces_per_block():
    # top face + front wall; the side wall only appears once the view is sheared
    kw = dict(base_height=0.8, wall_fade=0.0, draw_missing=False)
    straight = upright({"Fe": 1.0}, shear=0.0, **kw)
    sheared = upright({"Fe": 1.0}, shear=0.3, **kw)
    assert len(_blocks(straight)) == 2 and len(_blocks(sheared)) == 3
    plt.close(straight.fig), plt.close(sheared.fig)


def test_faded_walls_are_built_from_strips():
    flat = upright({"Fe": 1.0}, shear=0.0, wall_fade=0.0, draw_missing=False)
    faded = upright({"Fe": 1.0}, shear=0.0, wall_fade=0.8, draw_missing=False)
    assert len(_blocks(faded)) > len(_blocks(flat))          # gradient, still vector
    plt.close(flat.fig), plt.close(faded.fig)


def test_zero_height_block_has_no_walls():
    r = upright({"Fe": 1.0}, relief_height=0.0, base_height=0.0, draw_missing=False)
    assert len(_blocks(r)) == 1                              # the top face alone
    plt.close(r.fig)


def test_painted_back_to_front():
    # near rows must be drawn over far ones, or the occlusion comes out inverted
    r = upright({"Li": 1.0, "Na": 1.0, "K": 1.0}, draw_missing=False)
    zs = [p.get_zorder() for p in _blocks(r)]
    assert zs == sorted(zs)
    plt.close(r.fig)


def test_text_sits_above_its_own_block():
    r = upright({"Fe": 1.0}, draw_missing=False, fblock_labels=False)
    assert min(t.get_zorder() for t in r.ax.texts) > \
        max(p.get_zorder() for p in _blocks(r))
    plt.close(r.fig)


def test_labels_lie_on_the_surface_by_default():
    # default: glyph outlines projected onto the faces, not upright Text
    r = periodic_table_relief({"Fe": 1.0}, draw_missing=False, fblock_labels=False,
                             colorbar=False)
    assert len(_labels(r)) == 2                              # symbol + value
    assert [t.get_text() for t in r.ax.texts] == []
    plt.close(r.fig)


def test_surface_labels_sit_above_their_block():
    r = periodic_table_relief({"Fe": 1.0}, draw_missing=False, fblock_labels=False,
                             colorbar=False)
    assert min(p.get_zorder() for p in _labels(r)) > \
        max(p.get_zorder() for p in _blocks(r))
    plt.close(r.fig)


def test_surface_labels_follow_the_projection():
    # a label on a sheared, foreshortened face must be sheared and squashed too
    straight = periodic_table_relief({"Fe": 1.0}, shear=0.0, depth=1.0,
                                    colorbar=False)
    tipped = periodic_table_relief({"Fe": 1.0}, shear=0.4, depth=0.5, colorbar=False)
    bbox = lambda r: max(p.get_path().get_extents().height for p in _labels(r))
    assert bbox(tipped) < bbox(straight)
    plt.close(straight.fig), plt.close(tipped.fig)


def test_signed_sinks_negative_values():
    # same period: negative below the plane, positive above it, zero flat
    r = upright({"Fe": -1.0, "Co": 0.0, "Ni": 1.0}, cmap_norm="diverging", relief_signed=True)
    assert _text_y(r, "Ni") < _text_y(r, "Co") < _text_y(r, "Fe")
    plt.close(r.fig)


def test_signed_is_symmetric_about_zero():
    # Ru (equally deep, in front of Fe) keeps Fe's floor unclipped, so the label
    # anchor measures the pure relief offset
    r = upright({"Fe": -1.0, "Ru": -1.0, "Co": 0.0, "Ni": 1.0},
                cmap_norm="diverging", relief_signed=True)
    up = _text_y(r, "Co") - _text_y(r, "Ni")
    down = _text_y(r, "Fe") - _text_y(r, "Co")
    assert up == pytest.approx(down)
    plt.close(r.fig)


def test_signed_zero_value_has_no_walls():
    r = upright({"Fe": 0.0}, cmap_norm=(-1.0, 1.0), relief_signed=True, draw_missing=False)
    assert len(_blocks(r)) == 1                              # sits flat in the plane
    plt.close(r.fig)


def test_signed_needs_a_norm_that_places_zero():
    from matplotlib.colors import LogNorm
    with pytest.raises(ValueError, match="place 0"):
        periodic_table_relief({"Fe": 1.0, "O": 2.0}, cmap_norm=LogNorm(1.0, 10.0),
                             relief_signed=True, colorbar=False)


def _pit(**nb):
    """Patch count of one sunken cell, given its neighbours' floors."""
    from periodicplots.relief import _draw_block
    nb.setdefault("z_front", -0.5)               # something in front, hiding our near side
    fig, ax = plt.subplots()
    _draw_block(ax, 1, 2, -0.5, 0.5, 0.7, 0.0, "tab:blue", 1.0, 0.9, 0.4,
                "black", "0.75", "black", 0.6, 2.0, **nb)
    n = len(ax.patches)
    plt.close(fig)
    return n


def test_pit_inner_wall_stops_at_its_neighbour():
    assert _pit(z_back=0.0) > 1       # a step down from the table surface: wall drawn
    assert _pit(z_back=None) > 1      # no cell behind: cut down from the surface too
    assert _pit(z_back=-0.5) == 1     # sunk to the same level: one continuous trough
    assert _pit(z_back=-0.9) == 1     # neighbour deeper: that wall faces away from us


def test_the_open_front_of_the_table_shows_no_wall():
    # tiles simply end at the table's front edge -- there is no outer face there
    assert _pit(z_back=-0.5, z_front=None) == _pit(z_back=-0.5, z_front=-0.5)


def test_the_trench_side_edge_still_shows():
    # ...but the side boundary against the bare slab is a real cut face
    from periodicplots.relief import _draw_block
    def n(z_side_out):
        fig, ax = plt.subplots()
        _draw_block(ax, 1, 2, -0.5, 0.5, 0.7, -0.2, "tab:blue", 1.0, 0.9, 0.4,
                    "black", "0.75", "black", 0.6, 2.0, z_back=-0.5, z_front=None,
                    z_side_out=z_side_out, side_back_open=True)
        k = len(ax.patches)
        plt.close(fig)
        return k
    assert n(None) > n(-0.5)                     # slab beside us: wall; equal pit: none


def test_side_wall_follows_whether_the_neighbour_is_lower():
    # every cell shows its side face unless the cell that side is higher
    from periodicplots.relief import _draw_block
    def walls(z_side_out):
        # z_back at the surface: the trench does not continue behind, so the side
        # wall runs its full height rather than being capped to a step
        fig, ax = plt.subplots()
        _draw_block(ax, 1, 2, -0.5, 0.5, 0.7, -0.2, "tab:blue", 1.0, 0.9, 0.4,
                    "black", "0.75", "black", 0.6, 2.0, z_back=0.0, z_front=-0.5,
                    z_side_out=z_side_out)
        n = len(ax.patches)
        plt.close(fig)
        return n
    bare = walls(-0.2)                                       # that side is higher: hidden
    assert walls(None) > bare                                # no cell there: shown
    assert walls(-0.9) > bare                                # that side is lower: shown


def test_a_shallow_step_keeps_its_outline():
    # the wall above a barely-sunk row still gets an edge, like every other one
    from periodicplots.relief import _draw_block
    fig, ax = plt.subplots()
    _draw_block(ax, 1, 2, -0.02, 0.5, 0.7, 0.0, "tab:blue", 1.0, 0.9, 0.4,
                "black", "0.75", "black", 0.6, 2.0, z_back=0.0, z_front=-0.02,
                c_back="tab:blue")
    assert sum(not p.get_fill() for p in ax.patches) == 1
    plt.close(fig)


def _pit_wall_colours(**nb):
    from periodicplots.relief import _draw_block
    nb.setdefault("z_front", -0.5)
    fig, ax = plt.subplots()
    _draw_block(ax, 1, 2, -0.5, 0.5, 0.7, 0.0, "tab:blue", 1.0, 0.9, 0.4,
                "black", "0.75", "black", 0.6, 2.0, **nb)
    cols = [p.get_facecolor()[:3] for p in ax.patches[:-1] if p.get_fill()]
    plt.close(fig)
    return cols


def test_a_cut_into_the_bare_slab_is_grey():
    # no cell behind -> the pit cuts the plain slab, and that face is neutral
    cols = _pit_wall_colours(z_back=None)
    assert cols and all(c[0] == pytest.approx(c[2], abs=0.01) for c in cols)


def test_a_cut_into_another_element_takes_its_colour():
    # a cell behind -> the pit cuts that element's tile, so the wall is its colour
    cols = _pit_wall_colours(z_back=-0.2, c_back="tab:red")
    assert cols and all(c[0] > c[2] + 0.2 for c in cols)      # red, not grey


def test_side_wall_is_capped_by_the_trench_behind():
    # at full height a side wall would stand up in front of the row behind it
    from periodicplots.relief import _draw_block
    def highest(z_back):
        fig, ax = plt.subplots()
        _draw_block(ax, 1, 2, -0.5, 0.5, 0.7, -0.2, "tab:blue", 1.0, 0.9, 0.4,
                    "black", "0.75", "black", 0.6, 2.0, z_back=z_back,
                    z_front=-0.5, z_side_out=None)
        y = min(p.get_path().get_extents().y0 for p in ax.patches)
        plt.close(fig)
        return y                                             # inverted axis: smaller = higher
    assert highest(-0.4) > highest(0.0)                      # trench behind -> wall stops short


def test_values_outside_an_explicit_norm_are_clipped():
    r = upright({"Fe": -5.0, "Co": 0.5, "Ni": 9.0}, cmap_norm=(0.0, 1.0))
    assert _text_y(r, "Ni") < _text_y(r, "Co") < _text_y(r, "Fe")
    plt.close(r.fig)


def test_bad_depth_raises():
    with pytest.raises(ValueError):
        periodic_table_relief({"Fe": 1.0}, depth=0.0)


def test_compose_into_existing_axes():
    fig, (axa, axb) = plt.subplots(1, 2)
    flat = pp.periodic_table({"Fe": 1.0}, ax=axa, colorbar=False)
    relief = periodic_table_relief({"Fe": 1.0}, ax=axb, colorbar=False)
    assert flat.ax is axa and relief.ax is axb and relief.fig is fig
    plt.close(fig)


def test_all_optional_fields_draw():
    r = upright({"Fe": 1.0, "Ce": 2.0}, show_at_number=True, show_at_mass=True,
                show_name=True, show_group_period=True)
    texts = {t.get_text() for t in r.ax.texts}
    assert {"26", "Iron", "La-Lu", "18"}.issubset(texts)
    plt.close(r.fig)


def test_all_optional_fields_draw_on_the_surface():
    r = periodic_table_relief({"Fe": 1.0, "Ce": 2.0}, show_at_number=True,
                             show_at_mass=True, show_name=True, draw_missing=False,
                             colorbar=False)
    # symbol + value + number + mass + name, for both elements
    assert len(_labels(r)) == 10
    plt.close(r.fig)


def test_save_is_vector(tmp_path):
    # the table itself is pure vector -- with no colourbar the PDF holds no
    # images at all.  The colourbar gradient is the one raster matplotlib
    # emits (plus a soft mask, since the bar is clipped to its rounded frame),
    # so count against the bare figure rather than a magic number.
    bare = tmp_path / "bare.pdf"
    periodic_table_relief({"Fe": 1.0, "O": 2.0}, savepath=str(bare),
                          colorbar=False)
    assert bare.exists() and bare.stat().st_size > 0
    assert b"/Subtype /Image" not in bare.read_bytes()
    out = tmp_path / "relief.pdf"
    periodic_table_relief({"Fe": 1.0, "O": 2.0}, savepath=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_shade_and_mix():
    assert _shade("white", 0.5) == (0.5, 0.5, 0.5)
    assert _shade("white", 1.0) == "white"                   # untouched when not shading
    assert _mix("black", "white", 0.5) == (0.5, 0.5, 0.5)


def test_tile_style_flat_dispatches_to_the_relief_renderer():
    r = pp.periodic_table_3d({"Fe": -1.0, "O": 2.0}, tile_style="flat",
                             cmap_norm="diverging", relief_signed=True, colorbar=False)
    assert r.mappable is not None
    plt.close(r.fig)


def test_unknown_style_raises():
    import pytest as _pt
    with _pt.raises(ValueError, match="tile_style must be"):
        pp.periodic_table_3d({"Fe": 1.0}, tile_style="metallic")


def test_flat_style_is_reachable_only_through_the_dispatcher():
    # the renderer is no longer exported; tile_style="flat" is the way in
    assert not hasattr(pp, "periodic_table_relief")
    r = pp.periodic_table_3d({"Fe": -1.0, "O": 2.0}, tile_style="flat",
                             cmap_norm="diverging", relief_signed=True,
                             colorbar=False)
    assert r.mappable is not None
    assert len(r.ax.images) == 0                 # no rasters at all
    plt.close(r.fig)


def test_each_style_keeps_its_own_relief_height_default():
    # the two projections need different heights for the same apparent relief
    import periodicplots.relief as _rel
    seen = {}
    orig = _rel._draw_block
    def spy(ax, x, y, z, *a, **k):
        seen[(x, y)] = z
        return orig(ax, x, y, z, *a, **k)
    _rel._draw_block = spy
    try:
        r = pp.periodic_table_3d({"H": 0.0, "F": 1.0}, tile_style="flat",
                                 colorbar=False, draw_missing=False)
    finally:
        _rel._draw_block = orig
    plt.close(r.fig)
    assert max(seen.values()) == pytest.approx(1.0)      # flat default


def test_flat_without_data_says_so_clearly():
    # "3d" has a no-data (chemical-family) mode, "flat" does not -- it used to
    # fail deep inside _value_dict with a TypeError naming the wrong function
    with pytest.raises(ValueError, match='tile_style="flat" needs data'):
        pp.periodic_table_3d(tile_style="flat")
    pp.periodic_table_3d()                       # "3d" still fine with no data
    plt.close("all")


def test_version_is_consistent():
    import tomllib
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    meta = tomllib.loads((root / "pyproject.toml").read_text())
    assert pp.__version__ == meta["project"]["version"]
