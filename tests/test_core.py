import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import periodicplots as pp
from periodicplots.core import _cell_pos, _to_Z, _value_dict


def test_symbol_and_Z_inputs_equivalent():
    a = _value_dict({"Fe": 1.0, "O": 2.0}, None)
    b = _value_dict({26: 1.0, 8: 2.0}, None)
    assert a == b == {26: 1.0, 8: 2.0}


def test_two_sequence_input():
    assert _value_dict(["Fe", "O"], [1.0, 2.0]) == {26: 1.0, 8: 2.0}


def test_symbol_case_insensitive():
    assert _to_Z("fe") == _to_Z("FE") == _to_Z("Fe") == 26


def test_bad_symbol_raises():
    with pytest.raises(KeyError):
        _to_Z("Xx")


def test_fblock_detached_rows():
    # lanthanoids / actinoids land in the detached rows, not the main grid
    assert _cell_pos(57, 3, 6)[1] == 8.5      # La
    assert _cell_pos(92, 3, 7)[1] == 9.5      # U


def test_returns_axes_and_mappable():
    r = pp.periodic_table({"Fe": 1.0, "O": 2.0}, colorbar=False)
    assert r.ax is not None and r.mappable is not None
    plt.close(r.fig)


def test_compose_into_existing_axes():
    fig, ax = plt.subplots()
    r = pp.periodic_table({"Fe": 1.0}, ax=ax, colorbar=False)
    assert r.ax is ax and r.fig is fig
    plt.close(fig)


def test_diverging_norm_symmetric():
    r = pp.periodic_table({"Fe": -0.8, "O": 0.3}, cmap_norm="diverging", colorbar=False)
    assert r.mappable.norm.vmin == -0.8 and r.mappable.norm.vmax == 0.8
    plt.close(r.fig)


def test_save(tmp_path):
    out = tmp_path / "t.pdf"
    pp.periodic_table({"Fe": 1.0}, savepath=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_max_z_omits_superheavies():
    r = pp.periodic_table({"Fe": 1.0, "U": 2.0}, max_z=103, colorbar=False)
    syms = {t.get_text() for t in r.ax.texts}
    assert "U" in syms and "Rf" not in syms and "Og" not in syms
    plt.close(r.fig)


def test_group_period_labels():
    r = pp.periodic_table({"Fe": 1.0}, show_group_period=True, colorbar=False)
    texts = {t.get_text() for t in r.ax.texts}
    assert {"1", "18", "7"}.issubset(texts)      # group + period labels present
    plt.close(r.fig)


def test_tile_style_3d_swaps_flat_cells_for_tiles():
    # tile_style is cosmetic: "3d" renders shadowed tiles (raster images);
    # the default stays flat and fully vector -- rounded FancyBbox cells,
    # or plain sharp Rectangles with tile_shape="square"
    from matplotlib.patches import FancyBboxPatch, Rectangle
    flat = pp.periodic_table({"Fe": 1.0, "O": 2.0}, colorbar=False)
    assert any(isinstance(p, FancyBboxPatch) for p in flat.ax.patches)
    assert len(flat.ax.images) == 0              # rounded but still vector
    sharp = pp.periodic_table({"Fe": 1.0, "O": 2.0}, tile_shape="square",
                              colorbar=False)
    assert not any(isinstance(p, FancyBboxPatch) for p in sharp.ax.patches)
    assert any(type(p) is Rectangle for p in sharp.ax.patches)
    tiled = pp.periodic_table({"Fe": 1.0, "O": 2.0}, tile_style="3d",
                              colorbar=False)
    assert len(tiled.ax.images) > 0
    plt.close(flat.fig), plt.close(sharp.fig), plt.close(tiled.fig)


def test_tile_style_keeps_core_options():
    # the tile look must not swallow periodic_table's own features -- the
    # regression whole-function delegation to style3d would have caused
    r = pp.periodic_table({"Fe": 1.0}, tile_style="3d", show_group_period=True,
                          colorbar=False)
    texts = {t.get_text() for t in r.ax.texts}
    assert {"1", "18", "7"}.issubset(texts)      # group/period labels intact
    plt.close(r.fig)


def test_bad_tile_style_raises():
    with pytest.raises(ValueError, match="tile_style"):
        pp.periodic_table({"Fe": 1.0}, tile_style="metallic")
    with pytest.raises(ValueError, match="tile_shape"):
        pp.periodic_table({"Fe": 1.0}, tile_shape="oval")


def test_tile_shape_round_flat_cells_stay_vector():
    # rounded flat cells: FancyBbox patches, edge colour honoured, no rasters
    from matplotlib.patches import FancyBboxPatch
    r = pp.periodic_table({"Fe": 1.0}, tile_shape="round", colorbar=False)
    boxes = [p for p in r.ax.patches if isinstance(p, FancyBboxPatch)]
    assert boxes and len(r.ax.images) == 0
    plt.close(r.fig)


def test_tile_text_maps_onto_the_face_not_the_cell():
    # a 3d tile's face is smaller than its cell and sits on a lip, so labels
    # are scaled onto the face and dropped by half the lip; flat cells map 1:1
    kw = dict(show_at_number=True, colorbar=False)
    flat = pp.periodic_table({"Fe": 1.0}, **kw)
    tile = pp.periodic_table({"Fe": 1.0}, tile_style="3d", **kw)
    pos = lambda r, s: next(t.get_position() for t in r.ax.texts
                            if t.get_text() == s)
    # Fe sits at (8, 4); flat offsets are exactly the written ones
    assert pos(flat, "Fe")[1] == pytest.approx(4 - 0.18)
    assert pos(flat, "26") == pytest.approx((8 - 0.44, 4 - 0.44))
    # on a tile every label moves DOWN the page and IN from the corner
    assert pos(tile, "Fe")[1] > pos(flat, "Fe")[1]
    assert pos(tile, "26")[0] > pos(flat, "26")[0]     # in from the left edge
    assert pos(tile, "26")[1] > pos(flat, "26")[1]     # down from the top edge
    plt.close(flat.fig), plt.close(tile.fig)
