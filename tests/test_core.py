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


def test_mismatched_lengths_raise():
    # zip would silently drop the unmatched tail -- a dropped element is easy
    # to miss in a busy figure, so it must fail loudly instead
    with pytest.raises(ValueError, match="length"):
        _value_dict(["Fe", "O", "Si"], [1.0, 2.0])


def test_unknown_kwargs_are_rejected_without_savepath():
    # extra keywords are forwarded to savefig, which only runs with savepath=;
    # without one they can only be typos or options of the other renderer
    with pytest.raises(TypeError, match="cmap_nrom"):
        pp.periodic_table({"Fe": 1.0}, cmap_nrom="diverging")
    with pytest.raises(TypeError, match="tilt"):
        pp.periodic_table_3d({"Fe": 1.0}, tile_style="flat", tilt=0.3)
    with pytest.raises(TypeError, match="glow"):
        pp.periodic_table_3d({"Fe": 1.0}, glow=True)


def test_element_data_spot_checks():
    # bracketed-mass convention: the most stable isotope of each unstable
    # element (Rn once carried thoron's 220)
    assert pp.ELEMENTS[86][2] == 222.0           # Rn
    assert pp.ELEMENTS[84][2] == 209.0           # Po
    assert pp.ELEMENTS[103][2] == 266.0          # Lr


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


def test_cbar_loc_is_the_same_option_in_both_functions():
    # "gap" is the default everywhere: an inset axes inside the Be-B block,
    # so it is a child of the table axes rather than a separate figure axes
    for call in (lambda **k: pp.periodic_table({"Fe": 1.0, "O": 2.0}, **k),
                 lambda **k: pp.periodic_table_3d({"Fe": 1.0, "O": 2.0}, **k),
                 lambda **k: pp.periodic_table_3d({"Fe": 1.0, "O": 2.0},
                                                  tile_style="flat", **k)):
        r = call()
        assert r.ax.child_axes                   # an inset, inside the table
        band = r.ax.child_axes[0].get_position()
        assert band.height < band.width          # a thin horizontal bar
        plt.close(r.fig)
        # the outside placements are matplotlib's own, on their own axes
        for loc in ("right", "left", "top", "bottom"):
            r = call(cbar_loc=loc)
            assert not r.ax.child_axes and len(r.fig.axes) > 1
            plt.close(r.fig)
        with pytest.raises(ValueError, match="cbar_loc"):
            call(cbar_loc="middle")


def test_cbar_loc_beats_a_conflicting_cbar_kw_location():
    # cbar_kw is for the rest (fraction, pad, shrink); a "location" smuggled
    # through it must not override the validated cbar_loc argument
    r = pp.periodic_table({"Fe": 1.0, "O": 2.0}, cbar_loc="left",
                          cbar_kw={"location": "right"})
    cax = next(a for a in r.fig.axes if a is not r.ax)
    assert cax.get_position().x0 < r.ax.get_position().x0   # bar sits left
    plt.close(r.fig)


def test_gap_colorbar_follows_the_projection():
    # in relief the band leans and squashes with the table, so it sits in the
    # plane rather than floating flat over it
    flat = pp.periodic_table({"Fe": 1.0, "O": 2.0})
    tipped = pp.periodic_table_3d({"Fe": 1.0, "O": 2.0})
    fb = flat.ax.child_axes[0].get_position()
    tb = tipped.ax.child_axes[0].get_position()
    assert tb.y0 != pytest.approx(fb.y0, abs=1e-3)
    plt.close(flat.fig), plt.close(tipped.fig)


def test_gap_bar_is_the_same_weight_in_every_renderer():
    # the band's ends follow each projection, but its thickness is measured
    # against the cell WIDTH -- which nothing foreshortens -- so the bar does
    # not come out thinner in the tipped views
    def bar(r):
        r.fig.canvas.draw()
        bb = r.ax.child_axes[0].get_window_extent()
        T = r.ax.transData
        cw = abs(T.transform((1, 0))[0] - T.transform((0, 0))[0])
        return bb.width / cw, bb.height / cw
    v = {"Fe": 1.0, "O": 2.0}
    ref = bar(pp.periodic_table(v))
    for made in (pp.periodic_table(v, tile_style="3d"),
                 pp.periodic_table_3d(v),
                 pp.periodic_table_3d(v, tile_style="flat")):
        w, h = bar(made)
        assert w == pytest.approx(ref[0], abs=0.1)
        assert h == pytest.approx(ref[1], abs=0.02)
        plt.close(made.fig)


def test_accepts_a_pandas_series():
    # core._value_dict duck-types anything with .items(), and the docstring
    # promises pandas Series -- the `dev` extra carries pandas for this
    pd = pytest.importorskip("pandas")
    s = pd.Series({"Fe": 1.0, "O": 2.0, "Si": 3.0})
    for make in (lambda: pp.periodic_table(s, colorbar=False),
                 lambda: pp.periodic_table_3d(s, colorbar=False),
                 lambda: pp.periodic_table_3d(s, tile_style="flat",
                                              colorbar=False)):
        r = make()
        assert r.mappable is not None
        plt.close(r.fig)
