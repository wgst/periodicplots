import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import periodicplots as pp
from periodicplots.style3d import _family, _shadow_image


def test_poster_mode_no_data():
    r = pp.periodic_table_3d()
    assert r.ax is not None and r.mappable is None
    syms = {t.get_text() for t in r.ax.texts}
    assert {"H", "Fe", "Og"}.issubset(syms)
    plt.close(r.fig)


def test_value_mode_has_mappable():
    r = pp.periodic_table_3d({"Fe": 1.0, "O": 2.0}, colorbar=False)
    assert r.mappable is not None
    plt.close(r.fig)


def test_compose_into_existing_axes():
    fig, ax = plt.subplots()
    r = pp.periodic_table_3d(ax=ax)
    assert r.ax is ax and r.fig is fig
    plt.close(fig)


def test_save(tmp_path):
    out = tmp_path / "t.png"
    pp.periodic_table_3d(max_z=20, savepath=str(out), dpi=50)
    assert out.exists() and out.stat().st_size > 0


def test_family_assignment():
    assert _family(3, 1, 2) == "alkali"
    assert _family(26, 8, 4) == "late_tm"
    assert _family(60, 3, 6) == "lanthanoid"
    assert _family(8, 16, 2) == "nonmetal"


def test_poster_cmap_registered_and_default():
    assert plt.get_cmap("poster") is not None
    assert plt.get_cmap("poster_r") is not None
    r = pp.periodic_table_3d({"Fe": 1.0, "O": 2.0}, colorbar=False)
    assert r.mappable.get_cmap().name == "poster"
    plt.close(r.fig)


def test_poster_cmap_usable_in_core_style():
    r = pp.periodic_table({"Fe": 1.0, "O": 2.0}, cmap="poster", colorbar=False)
    assert r.mappable.get_cmap().name == "poster"
    plt.close(r.fig)


def test_relief_tilts_table_back():
    # in relief mode the only FancyBbox patches are the top faces (walls are
    # PathPatches); F (high) drawn before Cs (low)
    from matplotlib.patches import FancyBboxPatch
    kw = dict(draw_missing=False, colorbar=False, background=False)
    r = pp.periodic_table_3d({"F": 4.0, "Cs": 0.8}, relief_height=0.6, **kw)
    tops = [p for p in r.ax.patches if isinstance(p, FancyBboxPatch)]
    f_top, cs_top = tops[0], tops[1]
    squash = 1.0 - 0.20
    f_rise = (2 - 0.44) * squash - f_top.get_y()   # block height above ground
    cs_rise = (6 - 0.44) * squash - cs_top.get_y()
    assert f_rise > cs_rise + 0.3                  # higher value -> taller
    assert f_top.get_height() < 0.8                # foreshortened top
    flat = pp.periodic_table_3d({"F": 4.0, "Cs": 0.8}, relief_height=0.0, **kw)
    flats = [p for p in flat.ax.patches if isinstance(p, FancyBboxPatch)]
    assert flats[1].get_height() > 0.9             # straight-on face untouched
    plt.close(r.fig)
    plt.close(flat.fig)


def _body_x(res, idx):
    """Data-space x of the first outline vertex of the idx-th PathPatch
    (block bodies and painted text are PathPatches; the body comes first
    per tile, and its first vertex sits on the ground front-left edge)."""
    from matplotlib.patches import PathPatch
    pps = [p for p in res.ax.patches if type(p) is PathPatch]
    return pps[idx].get_path().vertices[0][0]


def test_relief_without_data_raises():
    import pytest
    with pytest.raises(ValueError):
        pp.periodic_table_3d(relief_height=0.6)


def test_side_tilt_yaws_table_and_adds_side_walls():
    from matplotlib.patches import Polygon
    kw = dict(draw_missing=False, colorbar=False, background=False)
    r = pp.periodic_table_3d({"F": 4.0, "Cs": 0.8}, relief_height=0.6, **kw)
    straight = pp.periodic_table_3d({"F": 4.0, "Cs": 0.8}, relief_height=0.6,
                                    side_tilt=0.0, **kw)
    # the darker side-region overlay exists only in the yawed view (the
    # left-gap occlusion strips are Polygons too, present in both views)
    n_on = sum(isinstance(p, Polygon) for p in r.ax.patches)
    n_off = sum(isinstance(p, Polygon) for p in straight.ax.patches)
    assert n_on == n_off + 2
    # the whole table leans: block bodies shift left by side_tilt * depth,
    # and Cs (4 rows deeper than F) leans further
    # per tile PathPatches: [body, number, symbol, mass]
    f_shift = _body_x(r, 0) - _body_x(straight, 0)
    cs_shift = _body_x(r, 4) - _body_x(straight, 4)
    assert f_shift < -0.25
    assert cs_shift < f_shift - 0.3
    plt.close(r.fig)
    plt.close(straight.fig)


def test_missing_tiles_share_one_grey():
    from matplotlib.patches import FancyBboxPatch
    r = pp.periodic_table_3d({"Fe": 1.0}, relief_height=0.0, colorbar=False,
                             background=False)
    boxes = [p for p in r.ax.patches if isinstance(p, FancyBboxPatch)]
    faces = {p.get_facecolor() for p in boxes[1::2]}   # flat mode: body, face
    assert len(faces) == 2                             # one grey + Fe's colour
    plt.close(r.fig)


def test_equal_values_share_exact_colour():
    from matplotlib.patches import FancyBboxPatch
    r = pp.periodic_table_3d({"At": 2.2, "Rn": 2.2}, relief_height=0.0, colorbar=False,
                             background=False, draw_missing=False)
    boxes = [p for p in r.ax.patches if isinstance(p, FancyBboxPatch)]
    at_face, rn_face = boxes[1], boxes[3]          # flat mode: body, face
    assert at_face.get_facecolor() == rn_face.get_facecolor()
    plt.close(r.fig)


def test_elements_filter():
    r = pp.periodic_table_3d(elements=["H", "He", 26])
    syms = {t.get_text() for t in r.ax.texts}
    assert {"H", "He", "Fe"}.issubset(syms) and "O" not in syms
    plt.close(r.fig)


def test_square_style_runs_both_modes():
    r = pp.periodic_table_3d(tile_shape="square", max_z=20)
    plt.close(r.fig)
    r = pp.periodic_table_3d({"F": 4.0, "Cs": 0.8}, tile_shape="square",
                             colorbar=False, draw_missing=False)
    plt.close(r.fig)


def test_unknown_style_raises():
    import pytest
    with pytest.raises(ValueError):
        pp.periodic_table_3d(tile_style="chrome")


def test_shadow_image_is_rgba_alpha_only():
    img = _shadow_image(n=32, radius=4)
    assert img.ndim == 3 and img.shape[2] == 4
    assert img[..., :3].max() == 0.0 and 0.0 < img[..., 3].max() <= 1.0


def test_tiles_are_perfectly_aligned():
    # every tile sits exactly on its grid cell -- tiles are never nudged
    import numpy as np
    def sym_y(r, sym):
        return next(t.get_position()[1] for t in r.ax.texts if t.get_text() == sym)
    for shp in ("round", "square"):
        r = pp.periodic_table_3d(tile_shape=shp, max_z=30)
        assert np.ptp([sym_y(r, s) for s in ("Co", "Ni", "Cu")]) < 1e-12
        plt.close(r.fig)


def test_relief_norm_drives_height_independently_of_colour():
    # diverging colour about zero, height from the MAGNITUDE: the most negative
    # element must stand as tall as the most positive one, not flat
    from matplotlib.colors import TwoSlopeNorm
    from periodicplots import style3d

    def lifts(**kw):
        seen = {}
        orig = style3d._draw_tile
        def spy(ax, c, r, face, *a, **k):
            seen[(round(c, 3), round(r, 3))] = k.get("lift", 0.0)
            return orig(ax, c, r, face, *a, **k)
        style3d._draw_tile = spy
        try:
            res = style3d.periodic_table_3d(
                {"Fe": -1.0, "Co": 0.0, "Ni": 1.0}, max_z=30, colorbar=False,
                cmap_norm=TwoSlopeNorm(0.0, -1.0, 1.0), relief_height=0.6, **kw)
        finally:
            style3d._draw_tile = orig
        plt.close(res.fig)
        return seen                              # keyed by (column, row)

    fe, co, ni = (8.0, 4.0), (9.0, 4.0), (10.0, 4.0)
    by_colour = lifts()                          # height follows the colour norm
    assert by_colour[fe] == pytest.approx(0.0)   # most negative -> flat
    assert by_colour[ni] > by_colour[co] > by_colour[fe]

    by_mag = lifts(relief_norm=lambda v: abs(v))
    assert by_mag[fe] == pytest.approx(by_mag[ni])    # equal magnitude, equal height
    assert by_mag[co] == pytest.approx(0.0)           # zero -> flat


def test_font_scale_shrinks_every_label():
    big = pp.periodic_table_3d({"Fe": 1.0}, relief_height=0.0, max_z=30, colorbar=False)
    small = pp.periodic_table_3d({"Fe": 1.0}, relief_height=0.0, max_z=30, colorbar=False,
                                 font_scale=0.5)
    sz = lambda r: max(t.get_fontsize() for t in r.ax.texts)
    assert sz(small) == pytest.approx(sz(big) * 0.5)
    plt.close(big.fig), plt.close(small.fig)


def test_signed_relief_sinks_negative_tiles():
    # positive lifts the tile out of the table, negative pulls it below: with
    # a norm whose zero sits at the top of the scale, every tile sinks
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from periodicplots import style3d

    def lifts(**kw):
        seen, orig = {}, style3d._draw_tile
        def spy(ax, c, r, face, *a, **k):
            seen[(round(c, 3), round(r, 3))] = k.get("lift", 0.0)
            return orig(ax, c, r, face, *a, **k)
        style3d._draw_tile = spy
        try:
            res = style3d.periodic_table_3d({"Fe": -1.0, "Co": 0.0, "Ni": 1.0},
                                            max_z=30, colorbar=False,
                                            relief_height=0.6, **kw)
        finally:
            style3d._draw_tile = orig
        plt.close(res.fig)
        return seen

    fe, co, ni = (8.0, 4.0), (9.0, 4.0), (10.0, 4.0)
    sg = lifts(cmap_norm=TwoSlopeNorm(0.0, -1.0, 1.0), relief_signed=True)
    assert sg[fe] == pytest.approx(-0.6)      # most negative: full depth down
    assert sg[co] == pytest.approx(0.0)       # zero: flush with the plane
    assert sg[ni] == pytest.approx(0.6)       # most positive: full height up
    # a scale ending at zero puts every value below it
    allneg = lifts(cmap_norm=Normalize(-1.0, 0.0), relief_signed=True)
    assert allneg[fe] < allneg[co] < 0 or allneg[fe] < 0


def test_signed_needs_a_norm_that_places_zero():
    import warnings
    from matplotlib.colors import LogNorm
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # the old path warned before raising
        with pytest.raises(ValueError, match="place 0"):
            pp.periodic_table_3d({"Fe": 1.0, "O": 2.0}, cmap_norm=LogNorm(1.0, 10.0),
                                 relief_height=0.6, relief_signed=True, colorbar=False)


def test_flat_view_limits_follow_the_drawn_rows():
    # a table restricted to the first periods must not keep the full-table
    # ylim and float in empty space
    r = pp.periodic_table_3d(elements=["H", "He", "Li"], relief_height=0.0)
    assert r.ax.get_ylim()[0] < 4.0
    plt.close(r.fig)
    full = pp.periodic_table_3d()
    assert full.ax.get_ylim()[0] > 10.0          # unrestricted: unchanged
    plt.close(full.fig)


def test_values_without_data_raise():
    with pytest.raises(ValueError, match="without data"):
        pp.periodic_table_3d(values=[1.0, 2.0])


def test_fblock_captions_only_when_the_block_is_drawn():
    # a "La-Lu" caption beside an empty row would float outside the trimmed
    # limits; every filter (max_z, elements, draw_missing) must silence it
    r = pp.periodic_table_3d({"Fe": 1.0}, relief_height=0.0, draw_missing=False,
                             fblock_labels=True, colorbar=False)
    assert not any("Lu" in t.get_text() for t in r.ax.texts)
    plt.close(r.fig)
    full = pp.periodic_table_3d(fblock_labels=True)
    assert any("Lu" in t.get_text() for t in full.ax.texts)
    plt.close(full.fig)


def test_ground_shadows_lie_under_every_block_of_their_row():
    # the contact shadow lies on the ground, so it is keyed to the row alone:
    # at the block's own zorder (which grows with column) the halo of a
    # right-hand tile was drawn over the body of the tile to its left
    r = pp.periodic_table_3d({"Fe": 1.0, "Co": 2.0}, relief_height=0.6,
                             draw_missing=False, colorbar=False,
                             background=False)
    row_base = 10 + 60 * 4                       # Fe and Co sit in period 4
    shadows = [im for im in r.ax.images if im.get_zorder() == row_base]
    assert len(shadows) == 2                     # one per tile, both at the row base
    assert row_base < min(p.get_zorder() for p in r.ax.patches)
    plt.close(r.fig)


def test_pit_geometry_replaces_the_standing_body():
    # a sunken tile draws the pit silhouette instead of a block: no ground
    # shadow (a hole casts none) and no left-edge occlusion strips
    from matplotlib.colors import TwoSlopeNorm
    kw = dict(draw_missing=False, colorbar=False, background=False,
              relief_height=0.6, relief_signed=True, cmap_norm=TwoSlopeNorm(0.0, -1.0, 1.0))
    up = pp.periodic_table_3d({"Fe": 1.0}, **kw)
    down = pp.periodic_table_3d({"Fe": -1.0}, **kw)
    assert len(down.ax.images) < len(up.ax.images)     # shadow raster dropped
    from matplotlib.patches import Polygon
    assert sum(isinstance(p, Polygon) for p in down.ax.patches) < \
        sum(isinstance(p, Polygon) for p in up.ax.patches)
    plt.close(up.fig), plt.close(down.fig)
