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
    r = pp.periodic_table({"Fe": -0.8, "O": 0.3}, norm="diverging", colorbar=False)
    assert r.mappable.norm.vmin == -0.8 and r.mappable.norm.vmax == 0.8
    plt.close(r.fig)


def test_save(tmp_path):
    out = tmp_path / "t.pdf"
    pp.periodic_table({"Fe": 1.0}, savepath=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_group_period_labels():
    r = pp.periodic_table({"Fe": 1.0}, show_group_period=True, colorbar=False)
    texts = {t.get_text() for t in r.ax.texts}
    assert {"1", "18", "7"}.issubset(texts)      # group + period labels present
    plt.close(r.fig)
