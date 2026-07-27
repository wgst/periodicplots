"""periodicplots — vector periodic-table heatmaps in matplotlib.

    import periodicplots as pp
    pp.periodic_table({"Fe": 1.2, "O": 3.4, "Si": 1.1}, label="my property")

The renderer is pure matplotlib (Rectangles + text), so figures stay fully
vector and compose into multi-panel layouts via the ``ax=`` argument.
"""
from .core import PeriodicTablePlot, periodic_table
from ._elements import ELEMENTS, SYMBOL_TO_Z

__all__ = ["periodic_table", "PeriodicTablePlot", "ELEMENTS", "SYMBOL_TO_Z"]
__version__ = "0.1.0"
