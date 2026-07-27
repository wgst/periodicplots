"""Generate the README images.

Hero: atomic mass for ALL 118 elements (data bundled with the package, so the
whole table is coloured).  Second: a diverging example.
"""
import periodicplots as pp
from periodicplots import ELEMENTS

# {Z: atomic mass} for every element -> the full table is filled
mass = {Z: meta[2] for Z, meta in ELEMENTS.items()}
pp.periodic_table(mass, label="atomic mass (u)", value_fmt="{:.0f}",
                  figsize=(10, 5.4), savepath="example.png", dpi=200)

# diverging example (property defined for a subset)
slope = {"H": -0.77, "Li": -0.27, "Be": -0.60, "B": -0.47, "C": -0.43, "N": -0.46,
         "O": -0.36, "F": -0.36, "Na": -0.45, "Al": -0.56, "Si": -0.45, "Cl": -0.28,
         "K": -0.39, "Ga": -0.52, "Ge": -0.44, "Cu": -0.16, "Zn": -0.46, "Ag": -0.22,
         "In": -0.41, "Ce": -0.17, "Gd": -0.34}
pp.periodic_table(slope, cmap="RdBu_r", norm="diverging", figsize=(10, 5.4),
                  label="mean d$E_g$/dT (meV/K)", value_fmt="{:+.2f}",
                  savepath="example_diverging.png", dpi=200)
print("wrote docs/example.png (all elements) + docs/example_diverging.png")
