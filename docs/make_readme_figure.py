"""Generate the README hero image (default look) + a diverging example."""
import matplotlib.pyplot as plt
import periodicplots as pp

# a fuller, nicer-looking sample property (mean band gap-ish, eV)
gap = {
    "H": 4.9, "Li": 2.1, "Be": 5.7, "B": 3.8, "C": 3.5, "N": 3.5, "O": 2.7, "F": 3.7,
    "Na": 3.2, "Mg": 3.1, "Al": 4.5, "Si": 3.8, "P": 2.7, "S": 2.1, "Cl": 3.3,
    "K": 3.3, "Ca": 3.3, "Sc": 3.5, "Ti": 2.4, "V": 1.2, "Cr": 1.4, "Mn": 1.6,
    "Fe": 0.9, "Co": 1.3, "Ni": 1.7, "Cu": 1.7, "Zn": 3.1, "Ga": 3.1, "Ge": 2.6,
    "As": 2.3, "Se": 1.5, "Br": 2.9, "Rb": 3.3, "Sr": 2.9, "Y": 3.7, "Zr": 3.3,
    "Nb": 2.3, "Mo": 2.0, "Ag": 2.0, "Cd": 2.8, "In": 2.5, "Sn": 2.5, "Sb": 2.2,
    "Te": 1.5, "I": 2.7, "Cs": 3.2, "Ba": 3.0, "Hf": 3.5, "Ta": 2.9, "W": 1.9,
    "Pb": 2.6, "Bi": 2.3, "La": 3.1, "Ce": 1.9, "Gd": 3.1, "Lu": 3.9, "Th": 4.0,
}

pp.periodic_table(gap, label="mean $E_g$ (eV)", figsize=(10, 5.4),
                  savepath="example.png", dpi=200)

# diverging example
slope = {"H": -0.77, "Li": -0.27, "Be": -0.60, "B": -0.47, "C": -0.43, "N": -0.46,
         "O": -0.36, "F": -0.36, "Na": -0.45, "Al": -0.56, "Si": -0.45, "Cl": -0.28,
         "K": -0.39, "Ga": -0.52, "Ge": -0.44, "Cu": -0.16, "Zn": -0.46, "Ag": -0.22,
         "In": -0.41, "Ce": -0.17, "Gd": -0.34}
pp.periodic_table(slope, cmap="RdBu_r", norm="diverging", figsize=(10, 5.4),
                  label="mean d$E_g$/dT (meV/K)", value_fmt="{:+.2f}",
                  savepath="example_diverging.png", dpi=200)
print("wrote docs/example.png + docs/example_diverging.png")
