"""Basic usage: a single periodic-table heatmap, saved to PDF (vector)."""
import periodicplots as pp

# values keyed by element symbol (atomic numbers work too)
values = {
    "H": 4.9, "Li": 2.1, "Be": 5.7, "B": 3.8, "C": 3.5, "N": 3.5, "O": 2.7,
    "F": 3.7, "Na": 3.2, "Mg": 3.1, "Al": 4.5, "Si": 3.8, "P": 2.7, "S": 2.1,
    "Cl": 3.3, "K": 3.3, "Ca": 3.3, "Sc": 3.5, "Ti": 2.4, "Fe": 0.9, "Cu": 1.7,
    "Zn": 3.1, "Ga": 3.1, "Ge": 2.6, "Se": 1.5, "Br": 2.9, "Ce": 1.9, "Gd": 3.1,
}

# default look == the reference figure (symbol + value + viridis + colourbar)
pp.periodic_table(values, label="mean $E_g$ (eV)",
                  savepath="basic.pdf")           # -> examples/basic.pdf
print("wrote basic.pdf")
