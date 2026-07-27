"""Compose two periodic tables into one multi-panel figure (stays vector).

Pass an existing ``ax`` so the table draws into your own layout; use the
returned ``.mappable`` if you want to place the colourbar yourself.
"""
import matplotlib.pyplot as plt
import periodicplots as pp

gap = {"H": 4.9, "C": 3.5, "O": 2.7, "Si": 3.8, "Fe": 0.9, "Cu": 1.7, "Ce": 1.9}
slope = {"H": -0.77, "C": -0.43, "O": -0.36, "Si": -0.45, "Fe": -0.18, "Cu": -0.16}

fig, (axa, axb) = plt.subplots(1, 2, figsize=(15, 4))

pp.periodic_table(gap, ax=axa, cmap="viridis",
                  label="mean $E_g$ (eV)")
pp.periodic_table(slope, ax=axb, cmap="RdBu_r", norm="diverging",
                  label="mean d$E_g$/dT (meV/K)")

fig.savefig("compose.pdf", bbox_inches="tight")   # fully vector, no rasterisation
print("wrote compose.pdf")
