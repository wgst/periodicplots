"""Show the optional per-cell fields: atomic number, mass and full name."""
import periodicplots as pp

values = {"Fe": 0.9, "O": 2.7, "Si": 3.8, "Cu": 1.7, "H": 4.9, "Ce": 1.9}

pp.periodic_table(
    values,
    label_cbar="mean $E_g$ (eV)",
    show_at_number=True,      # atomic number in the top-left corner
    show_at_mass=True,        # atomic mass in the top-right corner
    show_name=True,        # full element name along the bottom
    value_fmt="{:.1f}",
    savepath="options.pdf",
)
print("wrote options.pdf")
