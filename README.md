# periodicplots

Vector periodic-table heatmaps in **matplotlib**.

Colour each element by any property and get a clean, publication-quality
periodic table. Every cell is a matplotlib `Rectangle` + text — **no
rasterisation** — so the output stays fully vector (crisp PDF/SVG) and drops
straight into multi-panel figures via an `ax=` argument.

![periodicplots example](docs/example.png)

- 🎨 any matplotlib colormap; sequential, or symmetric **diverging** about 0
- 🧩 **composable** — draw into your own axes, keep vector quality
- 🪶 lightweight — only `matplotlib` + `numpy` (element data is bundled, no `pymatgen`)
- 🔤 optional per-cell atomic **number**, **mass** and full **name**
- 💾 save to any format (PDF/SVG/PNG) or use the returned figure

## Install

```bash
pip install periodicplots
# or, from a clone:
pip install -e .
```

## Quick start

```python
import periodicplots as pp

# values keyed by symbol (atomic numbers work too)
pp.periodic_table(
    {"Fe": 0.9, "O": 2.7, "Si": 3.8, "Cu": 1.7, "H": 4.9},
    label="mean $E_g$ (eV)",
    savepath="gap.pdf",          # vector output
)
```

By default each cell shows the **element symbol and the value**, viridis
colouring and a slim colourbar — the reference look.

### Two ways to pass data

```python
pp.periodic_table({"Fe": 0.9, "O": 2.7})     # mapping {symbol|Z: value}
pp.periodic_table([26, 8], [0.9, 2.7])       # parallel sequences (Z or symbol)
```

### Diverging property (centred at 0)

```python
pp.periodic_table(slopes, cmap="RdBu_r", norm="diverging",
                  label="d$E_g$/dT (meV/K)")
```

![diverging example](docs/example_diverging.png)

### Optional atomic number / mass / name

```python
pp.periodic_table(values, show_number=True, show_mass=True, show_name=True)
```

### Compose into a multi-panel figure (stays vector)

```python
import matplotlib.pyplot as plt
fig, (axa, axb) = plt.subplots(1, 2, figsize=(15, 4))

pp.periodic_table(gap,   ax=axa, cmap="viridis",  label="$E_g$ (eV)")
pp.periodic_table(slope, ax=axb, cmap="RdBu_r", norm="diverging",
                  label="d$E_g$/dT (meV/K)")

fig.savefig("panel.pdf", bbox_inches="tight")    # no quality loss
```

`periodic_table(...)` returns a `PeriodicTablePlot` with `.fig`, `.ax` and
`.mappable` (use the mappable to place your own colourbar when composing).

## Key options

| argument | default | meaning |
|---|---|---|
| `cmap` | `"viridis"` | colormap name or instance |
| `norm` | `None` | `None` (auto min–max), `"diverging"`, `(vmin, vmax)`, or a `Normalize` |
| `value_fmt` | `"{:.2f}"` | format of the in-cell value |
| `label` | `None` | colourbar label |
| `ax` | `None` | draw into an existing axes (compose) |
| `show_value` | `True` | print the value in each cell |
| `show_number` / `show_mass` / `show_name` | `False` | extra per-cell text |
| `draw_missing` | `True` | draw elements without a value in `missing_color` |
| `colorbar` | `True` | attach a colourbar |
| `savepath` | `None` | save the figure (extension picks the format) |

## Examples

See [`examples/`](examples/): `example_basic.py`, `example_options.py`,
`example_compose.py`.

## License

MIT
