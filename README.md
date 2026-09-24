# Bergomi–Guyon Expansion

[![CI](https://github.com/fbourgey/bergomi-guyon/actions/workflows/ci.yml/badge.svg)](https://github.com/fbourgey/bergomi-guyon/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fbourgey/bergomi-guyon/blob/main/bergomi_guyon_recursion.ipynb)

Code accompanying Bourgey, F., & Gatheral, J. (2026), *Demystifying the
Bergomi–Guyon expansion* and *Magic strikes for variance and gamma contracts,
and beyond*. Includes the exact coefficient recursion, coefficients through
order six, magic-strike approximations through order five, and research notebooks.

## Run the notebook

### Google Colab (no local installation)

Click `Open In Colab`, then `Runtime → Run all`. Setup runs automatically.

### Local installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). From a terminal:

```bash
git clone https://github.com/fbourgey/bergomi-guyon.git
cd bergomi-guyon
uv sync --locked
uv run jupyter lab bergomi_guyon_recursion.ipynb
```

## Magic-strike experiments

Open the research notebooks locally with `uv run jupyter lab`:

| Notebook | Contents |
| --- | --- |
| [heston.ipynb](heston.ipynb) | Heston variance, gamma, and implied power variance comparisons |
| [rough_bergomi.ipynb](rough_bergomi.ipynb) | Forward variance curves and Monte Carlo comparisons |
| [market_swap_estimates.ipynb](market_swap_estimates.ipynb) | SPX variance and gamma estimates against a ten-node Fukasawa benchmark |

Use `set_id = 1` or `set_id = 2` in the model notebooks. The rough Bergomi
notebook defaults to 300,000 paths, 300 time steps, 15 maturities, and five
batches; reduce `n_mc`, `n_disc`, `n_batch`, and `Ts` for a shorter run.

Four SPX snapshots are included in [data/](data/). In the market notebook,
replace the absolute CSV path with `f"data/{date}_spx_vol.csv"` to use them.
Create `figures/` before setting `SAVEFIG = True` to export plots.

## Generate coefficients

In a notebook (local or Colab), generate coefficients through order three:

```python
!python -m bergomi_guyon --order 3 --format text
```

Change `--order 3` to compute a different order.

Or, from a local terminal in the repository root:

```bash
uv run generate-bg-coefficients --order 3 --format text
```

The exact coefficients through order six are listed in
[bg_coefficients_order_6.txt](src/bergomi_guyon/bg_coefficients_order_6.txt).

## License

[MIT](LICENSE).
