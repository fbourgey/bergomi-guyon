# Bergomi–Guyon Expansion

[![CI](https://github.com/fbourgey/bergomi-guyon/actions/workflows/ci.yml/badge.svg)](https://github.com/fbourgey/bergomi-guyon/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Python research code for the Bergomi–Guyon expansion and magic-strike contract
approximations. Includes the exact coefficient recursion, coefficients through
order six, magic-strike approximations through order five, and research notebooks.

## References

This repository reproduces results from:

- Bourgey, F., & Gatheral, J. (2026). *Demystifying the Bergomi–Guyon expansion*.
  Available at [SSRN 7468158](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7468158)
  and [arXiv:2609.17869](https://arxiv.org/abs/2609.17869).

- Bourgey, F., & Gatheral, J. (2026). *Magic strikes for variance and gamma       contracts, and other attainable claims*. Forthcoming.

## Notebooks

### Google Colab (no local installation)

Choose a notebook below, click `Open In Colab`, then select `Runtime → Run all`.
Each notebook installs the package from GitHub automatically when running in Colab.

| Notebook | Contents | Google Colab |
| --- | --- | --- |
| [bergomi_guyon_recursion.ipynb](bergomi_guyon_recursion.ipynb) | Coefficient recursion through order three, with exact verification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fbourgey/bergomi-guyon/blob/main/bergomi_guyon_recursion.ipynb) |
| [heston.ipynb](heston.ipynb) | Heston variance, gamma, and implied power contracts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fbourgey/bergomi-guyon/blob/main/heston.ipynb) |
| [rough_bergomi.ipynb](rough_bergomi.ipynb) | Rough Bergomi magic-strike, Fukasawa, and Monte Carlo comparisons | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fbourgey/bergomi-guyon/blob/main/rough_bergomi.ipynb) |
| [market_swap_estimates.ipynb](market_swap_estimates.ipynb) | SPX variance and gamma estimates | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fbourgey/bergomi-guyon/blob/main/market_swap_estimates.ipynb) |

### Local installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). From a terminal:

```bash
git clone https://github.com/fbourgey/bergomi-guyon.git
cd bergomi-guyon
uv sync --locked
uv run jupyter lab
```

Open any of the notebooks in the table from JupyterLab.

### Magic-strike experiments

Use `set_id = 1` or `set_id = 2` in the model notebooks. The rough Bergomi
notebook defaults to 300,000 paths, 300 time steps, 15 maturities, and five
batches; reduce `n_mc`, `n_disc`, `n_batch`, and `Ts` for a shorter run.

Four SPX snapshots are included in [data/](data/). The market notebook loads
them from the repository locally, or directly from GitHub in Colab.
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
[bg_coefficients_order_6.txt](src/bergomi_guyon/bg_coefficients_order_6.txt)
and available as a generated
[Python module](src/bergomi_guyon/bg_coefficients_order_6.py).

To regenerate and verify coefficients through order six:

```bash
uv run generate-bg-coefficients --order 6 --verify --quiet
```

Verification uses exact rational arithmetic to check the PDE, boundary data,
tree weights and completeness, degree structure, and original matching identity.
The CLI supports `--format text`, `latex`, or `python`, and `--output PATH` to
write a file. LaTeX output requires the paper's tree macros, which are not bundled
here.

## Repository contents

- [src/bergomi_guyon/](src/bergomi_guyon/): exact coefficient generation,
  verification, rendering, and the command-line interface.
- [src/magic_strikes/](src/magic_strikes/): Heston and rough Bergomi models,
  magic-strike approximations for variance, gamma, and power contracts through
  order five, Fukasawa quadrature, and market-smile estimates.
- [data/](data/): four SPX implied-volatility snapshots used by the market notebook.
- [mathematica/](mathematica/): the `MagicStrikes.nb` notebook and `aBG.m`
  coefficient expressions.
- [tests/](tests/): tests for both Python packages.

## Development checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

CI runs these checks and exact coefficient verification through order six on
Python 3.12, 3.13, and 3.14.

## License

[MIT](LICENSE)
