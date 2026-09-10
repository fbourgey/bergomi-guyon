# Bergomi–Guyon Expansion

[![CI](https://github.com/fbourgey/bergomi-guyon/actions/workflows/ci.yml/badge.svg)](https://github.com/fbourgey/bergomi-guyon/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-6340ac.svg)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Python research code accompanying *Demystifying the Bergomi–Guyon expansion*.
The package computes the universal forest-expansion coefficients of the total
implied variance smile using exact rational arithmetic. It includes the
recursion, independent verification, coefficients through order six, and a
worked tutorial.

## References

- Bourgey, F., & Gatheral, J. (2026). *Demystifying the Bergomi–Guyon expansion*. Manuscript.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and requires Python 3.12 or
newer. Clone the repository and install from the lockfile:

```bash
git clone https://github.com/fbourgey/bergomi-guyon.git
cd bergomi-guyon
uv sync --locked
```

This installs the package in editable mode, together with the test and notebook
dependencies. Useful commands, run from the repository root:

```bash
uv run pytest -q
uv run pytest -q tests/bergomi_guyon/test_bg_coefficients.py
uv run generate-bg-coefficients --order 6 --verify --quiet
uv run jupyter lab
```

To install the package into an existing Python environment, use `pip install .`.
All dependencies, including SymPy, JupyterLab, and pytest, are installed
automatically. The core generator and verifier themselves use only the Python
standard library.

## Reproducing Results

Open the tutorial with Jupyter and select the managed environment's Python
kernel. Run its cells in order.

| Notebook | Purpose |
| --- | --- |
| [`bergomi_guyon_recursion.ipynb`](bergomi_guyon_recursion.ipynb) | Worked second-order recursion, an order-three extension, and exact checks of the PDE, boundary conditions, and matching identity |

To reproduce and independently verify the coefficients through order six:

```bash
uv run generate-bg-coefficients --order 6 --verify --format text
```

The test suite also checks the symbolic identities and freshness of the saved
Python and text coefficients. Regenerate these artifacts with:

```bash
uv run generate-bg-coefficients --order 6 --verify --format python --output src/bergomi_guyon/bg_coefficients_order_6.py
uv run generate-bg-coefficients --order 6 --verify --format text --output src/bergomi_guyon/bg_coefficients_order_6.txt
```

`--output` overwrites the named file and requires its parent directory to exist.
Status messages go to stderr. `--quiet` suppresses coefficient output to stdout;
an explicit output file is still written. The CLI is also available as
`uv run python -m bergomi_guyon`; use `--help` for all options.

### Using the coefficients in Python

```python
from bergomi_guyon import generate_coefficients, verify

result = generate_coefficients(6)
verify(result)
a_6 = result.coefficients[6]
```

`result` provides `trees`, `boundaries`, `sources`, and `coefficients`, indexed
from zero to the requested order. Each coefficient maps a tuple of diamond
trees to a polynomial in `zeta` and `theta` with exact `Fraction` coefficients.
`verify(result)` returns `None` on success and raises `AssertionError` if a
consistency check fails.

To use the saved symbolic expressions without regenerating them:

```python
from bergomi_guyon.bg_coefficients_order_6 import M, a, in_k_and_M

a_2 = in_k_and_M(a[2])
```

The expansion is for **total implied variance**,
`Sigma(k) = T * sigma_BS(k, T)**2 = M + sum(epsilon**ell * a_ell(k))`.
The formal variables are `zeta = 1/2 + k/M` and `theta = 1/M`. The correction
at index zero is empty (`a[0]` is `None` in the saved module); the leading term
is `M`. A model supplies the numerical diamond-tree values. After substitution,
set `epsilon = 1` and obtain volatility as `sqrt(Sigma / T)` for positive
maturity and total variance.

The generator supports higher orders, with rapidly increasing time and memory
cost. Routine regression checks cover orders through six. Formal verification
does not establish convergence or positivity of a truncated numerical smile.

## Repository Map

| File | Contents |
| --- | --- |
| `src/bergomi_guyon/recursion.py` | Sparse exact algebra, diamond trees, and coefficient recursion |
| `src/bergomi_guyon/verify.py` | PDE, boundary, degree, tree-completeness, and independent matching checks |
| `src/bergomi_guyon/render.py` | Text, LaTeX, and SymPy exporters |
| `src/bergomi_guyon/cli.py` | `generate-bg-coefficients` command-line interface |
| `src/bergomi_guyon/bg_coefficients_order_6.py` | Importable SymPy coefficients through order six |
| `src/bergomi_guyon/bg_coefficients_order_6.txt` | Readable exact coefficients through order six |
| `bergomi_guyon_recursion.ipynb` | Mathematical tutorial |
| `tests/bergomi_guyon/` | Algebra, verification, exports, artifact freshness, tutorial, and installation tests |

## License

This project is distributed under the [MIT License](LICENSE).
