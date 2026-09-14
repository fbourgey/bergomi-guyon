# Bergomi–Guyon Expansion

[![CI](https://github.com/fbourgey/bergomi-guyon/actions/workflows/ci.yml/badge.svg)](https://github.com/fbourgey/bergomi-guyon/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fbourgey/bergomi-guyon/blob/main/bergomi_guyon_recursion.ipynb)

Code accompanying Bourgey, F., & Gatheral, J. (2026), *Demystifying the
Bergomi–Guyon expansion*. Includes the exact coefficient recursion, coefficients
through order six, and a worked tutorial.

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
