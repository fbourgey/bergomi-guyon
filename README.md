# Bergomi–Guyon Expansion

[![CI](https://github.com/fbourgey/bergomi-guyon/actions/workflows/ci.yml/badge.svg)](https://github.com/fbourgey/bergomi-guyon/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Code accompanying Bourgey, F., & Gatheral, J. (2026), *Demystifying the
Bergomi–Guyon expansion*. Includes the exact coefficient recursion, coefficients
through order six, and a worked tutorial.

## Run the notebook

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). From a terminal:

```bash
git clone https://github.com/fbourgey/bergomi-guyon.git
cd bergomi-guyon
uv sync --locked
uv run jupyter lab bergomi_guyon_recursion.ipynb
```

## Generate coefficients

From the repository root, generate coefficients through any positive integer order
`N`:

```bash
uv run generate-bg-coefficients --order N --format text
```

The exact coefficients through order six are listed in
[bg_coefficients_order_6.txt](src/bergomi_guyon/bg_coefficients_order_6.txt).

## License

[MIT](LICENSE).
