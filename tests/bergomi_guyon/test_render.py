"""Rendering boundaries and zero/constant coefficient exports."""

from fractions import Fraction

import pytest
import sympy as sp

from bergomi_guyon.recursion import ONE, Polynomial, Tree
from bergomi_guyon.render import (
    factor_latex_prefactor,
    polynomial_to_python,
    polynomial_to_text,
    render_coefficients,
    render_latex,
    render_python_module,
    tree_to_latex,
)


@pytest.mark.parametrize("renderer", [render_python_module, render_latex,
                                          lambda a: render_coefficients(a, "text")])
@pytest.mark.parametrize("coefficients", [[], [{}], [{(): ONE}, {}]])
def test_export_rejects_invalid_series_layout(renderer, coefficients):
    with pytest.raises(ValueError, match="empty order zero"):
        renderer(coefficients)


def test_zero_and_constant_exports():
    coefficients = [{}, {}, {(): Polynomial.constant(Fraction(-2, 3))}]
    namespace = {}
    # Execute this package's generated source for fixed test coefficients.
    exec(render_python_module(coefficients), namespace)  # noqa: S102
    assert namespace["a"] == (None, sp.S.Zero, -sp.Rational(2, 3))
    assert namespace["in_k_and_M"](namespace["a"][1]) == 0
    assert "a_1:\n  0" in render_coefficients(coefficients, "text")
    assert "$a_{1}(k) = 0$" in render_latex(coefficients)
    assert "(-2/3) * 1" in render_coefficients(coefficients, "text")
    assert "$a_{2}(k) =" in render_latex(coefficients)
    zero = Polynomial()
    assert polynomial_to_text(zero) == "0"
    assert polynomial_to_python(zero) == "Rational(0)"
    factor, remainder = factor_latex_prefactor(zero)
    assert factor * remainder == zero
    assert "$a_{1}(k) = 0$" in render_latex([{}, {(): zero}])


def test_recursive_tree_latex():
    m = Tree.variance()
    assert tree_to_latex(Tree.unary(Tree.binary(m, m))) == (
        r"\left(X\diamond \left(M\diamond M\right)\right)"
    )


def test_export_rejects_unknown_format():
    with pytest.raises(ValueError, match="output_format"):
        render_coefficients([{}, {}], "html")
