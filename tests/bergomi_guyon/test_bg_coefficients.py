"""Paper identities, exact artifact reproduction, and BG export regressions."""

import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from bergomi_guyon import generate_coefficients, verify
from bergomi_guyon.recursion import Polynomial, Tree, heat, solve_zero_boundary
from bergomi_guyon.render import (
    factor_latex_prefactor,
    group_latex_terms,
    polynomial_to_latex,
    render_coefficients,
    render_python_module,
    tree_to_symbol,
)
from bergomi_guyon.verify import verify_matching

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def result():
    return generate_coefficients(6)


def as_sympy(poly, zeta, theta):
    return sp.Add(*(
        sp.Rational(value.numerator, value.denominator) * zeta**z * theta**t
        for (z, t), value in poly.terms.items()
    ))


def test_paper_identities_through_order_six(result):
    verify(result)


def test_matching_detects_wrong_nonlinear_coefficient(result):
    coefficients = list(result.coefficients)
    coefficients[3] = dict(coefficients[3])
    first = Tree.unary(Tree.variance())
    monomial = (first, first, first)
    coefficients[3][monomial] = coefficients[3][monomial].scale(2)
    with pytest.raises(AssertionError, match="matching.*order 3"):
        verify_matching(replace(result, coefficients=coefficients))


def test_verification_detects_missing_forest(result):
    coefficients = list(result.coefficients)
    coefficients[6] = dict(coefficients[6])
    coefficients[6].pop(next(iter(coefficients[6])))
    with pytest.raises(AssertionError, match="completeness.*order 6"):
        verify(replace(result, coefficients=coefficients))


def test_verification_detects_wrong_symmetry_weight(result):
    trees = list(result.trees)
    trees[3] = dict(trees[3])
    tree = next(iter(trees[3]))
    trees[3][tree] *= 2
    with pytest.raises(AssertionError, match="symmetry weight.*order 3"):
        verify(replace(result, trees=trees))


def test_verification_detects_wrong_reduced_polynomial(result):
    tree = next(iter(result.trees[3]))

    def rescale_tree(series):
        return [
            {
                monomial: polynomial.scale(2 ** monomial.count(tree))
                for monomial, polynomial in coefficient.items()
            }
            for coefficient in series
        ]

    # A consistent change of tree normalization preserves matching and the
    # PDE, but violates the paper's reduced polynomial for the given weight.
    altered = replace(
        result,
        boundaries=rescale_tree(result.boundaries),
        coefficients=rescale_tree(result.coefficients),
        sources=rescale_tree(result.sources),
    )
    verify_matching(altered)
    with pytest.raises(AssertionError, match="tree boundary formula.*order 3"):
        verify(altered)


def test_same_leaf_content_preserves_repetition_multiplicities(result):
    first, second = [tree for tree in result.trees[3] if tree.leaves == (2, 1)]
    w_first, w_second = (result.trees[3][tree] for tree in (first, second))
    repeated = result.coefficients[6][(first, first)].scale(1 / w_first**2)
    mixed = result.coefficients[6][tuple(sorted((first, second)))].scale(
        1 / (w_first * w_second)
    )
    assert mixed == repeated.scale(2)


def test_heat_and_duhamel_against_symbolic_differentiation():
    zeta, theta = sp.symbols("zeta theta")
    for z_power in range(9):
        boundary = Polynomial.monomial(z_power, 0, Fraction(2, 3))
        solution = as_sympy(heat(boundary), zeta, theta)
        assert sp.expand(sp.diff(solution, theta) + sp.diff(solution, zeta, 2) / 2) == 0
        assert solution.subs(theta, 0) == sp.Rational(2, 3) * zeta**z_power
        for t_power in range(4):
            source = Polynomial.monomial(z_power, t_power, Fraction(-3, 7))
            solution = as_sympy(solve_zero_boundary(source), zeta, theta)
            residual = sp.diff(solution, theta) + sp.diff(solution, zeta, 2) / 2
            assert sp.expand(residual - as_sympy(source, zeta, theta)) == 0
            assert solution.subs(theta, 0) == 0


@pytest.mark.parametrize("order", [1, 2, 6])
def test_generated_module_import_and_exact_expressions(result, order):
    namespace = {}
    # Execute this package's generated source to validate its symbolic exports.
    exec(render_python_module(result.coefficients[:order + 1]), namespace)  # noqa: S102
    assert namespace["a"][0] is None
    assert len(namespace["a"]) == order + 1
    for ell in range(1, order + 1):
        expected = sp.Add(*(
            as_sympy(poly, namespace["zeta"], namespace["theta"])
            * sp.prod(sp.Symbol(tree_to_symbol(tree)) for tree in monomial)
            for monomial, poly in result.coefficients[ell].items()
        ))
        assert sp.expand(namespace["a"][ell] - expected) == 0
        assert not namespace["a"][ell].atoms(sp.Float)
    assert namespace["in_k_and_M"](namespace["a"][1]) == (
        sp.Rational(1, 2) + namespace["k"] / namespace["M"]
    ) * namespace["MXd"]


def test_tree_symbols_are_unique(result):
    trees = [tree for order in result.trees for tree in order]
    assert len({tree_to_symbol(tree) for tree in trees}) == len(trees)


def test_checked_in_python_artifact_is_reproducible(result):
    assert render_python_module(result.coefficients) == (
        ROOT / "src/bergomi_guyon/bg_coefficients_order_6.py"
    ).read_text()


def test_checked_in_text_artifact_is_reproducible(result):
    rendered = render_coefficients(result.coefficients, "text")
    assert rendered == (
        ROOT / "src/bergomi_guyon/bg_coefficients_order_6.txt"
    ).read_text()
    assert "Sigma(k) = sum_{ell=0}^6 a_ell * epsilon^ell + O(epsilon^7)" in rendered
    assert "MXdXd    = (M diamond X) diamond X" in rendered
    assert "MXdMdXd  = ((M diamond X) diamond M) diamond X" in rendered
    assert "\n# Order 0\na_0:\n  M\n\n# Order 1\na_1:\n" in rendered
    assert "(zeta^2 - theta) * MXdXd" in rendered
    assert " * (M diamond M)" not in rendered


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "bergomi_guyon", *args],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )


def test_verified_python_stdout_is_importable():
    process = run_cli("--order", "1", "--verify", "--format", "python")
    namespace = {}
    # Execute Python emitted by the local CLI with fixed test arguments.
    exec(process.stdout, namespace)  # noqa: S102
    assert len(namespace["a"]) == 2
    assert "verified orders 1 through 1" in process.stderr


@pytest.mark.parametrize("output_format", ["text", "latex", "python"])
def test_quiet_suppresses_stdout_but_preserves_explicit_output(tmp_path, output_format):
    arguments = ("--order", "2", "--verify", "--quiet", "--format", output_format)
    assert run_cli(*arguments).stdout == ""
    output = tmp_path / "coefficients"
    process = run_cli(*arguments, "--output", str(output))
    assert process.stdout == ""
    assert output.read_text() == run_cli("--order", "2", "--format", output_format).stdout


def test_latex_braces_multi_digit_exponents():
    assert polynomial_to_latex(Polynomial.monomial(12, 10)) == r"\zeta^{12}\,\theta^{10}"


def test_latex_uses_exact_fractions():
    assert polynomial_to_latex(Polynomial.monomial(2, 1, Fraction(-3, 8))) == (
        r"-\frac{3}{8}\,\zeta^{2}\,\theta"
    )


def test_default_cli_generates_insertable_latex():
    output = run_cli("--order", "2").stdout
    assert output == run_cli("--order", "2", "--format", "latex").stdout
    assert r"$a_{1}(k) =" in output
    assert r"\tikz[" not in output
    assert r"\bigl(\MXd\bigr)^{2}" in output
    assert r"\mathcal{O}(\epsilon^{3})" in output


def test_latex_grouping_preserves_every_exact_coefficient(result):
    for coefficient in result.coefficients[1:]:
        recovered = {}
        for polynomial, products in group_latex_terms(coefficient):
            for monomial, scale in products:
                assert monomial not in recovered
                recovered[monomial] = polynomial.scale(scale)
        assert recovered == coefficient


def test_latex_factorization_is_exact(result):
    for coefficient in result.coefficients[1:]:
        for polynomial in coefficient.values():
            factor, remainder = factor_latex_prefactor(polynomial)
            assert factor * remainder == polynomial


@pytest.mark.parametrize("order", [0, -1, True, 1.5])
def test_generator_rejects_invalid_orders(order):
    with pytest.raises(ValueError, match="positive integer"):
        generate_coefficients(order)
