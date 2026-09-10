"""Sparse algebra, tree construction, and reciprocal-series checks."""

import pytest

from bergomi_guyon import generate_coefficients
from bergomi_guyon.recursion import (
    FOREST_ONE,
    ONE,
    THETA,
    ZETA,
    Polynomial,
    Tree,
    forest_add,
    forest_multiply,
    forest_scale,
    heat,
    reciprocal_coefficient,
)


@pytest.mark.parametrize("exponent", [(-1, 0), (0, -1), (1.5, 0), (True, 0), (1,), "xy"])
def test_invalid_polynomial_exponents(exponent):
    with pytest.raises(ValueError, match="two nonnegative integers"):
        Polynomial({exponent: 1})


@pytest.mark.parametrize("order", [-1, True, 0.5, "2", None])
@pytest.mark.parametrize("operation", ["power", "d_zeta"])
def test_invalid_algebra_orders(order, operation):
    with pytest.raises(ValueError, match="nonnegative integer"):
        getattr(ZETA, operation)(order)


def test_sparse_algebra_cancellation_and_derivatives():
    p = (ZETA + THETA).power(3)
    expected = Polynomial({(3, 0): 1, (2, 1): 3, (1, 2): 3, (0, 3): 1})
    assert p == expected
    assert p.d_zeta(2) == (ZETA + THETA).scale(6)
    assert p.d_theta() == (ZETA + THETA).power(2).scale(3)
    assert p.at_theta_zero() == ZETA.power(3)
    assert p.power(0) == ONE
    assert p.d_zeta(0) == p
    assert not p.d_zeta(4)
    assert not p - expected
    assert not p.scale(0)
    assert not Polynomial.constant(0)
    assert p != 0
    assert p == expected  # operations did not mutate the input


@pytest.mark.parametrize("kind,children", [
    ("X", ()), ("M", (Tree.variance(),)), ("U", ()),
    ("B", (Tree.variance(),)), ("U", (None,)), ("M", []),
])
def test_invalid_trees(kind, children):
    with pytest.raises(ValueError, match="tree must be"):
        Tree(kind, children)


def test_binary_trees_and_forest_products_commute():
    m = Tree.variance()
    u = Tree.unary(m)
    assert Tree("B", (u, m)) == Tree.binary(m, u) == Tree.binary(u, m)
    assert Tree.binary(m, u).leaves == (2, 1)
    assert Tree.binary(m, u).order == 3
    left = {(u,): ONE, (Tree.unary(u),): ONE}
    right = {(u,): ONE, (Tree.unary(u),): -ONE}
    product = forest_multiply(left, right)
    assert product == {(u, u): ONE, (Tree.unary(u), Tree.unary(u)): -ONE}
    assert forest_multiply(left, FOREST_ONE) == left
    assert not forest_add(left, forest_scale(left, -ONE))
    assert not forest_scale(left, Polynomial())


def test_heat_rejects_theta_dependent_data():
    with pytest.raises(ValueError, match="theta-free"):
        heat(ZETA + THETA)
    assert not heat(Polynomial())


def test_reciprocal_series_identity():
    coefficients = generate_coefficients(4).coefficients
    reciprocal = [FOREST_ONE]
    for ell in range(1, len(coefficients)):
        reciprocal.append(reciprocal_coefficient(ell, coefficients, reciprocal))
        residual = forest_add(reciprocal[ell], *(
            forest_scale(forest_multiply(coefficients[i], reciprocal[ell - i]), THETA)
            for i in range(1, ell + 1)
        ))
        assert not residual
