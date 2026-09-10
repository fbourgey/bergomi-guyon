"""Reject incomplete or inconsistent BG recursion results."""

from dataclasses import replace

import pytest

from bergomi_guyon import GenerationResult, generate_coefficients, verify
from bergomi_guyon.recursion import ONE, THETA, ZETA, Polynomial, forest_D
from bergomi_guyon.verify import verify_matching


@pytest.mark.parametrize("checker", [verify, verify_matching])
@pytest.mark.parametrize("size", [0, 1])
def test_verification_requires_positive_orders(checker, size):
    with pytest.raises(AssertionError, match="positive order"):
        checker(GenerationResult([{}] * size, [{}] * size, [{}] * size, [{}] * size))


@pytest.mark.parametrize("field", ["trees", "boundaries", "coefficients", "sources"])
def test_verification_rejects_mismatched_lengths(field):
    result = generate_coefficients(1)
    with pytest.raises(AssertionError, match="numbers of orders"):
        verify(replace(result, **{field: []}))


@pytest.mark.parametrize("field", ["boundaries", "coefficients", "sources"])
def test_verification_rejects_nonempty_order_zero(field):
    result = generate_coefficients(1)
    getattr(result, field)[0][()] = ONE
    with pytest.raises(AssertionError, match="order-zero"):
        verify(result)


def test_verification_rejects_wrong_base_tree():
    result = generate_coefficients(1)
    result.trees[0].clear()
    with pytest.raises(AssertionError, match="order-zero tree"):
        verify(result)


def test_verification_detects_missing_tree_beyond_fixed_counts():
    result = generate_coefficients(7)
    verify(result)
    tree = next(iter(result.trees[7]))
    # Remove the single-tree contribution consistently: PDE and matching
    # still hold, so completeness must be checked independently.
    del result.trees[7][tree]
    del result.boundaries[7][(tree,)]
    del result.coefficients[7][(tree,)]
    verify_matching(result)
    with pytest.raises(AssertionError, match="tree completeness.*order 7"):
        verify(result)


def test_verification_detects_wrong_source():
    result = generate_coefficients(2)
    result.sources[2] = {}
    with pytest.raises(AssertionError, match="PDE.*order 2"):
        verify(result)


def test_verification_detects_wrong_boundary_value():
    result = generate_coefficients(1)
    tree = next(iter(result.trees[1]))
    result.coefficients[1][(tree,)] = ZETA + ONE
    with pytest.raises(AssertionError, match="boundary verification"):
        verify(result)


@pytest.mark.parametrize(
    "polynomial,message",
    [
        (Polynomial(), "zero forest coefficient"),
        (THETA, "exact degree"),
        (THETA * ZETA.power(2), "positive leading coefficient"),
        (THETA.power(2) * ZETA.power(2), "positive leading coefficient"),
    ],
)
def test_verification_checks_degree_and_leading_sign(polynomial, message):
    result = generate_coefficients(2)
    first = next(iter(result.trees[1]))
    result.coefficients[2][(first, first)] = polynomial
    result.sources[2] = forest_D(result.coefficients[2])
    with pytest.raises(AssertionError, match=message):
        verify(result)
