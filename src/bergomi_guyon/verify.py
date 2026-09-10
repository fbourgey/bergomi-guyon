"""Exact checks of the BG recursion and its original matching condition."""

from fractions import Fraction
from math import factorial

from .recursion import (
    FOREST_ONE, LAMBDA, THETA, ZETA, ForestPolynomial, GenerationResult,
    Polynomial, Tree, forest_add, forest_D, forest_multiply, forest_scale,
    heat, heat_forest,
)


def verify(result: GenerationResult) -> None:
    """Check all generated orders; raise AssertionError on inconsistent data.

    Parameters
    ----------
    result : GenerationResult
        Trees, boundaries, coefficients, and sources to verify.

    Raises
    ------
    AssertionError
        If the PDE, boundary, completeness, grading, or matching checks fail.

    Notes
    -----
    Returns None on success. This validates formal algebra, not numerical
    model substitutions or convergence of a truncated smile expansion.
    """
    forests = result.trees
    boundaries = result.boundaries
    coefficients = result.coefficients
    sources = result.sources
    if not (len(forests) == len(boundaries) == len(coefficients) == len(sources)):
        raise AssertionError("inconsistent numbers of orders")
    _verify_correction_shape(result)
    if sources[0]:
        raise AssertionError("order-zero source must be empty")
    if forests[0] != {Tree.variance(): Fraction(1)}:
        raise AssertionError("order-zero tree must be M with weight one")
    _verify_forests(result)
    for ell in range(1, len(coefficients)):
        residual = forest_add(
            forest_D(coefficients[ell]),
            forest_scale(sources[ell], Polynomial.constant(-1)),
        )
        if residual:
            raise AssertionError(f"PDE verification failed at order {ell}")

        initial = {
            monomial: boundary
            for monomial, coefficient in coefficients[ell].items()
            if (boundary := coefficient.at_theta_zero())
        }
        if initial != boundaries[ell]:
            raise AssertionError(f"boundary verification failed at order {ell}")

        for monomial, coefficient in coefficients[ell].items():
            if not coefficient:
                raise AssertionError(f"zero forest coefficient at order {ell}: {monomial}")
            theta_factor = len(monomial) - 1
            if coefficient.min_theta_power() < theta_factor:
                raise AssertionError(f"theta factor failed at order {ell}: {monomial}")

            # For c = (-theta)^(j-1) q, check q's leading zeta term
            # directly in c, without constructing the quotient polynomial.
            if coefficient.max_zeta_power() != ell:
                raise AssertionError(f"exact degree failed at order {ell}: {monomial}")
            leading_terms = {
                t: value
                for (zeta_power, t), value in coefficient.terms.items()
                if zeta_power == ell
            }
            if (
                set(leading_terms) != {theta_factor}
                or leading_terms[theta_factor] * (-1) ** theta_factor <= 0
            ):
                raise AssertionError(
                    f"positive leading coefficient failed at order {ell}: {monomial}"
                )

    expected_counts = [1, 1, 2, 3, 6, 11, 23]
    for ell, expected in enumerate(expected_counts[: len(forests)]):
        if len(forests[ell]) != expected:
            raise AssertionError(
                f"tree count at order {ell}: got {len(forests[ell])}, "
                f"expected {expected}"
            )

    if len(coefficients) > 2:
        m = Tree.variance()
        first = Tree.unary(m)
        second_unary = Tree.unary(first)
        second_binary = Tree.binary(m, m)
        known = {
            (second_unary,): ZETA * ZETA - THETA,
            (second_binary,): (ZETA * ZETA - ZETA - THETA).scale(Fraction(1, 4)),
            (first, first): (
                THETA * (Polynomial.monomial(2, 0, -5) + ZETA.scale(3) + THETA.scale(3))
            ).scale(Fraction(1, 4)),
        }
        for monomial, expected in known.items():
            if coefficients[2].get(monomial) != expected:
                raise AssertionError(
                    f"known second-order coefficient failed: {monomial}"
                )

    verify_matching(result)


def _verify_forests(result: GenerationResult) -> None:
    """Check symmetry weights and all unordered products of positive-order trees."""
    def symmetric_nodes(tree: Tree) -> int:
        """Count binary nodes whose two children are identical trees."""
        return sum(symmetric_nodes(child) for child in tree.children) + (
            int(tree.children[0] == tree.children[1]) if tree.kind == "B" else 0
        )

    expected = [set() for _ in result.coefficients]
    expected[0].add(())
    for ell, trees in enumerate(result.trees):
        if ell:
            # Enumerate shapes, independently of the generator's accumulation
            # of symmetry weights. Fixed counts alone cover only orders <= 6.
            shapes = {Tree.unary(tree) for tree in result.trees[ell - 1]}
            shapes.update(
                Tree.binary(left, right)
                for i in range(ell - 1)
                for left in result.trees[i]
                for right in result.trees[ell - 2 - i]
            )
            if set(trees) != shapes:
                raise AssertionError(f"tree completeness failed at order {ell}")
        for tree, weight in trees.items():
            m_leaves, x_leaves = tree.leaves
            if tree.order != ell or ell != 2 * m_leaves + x_leaves - 2:
                raise AssertionError(f"tree grading failed at order {ell}")
            if weight != Fraction(1, 2 ** symmetric_nodes(tree)):
                raise AssertionError(f"symmetry weight failed at order {ell}")
            if ell == 0:
                continue
            # Ascending order allows repeated occurrences of this tree.
            for order in range(ell, len(expected)):
                expected[order].update(
                    tuple(sorted(monomial + (tree,)))
                    for monomial in expected[order - ell]
                )
    for ell in range(1, len(expected)):
        # Check the reduced-polynomial formula independently of the generator's
        # boundary_polynomial. Matching alone accepts any boundary.
        boundary = {
            (tree,): (
                LAMBDA.power(tree.leaves[0] - 1) * ZETA.power(tree.leaves[1])
            ).scale(weight)
            for tree, weight in result.trees[ell].items()
        }
        if result.boundaries[ell] != boundary:
            raise AssertionError(f"tree boundary formula failed at order {ell}")
        if set(result.coefficients[ell]) != expected[ell]:
            raise AssertionError(f"forest completeness/grading failed at order {ell}")


def _verify_correction_shape(result: GenerationResult) -> None:
    """Require matching series lengths and an empty order-zero correction."""
    if len(result.coefficients) < 2:
        raise AssertionError("at least one positive order is required")
    if len(result.boundaries) != len(result.coefficients):
        raise AssertionError("inconsistent numbers of orders")
    if result.boundaries[0] or result.coefficients[0]:
        raise AssertionError("order-zero correction and boundary must be empty")


def verify_matching(result: GenerationResult) -> None:
    """Check exponential matching without using the PDE sources or Duhamel solver.

    Parameters
    ----------
    result : GenerationResult
        Correction coefficients and boundary data to compare.

    Raises
    ------
    AssertionError
        If the series layout or any matching coefficient is inconsistent.

    Notes
    -----
    With b = psi_tilde/lambda and a = Sigma_tilde, compare coefficients of
    B[sum lambda**(n-1) b**n/n!] and sum a**n B[lambda**(n-1)]/n!.
    Only sparse arithmetic and the linear BG heat operator are shared with
    generation; the nonlinear identity is independent of source_coefficient.
    This check alone does not establish tree completeness or boundary validity;
    use verify for the full set of checks.
    """
    _verify_correction_shape(result)
    size = len(result.coefficients)

    def multiply_series(left, right):
        """Multiply forest series, truncating at the requested result size."""
        product: list[ForestPolynomial] = [{} for _ in range(size)]
        for i in range(size):
            for j in range(size - i):
                product[i + j] = forest_add(
                    product[i + j], forest_multiply(left[i], right[j])
                )
        return product

    boundary_power = [FOREST_ONE] + [{} for _ in range(size - 1)]
    coefficient_power = [FOREST_ONE] + [{} for _ in range(size - 1)]
    left: list[ForestPolynomial] = [{} for _ in range(size)]
    right: list[ForestPolynomial] = [{} for _ in range(size)]
    for n in range(1, size):
        boundary_power = multiply_series(boundary_power, result.boundaries)
        coefficient_power = multiply_series(coefficient_power, result.coefficients)
        factor = LAMBDA.power(n - 1).scale(Fraction(1, factorial(n)))
        heat_factor = heat(factor)
        for ell in range(n, size):
            left[ell] = forest_add(
                left[ell], heat_forest(forest_scale(boundary_power[ell], factor))
            )
            right[ell] = forest_add(
                right[ell], forest_scale(coefficient_power[ell], heat_factor)
            )
    for ell in range(1, size):
        if left[ell] != right[ell]:
            raise AssertionError(f"matching verification failed at order {ell}")
