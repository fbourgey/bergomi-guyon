"""Exact Bergomi--Guyon recursion, using only the standard library.

The iteration advances by epsilon order ell. A forest monomial separately
records its number of tree factors j; these are two separate gradings.
Variables: zeta = 1/2 + k/M, theta = 1/M, kappa = zeta - 1/2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import factorial

Exponent = tuple[int, int]  # powers of (zeta, theta)


class Polynomial:
    """A sparse polynomial in zeta and theta with rational coefficients.

    Parameters
    ----------
    terms : mapping of (int, int) to int or Fraction, optional
        Nonnegative powers of (zeta, theta) and their coefficients. Zero
        coefficients are discarded; omitted terms represent zero.

    Attributes
    ----------
    terms : dict
        Mutable exponent-to-Fraction mapping. Arithmetic returns polynomials
        without modifying the operands.
    """

    def __init__(self, terms: Mapping[Exponent, Fraction | int] | None = None):
        """Validate exponents and store the nonzero rational coefficients."""
        self.terms: dict[Exponent, Fraction] = {}
        for exponent, coefficient in (terms or {}).items():
            if (
                not isinstance(exponent, tuple) or len(exponent) != 2
                or any(isinstance(p, bool) or not isinstance(p, int) or p < 0
                       for p in exponent)
            ):
                raise ValueError("polynomial exponents must be two nonnegative integers")
            value = Fraction(coefficient)
            if value:
                self.terms[exponent] = value

    @staticmethod
    def constant(value: Fraction | int) -> Polynomial:
        """Return a constant polynomial with the given rational value."""
        return Polynomial({(0, 0): value})

    @staticmethod
    def monomial(
        zeta_power: int, theta_power: int, coefficient: Fraction | int = 1
    ) -> Polynomial:
        """Return coefficient times zeta**zeta_power times theta**theta_power."""
        return Polynomial({(zeta_power, theta_power): coefficient})

    def __bool__(self) -> bool:
        """Return whether the polynomial is nonzero."""
        return bool(self.terms)

    def __eq__(self, other: object) -> bool:
        """Compare polynomials by their exact nonzero terms."""
        return isinstance(other, Polynomial) and self.terms == other.terms

    def __add__(self, other: Polynomial) -> Polynomial:
        """Add two polynomials, dropping terms that cancel."""
        result = dict(self.terms)
        for exponent, coefficient in other.terms.items():
            result[exponent] = result.get(exponent, Fraction(0)) + coefficient
        return Polynomial(result)

    def __neg__(self) -> Polynomial:
        """Negate every coefficient."""
        return Polynomial({exponent: -value for exponent, value in self.terms.items()})

    def __sub__(self, other: Polynomial) -> Polynomial:
        """Subtract another polynomial."""
        return self + (-other)

    def __mul__(self, other: Polynomial) -> Polynomial:
        """Multiply polynomials and collect equal powers."""
        result: dict[Exponent, Fraction] = {}
        for (z1, t1), value1 in self.terms.items():
            for (z2, t2), value2 in other.terms.items():
                exponent = (z1 + z2, t1 + t2)
                result[exponent] = result.get(exponent, Fraction(0)) + value1 * value2
        return Polynomial(result)

    def scale(self, value: Fraction | int) -> Polynomial:
        """Multiply every coefficient by a rational scalar."""
        factor = Fraction(value)
        return Polynomial(
            {
                exponent: factor * coefficient
                for exponent, coefficient in self.terms.items()
            }
        )

    def power(self, exponent: int) -> Polynomial:
        """Raise the polynomial to a nonnegative integer power."""
        if isinstance(exponent, bool) or not isinstance(exponent, int) or exponent < 0:
            raise ValueError("polynomial powers must be nonnegative integers")
        result = ONE
        base = self
        power = exponent
        while power:
            if power & 1:
                result = result * base
            base = base * base
            power //= 2
        return result

    def d_zeta(self, times: int = 1) -> Polynomial:
        """Differentiate in zeta a nonnegative integer number of times."""
        if isinstance(times, bool) or not isinstance(times, int) or times < 0:
            raise ValueError("derivative order must be a nonnegative integer")
        result = self
        for _ in range(times):
            result = Polynomial(
                {
                    (zeta_power - 1, theta_power): coefficient * zeta_power
                    for (zeta_power, theta_power), coefficient in result.terms.items()
                    if zeta_power
                }
            )
        return result

    def d_theta(self) -> Polynomial:
        """Differentiate once in theta."""
        return Polynomial(
            {
                (zeta_power, theta_power - 1): coefficient * theta_power
                for (zeta_power, theta_power), coefficient in self.terms.items()
                if theta_power
            }
        )

    def at_theta_zero(self) -> Polynomial:
        """Evaluate at theta = 0, retaining a polynomial in zeta."""
        return Polynomial(
            {
                (zeta_power, 0): coefficient
                for (zeta_power, theta_power), coefficient in self.terms.items()
                if theta_power == 0
            }
        )

    def min_theta_power(self) -> int:
        """Return the smallest theta power; raise ValueError for zero."""
        return min(theta_power for _, theta_power in self.terms)

    def max_zeta_power(self) -> int:
        """Return the degree in zeta, or -1 for the zero polynomial."""
        return max((zeta_power for zeta_power, _ in self.terms), default=-1)

    def sorted_terms(self) -> list[tuple[Exponent, Fraction]]:
        """Return terms in descending lexicographic (zeta, theta) order."""
        return sorted(self.terms.items(), reverse=True)


ZERO = Polynomial()
ONE = Polynomial.constant(1)
ZETA = Polynomial.monomial(1, 0)
THETA = Polynomial.monomial(0, 1)
KAPPA = ZETA - Polynomial.constant(Fraction(1, 2))
LAMBDA = (ZETA * (ZETA - ONE)).scale(Fraction(1, 2))


@dataclass(frozen=True, order=True)
class Tree:
    """A diamond tree with canonical ordering of binary children.

    Parameters
    ----------
    kind : {'M', 'U', 'B'}
        Variance leaf, unary diamond with X, or binary diamond.
    children : tuple of Tree, optional
        Zero, one, or two children, respectively. Binary children are sorted
        because the diamond product is commutative.
    """

    kind: str
    children: tuple[Tree, ...] = ()

    def __post_init__(self):
        """Validate the tree arity and canonicalize binary children."""
        arity = {"M": 0, "U": 1, "B": 2}.get(self.kind)
        if (
            arity is None or not isinstance(self.children, tuple)
            or len(self.children) != arity
            or any(not isinstance(child, Tree) for child in self.children)
        ):
            raise ValueError("tree must be M (no children), U (one), or B (two)")
        if self.kind == "B":
            object.__setattr__(self, "children", tuple(sorted(self.children)))

    @staticmethod
    def variance() -> Tree:
        """Return the variance leaf M."""
        return Tree("M")

    @staticmethod
    def unary(child: Tree) -> Tree:
        """Return X diamond child."""
        return Tree("U", (child,))

    @staticmethod
    def binary(left: Tree, right: Tree) -> Tree:
        """Return the commutative diamond product of two trees."""
        return Tree("B", (left, right))

    @property
    def order(self) -> int:
        """Return the order, adding one per unary node and two per binary node."""
        if self.kind == "M":
            return 0
        if self.kind == "U":
            return self.children[0].order + 1
        return self.children[0].order + self.children[1].order + 2

    @property
    def leaves(self) -> tuple[int, int]:
        """Return the numbers of M and X leaves."""
        if self.kind == "M":
            return (1, 0)
        if self.kind == "U":
            m_leaves, x_leaves = self.children[0].leaves
            return (m_leaves, x_leaves + 1)
        left = self.children[0].leaves
        right = self.children[1].leaves
        return (left[0] + right[0], left[1] + right[1])


Monomial = tuple[Tree, ...]
ForestPolynomial = dict[Monomial, Polynomial]


def forest_add(*values: ForestPolynomial) -> ForestPolynomial:
    """Add forest polynomials and discard coefficients that cancel."""
    result: ForestPolynomial = {}
    for value in values:
        for monomial, coefficient in value.items():
            updated = result.get(monomial, ZERO) + coefficient
            if updated:
                result[monomial] = updated
            else:
                result.pop(monomial, None)
    return result


def forest_scale(value: ForestPolynomial, polynomial: Polynomial) -> ForestPolynomial:
    """Multiply every forest coefficient by a polynomial in zeta and theta."""
    return {
        monomial: product
        for monomial, coefficient in value.items()
        if (product := coefficient * polynomial)
    }


def forest_multiply(
    left: ForestPolynomial, right: ForestPolynomial
) -> ForestPolynomial:
    """Multiply forests, sorting tree factors and collecting multiplicities."""
    result: ForestPolynomial = {}
    for left_monomial, left_coefficient in left.items():
        for right_monomial, right_coefficient in right.items():
            monomial = tuple(sorted(left_monomial + right_monomial))
            updated = result.get(monomial, ZERO) + left_coefficient * right_coefficient
            if updated:
                result[monomial] = updated
            else:
                result.pop(monomial, None)
    return result


def forest_d_zeta(value: ForestPolynomial) -> ForestPolynomial:
    """Differentiate forest coefficients once in zeta, dropping zeros."""
    return {
        monomial: derivative
        for monomial, coefficient in value.items()
        if (derivative := coefficient.d_zeta())
    }


def forest_D(value: ForestPolynomial) -> ForestPolynomial:
    """Apply D = d_theta + d_zeta**2 / 2 to each forest coefficient."""
    return {
        monomial: derivative
        for monomial, coefficient in value.items()
        if (
            derivative := coefficient.d_theta()
            + coefficient.d_zeta(2).scale(Fraction(1, 2))
        )
    }


FOREST_ONE: ForestPolynomial = {(): ONE}


def generate_trees(max_order: int) -> Sequence[dict[Tree, Fraction]]:
    """Generate diamond trees and their exact symmetry weights.

    Parameters
    ----------
    max_order : int
        Highest perturbative order; must be positive and not a boolean.

    Returns
    -------
    sequence of dict
        Tree-to-weight mappings indexed from order zero through max_order.
    """
    if isinstance(max_order, bool) or not isinstance(max_order, int) or max_order < 1:
        raise ValueError("max_order must be a positive integer")
    forests: list[dict[Tree, Fraction]] = [{Tree.variance(): Fraction(1)}]
    for ell in range(1, max_order + 1):
        current: dict[Tree, Fraction] = {}

        for tree, weight in forests[ell - 1].items():
            unary = Tree.unary(tree)
            current[unary] = current.get(unary, Fraction(0)) + weight

        for j in range(ell - 1):
            left_order = ell - 2 - j
            right_order = j
            for left, left_weight in forests[left_order].items():
                for right, right_weight in forests[right_order].items():
                    binary = Tree.binary(left, right)
                    weight = Fraction(1, 2) * left_weight * right_weight
                    current[binary] = current.get(binary, Fraction(0)) + weight

        forests.append(current)
    return forests


def boundary_polynomial(weighted_trees: Mapping[Tree, Fraction]) -> ForestPolynomial:
    """Build single-tree boundary terms w * lambda**(m - 1) * zeta**x."""
    result: ForestPolynomial = {}
    for tree, weight in weighted_trees.items():
        m_leaves, x_leaves = tree.leaves
        r_tree = LAMBDA.power(m_leaves - 1) * ZETA.power(x_leaves)
        result[(tree,)] = r_tree.scale(weight)
    return result


def heat(polynomial: Polynomial) -> Polynomial:
    """Apply the finite backward heat transform to boundary data.

    Parameters
    ----------
    polynomial : Polynomial
        Boundary polynomial independent of theta.

    Returns
    -------
    Polynomial
        exp(-theta/2 d_zeta**2) applied to the boundary polynomial.

    Raises
    ------
    ValueError
        If the boundary depends on theta.
    """
    result = ZERO
    for (zeta_power, theta_power), coefficient in polynomial.terms.items():
        if theta_power:
            raise ValueError("heat boundary data must be theta-free")
        for j in range(zeta_power // 2 + 1):
            multiplier = Fraction(
                (-1) ** j * factorial(zeta_power),
                2**j * factorial(j) * factorial(zeta_power - 2 * j),
            )
            result += Polynomial.monomial(
                zeta_power - 2 * j, j, coefficient * multiplier
            )
    return result


def solve_zero_boundary(source: Polynomial) -> Polynomial:
    """Solve Df=source, f(zeta,0)=0 by finite polynomial Duhamel integration.

    Parameters
    ----------
    source : Polynomial
        Polynomial right-hand side of Df = source.

    Returns
    -------
    Polynomial
        Exact solution with zero boundary at theta = 0.

    Notes
    -----
    This sums the Taylor recurrence exactly:
    f^(n+1)(0) = source^(n)(0) - (1/2) d_zeta^2 f^(n)(0).
    """
    result = ZERO
    for (zeta_power, theta_power), coefficient in source.terms.items():
        for j in range(zeta_power // 2 + 1):
            multiplier = Fraction(
                (-1) ** j * factorial(zeta_power) * factorial(theta_power),
                2**j * factorial(zeta_power - 2 * j) * factorial(theta_power + j + 1),
            )
            result += Polynomial.monomial(
                zeta_power - 2 * j,
                theta_power + j + 1,
                coefficient * multiplier,
            )
    return result


def heat_forest(boundary: ForestPolynomial) -> ForestPolynomial:
    """Apply the backward heat transform to each forest boundary coefficient."""
    return {monomial: heat(coefficient) for monomial, coefficient in boundary.items()}


def solve_source(source: ForestPolynomial) -> ForestPolynomial:
    """Solve each forest source coefficient with zero boundary data."""
    return {
        monomial: solve_zero_boundary(coefficient)
        for monomial, coefficient in source.items()
    }


def reciprocal_coefficient(
    ell: int,
    coefficients: Sequence[ForestPolynomial],
    reciprocal: Sequence[ForestPolynomial],
) -> ForestPolynomial:
    """Extract order ell of (1 + theta*A)**(-1) from A and lower orders."""
    total: ForestPolynomial = {}
    for i in range(1, ell + 1):
        total = forest_add(total, forest_multiply(coefficients[i], reciprocal[ell - i]))
    return forest_scale(total, -THETA)


def square_series_coefficient(
    ell: int, series: Sequence[ForestPolynomial]
) -> ForestPolynomial:
    """Return order ell of a squared forest series by finite convolution."""
    total: ForestPolynomial = {}
    for i in range(ell + 1):
        total = forest_add(total, forest_multiply(series[i], series[ell - i]))
    return total


def source_coefficient(
    ell: int,
    coefficients: Sequence[ForestPolynomial],
    reciprocal: Sequence[ForestPolynomial],
) -> ForestPolynomial:
    """Extract epsilon**ell from the nonlinear PDE using the closed form of K.

    Parameters
    ----------
    ell : int
        Positive perturbative order to extract.
    coefficients : sequence of ForestPolynomial
        Correction coefficients, known through ell - 1, with empty order zero.
    reciprocal : sequence of ForestPolynomial
        Coefficients of (1 + theta*A)**(-1), known through ell - 2.

    Returns
    -------
    ForestPolynomial
        Source at order ell, determined entirely by lower correction orders.

    Notes
    -----
    For A = Sigma_tilde, R = (1 + theta*A)**(-1), and A' = d_zeta A,
    the source is -kappa**2*(A')**2*R**2/4 + theta*(A')**2*R/4
    - kappa*A*A'*R + (A')**2/16.
    """
    derivatives = [{}] + [forest_d_zeta(coefficients[i]) for i in range(1, ell)]
    reciprocal_square = [
        square_series_coefficient(i, reciprocal) for i in range(ell - 1)
    ]
    source: ForestPolynomial = {}

    for i in range(1, ell):
        for j in range(1, ell - i + 1):
            m = ell - i - j
            derivative_product = forest_multiply(derivatives[i], derivatives[j])
            source = forest_add(
                source,
                forest_scale(
                    forest_multiply(derivative_product, reciprocal_square[m]),
                    (KAPPA * KAPPA).scale(Fraction(-1, 4)),
                ),
                forest_scale(
                    forest_multiply(derivative_product, reciprocal[m]),
                    THETA.scale(Fraction(1, 4)),
                ),
                forest_scale(
                    forest_multiply(
                        forest_multiply(coefficients[i], derivatives[j]),
                        reciprocal[m],
                    ),
                    -KAPPA,
                ),
            )
            if m == 0:
                source = forest_add(
                    source,
                    forest_scale(derivative_product, Polynomial.constant(Fraction(1, 16))),
                )
    return source


@dataclass(frozen=True)
class GenerationResult:
    """Order-indexed recursion data for the correction Sigma - M.

    Attributes
    ----------
    trees : sequence of dict
        Tree-to-symmetry-weight mappings, including the order-zero leaf M.
    boundaries : sequence of ForestPolynomial
        Single-tree boundary data at theta = 0.
    coefficients : sequence of ForestPolynomial
        Solutions of the coefficientwise nonlinear PDE.
    sources : sequence of ForestPolynomial
        Right-hand sides of the coefficientwise PDE.

    Notes
    -----
    Index zero of boundaries, coefficients, and sources is empty; a_0 = M
    belongs to the full smile, not to this correction. Trees include order zero.
    All four sequences have length max_order + 1. Forest keys are sorted tuples
    of trees, and their values are polynomials in (zeta, theta). The dataclass
    is frozen, but its nested dictionaries and polynomials remain mutable.
    """

    trees: Sequence[dict[Tree, Fraction]]
    boundaries: Sequence[ForestPolynomial]
    coefficients: Sequence[ForestPolynomial]
    sources: Sequence[ForestPolynomial]


def generate_coefficients(max_order: int) -> GenerationResult:
    """Solve the BG nonlinear PDE, retaining all tree-product multiplicities.

    Parameters
    ----------
    max_order : int
        Highest perturbative order; must be positive and not a boolean.

    Returns
    -------
    GenerationResult
        Trees, boundaries, coefficients, and sources indexed through max_order.

    Raises
    ------
    ValueError
        If max_order is not a positive Python integer.

    Notes
    -----
    We advance by epsilon order ell rather than tree count j.
    Every nonlinear source factor has positive epsilon order, so this is
    another triangular evaluation of the same polynomial solution. Symmetry
    weights enter at the boundary; factoring them out and restoring them at
    the end gives identical coefficients.

    max_order must be a positive Python integer (booleans are rejected).
    Coefficients describe total implied variance, not volatility. This is a
    formal expansion; generation does not establish numerical convergence.
    """
    forests = generate_trees(max_order)
    boundaries = [{}] + [
        boundary_polynomial(forests[ell]) for ell in range(1, max_order + 1)
    ]
    coefficients: list[ForestPolynomial] = [{}]
    reciprocal: list[ForestPolynomial] = [FOREST_ONE]
    sources: list[ForestPolynomial] = [{}]

    for ell in range(1, max_order + 1):
        source = source_coefficient(ell, coefficients, reciprocal)
        sources.append(source)
        coefficients.append(forest_add(
            heat_forest(boundaries[ell]), solve_source(source)
        ))
        if ell < max_order:
            reciprocal.append(reciprocal_coefficient(ell, coefficients, reciprocal))

    return GenerationResult(forests, boundaries, coefficients, sources)
