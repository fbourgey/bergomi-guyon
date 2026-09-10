"""Deterministic text, LaTeX, and SymPy-source rendering of BG coefficients."""

import re
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from math import gcd, lcm

from .recursion import ForestPolynomial, Monomial, Polynomial, Tree


def polynomial_to_text(polynomial: Polynomial) -> str:
    """Render a rational polynomial with explicit zeta and theta powers."""
    if not polynomial:
        return "0"
    pieces = []
    for (zeta_power, theta_power), coefficient in polynomial.sorted_terms():
        sign = "-" if coefficient < 0 else "+"
        magnitude = abs(coefficient)
        variables = []
        if zeta_power:
            variables.append("zeta" if zeta_power == 1 else f"zeta^{zeta_power}")
        if theta_power:
            variables.append(
                "theta" if theta_power == 1 else f"theta^{theta_power}"
            )
        variable_part = "*".join(variables)
        if magnitude == 1 and variable_part:
            body = variable_part
        else:
            body = str(magnitude)
            if variable_part:
                body += "*" + variable_part
        pieces.append((sign, body))
    first_sign, first_body = pieces[0]
    output = ("-" if first_sign == "-" else "") + first_body
    for sign, body in pieces[1:]:
        output += f" {sign} {body}"
    return output


def polynomial_to_latex(polynomial: Polynomial) -> str:
    """Render a polynomial with exact fractions and braced LaTeX powers."""
    return (
        re.sub(r"(\d+)/(\d+)", r"\\frac{\1}{\2}",
               re.sub(r"\^(\d+)", r"^{\1}", polynomial_to_text(polynomial)))
        .replace("zeta", r"\zeta")
        .replace("theta", r"\theta")
        .replace("*", r"\,")
    )


def polynomial_to_python(polynomial: Polynomial) -> str:
    """Return an exact SymPy expression using Rational, zeta, and theta."""
    if not polynomial:
        return "Rational(0)"
    terms = []
    for (zeta_power, theta_power), coefficient in polynomial.sorted_terms():
        factors = [f"Rational({coefficient.numerator}, {coefficient.denominator})"]
        if zeta_power:
            factors.append("zeta" if zeta_power == 1 else f"zeta**{zeta_power}")
        if theta_power:
            factors.append("theta" if theta_power == 1 else f"theta**{theta_power}")
        terms.append(" * ".join(factors))
    return " + ".join(terms).replace("+ Rational(-", "- Rational(")


def tree_to_text(tree: Tree) -> str:
    """Render a tree as a fully parenthesized diamond expression."""
    if tree.kind == "M":
        return "M"
    if tree.kind == "U":
        return f"(X diamond {tree_to_text(tree.children[0])})"
    return f"({tree_to_text(tree.children[0])} diamond {tree_to_text(tree.children[1])})"


def tree_to_latex(tree: Tree) -> str:
    """Render a tree using explicit LaTeX diamond products."""
    if tree.kind == "M":
        return "M"
    if tree.kind == "U":
        return rf"\left(X\diamond {tree_to_latex(tree.children[0])}\right)"
    return rf"\left({tree_to_latex(tree.children[0])}\diamond {tree_to_latex(tree.children[1])}\right)"


def tree_to_symbol(tree: Tree) -> str:
    """Return the unambiguous postfix symbol used in generated formulas."""
    if tree.kind == "M":
        return "M"
    if tree.kind == "U":
        return f"{tree_to_symbol(tree.children[0])}Xd"
    return f"{tree_to_symbol(tree.children[0])}{tree_to_symbol(tree.children[1])}d"


def monomial_text(monomial: Monomial) -> str:
    """Render a forest product using paper tree names, or 1 if empty."""
    return " * ".join(tree_to_paper_symbol(tree) for tree in monomial) or "1"


def tree_to_paper_symbol(tree: Tree) -> str:
    """Return a diamond-tree macro name with the larger subtree first."""
    if tree.kind == "M":
        return "M"
    if tree.kind == "U":
        return tree_to_paper_symbol(tree.children[0]) + "Xd"
    children = sorted(
        tree.children, key=lambda t: (t.order, tree_to_paper_symbol(t)), reverse=True
    )
    return "".join(tree_to_paper_symbol(child) for child in children) + "d"


def factor_latex_prefactor(polynomial: Polynomial):
    """Extract the common rational coefficient and powers of zeta and theta."""
    if not polynomial:
        return Polynomial.constant(0), Polynomial.constant(1)
    values = list(polynomial.terms.values())
    rational = Fraction(gcd(*(v.numerator for v in values)),
                        lcm(*(v.denominator for v in values)))
    z = min(z for z, t in polynomial.terms)
    t = min(t for z, t in polynomial.terms)
    factor = Polynomial.monomial(z, t, rational)
    remainder = Polynomial({(zp - z, tp - t): v / rational
                            for (zp, tp), v in polynomial.terms.items()})
    return factor, remainder


def group_latex_terms(coefficient: ForestPolynomial):
    """Collect proportional prefactors without changing their exact values."""
    groups = {}
    for monomial, polynomial in sorted(coefficient.items()):
        if not polynomial:
            continue
        leading = polynomial.sorted_terms()[0][1]
        key = tuple(polynomial.scale(1 / leading).sorted_terms())
        if key not in groups:
            groups[key] = (polynomial, [])
        representative, products = groups[key]
        scale = leading / representative.sorted_terms()[0][1]
        products.append((monomial, scale))
    return list(groups.values())


def diagram_product(monomial: Monomial) -> str:
    """Render a product of tree macros, grouping repeated trees into powers."""
    if not monomial:
        return "1"
    return r"\,".join(
        rf"\{tree_to_paper_symbol(tree)}" if count == 1 else
        rf"\bigl(\{tree_to_paper_symbol(tree)}\bigr)^{{{count}}}"
        for tree, count in Counter(monomial).items()
    )


def _latex_polynomial(coefficient: Polynomial) -> str:
    """Format a nonzero prefactor, wrapping long expressions in aligned rows."""
    terms = []
    for exponent, value in coefficient.sorted_terms():
        term = polynomial_to_latex(Polynomial({exponent: abs(value)}))
        sign = "-" if value < 0 else "+"
        terms.append((sign, term))
    chunks = []
    for start in range(0, len(terms), 5):
        chunk = " ".join(
            ("" if i == 0 and sign == "+" else sign + " ") + term
            for i, (sign, term) in enumerate(terms[start:start + 5], start)
        )
        chunks.append("&" + chunk)
    polynomial = (
        r"\left(" + chunks[0][1:] + r"\right)" if len(chunks) == 1 else
        r"\left(\begin{aligned}" + "\n  " + " \\\\\n  ".join(chunks)
        + "\n" + r"\end{aligned}\right)"
    )
    if coefficient == Polynomial.monomial(0, 0):
        polynomial = ""
    return polynomial


def _latex_forests(products: Sequence[tuple[Monomial, Fraction]]) -> str:
    """Format a nonempty group of weighted forest products in aligned rows."""
    forests = []
    for j, (monomial, scale) in enumerate(products):
        sign = "-" if scale < 0 else "+" if j else ""
        weight = "" if abs(scale) == 1 else polynomial_to_latex(
            Polynomial.monomial(0, 0, abs(scale))) + r"\,"
        forests.append(sign + weight + diagram_product(monomial))
    if len(forests) == 1:
        product = forests[0]
    elif len(forests) == 2:
        product = r"\left[" + " ".join(forests) + r"\right]"
    else:
        product = (
            r"\left[\begin{aligned}" + "\n  "
            + " \\\\\n  ".join("&" + " ".join(forests[j:j + 2])
                                for j in range(0, len(forests), 2))
            + "\n" + r"\end{aligned}\right]"
        )
    return product


def render_latex(coefficients: Sequence[ForestPolynomial]) -> str:
    """Return a compact LaTeX expansion using diamond-tree macros.

    Parameters
    ----------
    coefficients : sequence of ForestPolynomial
        Correction coefficients with an empty order zero and at least order one.

    Returns
    -------
    str
        LaTeX requiring amsmath and the paper's ForestCommands.tex dependencies.
        The supplied tree macros cover orders through six.
    """
    _validate_coefficients(coefficients)
    lines = [
        "% Generated by the bergomi_guyon CLI; do not edit.",
        "% Requires amsmath and ForestCommands.tex (tree macros through order six).",
        r"\begingroup",
        r"\footnotesize\tkz\raggedright",
        r"\everymath{\displaystyle}",
        r"\setlength{\jot}{1pt}",
        r"\thinmuskip=2mu \medmuskip=2mu \thickmuskip=3mu",
        r"\setlength{\abovedisplayskip}{5pt}\setlength{\belowdisplayskip}{5pt}",
        r"\setlength{\parskip}{3pt}",
    ]
    lines.extend([
        r"\[",
        r"\zeta=\frac12+\frac{k}{M},\qquad\theta=\frac1M,\qquad",
        r"\Sigma(k)=M+\sum_{\ell=1}^{" + str(len(coefficients) - 1)
        + r"}\epsilon^\ell a_\ell(k)+\mathcal{O}(\epsilon^{"
        + str(len(coefficients)) + r"}).",
        r"\]",
        "",
    ])
    for ell in range(1, len(coefficients)):
        lines.append(r"\addvspace{\baselineskip}")
        lines.append(rf"\textbf{{Order {ell}}}\\*[\baselineskip]")
        groups = group_latex_terms(coefficients[ell])
        if not groups:
            lines.append(rf"$a_{{{ell}}}(k) = 0$")
        for index, (coefficient, products) in enumerate(groups):
            factor, coefficient = factor_latex_prefactor(coefficient)
            factor_text = polynomial_to_latex(factor)
            if factor_text == "1":
                factor_text = ""
            polynomial = _latex_polynomial(coefficient)
            product = _latex_forests(products)
            prefix = rf"a_{{{ell}}}(k) =" if index == 0 else "+"
            lines.append("$" + prefix + factor_text
                         + polynomial + r"\," + product + "$")
        lines.append("")
    lines.append(r"\endgroup")
    return "\n".join(lines) + "\n"


def _validate_coefficients(coefficients: Sequence[ForestPolynomial]) -> None:
    """Reject series without an empty order zero and a positive-order entry."""
    if len(coefficients) < 2 or coefficients[0]:
        raise ValueError("coefficients must include an empty order zero and at least order one")


def render_coefficients(
    coefficients: Sequence[ForestPolynomial], output_format: str
) -> str:
    """Render the correction series, with an empty entry at index zero.

    Parameters
    ----------
    coefficients : sequence of ForestPolynomial
        Correction coefficients with an empty order zero and at least order one.
    output_format : {'text', 'latex'}
        Format of the returned expansion.

    Returns
    -------
    str
        Deterministically formatted expansion, including the base variance M.

    Raises
    ------
    ValueError
        If the series layout or output format is invalid.

    Notes
    -----
    LaTeX uses the paper's custom macros (provided through order six).
    Text uses larger-subtree-first names; Python exports use canonical tree
    ordering, so equivalent binary trees can have different postfix names.
    """
    if output_format not in {"text", "latex"}:
        raise ValueError("output_format must be text or latex")
    if output_format == "latex":
        return render_latex(coefficients)
    _validate_coefficients(coefficients)
    lines: list[str] = []
    emit = lines.append

    order = len(coefficients) - 1
    emit(f"Bergomi--Guyon total implied variance expansion through order {order}:")
    emit(
        f"  Sigma(k) = sum_{{ell=0}}^{order} a_ell * epsilon^ell "
        f"+ O(epsilon^{order + 1})"
    )
    emit("  Diamond trees use postfix names: Xd appends diamond X;")
    emit("  d joins the two preceding tree expressions. Examples:")
    emit("    MMd      = M diamond M")
    emit("    MXdXd    = (M diamond X) diamond X")
    emit("    MXdMdXd  = ((M diamond X) diamond M) diamond X")
    emit("")
    emit("variables:")
    emit("  zeta  = 1/2 + k/M")
    emit("  theta = 1/M")
    emit("  kappa = zeta - 1/2 = k/M")
    emit("")
    emit("# Order 0")
    emit("a_0:\n  M")
    for ell in range(1, len(coefficients)):
        emit("")
        emit(f"# Order {ell}")
        emit(f"a_{ell}:")
        if not coefficients[ell]:
            emit("  0")
        for monomial, coefficient in sorted(coefficients[ell].items()):
            emit(f"  ({polynomial_to_text(coefficient)}) * {monomial_text(monomial)}")

    return "\n".join(lines) + "\n"


def render_python_module(coefficients: Sequence[ForestPolynomial]) -> str:
    """Render explicit orders as an importable SymPy module.

    Parameters
    ----------
    coefficients : sequence of ForestPolynomial
        Correction coefficients with an empty order zero and at least order one.

    Returns
    -------
    str
        Python source defining exact expressions, tree symbols, the tuple a
        with a[0] = None, and the substitution helper in_k_and_M.

    Notes
    -----
    Rendering uses only the standard library; importing the output needs SymPy.
    """
    _validate_coefficients(coefficients)
    trees = sorted(
        {
            tree
            for coefficient in coefficients[1:]
            for monomial in coefficient
            for tree in monomial
        }
    )
    symbol_names = [tree_to_symbol(tree) for tree in trees]
    lines = [
        f'"""Generated exact Bergomi--Guyon coefficients through order {len(coefficients) - 1}.',
        "",
        "Usage::",
        "",
        f"    from bergomi_guyon.bg_coefficients_order_{len(coefficients) - 1} import a, in_k_and_M",
        f"    coefficient = a[{len(coefficients) - 1}]",
        "    coefficient_in_k_M = in_k_and_M(coefficient)",
        "",
        f"The tuple a contains orders 1 through {len(coefficients) - 1}, with a[0] = None.",
        "The zeroth-order coefficient is the separate symbol M.",
        "The formulas use zeta = 1/2 + k/M and theta = 1/M. Tree names use",
        "postfix notation: Xd is the unary diamond operation and d joins the",
        'two preceding tree expressions. This file is generated; do not edit."""',
        "",
        "from sympy import Rational, symbols",
        "",
        'zeta, theta, k, M = symbols("zeta theta k M")',
        "",
    ]
    if symbol_names:
        lines.extend(
            [
                "(",
                *(f"    {name}," for name in symbol_names),
                f') = symbols("{" ".join(symbol_names)}", seq=True)',
                "",
            ]
        )

    for ell in range(1, len(coefficients)):
        lines.append(f"a_{ell} = (")
        if not coefficients[ell]:
            lines.append("    Rational(0)")
        for index, (monomial, coefficient) in enumerate(
            sorted(coefficients[ell].items())
        ):
            tree_product = " * ".join(tree_to_symbol(tree) for tree in monomial) or "1"
            prefix = "    " if index == 0 else "    + "
            lines.append(f"{prefix}({polynomial_to_python(coefficient)}) * {tree_product}")
        lines.extend([")", ""])

    aliases = ", ".join(f"a_{ell}" for ell in range(1, len(coefficients)))
    lines.extend(
        [
            f"a = (None, {aliases})",
            "",
            "def in_k_and_M(expression):",
            '    """Substitute zeta = 1/2 + k/M and theta = 1/M.',
            "",
            "    Parameters",
            "    ----------",
            "    expression : sympy.Expr",
            "        Coefficient expression in zeta, theta, and tree symbols.",
            "",
            "    Returns",
            "    -------",
            "    sympy.Expr",
            "        Expression in k and M, with tree symbols unchanged.",
            '    """',
            "    return expression.subs({zeta: Rational(1, 2) + k / M, theta: 1 / M})",
            "",
        ]
    )
    return "\n".join(lines)
