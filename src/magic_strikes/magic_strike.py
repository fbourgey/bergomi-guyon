import warnings

import numpy as np

_SWAP_OPTS = (
    "variance",
    "gamma",
    "power",
)
_ALL_SWAP_OPTS = (
    "variance",
    "gamma",
)


def _orders_for_opt(opt):
    """Return the supported approximation orders for one contract.

    Parameters
    ----------
    opt : str
        Contract name. This helper accepts an individual contract, not ``"all"``.

    Returns
    -------
    tuple of int
        Supported orders, currently 1 through 5 for every contract.

    Raises
    ------
    ValueError
        If ``opt`` is not a supported contract name.
    """
    if opt == "variance":
        return (1, 2, 3, 4, 5)
    if opt == "gamma":
        return (1, 2, 3, 4, 5)
    if opt == "power":
        return (1, 2, 3, 4, 5)
    raise ValueError("opt must be either 'variance', 'gamma', 'power', or 'all'")


def _validate_power_parameter(opt, p):
    """Validate the scalar power parameter for one contract selection."""
    if opt == "power":
        if p is None:
            raise ValueError("p is required when opt='power'")
        p_array = np.asarray(p, dtype=float)
        if p_array.ndim != 0:
            raise ValueError("p must be a scalar")
        p = float(p_array)
        if not np.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError("p must be finite and between 0 and 1")
        return p
    if p is not None:
        raise ValueError("p may only be specified when opt='power'")
    return None


def _normalize_orders(opt, order):
    """Normalize an order selector to a validated tuple of integers.

    ``None`` and ``"max"`` select every order supported by ``opt``.

    Parameters
    ----------
    opt : str
        Individual contract name.
    order : int, array_like, ``"max"``, or None
        Requested approximation order or orders.

    Returns
    -------
    tuple of int
        Requested orders in input order.

    Raises
    ------
    ValueError
        If the selection is empty or contains an unsupported order.
    """
    supported = _orders_for_opt(opt)
    if order is None or order == "max":
        return supported

    orders = tuple(int(order_i) for order_i in np.atleast_1d(order))
    if not orders:
        raise ValueError("order must contain at least one order")

    unsupported = sorted(set(orders).difference(supported))
    if unsupported:
        supported_str = ", ".join(str(order_i) for order_i in supported)
        raise ValueError(f"order must be one of {supported_str} for opt='{opt}'")
    return orders


def _to_scalar(value) -> float:
    """Convert a size-one numeric smile result to ``float``.

    Raises
    ------
    ValueError
        If ``value`` does not contain exactly one element.
    """
    arr = np.asarray(value, dtype=float)
    if arr.size != 1:
        raise ValueError("Magic-strike formulas require scalar smile evaluations.")
    return float(arr.reshape(-1)[0])


def _cached_smile(tot_impvar, *, decimals: int = 12):
    """Return a scalar smile wrapper with rounded-strike memoization.

    Parameters
    ----------
    tot_impvar : callable
        Scalar total-implied-variance smile ``f(k)``.
    decimals : int, default 12
        Number of decimal places used to form cache keys.

    Returns
    -------
    callable
        Scalar smile function whose values are cached by rounded strike.
    """
    cache = {}

    def scalar(k):
        """Return one cached scalar smile evaluation."""
        key = round(float(k), decimals)
        if key not in cache:
            cache[key] = _to_scalar(tot_impvar(key))
        return cache[key]

    return scalar


def _initial_guess(f) -> float:
    """Return positive ATM total implied variance as a fixed-point seed.

    Parameters
    ----------
    f : callable
        Scalar total-implied-variance smile.

    Returns
    -------
    float
        ``f(0)``.

    Raises
    ------
    ValueError
        If the ATM value is nonfinite or nonpositive.
    """
    guess = f(0.0)
    if not np.isfinite(guess) or guess <= 0.0:
        raise ValueError(
            "Initial total implied variance at k=0 must be finite and > 0."
        )
    return guess


def _relaxed_fixed_point(fun, x0, tol, max_iter, relax, *, positive: bool = True):
    """Solve a scalar fixed point with damping and finite-value fallback.

    The iteration applies ``x <- (1-relax)*x + relax*fun(x)``. Nonfinite or,
    when requested, nonpositive candidates halve the damping factor. Failure
    returns the last finite iterate and emits :class:`RuntimeWarning`.

    Parameters
    ----------
    fun : callable
        Scalar fixed-point map.
    x0 : float
        Initial iterate.
    tol : float
        Absolute stopping tolerance between consecutive iterates.
    max_iter : int
        Maximum number of iterations.
    relax : float
        Initial damping factor.
    positive : bool, default True
        Whether accepted iterates must remain strictly positive.

    Returns
    -------
    float
        Converged value or last finite fallback.

    Warns
    -----
    RuntimeWarning
        If convergence is not reached.
    """
    x = float(x0)
    last_finite = x
    damping = float(relax)

    for _ in range(max_iter):
        candidate = fun(x)
        if not np.isfinite(candidate):
            damping *= 0.5
            if damping < 1e-4:
                break
            continue

        x_new = (1.0 - damping) * x + damping * float(candidate)
        if positive and x_new <= 0.0:
            damping *= 0.5
            if damping < 1e-4:
                break
            continue

        if not np.isfinite(x_new):
            damping *= 0.5
            if damping < 1e-4:
                break
            continue

        last_finite = x_new
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    else:
        warnings.warn(
            "Magic-strike fixed-point iteration reached max_iter without "
            "converging; returning last finite iterate.",
            RuntimeWarning,
            stacklevel=2,
        )
        return last_finite

    warnings.warn(
        "Magic-strike fixed-point iteration did not converge; returning last finite "
        "iterate.",
        RuntimeWarning,
        stacklevel=2,
    )
    return last_finite


def _relaxed_log_fixed_point(fun, x0, tol, max_iter, relax):
    """Solve a positive scalar fixed point using log-space damping.

    Nonfinite or nonpositive candidates halve the damping factor. Failure
    returns the last finite positive iterate and emits :class:`RuntimeWarning`.

    Parameters
    ----------
    fun : callable
        Positive scalar fixed-point map.
    x0 : float
        Strictly positive initial iterate.
    tol : float
        Absolute stopping tolerance in the original value space.
    max_iter : int
        Maximum number of iterations.
    relax : float
        Initial damping factor in log space.

    Returns
    -------
    float
        Converged value or last finite fallback.

    Raises
    ------
    ValueError
        If ``x0`` is nonfinite or nonpositive.

    Warns
    -----
    RuntimeWarning
        If convergence is not reached.
    """
    x0 = float(x0)
    if not np.isfinite(x0) or x0 <= 0.0:
        raise ValueError("Log fixed-point iteration requires a positive initial guess.")

    y = np.log(x0)
    last_finite = x0
    damping = float(relax)

    for _ in range(max_iter):
        x = np.exp(y)
        candidate = fun(x)
        if not np.isfinite(candidate) or candidate <= 0.0:
            damping *= 0.5
            if damping < 1e-4:
                break
            continue

        y_new = (1.0 - damping) * y + damping * np.log(float(candidate))
        x_new = np.exp(y_new)
        if not np.isfinite(x_new):
            damping *= 0.5
            if damping < 1e-4:
                break
            continue

        last_finite = x_new
        if abs(x_new - x) < tol:
            return x_new
        y = y_new
    else:
        warnings.warn(
            "Magic-strike fixed-point iteration reached max_iter without "
            "converging; returning last finite iterate.",
            RuntimeWarning,
            stacklevel=2,
        )
        return last_finite

    warnings.warn(
        "Magic-strike fixed-point iteration did not converge; returning last finite "
        "iterate.",
        RuntimeWarning,
        stacklevel=2,
    )
    return last_finite


def _finite_differences(
    f, center, spacing, max_order
) -> tuple[float, float, float, float, float, float]:
    """Evaluate the paper's normalized unit-stencil finite differences.

    Values are sampled around ``center`` at integer multiples of ``spacing``.
    Orders 1--2 use the inner three points, orders 3--4 use five points, and
    order 5 uses seven points. These are the normalized combinations entering
    the magic-strike formulas; they should not be interpreted as a generic
    finite-difference API. Components above ``max_order`` are returned as NaN.

    Parameters
    ----------
    f : callable
        Scalar total-implied-variance smile.
    center : float
        Stencil center.
    spacing : float
        Strictly positive unit-stencil spacing.
    max_order : int
        Highest normalized difference to evaluate, from 0 through 5.

    Returns
    -------
    tuple of float
        Six entries ``(d0, d1, d2, d3, d4, d5)``. Unevaluated entries are NaN.

    Raises
    ------
    ValueError
        If ``max_order``, ``center``, or ``spacing`` is invalid.
    """
    if max_order not in [0, 1, 2, 3, 4, 5]:
        raise ValueError("max_order must be between 0 and 5")
    if not np.isfinite(center):
        raise ValueError("finite-difference center must be finite")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("finite-difference spacing must be finite and positive")
    center = float(center)
    spacing = float(spacing)

    missing = np.nan

    d0 = f(center)
    if max_order == 0:
        return (d0, missing, missing, missing, missing, missing)

    f_plus = f(center + spacing)
    f_minus = f(center - spacing)
    d1 = (f_plus - f_minus) / (2 * spacing)
    if max_order == 1:
        return (d0, d1, missing, missing, missing, missing)

    d2 = (f_plus + f_minus - 2 * d0) / (2 * spacing**2)
    if max_order == 2:
        return (d0, d1, d2, missing, missing, missing)

    f_2plus = f(center + 2 * spacing)
    f_2minus = f(center - 2 * spacing)
    d3 = (f_2plus - 2 * f_plus + 2 * f_minus - f_2minus) / (12 * spacing**3)
    if max_order == 3:
        return (d0, d1, d2, d3, missing, missing)

    d4 = (f_2plus - 4 * f_plus - 4 * f_minus + f_2minus + 6 * d0) / (12 * spacing**2)
    if max_order == 4:
        return (d0, d1, d2, d3, d4, missing)

    f_3plus = f(center + 3 * spacing)
    f_3minus = f(center - 3 * spacing)
    d5 = (
        19
        * (f_3plus - 4 * f_2plus + 5 * f_plus - 5 * f_minus + 4 * f_2minus - f_3minus)
        / (240 * spacing**5)
    )
    return (d0, d1, d2, d3, d4, d5)


def swap(
    tot_impvar,
    order=None,
    opt="variance",
    tol=1e-8,
    max_iter=100,
    relax=0.8,
    p=None,
):
    """
    Compute magic-strike approximations from a scalar total-variance smile.

    Parameters
    ----------
    tot_impvar : callable
        Scalar function of log-moneyness returning Black total implied variance
        ``sigma_BS(k, T)**2 * T`` for one fixed maturity.
    order : int, array_like, ``"max"``, or None, optional
        Approximation order or orders. ``None`` and ``"max"`` select every
        supported order.
    opt : {'variance', 'gamma', 'power',
        'all'}, default 'variance'
        Contract to compute. ``"gamma"`` is the direct gamma fixed point.
        ``"all"`` returns variance and gamma.
    p : float, optional
        Power-payoff parameter in ``[0, 1]``. Required exactly when
        ``opt="power"``. The returned quantity is the implied power variance
        ``W_p`` from the paper, not the power-payoff price.
    tol : float, default 1e-8
        Absolute fixed-point stopping tolerance.
    max_iter : int, default 100
        Maximum number of fixed-point iterations.
    relax : float, default 0.8
        Initial fixed-point damping factor.

    Returns
    -------
    float or dict
        A scalar when one order of one contract is requested, an order-to-value
        dictionary for several orders, or a nested contract/order dictionary
        when ``opt="all"``.

    Raises
    ------
    ValueError
        If the contract, order selection, smile values, or fixed-point inputs
        are invalid.
    """
    if opt == "all":
        _validate_power_parameter(opt, p)
        return swap_many(
            tot_impvar=tot_impvar,
            orders_by_opt={
                opt_i: _normalize_orders(opt_i, order) for opt_i in _ALL_SWAP_OPTS
            },
            tol=tol,
            max_iter=max_iter,
            relax=relax,
        )

    if opt not in _SWAP_OPTS:
        raise ValueError("opt must be either 'variance', 'gamma', 'power', or 'all'")

    p = _validate_power_parameter(opt, p)
    orders = _normalize_orders(opt, order)
    out = swap_many(
        tot_impvar=tot_impvar,
        orders_by_opt={opt: orders},
        p=p,
        tol=tol,
        max_iter=max_iter,
        relax=relax,
    )
    if len(orders) == 1:
        return out[opt][orders[0]]
    return out[opt]


def swap_many(
    tot_impvar,
    orders_by_opt,
    tol=1e-8,
    max_iter=100,
    relax=0.8,
    p=None,
):
    """Compute several contracts and orders from one cached scalar smile.

    Smile evaluations are cached across contracts and orders. Direct
    gamma orders are computed successively so that each lower-order solution
    initializes the next fixed point.

    Parameters
    ----------
    tot_impvar : callable
        Scalar total-implied-variance smile for one maturity.
    orders_by_opt : mapping
        Contract names mapped to requested order selectors.
    p : float, optional
        Scalar power parameter, required exactly when ``orders_by_opt`` contains
        ``"power"``.
    tol : float, default 1e-8
        Absolute fixed-point stopping tolerance.
    max_iter : int, default 100
        Maximum number of fixed-point iterations.
    relax : float, default 0.8
        Initial fixed-point damping factor.

    Returns
    -------
    dict
        Nested dictionary ``result[contract][order]`` of scalar total values.

    Raises
    ------
    ValueError
        If a contract or order selection is unsupported.
    """
    f = _cached_smile(tot_impvar)
    has_power = "power" in orders_by_opt
    p = _validate_power_parameter("power" if has_power else "variance", p)
    variance_cache = {}

    def variance(order):
        """Return one cached variance approximation."""
        order = int(order)
        if order not in variance_cache:
            variance_cache[order] = _variance_swap(
                f=f,
                order=order,
                tol=tol,
                max_iter=max_iter,
                relax=relax,
            )
        return variance_cache[order]

    gamma_cache = {}
    power_cache = {}

    def gamma(order):
        """Return one cached direct-gamma approximation."""
        order = int(order)
        if order not in gamma_cache:
            x0 = None if order == 1 else gamma(order - 1)
            gamma_cache[order] = _gamma_swap(
                f=f,
                order=order,
                tol=tol,
                max_iter=max_iter,
                relax=relax,
                x0=x0,
            )
        return gamma_cache[order]

    def power(order):
        """Return one cached implied-power-variance approximation."""
        order = int(order)
        if order not in power_cache:
            x0 = None if order == 1 else power(order - 1)
            power_cache[order] = _power_swap(
                f=f,
                order=order,
                p=p,
                tol=tol,
                max_iter=max_iter,
                relax=relax,
                x0=x0,
            )
        return power_cache[order]

    out = {}
    for opt, orders in orders_by_opt.items():
        orders = _normalize_orders(opt, orders)
        if opt == "variance":
            out[opt] = {order: variance(order) for order in orders}
        elif opt == "gamma":
            out[opt] = {order: gamma(order) for order in orders}
        elif opt == "power":
            out[opt] = {order: power(order) for order in orders}
        else:
            raise ValueError(
                "opt must be either 'variance', 'gamma', 'power', or 'all'"
            )
    return out


def _variance_swap(f, order, tol=1e-14, max_iter=300, relax=0.8):
    """Solve one direct variance-contract magic-strike fixed point.

    The unit stencil is centered at ``-M/2`` with spacing ``sqrt(M)``. Orders
    1--5 use the corresponding formulas from the paper.

    Parameters
    ----------
    f : callable
        Cached scalar total-implied-variance smile.
    order : int
        Approximation order from 1 through 5.
    tol : float, default 1e-14
        Absolute fixed-point stopping tolerance.
    max_iter : int, default 300
        Maximum number of iterations.
    relax : float, default 0.8
        Initial damping factor.

    Returns
    -------
    float
        Total variance-contract approximation.

    Raises
    ------
    ValueError
        If the order or an iterated total-variance level is invalid.
    """
    if order not in [1, 2, 3, 4, 5]:
        raise ValueError("order must be either 1, 2, 3, 4, or 5")

    def derivatives(m):
        """Evaluate the normalized variance stencil at level ``m``."""
        if not np.isfinite(m) or m <= 0.0:
            raise ValueError(
                "Variance swap iteration requires positive total variance."
            )
        max_derivative_order = 0 if order == 1 else 2 if order <= 3 else 4
        return _finite_differences(
            f,
            center=-0.5 * m,
            spacing=m**0.5,
            max_order=max_derivative_order,
        )

    def fun(m):
        """Return the selected-order variance fixed-point update."""
        d0, d1, d2, d3, d4, _d5 = derivatives(m)

        if order == 1:
            return d0

        M1 = d0

        M2 = m * d2 + 0.5 * d1**2 + M1

        if order == 2:
            return M2

        M3 = -m * d1 * d2 - 0.25 * d1**3 + M2

        if order == 3:
            return M3

        M4 = (
            m * d4
            + 5 * m * d3 * d1
            + 2 * m * d2**2
            + 0.25 * (10 + 3 * m) * d2 * d1**2
            + (1 / 8) * d1**4
            + M3
        )

        if order == 4:
            return M4

        M5 = (
            -(5 / 2) * m * d4 * d1
            - 5 * m**2 * d3 * d2
            - (21 / 2) * m * d3 * d1**2
            - 10 * m * d2**2 * d1
            - 0.5 * (9 + m) * d2 * d1**3
            - (1 / 16) * d1**5
            + M4
        )

        return M5

    return _relaxed_fixed_point(fun, _initial_guess(f), tol, max_iter, relax)


def _gamma_swap(f, order, tol=1e-14, max_iter=300, relax=0.8, x0=None):
    """Solve one direct gamma-contract magic-strike fixed point.

    The positive fixed point is solved in log space. Its unit stencil is
    centered at ``G/2`` with spacing ``sqrt(G)``.

    Parameters
    ----------
    f : callable
        Cached scalar total-implied-variance smile.
    order : int
        Approximation order from 1 through 5.
    tol : float, default 1e-14
        Absolute stopping tolerance in total-variance units.
    max_iter : int, default 300
        Maximum number of iterations.
    relax : float, default 0.8
        Initial log-space damping factor.
    x0 : float or None, optional
        Positive initial level. If omitted, use ATM total implied variance.

    Returns
    -------
    float
        Total direct gamma-contract approximation.

    Raises
    ------
    ValueError
        If the order, initial level, or an iterated level is invalid.
    """
    if order not in [1, 2, 3, 4, 5]:
        raise ValueError("order must be either 1, 2, 3, 4, or 5")

    def fun(x):
        """Return the selected-order gamma fixed-point update."""
        if not np.isfinite(x) or x <= 0.0:
            raise ValueError("Gamma swap iteration requires positive total variance.")

        max_derivative_order = 0 if order == 1 else 2 if order <= 3 else 4
        d0, d1, d2, d3, d4, _ = _finite_differences(
            f,
            center=0.5 * x,
            spacing=x**0.5,
            max_order=max_derivative_order,
        )

        G1 = d0
        if order == 1:
            return G1

        G2 = G1 + x * d2 + (1 / 2) * d1**2
        if order == 2:
            return G2

        G3 = G2 + x * d2 * d1 + (1 / 4) * d1**3
        if order == 3:
            return G3

        G4 = (
            x * d4
            + 5 * x * d3 * d1
            + 2 * x * d2**2
            + 0.25 * (10 + 3 * x) * d2 * d1**2
            + (1 / 8) * d1**4
            + G3
        )
        if order == 4:
            return G4

        G5 = (
            (5 / 2) * x * d4 * d1
            + 5 * x**2 * d3 * d2
            + (21 / 2) * x * d3 * d1**2
            + 10 * x * d2**2 * d1
            + 0.5 * (x + 9) * d2 * d1**3
            + (1 / 16) * d1**5
            + G4
        )
        return G5

    if x0 is None:
        x0 = _initial_guess(f)
    elif not np.isfinite(x0) or x0 <= 0.0:
        raise ValueError("Gamma swap initial guess must be finite and positive.")

    return _relaxed_log_fixed_point(fun, x0, tol, max_iter, relax)


def _power_swap(f, order, p, tol=1e-14, max_iter=300, relax=0.8, x0=None):
    """Solve the paper's implied-power-variance fixed point through order 5."""
    if order not in [1, 2, 3, 4, 5]:
        raise ValueError("order must be either 1, 2, 3, 4, or 5")
    p = _validate_power_parameter("power", p)
    lambda_p = 0.5 * p * (p - 1.0)
    q = 2.0 * p - 1.0

    def fun(x):
        if not np.isfinite(x) or x <= 0.0:
            raise ValueError(
                "Power iteration requires positive implied power variance."
            )

        max_derivative_order = 0 if order == 1 else 2 if order <= 3 else 4
        d0, d1, d2, d3, d4, _ = _finite_differences(
            f,
            center=(p - 0.5) * x,
            spacing=x**0.5,
            max_order=max_derivative_order,
        )
        omega = lambda_p * x

        P1 = d0
        if order == 1:
            return P1

        P2 = x * d2 + 0.5 * (1.0 + omega) * d1**2 + P1
        if order == 2:
            return P2

        P3 = q * (x * d2 * d1 + 0.25 * (1.0 + omega) * d1**3) + P2
        if order == 3:
            return P3

        P4 = (
            x * d4
            + (5.0 + 2.0 * omega) * x * d3 * d1
            + (2.0 + omega) * x * d2**2
            + (2.5 + 0.75 * x + 10.5 * omega + omega**2) * d2 * d1**2
            + 0.125 * (1.0 + 11.0 * lambda_p + (10.0 * lambda_p + 1.0) * omega) * d1**4
            + P3
        )
        if order == 4:
            return P4

        P5 = (
            q
            * (
                2.5 * x * d4 * d1
                + 5.0 * x**2 * d3 * d2
                + (10.5 + 4.5 * omega) * x * d3 * d1**2
                + (10.0 + 4.5 * omega) * x * d2**2 * d1
                + (4.5 + 0.5 * x + 12.0 * omega + 1.5 * omega**2) * d2 * d1**3
                + (1.0 / 16.0)
                * (1.0 + 17.0 * lambda_p + (14.0 * lambda_p + 1.0) * omega)
                * d1**5
            )
            + P4
        )
        return P5

    if x0 is None:
        x0 = _initial_guess(f)
    elif not np.isfinite(x0) or x0 <= 0.0:
        raise ValueError("Power initial guess must be finite and positive.")

    return _relaxed_log_fixed_point(fun, x0, tol, max_iter, relax)


def _magic_strike_grid(
    total_variance: float, order: int, opt: str = "variance"
) -> np.ndarray:
    """Return the fixed unit-stencil variance or gamma magic strikes.

    The grid is centered at ``-total_variance/2`` for a variance contract and
    ``total_variance/2`` for a gamma contract, with spacing
    ``sqrt(total_variance)``. Order 1 uses one strike, orders 2--3 use three,
    and orders 4--5 use five.

    Parameters
    ----------
    total_variance : float
        Total variance level determining the center and spacing.
    order : int
        Approximation order from 1 through 5.
    opt : {'variance', 'gamma'}, default 'variance'
        Contract determining the fixed stencil center.

    Returns
    -------
    ndarray
        Increasing array of log-moneyness strikes.

    Raises
    ------
    ValueError
        If ``order`` is outside 1 through 5 or ``opt`` is unsupported.
    """
    if opt not in {"variance", "gamma"}:
        raise ValueError("opt must be either 'variance' or 'gamma'")

    k0 = (-0.5 if opt == "variance" else 0.5) * total_variance
    dk = np.sqrt(total_variance)

    if order == 1:
        return np.array([k0])
    if order in [2, 3]:
        return np.array([k0 - dk, k0, k0 + dk])
    if order in [4, 5]:
        return np.array(
            [
                k0 - 2 * dk,
                k0 - dk,
                k0,
                k0 + dk,
                k0 + 2 * dk,
            ]
        )
    raise ValueError("order must be 1, 2, 3, 4, or 5")
