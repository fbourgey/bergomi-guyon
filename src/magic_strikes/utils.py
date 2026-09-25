import warnings
from functools import cache
from operator import index

import numpy as np
from numfracpy import Mittag_Leffler_two
from scipy import stats
from scipy.integrate import quad_vec

# Module-level constants
IMPVOL_MIN = 1e-10
IMPVOL_MAX = 5.0


def _validate_quadrature_order(n: int) -> int:
    """Validate and normalize a positive quadrature order."""
    if isinstance(n, (bool, np.bool_)):
        raise ValueError("n must be a positive integer.")  # noqa: TRY004

    try:
        n = index(n)
    except TypeError as exc:
        raise TypeError("n must be an integer.") from exc

    if n <= 0:
        raise ValueError("n must be a positive integer.")

    return n


@cache
def _cached_hermgauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached probabilist-scaled Gauss-Hermite nodes and weights."""
    knots, weights = np.polynomial.hermite.hermgauss(n)
    return knots * np.sqrt(2.0), weights / np.sqrt(np.pi)


@cache
def _cached_leggauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached Gauss-Legendre nodes and weights on ``[-1, 1]``."""
    return np.polynomial.legendre.leggauss(n)


def black_price(K, T, F, vol, opttype: float | np.ndarray = 1.0):
    """
    Calculate the Black option price.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    vol : float
        Volatility of the underlying asset.
    opttype : float or np.ndarray, optional
        Option type: 1 for call options, -1 for put options. Default is 1.

    Returns
    -------
    ndarray
        Black option prices with the broadcasted input shape.
    """
    T = float(T)
    K, F, vol, opttype = np.broadcast_arrays(
        np.asarray(K, dtype=float),
        np.asarray(F, dtype=float),
        np.asarray(vol, dtype=float),
        np.asarray(opttype, dtype=float),
    )

    if not np.all(np.abs(opttype) == 1.0):
        raise ValueError("opttype must be either 1 or -1.")

    valid = (K > 0.0) & (F > 0.0) & np.isfinite(K) & np.isfinite(F)
    intrinsic = np.maximum(opttype * (F - K), 0.0)
    price = np.where(valid, intrinsic, np.nan)

    positive_var = valid & (T > 0.0) & (vol > 0.0) & np.isfinite(vol)
    if np.any(positive_var):
        s = vol[positive_var] * T**0.5
        d1 = np.log(F[positive_var] / K[positive_var]) / s + 0.5 * s
        d2 = d1 - s
        price[positive_var] = opttype[positive_var] * (
            F[positive_var] * stats.norm.cdf(opttype[positive_var] * d1)
            - K[positive_var] * stats.norm.cdf(opttype[positive_var] * d2)
        )

    return price


def black_impvol(
    K, T, F, value, opttype: int | np.ndarray = 1, TOL=1e-10, MAX_ITER=1000
):
    """
    Calculate the Black implied volatility using a bisection method.

    Parameters
    ----------
    K : ndarray or float
        Strike price(s) of the option(s).
    T : float
        Time to maturity of the option(s).
    F : float
        Forward price of the underlying asset.
    value : ndarray or float
        Observed market price(s) of the option(s).
    opttype : int or ndarray, optional
        Option type: 1 for call options, -1 for put options. Default is 1.
    TOL : float, optional
        Tolerance for convergence of the implied volatility. Default is 1e-10.
    MAX_ITER : int, optional
        Maximum number of iterations for the bisection method. Default is 1000.

    Returns
    -------
    ndarray or float
        Implied volatility(ies) corresponding to the input option prices. If the
        input arrays are multidimensional, the output will have the same shape.
        Returns NaN if the implied volatility does not converge or if invalid
        inputs are provided.

    Raises
    ------
    ValueError
        If `K` and `value` do not have the same shape.
        If `opttype` is not 1 or -1.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    value = np.atleast_1d(np.asarray(value, dtype=float))
    try:
        opttype = np.broadcast_to(np.asarray(opttype, dtype=float), K.shape)
    except ValueError as exc:
        raise ValueError("opttype must be scalar or have the same shape as K.") from exc

    if K.shape != value.shape:
        raise ValueError("K and value must have the same shape.")

    if not np.all(np.abs(opttype) == 1):
        raise ValueError("opttype must be either 1 or -1.")

    F = float(F)
    T = float(T)

    if T <= 0 or F <= 0:
        return np.full_like(K, np.nan, dtype=float)

    low = IMPVOL_MIN * np.ones_like(K, dtype=float)
    high = IMPVOL_MAX * np.ones_like(K, dtype=float)
    mid = 0.5 * (low + high)

    low_price = black_price(K, T, F, low, opttype)
    high_price = black_price(K, T, F, high, opttype)
    price_tol = TOL * np.maximum(np.abs(value), 1.0)
    valid = (
        (K > 0.0)
        & np.isfinite(value)
        & (value >= low_price - price_tol)
        & (value <= high_price + price_tol)
    )
    impvol = np.full_like(K, np.nan, dtype=float)
    active = valid.copy()

    for _ in range(MAX_ITER):
        if not np.any(active):
            return impvol

        active_idx = np.flatnonzero(active)
        price = black_price(K[active_idx], T, F, mid[active_idx], opttype[active_idx])
        diff = price - value[active_idx]
        converged = (np.abs(diff) <= price_tol[active_idx]) | (
            high[active_idx] - low[active_idx] <= TOL
        )

        converged_idx = active_idx[converged]
        impvol[converged_idx] = mid[converged_idx]
        active[converged_idx] = False

        remaining_idx = active_idx[~converged]
        high_idx = remaining_idx[diff[~converged] > 0.0]
        low_idx = remaining_idx[diff[~converged] <= 0.0]
        high[high_idx] = mid[high_idx]
        low[low_idx] = mid[low_idx]
        mid[remaining_idx] = 0.5 * (low[remaining_idx] + high[remaining_idx])

    warnings.warn(
        "Implied volatility did not converge for all log(K/F) values.",
        RuntimeWarning,
        stacklevel=2,
    )

    return impvol


def lewis_formula_otm_price(phi, k, T, epsrel: float = 1e-10, limit: int = 1000):
    """
    Compute the OTM (Out-of-The-Money) option price using the Lewis formula.

    The Lewis formula is used to price European options by applying Fourier transform
    methods. It calculates the price of OTM options given the characteristic function
    of the log price.

    Parameters
    ----------
    phi : callable
        The characteristic function of the log price process.
        Should take two arguments: complex number u and time to maturity T.
    k : float or array_like
        Log strike price(s). k = log(K/S) where K is strike and S is spot price.
    T : float or array_like
        Time(s) to maturity in years.

    Returns
    -------
    ndarray
        OTM option prices. For k < 0, returns put prices; for k >= 0, returns
        call prices.

    Notes
    -----
    The formula uses the following representation:
    For k >= 0 (calls): C(k,T) = 1/π ∫[0,∞] Re[e^(-iuk)φ(u-i/2,T)/(u^2+1/4)]du
    For k < 0 (puts): P(k,T) = e^k - 1/π ∫[0,∞] Re[e^(-iuk)φ(u-i/2,T)/(u^2+1/4)]du

    References
    ----------
    Lewis, A. L. (2000). Option Valuation under Stochastic Volatility.
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    T = np.atleast_1d(np.asarray(T, dtype=float))

    if not np.all(T > 0.0):
        raise ValueError("T must be positive.")

    def integrand(u):
        return np.real(np.exp(-1j * u * k) * phi(u - 0.5j, T) / (u**2 + 0.25))

    integral, _ = quad_vec(
        integrand,
        0,
        np.inf,
        epsrel=epsrel,
        limit=limit,
    )
    k_minus = k * (k < 0)
    return np.exp(k_minus) - np.exp(k / 2) / np.pi * integral


def implied_vol_from_paths(
    k: float | np.ndarray,
    T: float,
    int_v_dt: np.ndarray,
    int_sqrt_v_dw: np.ndarray,
    s0: float,
    conditioning: bool = False,
    return_skew: bool = False,
    rho_cond: float | None = None,
):
    """
    Estimate the implied volatility (and optionally its skew) from simulated paths.

    Parameters
    ----------
    k : float or np.ndarray, optional
        Log-strike k = log(S/S_0).
    T : float
        Maturity.
    int_v_dt : np.ndarray
        Array of time-integrated variances (shape: n_samples,).
    int_sqrt_v_dw : np.ndarray
        Array of stochastic integrals `int sqrt(v) dW` (shape: n_samples,).
    s0 : float
        Initial stock price.
    conditioning : bool, optional
        If True, use conditioning technique. See Bergomi (2016) - Stochastic Volatility
        Modeling - Chapter 8 - Appendix A.
    return_skew : bool, optional
        If True, also return the estimated implied volatility skew.
        Default is False. Monte Carlo estimation is used, see Bourgey et al. (2024)
        - Local volatility under rough volatility - Section 4.
    rho_cond : float, optional
        Correlation parameter for conditioning. Required if `conditioning` is True.

    Returns
    -------
    float or tuple
        Estimated implied volatility. If `return_skew` is True, returns a
        tuple (implied_vol, implied_vol_skew).
    """
    if T <= 0.0:
        raise ValueError("T must be positive.")

    if s0 <= 0.0:
        raise ValueError("s0 must be positive.")

    if int_v_dt.shape != int_sqrt_v_dw.shape:
        raise ValueError("int_v_dt and int_sqrt_v_dw must have the same shape.")

    k = np.atleast_1d(np.asarray(k, dtype=float))
    int_v_dt = np.asarray(int_v_dt).flatten()
    int_sqrt_v_dw = np.asarray(int_sqrt_v_dw).flatten()

    if conditioning:
        if rho_cond is None:
            raise ValueError("rho_cond must be provided when conditioning is True.")

        if not (-1.0 <= rho_cond <= 1.0):
            raise ValueError("rho_cond must be between -1 and 1.")

        s0_cond = s0 * np.exp(-0.5 * rho_cond**2 * int_v_dt + rho_cond * int_sqrt_v_dw)
        vol_cond = np.sqrt((1.0 - rho_cond**2) * int_v_dt / T)
        F = s0_cond.mean()
        K = F * np.exp(k)
        opttype = 2.0 * (K >= F) - 1.0
        price_cond = black_price(
            K=K[None, :],
            T=T,
            F=s0_cond[:, None],
            vol=vol_cond[:, None],
            opttype=opttype[None, :],
        ).mean(axis=0)
        impvol = black_impvol(K=K, T=T, F=F, value=price_cond, opttype=opttype)

        if return_skew:
            # Control variate to compute digital prices.
            w_cond = vol_cond * T**0.5
            d2_cond = (
                np.log(s0_cond[:, None] / K[None, :]) / w_cond[:, None]
                - 0.5 * w_cond[:, None]
            )
            digit = stats.norm.cdf(d2_cond).mean(axis=0)
    else:
        S = s0 * np.exp(-0.5 * int_v_dt + int_sqrt_v_dw)
        F = S.mean()
        K = F * np.exp(k)
        opttype = 2.0 * (K >= F) - 1.0
        payoff = np.maximum(opttype[None, :] * (S[:, None] - K[None, :]), 0.0)
        otm_price = np.mean(payoff, axis=0)
        impvol = black_impvol(K=K, T=T, F=F, value=otm_price, opttype=opttype)

        if return_skew:
            digit = np.mean(S[:, None] >= K[None, :], axis=0)

    if return_skew:
        w = impvol * T**0.5
        d2 = -k / w - 0.5 * w
        skew = (stats.norm.cdf(d2) - digit) / (stats.norm.pdf(d2) * T**0.5)
        return impvol, skew

    return impvol


def gauss_legendre(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Gauss-Legendre quadrature points and weights on the interval [a, b].

    Parameters
    ----------
    a : float
        Lower bound of the integration interval.
    b : float
        Upper bound of the integration interval.
    n : int
        Number of quadrature points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two 1-D arrays:
        - Quadrature points on [a, b].
        - Quadrature weights on [a, b].
    """
    n = _validate_quadrature_order(n)
    knots, weights = _cached_leggauss(n)
    knots_a_b = 0.5 * (b - a) * knots + 0.5 * (b + a)
    weights_a_b = 0.5 * (b - a) * weights
    return knots_a_b, weights_a_b


def mittag_leffler_two(z, alpha, beta):
    """Mittag-Leffler function E_{alpha, beta}(z) for alpha > 0 and beta > 0."""
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive.")

    return np.vectorize(
        lambda u: Mittag_Leffler_two(u, alpha, beta),
        otypes=[complex],
    )(z)


def cholesky_from_svd(a: np.ndarray) -> np.ndarray:
    """
    Compute a square-root factor of a positive semi-definite matrix.

    This function is a fallback for numerically positive semi-definite matrices.

    Parameters
    ----------
    a : np.ndarray
        The input matrix.

    Returns
    -------
    np.ndarray
        Matrix factor `b` such that `b @ b.T` approximates the input matrix.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a must be a square matrix.")

    a = 0.5 * (a + a.T)
    eigvals, eigvecs = np.linalg.eigh(a)
    tol = np.finfo(float).eps * a.shape[0] * max(1.0, np.max(np.abs(eigvals)))

    if np.any(eigvals < -tol):
        raise np.linalg.LinAlgError("Matrix is not positive semi-definite.")

    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs * np.sqrt(eigvals)


def _xi0_heston(t, lbd, vbar, v):
    """
    Forward variance curve for the Heston model.

    Parameters
    ----------
    t : array_like
        Time to maturity
    lbd : float
        Mean reversion rate
    vbar : float
        Long-term variance
    v : float
        Initial variance
    Returns
    -------
    array_like
        Forward variance curve values at time t
    """
    if lbd < 0:
        raise ValueError("lbd must be non-negative.")
    if vbar < 0:
        raise ValueError("vbar must be non-negative.")
    if v < 0:
        raise ValueError("v must be non-negative.")
    if lbd == 0.0:
        return np.full_like(t, v)
    return vbar + (v - vbar) * np.exp(-lbd * t)


def relative_error(true_value, approx_value):
    """
    Calculate the relative error between a true value and an approximate value.

    Parameters
    ----------
    true_value : float
        The true value.
    approx_value : float
        The approximate value.

    Returns
    -------
    float
        The relative error, defined as (approx_value - true_value) / true_value.
        Returns NaN if true_value is zero to avoid division by zero.
    """
    return np.where(true_value == 0, np.nan, (approx_value - true_value) / true_value)
