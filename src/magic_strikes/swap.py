import time
import warnings
from collections.abc import Callable
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

from . import magic_strike
from .utils import _validate_quadrature_order, gauss_legendre

MagicStrikeOpt = Literal[
    "variance",
    "gamma",
    "power",
]
MagicStrikeResult = dict[int, np.ndarray]
MagicStrikeAllResult = dict[str, MagicStrikeResult]


def _as_1d_float_array(values) -> np.ndarray:
    """Normalize scalar- or array-like numeric outputs to a 1-D float array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return np.atleast_1d(arr)


def _sanitize_xy_grid(x, y):
    """Return a sorted finite interpolation grid with at least three points."""
    x = _as_1d_float_array(x)
    y = _as_1d_float_array(y)
    if x.shape != y.shape:
        raise ValueError("Interpolation x and y values must have the same shape.")

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 3:
        raise ValueError("Need at least three finite interpolation points.")

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    x_unique, unique_idx = np.unique(x, return_index=True)
    y = y[unique_idx]

    if x_unique.size < 3:
        raise ValueError("Need at least three distinct interpolation points.")

    return x_unique, y


def _sanitize_smile_grid(ks, tot_impvar_values):
    """Return a finite positive smile grid with at least three points."""
    ks, tot_impvar_values = _sanitize_xy_grid(ks, tot_impvar_values)
    mask = tot_impvar_values > 0.0
    ks = ks[mask]
    tot_impvar_values = tot_impvar_values[mask]
    if ks.size < 3:
        raise ValueError("Need at least three finite positive smile points.")
    return ks, tot_impvar_values


def _validate_extrapolation(extrapolation):
    """Validate the interpolation extrapolation policy."""
    if extrapolation not in {"raise", "flat"}:
        raise ValueError("extrapolation must be either 'raise' or 'flat'")
    return extrapolation


def _validate_market_swap_opt(opt):
    """Validate the market swap contract selector."""
    if opt not in {
        "variance",
        "gamma",
        "all",
    }:
        raise ValueError("opt must be either 'variance', 'gamma', or 'all'")
    return opt


def _magic_strike_orders_for_opt(opt):
    """Return implemented magic-strike orders for a market contract."""
    if opt == "variance":
        return (1, 2, 3, 4, 5)
    if opt == "gamma":
        return (1, 2, 3, 4, 5)
    return (1, 2, 3, 4, 5)


def _magic_strike_orders_for_call(opt):
    """Return implemented magic-strike orders for direct magic-strike calls."""
    if opt in {
        "variance",
        "gamma",
        "power",
    }:
        return (1, 2, 3, 4, 5)
    raise ValueError("opt must be either 'variance', 'gamma', 'power', or 'all'")


def _validate_magic_strike_orders(opt, magic_orders):
    """Validate market magic-strike orders against implemented contract support."""
    default_orders = _magic_strike_orders_for_opt(opt)
    supported_orders = _magic_strike_orders_for_call(opt)
    if magic_orders is None:
        return default_orders
    if magic_orders == "max":
        return supported_orders

    orders = tuple(int(order) for order in np.atleast_1d(magic_orders))
    if not orders:
        raise ValueError("magic_orders must contain at least one order.")

    unsupported = sorted(set(orders).difference(supported_orders))
    if unsupported:
        raise ValueError(
            "magic_orders contains unsupported orders "
            f"{unsupported} for opt='{opt}'. Supported orders are {supported_orders}."
        )
    return orders


def _validate_direct_magic_strike_orders(opt, order):
    """Validate direct magic-strike orders."""
    default_orders = _magic_strike_orders_for_call(opt)
    if order is None or order == "max":
        return default_orders

    orders = tuple(int(order_i) for order_i in np.atleast_1d(order))
    if not orders:
        raise ValueError("order must contain at least one order.")

    unsupported = sorted(set(orders).difference(default_orders))
    if unsupported:
        raise ValueError(
            "order contains unsupported orders "
            f"{unsupported} for opt='{opt}'. Supported orders are {default_orders}."
        )
    return orders


def _rename_market_result_for_opt(result: dict, opt: str) -> dict:
    """Rename legacy vs_* keys to contract-specific keys for opt='all' output."""
    renamed = {}
    prefix = "vs_"
    for key, value in result.items():
        if key.startswith(prefix):
            renamed[f"{opt}_{key[len(prefix) :]}"] = value
        else:
            renamed[key] = value
    return renamed


def _validate_interpolation_grid_choice(ks_interp, std):
    """Raise if the interpolation grid is specified in two incompatible ways."""
    if ks_interp is not None and std is not None:
        raise ValueError(
            "Cannot specify both ks_interp and std; they both define "
            "the interpolation grid."
        )
    if ks_interp is None and std is None:
        raise ValueError("std must be specified when ks_interp is None.")
    if ks_interp is not None:
        ks_interp = np.asarray(ks_interp)
        if ks_interp.ndim != 1:
            raise ValueError("ks_interp must be one-dimensional.")


def _build_bounded_pchip_interpolator(x, y, extrapolation="raise"):
    """Build a PCHIP interpolator with explicit out-of-domain behavior."""
    extrapolation = _validate_extrapolation(extrapolation)
    x, y = _sanitize_xy_grid(x, y)
    interpolator = PchipInterpolator(x, y, extrapolate=False)

    x_min = float(x[0])
    x_max = float(x[-1])

    def evaluate(x_new):
        x_new_arr = np.asarray(x_new, dtype=float)
        if np.any(x_new_arr < x_min) or np.any(x_new_arr > x_max):
            if extrapolation == "raise":
                raise ValueError(f"Interpolation input must be in [{x_min}, {x_max}].")
            x_new_arr = np.clip(x_new_arr, x_min, x_max)
        return np.asarray(interpolator(x_new_arr), dtype=float)

    return evaluate, x, y


def _build_market_tot_impvar(log_moneyness, vol, T, extrapolation="raise"):
    """
    Build a bounded market total-implied-variance interpolator.

    Parameters
    ----------
    log_moneyness : array_like
        Log-strike grid.
    vol : array_like
        Implied volatilities on the same grid.
    T : float
        Time to maturity.
    extrapolation : {'raise', 'flat'}, default 'raise'
        Out-of-domain interpolation policy. ``'raise'`` rejects inputs outside
        the fitted log-moneyness grid; ``'flat'`` clips them to the nearest
        boundary.

    Returns
    -------
    tuple[callable, ndarray]
        A total-implied-variance interpolator and the sanitized log-moneyness
        grid used to build it.
    """
    vol = _as_1d_float_array(vol)
    if T <= 0.0 or not np.isfinite(T):
        raise ValueError("T must be finite and positive.")

    ks_clean, total_variance_clean = _sanitize_smile_grid(log_moneyness, vol**2 * T)
    evaluate, _, _ = _build_bounded_pchip_interpolator(
        ks_clean, total_variance_clean, extrapolation=extrapolation
    )

    def tot_impvar(k, T=None):
        return evaluate(k)

    return tot_impvar, ks_clean


def _largest_in_bounds_n_quad(T, tot_impvar, ks_interp, n_quad_max, opt="variance"):
    """Return the largest quadrature order supported by an observed smile grid."""
    n_quad_max = _validate_quadrature_order(n_quad_max)
    opt = _validate_market_swap_opt(opt)
    d_values = black_d(
        k=ks_interp,
        T=T,
        tot_impvar=tot_impvar,
        opt="plus" if opt == "gamma" else "minus",
    )
    d_values = d_values[np.isfinite(d_values)]
    if d_values.size < 3:
        return None

    d_min = float(np.min(d_values))
    d_max = float(np.max(d_values))
    for n_quad in range(n_quad_max, 0, -1):
        x_leg, _ = gauss_legendre(0, 1, n_quad)
        x_leg = norm.ppf(x_leg)
        if np.all((d_min <= x_leg) & (x_leg <= d_max)):
            return n_quad
    return None


def black_d(k, T, tot_impvar, opt="plus"):
    """
    Compute Black-Scholes d+ or d- parameter.

    It computes:
        d±(k,T) = -k / tot_impvol(k,T) ± 0.5 * tot_impvol(k,T)

    Parameters
    ----------
    k : array_like
        Log-moneyness
    T : float
        Time to maturity
    tot_impvar : callable
        Total implied variance function sig_BS(k,T)^2 * T, taking parameters (k, T).
    opt : {'plus', 'minus'}, default 'plus'
        Which parameter to compute

    Returns
    -------
    ndarray
        d+ or d- values, same shape as k
    """
    if opt not in ["plus", "minus"]:
        raise ValueError("opt must be either 'plus' or 'minus'")

    tot_impvar_values = np.asarray(tot_impvar(k, T), dtype=float)

    if np.any(~np.isfinite(tot_impvar_values)) or np.any(tot_impvar_values <= 0.0):
        raise ValueError("Total implied variance must be finite and positive.")

    tot_impvol = np.sqrt(tot_impvar_values)
    sgn = 1.0 if opt == "plus" else -1.0

    return -k / tot_impvol + sgn * 0.5 * tot_impvol


def func_g(
    T,
    tot_impvar,
    ks_interp=None,
    opt="plus",
    std: None | float = 5,
    n_interp: int = 1001,
    extrapolation="raise",
):
    """
    Create an interpolator mapping Black ``d`` values to log-strikes.

    The interpolation grid is built in log-strike space, then the corresponding
    ``d_+`` or ``d_-`` values are computed with :func:`black_d`. The resulting
    mapping is inverted numerically to obtain a function ``g`` such that
    ``g(d)`` returns the corresponding log-strike. Evaluating the returned
    interpolator outside its fitted ``d`` range raises ``ValueError`` by
    default.

    Parameters
    ----------
    T : float
        Time to maturity
    tot_impvar : callable
        Total implied variance function sig_BS(k,T)^2 * T, taking parameters (k, T).
    ks_interp : array_like, optional
        Log-strikes used to build interpolation grid.
        If None, a symmetric grid ``[-std * sqrt(w(0,T)), std * sqrt(w(0,T))]``
        with ``n_interp`` points is used, where ``w(0,T)`` is the ATM total
        implied variance. Cannot be specified together with ``std``.
    opt : {'plus', 'minus'}, default 'plus'
        Which Black parameter to invert.
    std : float or None, default 5
        Width multiplier for the automatically generated interpolation grid when
        ``ks_interp`` is None. Set to None when passing ``ks_interp``.
    n_interp : int, default 1001
        Number of interpolation points when ``ks_interp`` is None.
    extrapolation : {'raise', 'flat'}, default 'raise'
        Out-of-domain behavior for the inverted ``d`` interpolator.

    Returns
    -------
    callable
        Function mapping ``d`` values to log-strikes.

    Raises
    ------
    ValueError
        If the ATM total implied variance is not finite and positive, if fewer
        than three valid interpolation points are available, or if the returned
        interpolator is evaluated outside its fitted domain.
    """
    extrapolation = _validate_extrapolation(extrapolation)
    _validate_interpolation_grid_choice(ks_interp, std)

    if ks_interp is None and std is not None:
        x_lim = np.atleast_1d(np.asarray(tot_impvar(0.0, T), dtype=float) ** 0.5)
        x_lim = x_lim[0]
        if not np.isfinite(x_lim) or x_lim <= 0.0:
            raise ValueError("ATM total implied variance must be finite and positive.")
        ks_interp = np.linspace(-std * x_lim, std * x_lim, n_interp)
    else:
        ks_interp = np.asarray(ks_interp)

    d_values = black_d(k=ks_interp, T=T, tot_impvar=tot_impvar, opt=opt)
    g, _, _ = _build_bounded_pchip_interpolator(
        d_values, ks_interp, extrapolation=extrapolation
    )
    return g


def swap_fukasawa(
    T,
    tot_impvar,
    n_quad=5,
    ks_interp=None,
    opt="variance",
    std: None | float = 5,
    n_interp: int = 1000,
    extrapolation="raise",
    verbose: bool = False,
):
    """
    Calculate swap strikes using Fukasawa's formula.

    Parameters
    ----------
    T : float or array_like
        Time to maturity
    tot_impvar : callable
        Total implied variance function sig_BS(k,T)^2 * T, taking parameters (k, T).
    n_quad : int, default 5
        Number of Gauss quadrature points
    ks_interp : array_like, optional
        Log-strikes for interpolation grid in func_g. Cannot be specified
        together with ``std``.
    opt : {'variance', 'gamma', 'all'}, default 'variance'
        Type of swap to price. Use ``'all'`` to return all supported contracts.
    std : float or None, default 5
        Standard deviation for interpolation grid if ks_interp is None. Set to
        None when passing ``ks_interp``.
    n_interp : int, default 1000
        Number of interpolation points if ks_interp is None
    extrapolation : {'raise', 'flat'}, default 'raise'
        Out-of-domain behavior for the inverted ``d`` interpolator.
    verbose : bool, default False
        If True, print the current maturity as it is computed.

    Returns
    -------
    ndarray or dict
        Fair strike of the selected swap. If ``opt='all'``, returns a dictionary
        with keys ``'variance'`` and ``'gamma'``.

    Raises
    ------
    ValueError
        If fewer than three valid interpolation points are available, or if the
        implied-variance interpolator is evaluated outside its fitted
        strike or ``d`` domain.
    """
    if opt not in {
        "variance",
        "gamma",
        "all",
    }:
        raise ValueError("opt must be either 'variance', 'gamma', or 'all'")

    extrapolation = _validate_extrapolation(extrapolation)
    _validate_interpolation_grid_choice(ks_interp, std)
    T = np.atleast_1d(np.asarray(T))

    def _variance_terms(Ti, x_leg, w_leg):
        g_minus = func_g(
            T=Ti,
            tot_impvar=tot_impvar,
            ks_interp=ks_interp,
            opt="minus",
            std=std,
            n_interp=n_interp,
            extrapolation=extrapolation,
        )
        sig_minus_leg2 = tot_impvar(g_minus(x_leg), Ti)
        variance = np.sum(w_leg * sig_minus_leg2)
        return variance, sig_minus_leg2

    def _gamma_value(Ti, x_leg, w_leg):
        g_plus = func_g(
            T=Ti,
            tot_impvar=tot_impvar,
            ks_interp=ks_interp,
            opt="plus",
            std=std,
            n_interp=n_interp,
            extrapolation=extrapolation,
        )
        return np.sum(w_leg * tot_impvar(g_plus(x_leg), Ti))

    def compute_swap(Ti):
        x_quad, w_leg = gauss_legendre(0, 1, n_quad)
        x_leg = norm.ppf(x_quad)

        variance = None
        _sig_minus_leg2 = None
        gamma = None

        if opt in {
            "variance",
            "all",
        }:
            variance, _sig_minus_leg2 = _variance_terms(Ti, x_leg, w_leg)
        if opt in {"gamma", "all"}:
            gamma = _gamma_value(Ti, x_leg, w_leg)

        if opt == "variance":
            return variance
        if opt == "gamma":
            return gamma

        return {
            "variance": variance,
            "gamma": gamma,
        }

    values = []
    for i, Ti in enumerate(T, start=1):
        if verbose:
            print(f"Computing Fukasawa {opt} swap for T={Ti:.4f}... ({i}/{len(T)})")
        values.append(compute_swap(Ti))
    if opt == "all":
        return {
            key: np.asarray([value[key] for value in values], dtype=float)
            for key in (
                "variance",
                "gamma",
            )
        }
    return np.asarray(values, dtype=float)


def swap_market(
    ivol_data: pd.DataFrame,
    slices=None,
    opt="variance",
    n_quad=2,
    n_quad_auto_max=25,
    extrapolation="raise",
    magic_orders=None,
    magic_tol: float = 1e-8,
    magic_max_iter: int = 1000,
    relax: float = 0.8,
    record_compute_time=False,
):
    """
    Estimate market swap quotes using Fukasawa's formula and magic
    strikes.

    This function calculates variance and gamma swap rates from
    implied-volatility smiles. It handles bid-ask spreads and
    computes mid, bid, and ask rates for selected expiries. Rows with missing
    required fields or non-positive bid-ask spreads are dropped before pricing.
    Global input problems raise immediately; recoverable per-expiry/per-side
    estimator failures emit ``RuntimeWarning`` and return ``NaN`` for the
    affected output.

    Parameters
    ----------
    ivol_data : pandas.DataFrame
        DataFrame containing implied volatility data with columns:
        - 'Bid' : float
            Positive finite bid implied volatilities
        - 'Ask' : float
            Positive finite ask implied volatilities. Rows where ``Ask <= Bid``
            are dropped before pricing.
        - 'Texp' : float
            Positive finite time to expiry for each option
        - 'Strike' : float
            Positive finite strike prices
        - 'Fwd' : float
            Positive finite forward prices, identical within each expiry
    slices : array-like, optional
        Integer positions into the sorted unique ``Texp`` values to compute. If
        None, uses all available expiry dates. A scalar integer is accepted.
        Default is None.
    opt : {'variance', 'gamma', 'all'}, default 'variance'
        Contract to estimate. Existing ``vs_*`` return keys are retained for all
        contracts for backward compatibility.
    n_quad : int or None, default 2
        Number of Gauss quadrature points used in the Fukasawa estimate. If
        None, select the largest order up to ``n_quad_auto_max`` whose normal
        quadrature nodes stay within each observed smile's fitted Fukasawa
        domain for the selected contract, then use the shared minimum order
        across bid, ask, and mid for that expiry.
    n_quad_auto_max : int, default 25
        Maximum order considered when ``n_quad`` is None.
    extrapolation : {'raise', 'flat'}, default 'raise'
        Out-of-domain interpolation policy for market smiles. ``'raise'``
        rejects evaluations outside the observed log-moneyness grid; ``'flat'``
        uses the nearest endpoint total implied variance.
    magic_orders : int or array_like, optional
        Magic-strike approximation orders to compute. Defaults to orders 1
        through 5 for variance and gamma.
    magic_tol : float, default 1e-12
        Tolerance forwarded to :func:`swap_magic_strike`.
    magic_max_iter : int, default 1000
        Maximum fixed-point iterations forwarded to :func:`swap_magic_strike`.
    relax : float, default 0.8
        Relaxation parameter forwarded to :func:`swap_magic_strike`.
    record_compute_time : bool, default False
        If True, include total elapsed seconds for the whole Fukasawa procedure
        and for each magic-strike order in the returned dictionary.
    Returns
    -------
    dict
        Dictionary containing:
        - 'expiries' : ndarray
            Array of expiry dates
        - 'vs_mid' : ndarray
            Array of mid contract rates
        - 'vs_bid' : ndarray
            Array of bid contract rates
        - 'vs_ask' : ndarray
            Array of ask contract rates
        - 'vs_mid_magic', 'vs_bid_magic', 'vs_ask_magic' : ndarray
            Magic-strike approximations, using orders 1 through 5 for both
            variance and gamma.
        - 'n_quad_used' : ndarray
            Quadrature order used for each expiry. This equals ``n_quad`` for
            explicit orders and is selected per expiry when ``n_quad`` is None.
            It is ``NaN`` when no in-domain automatic order is available.
        - 'compute_time' : dict, optional
            Present only when ``record_compute_time=True``. Contains
            ``fukasawa`` as a scalar total across all selected maturities and
            sides, and ``magic_strike`` as one total per magic-strike order
            across all selected maturities and sides.
    """
    opt = _validate_market_swap_opt(opt)
    if opt == "all":
        return {
            opt_i: _rename_market_result_for_opt(
                swap_market(
                    ivol_data=ivol_data,
                    slices=slices,
                    opt=opt_i,
                    n_quad=n_quad,
                    n_quad_auto_max=n_quad_auto_max,
                    extrapolation=extrapolation,
                    magic_orders=magic_orders,
                    magic_tol=magic_tol,
                    magic_max_iter=magic_max_iter,
                    relax=relax,
                    record_compute_time=record_compute_time,
                ),
                opt_i,
            )
            for opt_i in (
                "variance",
                "gamma",
            )
        }

    magic_orders = _validate_magic_strike_orders(opt, magic_orders)
    magic_tol = float(magic_tol)
    if not np.isfinite(magic_tol) or magic_tol <= 0.0:
        raise ValueError("magic_tol must be finite and positive.")
    if int(magic_max_iter) < 1:
        raise ValueError("magic_max_iter must be positive.")
    magic_max_iter = int(magic_max_iter)
    auto_n_quad = n_quad is None
    extrapolation = _validate_extrapolation(extrapolation)
    if auto_n_quad:
        n_quad_auto_max = _validate_quadrature_order(n_quad_auto_max)
    else:
        n_quad = _validate_quadrature_order(n_quad)

    data = ivol_data.copy()
    required_cols = {"Bid", "Ask", "Texp", "Strike", "Fwd"}
    missing_cols = required_cols.difference(data.columns)
    if missing_cols:
        raise ValueError(
            f"ivol_data is missing required columns: {sorted(missing_cols)}"
        )

    data = data.dropna(subset=sorted(required_cols)).copy()
    if data.empty:
        raise ValueError("ivol_data must contain at least one complete quote row.")

    numeric = data.loc[:, sorted(required_cols)].apply(pd.to_numeric, errors="raise")
    for col in sorted(required_cols):
        data[col] = numeric[col].astype(float)

    if np.any(~np.isfinite(numeric["Texp"])) or np.any(numeric["Texp"] <= 0.0):
        raise ValueError("Expiries must be finite and positive.")
    if np.any(~np.isfinite(numeric["Strike"])) or np.any(numeric["Strike"] <= 0.0):
        raise ValueError("Strikes must be finite and positive.")
    if np.any(~np.isfinite(numeric["Fwd"])) or np.any(numeric["Fwd"] <= 0.0):
        raise ValueError("Forward prices must be finite and positive.")
    if np.any(~np.isfinite(numeric["Bid"])) or np.any(numeric["Bid"] <= 0.0):
        raise ValueError("Bid implied volatilities must be finite and positive.")
    if np.any(~np.isfinite(numeric["Ask"])) or np.any(numeric["Ask"] <= 0.0):
        raise ValueError("Ask implied volatilities must be finite and positive.")
    data = data.loc[data["Ask"] > data["Bid"]].copy()
    if data.empty:
        raise ValueError(
            "ivol_data must contain at least one quote row with Ask greater than Bid."
        )

    expiries = np.sort(data["Texp"].unique())

    if slices is None:
        idx = np.arange(len(expiries))
    else:
        try:
            idx = np.atleast_1d(np.asarray(list(slices), dtype=int))
        except TypeError:
            idx = np.asarray([slices], dtype=int)
    if np.any((idx < 0) | (idx >= len(expiries))):
        raise IndexError("slices contains an out-of-range expiry index")

    vs_mid = np.empty(len(idx))
    vs_bid = np.empty(len(idx))
    vs_ask = np.empty(len(idx))
    vs_mid_magic = np.empty((len(magic_orders), len(idx)))
    vs_bid_magic = np.empty((len(magic_orders), len(idx)))
    vs_ask_magic = np.empty((len(magic_orders), len(idx)))
    n_quad_used = np.empty(len(idx))
    if record_compute_time:
        fukasawa_time = 0.0
        magic_strike_time = np.zeros(len(magic_orders))

    def _estimate_fukasawa(T, tot_impvar, ks_clean, n_quad_t):
        return swap_fukasawa(
            T=T,
            tot_impvar=tot_impvar,
            ks_interp=ks_clean,
            std=None,
            n_quad=n_quad_t,
            opt=opt,
            extrapolation=extrapolation,
        )[0]

    def _estimate_magic(T, tot_impvar, ks_clean, orders):
        kwargs = dict(  # noqa: C408
            T=T,
            tot_impvar=tot_impvar,
            order=orders,
            ks_interp=ks_clean,
            std=None,
            max_iter=magic_max_iter,
            tol=magic_tol,
            relax=relax,
            extrapolation=extrapolation,
        )
        return swap_magic_strike(opt=opt, **kwargs)

    for out_i, exp_i in enumerate(idx):
        t = expiries[exp_i]
        slice_t = data[data["Texp"] == t].copy()

        bid_t = slice_t["Bid"].to_numpy()
        ask_t = slice_t["Ask"].to_numpy()
        mid_t = 0.5 * (bid_t + ask_t)

        fwd_values = slice_t["Fwd"].to_numpy(dtype=float)
        fwd_t = float(fwd_values[0])
        if not np.allclose(fwd_values, fwd_t, rtol=0.0, atol=0.0):
            raise ValueError("Forward prices must be identical within each expiry.")
        k_t = np.log(slice_t["Strike"].to_numpy() / fwd_t)

        vol_triplet = [("bid", bid_t), ("ask", ask_t), ("mid", mid_t)]
        if auto_n_quad:
            expiry_n_quads = []
            for side, vol in vol_triplet:
                try:
                    tot_impvar, ks_clean = _build_market_tot_impvar(
                        k_t, vol, t, extrapolation=extrapolation
                    )
                    expiry_n_quads.append(
                        _largest_in_bounds_n_quad(
                            T=t,
                            tot_impvar=tot_impvar,
                            ks_interp=ks_clean,
                            n_quad_max=n_quad_auto_max,
                            opt=opt,
                        )
                    )
                except Exception as e:  # noqa: BLE001 — preserve source fallback behavior
                    warnings.warn(
                        "Could not determine automatic Fukasawa quadrature order "
                        f"for {side} quotes at expiry {t:.4f} "
                        f"(index {exp_i}): {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    expiry_n_quads.append(None)
            if any(n is None for n in expiry_n_quads):
                n_quad_t = None
                n_quad_used[out_i] = np.nan
                warnings.warn(
                    "No in-domain automatic Fukasawa quadrature order is available "
                    f"for expiry {t:.4f} (index {exp_i}); Fukasawa estimates "
                    "will be set to NaN.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                n_quad_t = min(expiry_n_quads)
                n_quad_used[out_i] = n_quad_t
        else:
            n_quad_t = n_quad
            n_quad_used[out_i] = n_quad

        if np.isfinite(n_quad_used[out_i]):
            print(
                f"\nExpiry {t:.4f}: {opt} swap (Fukasawa) using "
                f"n_quad={int(n_quad_used[out_i])}.\n"
            )
        print(f"Expiry {t:.4f}: {opt} swap (magic strikes).\n")

        for (side, vol), vs, vs_magic in zip(
            vol_triplet,
            [vs_bid, vs_ask, vs_mid],
            [vs_bid_magic, vs_ask_magic, vs_mid_magic],
            strict=True,
        ):
            try:
                tot_impvar, ks_clean = _build_market_tot_impvar(
                    k_t, vol, t, extrapolation=extrapolation
                )
            except Exception as e:  # noqa: BLE001 — preserve source fallback behavior
                warnings.warn(
                    "Could not build market total-implied-variance interpolator "
                    f"for {side} quotes at expiry {t:.4f} (index {exp_i}): {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                vs[out_i] = np.nan
                vs_magic[:, out_i] = np.nan
                continue

            # Selected contract via Fukasawa's formula.
            start = None
            try:
                if n_quad_t is None:
                    vs[out_i] = np.nan
                else:
                    if record_compute_time:
                        start = time.perf_counter()
                    vs[out_i] = _estimate_fukasawa(t, tot_impvar, ks_clean, n_quad_t)
                    if record_compute_time:
                        fukasawa_time += time.perf_counter() - start
            except Exception as e:  # noqa: BLE001 — preserve source fallback behavior
                if record_compute_time and start is not None:
                    fukasawa_time += time.perf_counter() - start
                warnings.warn(
                    f"Fukasawa {opt}-swap estimate failed for "
                    f"{side} quotes at expiry {t:.4f} (index {exp_i}): {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                vs[out_i] = np.nan

            # magic strikes
            start = None
            order_i = None
            try:
                if record_compute_time:
                    for order_i, order in enumerate(magic_orders):
                        start = time.perf_counter()
                        res = _estimate_magic(t, tot_impvar, ks_clean, [order])
                        magic_strike_time[order_i] += time.perf_counter() - start
                        vs_magic[order_i, out_i] = res[order][0]
                else:
                    res = _estimate_magic(t, tot_impvar, ks_clean, magic_orders)
                    for i, order in enumerate(magic_orders):
                        vs_magic[i, out_i] = res[order][0]
            except Exception as e:  # noqa: BLE001 — preserve source fallback behavior
                if record_compute_time and start is not None and order_i is not None:
                    magic_strike_time[order_i] += time.perf_counter() - start
                warnings.warn(
                    f"Magic-strike {opt}-swap estimate failed for "
                    f"{side} quotes at expiry {t:.4f} (index {exp_i}): {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                vs_magic[:, out_i] = np.nan

    result = {
        "expiries": expiries[idx],
        "vs_mid": vs_mid,
        "vs_bid": vs_bid,
        "vs_ask": vs_ask,
        "n_quad_used": n_quad_used,
        "vs_mid_magic": vs_mid_magic,
        "vs_bid_magic": vs_bid_magic,
        "vs_ask_magic": vs_ask_magic,
    }
    if record_compute_time:
        result["compute_time"] = {
            "fukasawa": fukasawa_time,
            "magic_strike": magic_strike_time,
        }
    return result


@overload
def swap_magic_strike(
    T: Any,
    tot_impvar: Callable[..., Any],
    order: Any = None,
    opt: MagicStrikeOpt = "variance",
    tol: float = 1e-8,
    max_iter: int = 100,
    relax: float = 0.8,
    std: float = 5.0,
    ks_interp: Any = None,
    n_interp: int = 1001,
    extrapolation: str = "raise",
    p: float | None = None,
) -> MagicStrikeResult: ...


@overload
def swap_magic_strike(
    T: Any,
    tot_impvar: Callable[..., Any],
    order: Any = None,
    opt: Literal["all"] = "all",
    tol: float = 1e-8,
    max_iter: int = 100,
    relax: float = 0.8,
    std: float = 5.0,
    ks_interp: Any = None,
    n_interp: int = 1001,
    extrapolation: str = "raise",
    p: None = None,
) -> MagicStrikeAllResult: ...


@overload
def swap_magic_strike(
    T: Any,
    tot_impvar: Callable[..., Any],
    order: Any = None,
    opt: str = "variance",
    tol: float = 1e-8,
    max_iter: int = 100,
    relax: float = 0.8,
    std: float = 5.0,
    ks_interp: Any = None,
    n_interp: int = 1001,
    extrapolation: str = "raise",
    p: float | None = None,
) -> MagicStrikeResult | MagicStrikeAllResult: ...


def swap_magic_strike(
    T,
    tot_impvar,
    order=None,
    opt: str = "variance",
    tol: float = 1e-8,
    max_iter: int = 100,
    relax: float = 0.8,
    std: float = 5.0,
    ks_interp=None,
    n_interp: int = 1001,
    extrapolation: str = "raise",
    p=None,
):
    """
    Compute swap rates using magic strikes.

    Parameters
    ----------
    T : float or array_like
        Time to maturity in years.
    tot_impvar : callable
        Total implied variance function sig_BS(k,T)^2 * T, taking parameters (k, T).
    order : int or array_like
        Approximation order(s).
    opt : {'variance', 'gamma', 'power',
        'all'},
        optional
        Swap type to compute. ``'gamma'`` uses the gamma fixed-point method.
        Use ``'all'`` to compute variance and gamma from the same smile. Default
        is 'variance'.
    p : float, optional
        Power-payoff parameter in ``[0, 1]``. Required exactly when
        ``opt='power'``. The returned values are implied power variances.
    tol : float, optional
        Tolerance for iterative solver. Default is 1e-8.
    max_iter : int, optional
        Maximum number of iterations for iterative solver. Default is 100.
    relax : float, optional
        Relaxation parameter for iterative solver. Default is 0.8.
    std : float or None, optional
        Standard deviation for interpolation range. Default is 5. Set to None
        when passing ``ks_interp``.
    ks_interp : array_like, optional
        Log-strikes for the interpolation grid. If provided, evaluations outside
        the fitted grid are not clipped and raise ``ValueError``. Cannot be
        specified together with ``std``.
    n_interp : int, optional
        Number of points for interpolation. Default is 1001.
    extrapolation : {'raise', 'flat'}, default 'raise'
        Out-of-domain interpolation policy when the magic-strike stencil
        evaluates outside the fitted strike grid. ``'raise'`` rejects those
        evaluations; ``'flat'`` uses the nearest endpoint total implied variance.

    Returns
    -------
    dict
        Dictionary mapping each order to the corresponding swap rate(s). Keys are
        the requested orders and values are arrays of swap rates.

    Raises
    ------
    ValueError
        If fewer than three valid interpolation points are available, or if the
        smile interpolator is evaluated outside its fitted strike domain.
    """

    p = magic_strike._validate_power_parameter(opt, p)
    if opt == "all":
        orders_by_opt = {
            opt_i: _validate_direct_magic_strike_orders(opt_i, order)
            for opt_i in (
                "variance",
                "gamma",
            )
        }
    else:
        orders = _validate_direct_magic_strike_orders(opt, order)
        orders_by_opt = {opt: orders}

    extrapolation = _validate_extrapolation(extrapolation)
    _validate_interpolation_grid_choice(ks_interp, std)

    def _swap(Ti):
        if ks_interp is None and std is not None:
            x_lim = np.atleast_1d(np.asarray(tot_impvar(0.0, Ti), dtype=float) ** 0.5)
            x_lim = x_lim[0]
            if not np.isfinite(x_lim) or x_lim <= 0.0:
                raise ValueError(
                    "ATM total implied variance must be finite and positive."
                )
            ks_grid = np.linspace(-std * x_lim, std * x_lim, n_interp)
        else:
            ks_grid = np.asarray(ks_interp)

        tot_impvar_values = tot_impvar(k=ks_grid, T=Ti)
        ks_grid_clean, tot_impvar_values_clean = _sanitize_smile_grid(
            ks_grid, tot_impvar_values
        )
        tot_impvar_interp, _, _ = _build_bounded_pchip_interpolator(
            ks_grid_clean,
            tot_impvar_values_clean,
            extrapolation=extrapolation,
        )

        return magic_strike.swap_many(
            tot_impvar=tot_impvar_interp,
            orders_by_opt=orders_by_opt,
            p=p,
            tol=tol,
            max_iter=max_iter,
            relax=relax,
        )

    swap_T = []
    for i, Ti in enumerate(np.atleast_1d(T)):
        print(
            f"Computing magic strikes for T={Ti:.4f}... "
            f"({i + 1}/{len(np.atleast_1d(T))})"
        )
        # t0 = time.time()
        swap_T.append(_swap(Ti))
        # print(f"Done in {time.time() - t0:.2f} seconds.\n")

    if opt == "all":
        return {
            opt_i: {
                order_i: np.asarray(
                    [swap_T_i[opt_i][order_i] for swap_T_i in swap_T], dtype=float
                )
                for order_i in orders_by_opt[opt_i]
            }
            for opt_i in orders_by_opt
        }

    return {
        order_i: np.asarray(
            [swap_T_i[opt][order_i] for swap_T_i in swap_T], dtype=float
        )
        for order_i in orders_by_opt[opt]
    }
