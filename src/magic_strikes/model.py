from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal, overload

import numpy as np
from scipy import integrate

from . import swap
from .utils import implied_vol_from_paths

MagicStrikeOpt = Literal[
    "variance",
    "gamma",
    "power",
]
MagicStrikeResult = dict[int, np.ndarray]
MagicStrikeAllResult = dict[str, MagicStrikeResult]


@dataclass(frozen=True)
class MonteCarloConfig:
    """Configuration shared by the model Monte Carlo estimators.

    Parameters
    ----------
    n_mc : int
        Total number of paths simulated per maturity and batch.
    n_disc : int
        Number of time intervals in the uniform simulation grid.
    n_loop : int, default 1
        Number of memory-saving simulation chunks. It must divide ``n_mc``.
    seed : int or None, optional
        Seed used directly for one batch or to generate independent batch seeds.
    antithetic : bool, default False
        Whether the model simulator should use antithetic shocks.
    """

    n_mc: int
    n_disc: int
    n_loop: int = 1
    seed: int | None = None
    antithetic: bool = False


def require_params(params: dict, required: tuple[str, ...]) -> tuple:
    """Return required parameter values or raise a consistent error."""
    missing = [p for p in required if p not in params]
    if missing:
        raise ValueError(f"Missing parameters: {missing}.")
    return tuple(params[p] for p in required)


def validate_positive(name: str, value: float) -> None:
    """Raise if ``value`` is not strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")


def validate_nonnegative(name: str, value: float) -> None:
    """Raise if ``value`` is negative."""
    if value < 0:
        raise ValueError(f"{name} must be >= 0.")


def validate_interval(
    name: str,
    value: float,
    lower: float,
    upper: float,
    *,
    closed: bool = True,
) -> None:
    """Raise if ``value`` lies outside the requested interval."""
    if closed:
        valid = lower <= value <= upper
        bounds = f"[{lower}, {upper}]"
    else:
        valid = lower < value < upper
        bounds = f"({lower}, {upper})"
    if not valid:
        raise ValueError(f"{name} must be in {bounds}.")


def validate_interpolation_grid_choice(ks_interp, std) -> None:
    """Raise if an interpolation grid is specified two ways."""
    if ks_interp is not None and std is not None:
        raise ValueError(
            "Cannot specify both ks_interp and std; they both define "
            "the interpolation grid."
        )
    if ks_interp is None and std is None:
        raise ValueError("std must be specified when ks_interp is None.")


def _as_1d_float_array(values) -> np.ndarray:
    """Normalize scalar- or array-like numeric outputs to a 1-D float array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return np.atleast_1d(arr)


def _reject_conditioning_kwargs(kwargs: dict, context: str) -> None:
    """Raise if conditioning-only arguments are passed to deterministic wrappers."""
    unsupported = sorted({"conditioning", "rho_cond"}.intersection(kwargs))
    if unsupported:
        raise NotImplementedError(
            f"{', '.join(unsupported)} is not implemented for {context}."
        )


def resolve_mc_config(
    config: MonteCarloConfig | None,
    overrides: dict,
) -> MonteCarloConfig:
    """Return a Monte Carlo configuration after applying keyword overrides.

    When ``config`` is omitted, ``overrides`` must contain ``n_mc`` and
    ``n_disc``. When it is supplied, only ``n_loop``, ``seed``, and
    ``antithetic`` may override existing fields. Consumed entries are removed
    from ``overrides``.
    """
    if config is None:
        required = ("n_mc", "n_disc")
        missing = [p for p in required if p not in overrides]
        if missing:
            raise ValueError(f"Missing Monte Carlo config values: {missing}.")
        config = MonteCarloConfig(
            n_mc=overrides.pop("n_mc"),
            n_disc=overrides.pop("n_disc"),
        )
    if not isinstance(config, MonteCarloConfig):
        raise TypeError("config must be a MonteCarloConfig instance.")

    allowed = ("n_loop", "seed", "antithetic")
    updates = {key: overrides.pop(key) for key in list(overrides) if key in allowed}
    if overrides:
        raise TypeError(f"Unexpected Monte Carlo config values: {sorted(overrides)}.")
    if updates:
        config = replace(config, **updates)
    return config


class ForwardVarianceModel(ABC):
    def __init__(
        self,
        params: dict,
        xi0: Callable[[np.ndarray], np.ndarray] = lambda t: 0.2**2 * np.ones_like(t),
        s0: float = 1.0,
    ) -> None:
        """
        Initialize a forward-variance model.

        Parameters
        ----------
        params : dict
            Model-specific parameters retained by the instance.
        xi0 : callable, optional
            Initial forward-variance curve ``xi0(t)``. It must accept NumPy
            arrays and return positive values compatible with its input. The
            default is the flat curve ``0.2**2``.
        s0 : float, default 1
            Strictly positive initial spot price.

        Raises
        ------
        ValueError
            If ``s0`` is not positive, ``xi0`` is not callable, or sampled
            forward-variance values are not positive.
        """
        if s0 <= 0.0:
            raise ValueError("Initial spot price s0 must be positive.")

        if not callable(xi0):
            raise ValueError("xi0 must be a callable function.")  # noqa: TRY004

        self.xi0 = xi0
        self._is_xi0_positive()
        self.xi0_0 = _as_1d_float_array(self.xi0(np.zeros(1)))[0]
        self.xi0_flat = self._is_xi0_flat()
        self.params = params
        self.s0 = s0
        self.delta_vix = 1.0 / 12.0

    def __repr__(self) -> str:
        """Return a string representation of the model and its parameters."""
        msg = f"{self.__class__.__name__} with parameters:\n"
        for key, value in self.params.items():
            msg += f"{key}={value}, "
        msg += f"s0={self.s0}."
        return msg

    __str__ = __repr__

    @abstractmethod
    def kernel(self, u, t) -> float | np.ndarray:
        """
        Compute the model-specific kernel function.

        Parameters
        ----------
        u : float or np.ndarray
            Upper time(s) (must satisfy u >= t).
        t : float or np.ndarray
            Lower time(s).

        Returns
        -------
        float or np.ndarray
            Value(s) of the kernel function evaluated at (u, t).
        """

    def _clone_with_params(self, **updates):
        """Return a new instance after applying model-parameter updates.

        Parameters in ``updates`` replace entries in a shallow copy of
        ``self.params``. Subclasses may customize reconstruction through
        :meth:`_clone_init_kwargs`.
        """
        params = self.params.copy()
        params.update(updates)
        return self.__class__(**self._clone_init_kwargs(params))

    def _clone_init_kwargs(self, params: dict) -> dict:
        """Return constructor arguments used by :meth:`_clone_with_params`.

        Subclasses whose forward-variance curve is derived from their parameters
        should override this method rather than reuse the current ``xi0``.
        """
        return {"params": params, "s0": self.s0, "xi0": self.xi0}

    def _is_xi0_flat(self) -> bool:
        """Return whether ``xi0`` is numerically flat on the diagnostic grid.

        The diagnostic samples 1,000 times between approximately zero and ten
        years and compares them with ``xi0(0)`` using :func:`numpy.allclose`.
        """
        t_test = np.linspace(1e-10, 10, 1000)
        return np.allclose(_as_1d_float_array(self.xi0(t_test)), self.xi0_0)

    def _is_xi0_positive(self):
        """Validate positivity of ``xi0`` on the diagnostic time grid.

        Raises
        ------
        ValueError
            If any of 1,000 sampled values between approximately zero and ten
            years is nonpositive.
        """
        t_test = np.linspace(1e-10, 10, 1000)
        if not np.all(_as_1d_float_array(self.xi0(t_test)) > np.array([0.0])):
            raise ValueError("xi0 must be positive for all t >= 0.")

    def simulate_mc(
        self,
        tab_t: np.ndarray,
        n_mc: int,
        n_loop: int,
        seed: int | None,
        antithetic: bool = False,
    ) -> dict:
        """
        Simulate path integrals required by the Monte Carlo estimators.

        The base class defines the required output contract but does not provide
        a simulation scheme. Model subclasses must override this method.

        Parameters
        ----------
        tab_t : ndarray
            One-dimensional simulation grid, including the initial and terminal
            times.
        n_mc : int
            Total number of Monte Carlo paths to simulate.
        n_loop : int
            Number of memory-saving simulation chunks. It must divide ``n_mc``.
        seed : int or None
            Random seed for reproducibility.
        antithetic : bool, default False
            Whether to use antithetic shocks when supported by the model.

        Returns
        -------
        dict
            Arrays ``"int_v_dt"`` and ``"int_sqrt_v_dw"``, each with one
            value per path. They represent integrated variance and the spot
            stochastic integral, respectively.

        Raises
        ------
        NotImplementedError
            Always raised by the base implementation.
        """
        raise NotImplementedError(
            "Monte Carlo simulation not implemented for this model."
        )

    def impvol_mc(
        self,
        k: float | np.ndarray,
        T: float,
        n_mc: int,
        n_disc: int,
        n_loop: int = 1,
        seed=None,
        antithetic: bool = False,
        return_skew: bool = False,
    ):
        """
        Estimate an implied-volatility smile from simulated path integrals.

        Parameters
        ----------
        k : float or array_like
            Log-moneyness values ``log(K/F)``.
        T : float
            Positive scalar maturity.
        n_mc : int
            Total number of simulated paths.
        n_disc : int
            Number of intervals in the uniform time grid.
        n_loop : int, default 1
            Number of memory-saving simulation chunks.
        seed : int or None, optional
            Random seed for reproducibility.
        antithetic : bool, default False
            Whether to use antithetic shocks when supported by the model.
        return_skew : bool, default False
            If true, also return the implied-volatility skew produced by
            :func:`utils.implied_vol_from_paths`.

        Returns
        -------
        float, ndarray, or tuple
            Implied volatility at ``k``. If ``return_skew`` is true, returns the
            implied volatility together with its skew.
        """

        tab_t = np.linspace(0.0, T, n_disc + 1)
        paths = self.simulate_mc(
            tab_t=tab_t,
            n_mc=n_mc,
            n_loop=n_loop,
            seed=seed,
            antithetic=antithetic,
        )

        return implied_vol_from_paths(
            k=k,
            T=T,
            int_v_dt=paths["int_v_dt"],
            int_sqrt_v_dw=paths["int_sqrt_v_dw"],
            s0=self.s0,
            conditioning=False,
            return_skew=return_skew,
        )

    @staticmethod
    def _swap_estimates_from_path_integrals(
        int_v_dt: np.ndarray,
        int_sqrt_v_dw: np.ndarray,
    ) -> dict:
        """Estimate total swap quantities from matching path integrals.

        This unadjusted estimator computes variance and gamma directly from
        the supplied paths. It does not apply the variance control variate used by
        :meth:`_swap_estimate_from_simulated_paths`.

        Parameters
        ----------
        int_v_dt, int_sqrt_v_dw : array_like
            Integrated variance and spot stochastic integral for each path.
            Both inputs must contain the same number of values.

        Returns
        -------
        dict
            Scalar estimates keyed by ``"variance"`` and ``"gamma"``.

        Raises
        ------
        ValueError
            If the flattened path-integral arrays have different shapes.
        """
        int_v_dt = np.asarray(int_v_dt, dtype=float).ravel()
        int_sqrt_v_dw = np.asarray(int_sqrt_v_dw, dtype=float).ravel()
        if int_v_dt.shape != int_sqrt_v_dw.shape:
            raise ValueError("int_v_dt and int_sqrt_v_dw must have the same shape.")

        variance = np.mean(int_v_dt)
        x = -0.5 * int_v_dt + int_sqrt_v_dw
        gamma = 2.0 * np.mean(x * np.exp(x))

        return {
            "variance": variance,
            "gamma": gamma,
        }

    @staticmethod
    def _control_variate_estimate(payoff, control, control_mean):
        """Return a scalar control-variate estimate of a payoff mean.

        The regression coefficient is estimated from the same sample. If the
        control has zero variance or the coefficient is nonfinite, the raw
        payoff mean is returned.

        Parameters
        ----------
        payoff, control : array_like
            Pathwise payoff and control samples.
        control_mean : float
            Known expectation of the control.

        Returns
        -------
        float
            Control-variate-adjusted estimate of the payoff expectation.
        """
        payoff_mean = np.mean(payoff)
        control_mean_mc = np.mean(control)
        control_centered = control - control_mean_mc
        control_var = np.mean(control_centered * control_centered)
        if not np.isfinite(control_var) or control_var <= 0.0:
            return payoff_mean

        payoff_centered = payoff - payoff_mean
        beta = np.mean(payoff_centered * control_centered) / control_var
        if not np.isfinite(beta):
            return payoff_mean
        return payoff_mean - beta * (control_mean_mc - control_mean)

    def _validate_swap_mc_inputs(
        self,
        T: float | np.ndarray,
        config: MonteCarloConfig,
        n_batch: int,
    ):
        """Normalize and validate inputs shared by swap Monte Carlo methods.

        Returns the maturities as a one-dimensional float array together with
        integer ``n_batch`` and ``n_loop`` values. ``config.n_mc`` must be
        divisible by ``n_loop``.

        Raises
        ------
        ValueError
            If a count is invalid, path chunks do not divide evenly, or a
            maturity is nonfinite or nonpositive.
        """
        n_batch = int(n_batch)
        if n_batch <= 0:
            raise ValueError("n_batch must be a positive integer.")

        n_loop = int(config.n_loop)
        if n_loop <= 0:
            raise ValueError("n_loop must be a positive integer.")

        n_mc_loop, remainder = divmod(config.n_mc, n_loop)
        if n_mc_loop <= 0 or remainder != 0:
            raise ValueError("n_mc must be divisible by n_loop.")

        T = _as_1d_float_array(T)
        if np.any(~np.isfinite(T)) or np.any(T <= 0.0):
            raise ValueError("T must be finite and positive.")

        return T, n_batch, n_loop

    def _swap_estimate_from_simulated_paths(
        self,
        tab_t: np.ndarray,
        paths: dict,
        *,
        variance: float,
    ) -> dict:
        """Estimate swaps from one simulation using variance control variates.

        Gamma uses the simulated integrated variance as a
        control whose expectation is evaluated on the same trapezoidal time
        grid. The returned variance is the supplied deterministic benchmark.

        Parameters
        ----------
        tab_t : array_like
            Simulation time grid corresponding to ``paths``.
        paths : dict
            Path arrays ``"int_v_dt"`` and ``"int_sqrt_v_dw"``.
        variance : float
            Deterministic total-variance value returned in the result.

        Returns
        -------
        dict
            Scalar estimates keyed by ``"variance"`` and ``"gamma"``.

        Raises
        ------
        ValueError
            If the two path-integral arrays have different shapes or ``xi0`` is
            incompatible with ``tab_t``.
        """
        int_v_dt = np.asarray(paths["int_v_dt"], dtype=float).ravel()
        int_sqrt_v_dw = np.asarray(paths["int_sqrt_v_dw"], dtype=float).ravel()
        if int_v_dt.shape != int_sqrt_v_dw.shape:
            raise ValueError("int_v_dt and int_sqrt_v_dw must have the same shape.")

        variance_control_mean = self._var_swap_trapezoidal_grid(tab_t)
        x = -0.5 * int_v_dt + int_sqrt_v_dw
        gamma_payoff = 2.0 * x * np.exp(x)

        gamma = self._control_variate_estimate(
            gamma_payoff, int_v_dt, variance_control_mean
        )
        return {
            "variance": variance,
            "gamma": gamma,
        }

    def swap_mc_all(
        self,
        T: float | np.ndarray,
        config: MonteCarloConfig | None = None,
        *,
        n_batch: int = 1,
        verbose: bool = False,
        **mc_kwargs,
    ) -> dict:
        """
        Estimate variance and gamma by maturity.

        Variance is evaluated deterministically with :meth:`var_swap_quad`.
        Gamma is a Monte Carlo estimate using integrated
        variance as a control variate.

        Parameters
        ----------
        T : float or array_like
            Positive maturity or maturities.
        config : MonteCarloConfig, optional
            Simulation configuration. If omitted, ``n_mc`` and ``n_disc`` must
            be supplied through ``mc_kwargs``.
        n_batch : int, default 1
            Number of independent simulations, each with ``config.n_mc`` paths
            per maturity.
        verbose : bool, default False
            Print simulation and batch progress.
        **mc_kwargs
            Monte Carlo configuration values accepted by
            :func:`resolve_mc_config`.

        Returns
        -------
        dict
            Arrays keyed by ``"variance"`` and ``"gamma"``. With multiple
            batches, each key also has
            ``"*_low"`` and ``"*_high"`` 95% confidence bounds. Variance
            bounds equal its deterministic value.

        Notes
        -----
        Confidence intervals are mean plus or minus 1.96 independent-batch
        standard errors. They quantify sampling error, not time-discretization
        bias.
        """
        config = resolve_mc_config(config, mc_kwargs)

        T, n_batch, n_loop = self._validate_swap_mc_inputs(T, config, n_batch)
        keys = ("variance", "gamma")
        mc_keys = ("gamma",)
        n_expiries = len(T)
        variance_quad = _as_1d_float_array(self.var_swap_quad(T))

        if verbose:
            print("\nComputing Monte Carlo swaps:")
            print("----------------------------")
            print(f"Model: {self.__class__.__name__}")
            print("Maturity T: ", T)
            print(f"Monte Carlo paths: {config.n_mc}")
            print(f"Number of time steps: {config.n_disc}")
            print(f"Number of simulation loops: {n_loop}")
            print(f"Number of independent batches: {n_batch}\n")

        def _estimate_for_maturity(i, Ti, batch_seed):
            """Simulate and estimate every direct contract at one maturity."""
            if verbose:
                print(f"{i}/{n_expiries}: expiry {Ti:.4f}")
            tab_t = np.linspace(0.0, Ti, config.n_disc + 1)
            paths = self.simulate_mc(
                tab_t=tab_t,
                n_mc=config.n_mc,
                n_loop=n_loop,
                seed=batch_seed,
                antithetic=config.antithetic,
            )
            return self._swap_estimate_from_simulated_paths(
                tab_t,
                paths,
                variance=variance_quad[i - 1],
            )

        def _run_batch(batch_seed):
            """Return direct-contract arrays from one independent batch."""
            estimates = [
                _estimate_for_maturity(i, Ti, batch_seed)
                for i, Ti in enumerate(T, start=1)
            ]
            return {
                key: np.asarray([estimate[key] for estimate in estimates], dtype=float)
                for key in keys
            }

        if n_batch == 1:
            out = _run_batch(config.seed)
            out["variance"] = variance_quad
            return out

        rng = np.random.default_rng(config.seed)
        batch_seeds = rng.integers(0, 2**32 - 1, size=n_batch)
        out_batch = []
        for i, batch_seed in enumerate(batch_seeds, start=1):
            batch_seed = int(batch_seed)
            if verbose:
                print(f"\nRunning batch {i}/{n_batch} with seed {batch_seed}")
            out_batch.append(_run_batch(batch_seed))

        out = {}
        for key in mc_keys:
            values = np.asarray([batch_out[key] for batch_out in out_batch])
            out[key] = values.mean(axis=0)
            err = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(n_batch)
            out[f"{key}_low"] = out[key] - err
            out[f"{key}_high"] = out[key] + err
        out["variance"] = variance_quad
        out["variance_low"] = variance_quad
        out["variance_high"] = variance_quad
        return out

    def swap_mc_estimators(
        self,
        T: float | np.ndarray,
        config: MonteCarloConfig | None = None,
        *,
        n_batch: int = 1,
        verbose: bool = False,
        magic_order=None,
        magic_tol: float = 1e-8,
        magic_max_iter: int = 100,
        magic_std: float | None = 5.0,
        magic_ks_interp=None,
        magic_n_interp: int = 1001,
        magic_extrapolation: str = "raise",
        fukasawa_n_quad: int = 5,
        fukasawa_ks_interp=None,
        fukasawa_std: float | None = 5.0,
        fukasawa_n_interp: int = 1001,
        fukasawa_extrapolation: str = "raise",
        **mc_kwargs,
    ) -> dict:
        """
        Estimate swap quantities directly and from the simulated option smile.

        For each maturity and batch, one set of paths is used to compute direct
        Monte Carlo estimates and a Monte Carlo implied-volatility smile. The
        magic-strike and Fukasawa estimates are both evaluated on that same
        smile, so estimator comparisons within a batch use common random
        numbers. Variance in the direct-Monte-Carlo output is the deterministic
        value returned by :meth:`var_swap_quad`.

        Parameters
        ----------
        T : float or array_like
            Positive maturity or maturities. Returned estimate arrays follow
            this one-dimensional maturity order.
        config : MonteCarloConfig, optional
            Simulation configuration. If omitted, ``n_mc`` and ``n_disc`` must
            be supplied through ``mc_kwargs``. The optional ``n_loop``, ``seed``,
            and ``antithetic`` values may also be supplied through ``mc_kwargs``;
            when ``config`` is provided, those three values override its fields.
        n_batch : int, default 1
            Number of independent simulations, each containing ``config.n_mc``
            paths per maturity. For more than one batch, estimates are averaged
            and normal-approximation 95% confidence intervals are computed from
            the independent batch estimates.
        verbose : bool, default False
            Print simulation and Fukasawa progress information.
        magic_order : int, array_like, ``"max"``, or None, optional
            Magic-strike order or orders. ``None`` and ``"max"`` request every
            supported order for each contract.
        magic_tol : float, default 1e-8
            Fixed-point convergence tolerance for the magic-strike estimates.
        magic_max_iter : int, default 100
            Maximum number of magic-strike fixed-point iterations.
        magic_std : float or None, default 5
            Half-width of the automatically generated magic-strike smile grid,
            measured in ATM total standard deviations. Set to ``None`` when
            passing ``magic_ks_interp``.
        magic_ks_interp : array_like, optional
            Explicit log-strike grid for the magic-strike smile interpolator.
        magic_n_interp : int, default 1001
            Number of points in the automatic magic-strike interpolation grid.
        magic_extrapolation : {"raise", "flat"}, default "raise"
            Policy for magic-strike smile evaluations outside the fitted grid.
        fukasawa_n_quad : int, default 5
            Number of quadrature nodes in each Fukasawa estimate.
        fukasawa_ks_interp : array_like, optional
            Explicit log-strike grid used to construct the Fukasawa inverse
            normalizing transformation.
        fukasawa_std : float or None, default 5
            Half-width of the automatic Fukasawa interpolation grid, measured
            in ATM total standard deviations. Set to ``None`` when passing
            ``fukasawa_ks_interp``.
        fukasawa_n_interp : int, default 1001
            Number of points in the automatic Fukasawa interpolation grid.
        fukasawa_extrapolation : {"raise", "flat"}, default "raise"
            Policy for Fukasawa evaluations outside the fitted interpolation
            domains.
        **mc_kwargs
            Monte Carlo configuration values accepted by
            :func:`resolve_mc_config`.

        Returns
        -------
        dict
            A dictionary with three entries:

            ``"mc"``
                Direct estimates keyed by ``"variance"`` and ``"gamma"``.
            ``"magic_strikes"``
                Nested as ``result["magic_strikes"][contract][order]`` for
                variance and gamma.
            ``"fukasawa"``
                Fukasawa estimates keyed by the same two contract names.

            Every estimate is an array with one value per maturity. If
            ``n_batch > 1``, each point estimate also has ``"*_low"`` and
            ``"*_high"`` entries containing mean plus or minus 1.96 batch
            standard errors. For magic strikes, these keys are strings such as
            ``"5_low"`` and ``"5_high"`` inside the contract dictionary.

        Notes
        -----
        Confidence intervals quantify sampling error across batches. They do
        not include time-discretization bias, smile interpolation error,
        quadrature error, or magic-strike truncation error.
        """
        config = resolve_mc_config(config, mc_kwargs)
        T, n_batch, n_loop = self._validate_swap_mc_inputs(T, config, n_batch)

        keys = ("variance", "gamma")
        mc_keys = ("gamma",)
        n_expiries = len(T)
        variance_quad = _as_1d_float_array(self.var_swap_quad(T))

        if verbose:
            print("\nComputing shared Monte Carlo swap estimators:")
            print("---------------------------------------------")
            print(f"Model: {self.__class__.__name__}")
            print("Maturity T: ", T)
            print(f"Monte Carlo paths: {config.n_mc}")
            print(f"Number of time steps: {config.n_disc}")
            print(f"Number of simulation loops: {n_loop}")
            print(f"Number of independent batches: {n_batch}\n")

        def _simulate_for_maturity(i, Ti, batch_seed):
            """Return the grid, paths, and direct estimates for one maturity."""
            if verbose:
                print(f"{i}/{n_expiries}: expiry {Ti:.4f}")
            tab_t = np.linspace(0.0, Ti, config.n_disc + 1)
            paths = self.simulate_mc(
                tab_t=tab_t,
                n_mc=config.n_mc,
                n_loop=n_loop,
                seed=batch_seed,
                antithetic=config.antithetic,
            )
            estimate = self._swap_estimate_from_simulated_paths(
                tab_t,
                paths,
                variance=variance_quad[i - 1],
            )
            return tab_t, paths, estimate

        def _run_batch(batch_seed, *, keep_paths: bool = False):
            """Run one batch and optionally retain paths for smile estimation."""
            estimates = []
            kept_paths = []
            for i, Ti in enumerate(T, start=1):
                _tab_t, paths, estimate = _simulate_for_maturity(i, Ti, batch_seed)
                estimates.append(estimate)
                if keep_paths:
                    kept_paths.append((float(Ti), paths))
            batch_out = {
                key: np.asarray([estimate[key] for estimate in estimates], dtype=float)
                for key in keys
            }
            return batch_out, kept_paths

        def _estimate_smile_from_paths(kept_paths):
            """Compute magic-strike and Fukasawa results from retained paths."""
            smile_paths = {Ti: paths for Ti, paths in kept_paths}

            def _shared_tot_impvar(k, T):
                """Recover total implied variance from one maturity's paths."""
                Ti = float(T)
                paths = smile_paths[Ti]
                impvol = implied_vol_from_paths(
                    k=k,
                    T=Ti,
                    int_v_dt=paths["int_v_dt"],
                    int_sqrt_v_dw=paths["int_sqrt_v_dw"],
                    s0=self.s0,
                    conditioning=False,
                )
                return np.asarray(impvol, dtype=float) ** 2 * Ti

            magic_out = swap.swap_magic_strike(
                T=T,
                tot_impvar=_shared_tot_impvar,
                order=magic_order,
                opt="all",
                tol=magic_tol,
                max_iter=magic_max_iter,
                std=magic_std,
                ks_interp=magic_ks_interp,
                n_interp=magic_n_interp,
                extrapolation=magic_extrapolation,
            )
            fukasawa_out = swap.swap_fukasawa(
                T=T,
                tot_impvar=_shared_tot_impvar,
                n_quad=fukasawa_n_quad,
                ks_interp=fukasawa_ks_interp,
                opt="all",
                std=fukasawa_std,
                n_interp=fukasawa_n_interp,
                extrapolation=fukasawa_extrapolation,
                verbose=verbose,
            )
            return magic_out, fukasawa_out

        def _mean_ci(values):
            """Return a batch mean and normal-approximation 95% interval."""
            values = np.asarray(values, dtype=float)
            mean = values.mean(axis=0)
            err = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(n_batch)
            return mean, mean - err, mean + err

        def _aggregate_magic(batch_values):
            """Aggregate nested magic-strike results across batches."""
            out = {}
            for opt in batch_values[0]:
                out[opt] = {}
                for order in batch_values[0][opt]:
                    values = [batch[opt][order] for batch in batch_values]
                    mean, low, high = _mean_ci(values)
                    out[opt][order] = mean
                    out[opt][f"{order}_low"] = low
                    out[opt][f"{order}_high"] = high
            return out

        def _aggregate_fukasawa(batch_values):
            """Aggregate Fukasawa contract results across batches."""
            out = {}
            for opt in batch_values[0]:
                values = [batch[opt] for batch in batch_values]
                mean, low, high = _mean_ci(values)
                out[opt] = mean
                out[f"{opt}_low"] = low
                out[f"{opt}_high"] = high
            return out

        if n_batch == 1:
            mc_out, kept_paths = _run_batch(config.seed, keep_paths=True)
            mc_out["variance"] = variance_quad
            magic_out, fukasawa_out = _estimate_smile_from_paths(kept_paths)
            return {
                "mc": mc_out,
                "magic_strikes": magic_out,
                "fukasawa": fukasawa_out,
            }
        else:
            rng = np.random.default_rng(config.seed)
            batch_seeds = rng.integers(0, 2**32 - 1, size=n_batch)
            out_batch = []
            magic_batch = []
            fukasawa_batch = []
            for i, batch_seed in enumerate(batch_seeds, start=1):
                batch_seed = int(batch_seed)
                if verbose:
                    print(f"\nRunning batch {i}/{n_batch} with seed {batch_seed}")
                batch_out, kept_paths = _run_batch(batch_seed, keep_paths=True)
                out_batch.append(batch_out)
                magic_out, fukasawa_out = _estimate_smile_from_paths(kept_paths)
                magic_batch.append(magic_out)
                fukasawa_batch.append(fukasawa_out)

            mc_out = {}
            for key in mc_keys:
                values = np.asarray([batch_out[key] for batch_out in out_batch])
                mc_out[key] = values.mean(axis=0)
                err = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(n_batch)
                mc_out[f"{key}_low"] = mc_out[key] - err
                mc_out[f"{key}_high"] = mc_out[key] + err
            mc_out["variance"] = variance_quad
            mc_out["variance_low"] = variance_quad
            mc_out["variance_high"] = variance_quad
            magic_out = _aggregate_magic(magic_batch)
            fukasawa_out = _aggregate_fukasawa(fukasawa_batch)

        return {
            "mc": mc_out,
            "magic_strikes": magic_out,
            "fukasawa": fukasawa_out,
        }

    @abstractmethod
    def impvol(self, k, T, **kwargs) -> float | np.ndarray:
        """Return model implied volatility at log-moneyness and maturity.

        Parameters
        ----------
        k : float or array_like
            Log-moneyness ``log(K/F)``.
        T : float
            Time to maturity.
        **kwargs
            Model-specific pricing arguments.

        Returns
        -------
        float or ndarray
            Black implied volatility with shape compatible with ``k``.
        """

    def total_impvar(self, k, T, **kwargs) -> float | np.ndarray:
        """Return Black total implied variance ``impvol(k, T)**2 * T``.

        Parameters
        ----------
        k : float or array_like
            Log-moneyness ``log(K/F)``.
        T : float
            Time to maturity.
        **kwargs
            Model-specific arguments forwarded to :meth:`impvol`.

        Returns
        -------
        float or ndarray
            Total implied variance with shape compatible with ``k``.
        """
        return self.impvol(k, T, **kwargs) ** 2 * T

    def var_swap_quad(self, T):
        """
        Integrate the initial forward-variance curve over each maturity.

        The total variance-swap strike is
        ``integral_0**T xi0(t) dt``. Each integral is evaluated by adaptive
        quadrature after rescaling its domain to the unit interval.

        Parameters
        ----------
        T : float or array_like
            Maturity or maturities.

        Returns
        -------
        ndarray
            One total variance-swap strike per maturity.
        """
        T = np.atleast_1d(T)
        return T * np.array(
            [integrate.quad(lambda u, Ti=Ti: self.xi0(Ti * u), 0, 1)[0] for Ti in T]
        )

    def _var_swap_trapezoidal_grid(self, tab_t: np.ndarray) -> float:
        """Return the trapezoidal integral of ``xi0`` on a simulation grid.

        This is the expectation of the discretized integrated-variance control
        used by the Monte Carlo swap estimators.

        Parameters
        ----------
        tab_t : array_like
            One-dimensional simulation time grid.

        Returns
        -------
        float
            Trapezoidal forward-variance integral.

        Raises
        ------
        ValueError
            If ``xi0(tab_t)`` is neither scalar nor shape-compatible with the
            flattened grid.
        """
        tab_t = np.asarray(tab_t, dtype=float).ravel()
        dt = np.diff(tab_t)
        xi0_grid = np.asarray(self.xi0(tab_t), dtype=float).reshape(-1)
        if xi0_grid.size == 1:
            xi0_grid = np.full_like(tab_t, xi0_grid[0], dtype=float)
        if xi0_grid.shape != tab_t.shape:
            raise ValueError("xi0 must return values compatible with tab_t.")
        return float(0.5 * np.sum(dt * (xi0_grid[:-1] + xi0_grid[1:])))

    def swap_fukasawa(
        self,
        T,
        opt="all",
        n_quad: int = 5,
        ks_interp=None,
        std: None | float = 5,
        n_interp: int = 1001,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Compute one or all Fukasawa estimates from the model smile.

        Parameters
        ----------
        T : float or array_like
            Positive maturity or maturities.
        opt : {"all", "variance", "gamma"}, default "all"
            Contract or collection of contracts to estimate.
        n_quad : int, default 5
            Number of quadrature nodes.
        ks_interp : array_like, optional
            Explicit log-strike grid used to invert the Fukasawa normalizing
            transformation. It cannot be combined with ``std``.
        std : float or None, default 5
            Width of an automatically generated interpolation grid in ATM total
            standard deviations. Set to ``None`` when passing ``ks_interp``.
        n_interp : int, default 1001
            Number of points in the automatic interpolation grid.
        verbose : bool, default False
            Print Fukasawa progress information.
        **kwargs
            Model-specific arguments forwarded to :meth:`total_impvar`.

        Returns
        -------
        ndarray or dict
            A maturity array for one contract. For ``opt="all"``, returns arrays
            keyed by variance and gamma.

        Raises
        ------
        ValueError
            If maturities, grid settings, or interpolation inputs are invalid.
        """
        return self._swap_fukasawa(
            T=T,
            opt=opt,
            n_quad=n_quad,
            ks_interp=ks_interp,
            std=std,
            n_interp=n_interp,
            verbose=verbose,
            **kwargs,
        )

    def var_swap_fukasawa(
        self, T, n_quad=5, ks_interp=None, std=5, n_interp=1001, **kwargs
    ):
        """
        Compute total variance-swap strikes using Fukasawa's representation.

        Parameters
        ----------
        T : float or array_like
            Positive maturity or maturities.
        n_quad : int, default 5
            Number of quadrature nodes.
        ks_interp : array_like, optional
            Explicit log-strike grid. It cannot be combined with ``std``.
        std : float or None, default 5
            Width of the automatic grid in ATM total standard deviations. Set
            to ``None`` when passing ``ks_interp``.
        n_interp : int, default 1001
            Number of points in the automatic interpolation grid.
        **kwargs
            Model-specific arguments forwarded to :meth:`total_impvar`.

        Returns
        -------
        ndarray
            One total variance-swap strike per maturity.

        Raises
        ------
        ValueError
            If maturities, grid settings, or interpolation inputs are invalid.
        """
        return self._swap_fukasawa(
            T=T,
            n_quad=n_quad,
            ks_interp=ks_interp,
            opt="variance",
            std=std,
            n_interp=n_interp,
            **kwargs,
        )

    def gamma_swap_fukasawa(
        self, T, n_quad=5, ks_interp=None, std=5, n_interp=1001, **kwargs
    ):
        """
        Compute total gamma-swap strikes using Fukasawa's representation.

        Parameters
        ----------
        T : float or array_like
            Positive maturity or maturities.
        n_quad : int, default 5
            Number of quadrature nodes.
        ks_interp : array_like, optional
            Explicit log-strike grid. It cannot be combined with ``std``.
        std : float or None, default 5
            Width of the automatic grid in ATM total standard deviations. Set
            to ``None`` when passing ``ks_interp``.
        n_interp : int, default 1001
            Number of points in the automatic interpolation grid.
        **kwargs
            Model-specific arguments forwarded to :meth:`total_impvar`.

        Returns
        -------
        ndarray
            One total gamma-swap strike per maturity.

        Raises
        ------
        ValueError
            If maturities, grid settings, or interpolation inputs are invalid.
        """
        return self._swap_fukasawa(
            T=T,
            n_quad=n_quad,
            ks_interp=ks_interp,
            opt="gamma",
            std=std,
            n_interp=n_interp,
            **kwargs,
        )

    def _swap_fukasawa(
        self,
        T,
        *,
        opt,
        n_quad: int = 5,
        ks_interp=None,
        std: None | float = None,
        n_interp: int = 1000,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Validate model inputs and delegate to :func:`swap.swap_fukasawa`.

        Parameters
        ----------
        T : float or array_like
            Positive maturity or maturities.
        opt : str
            Contract selector accepted by :func:`swap.swap_fukasawa`.
        n_quad : int, default 5
            Number of quadrature nodes.
        ks_interp : array_like, optional
            Explicit log-strike interpolation grid.
        std : float or None, optional
            Automatic-grid width in ATM total standard deviations.
        n_interp : int, default 1000
            Number of automatic interpolation points.
        verbose : bool, default False
            Print Fukasawa progress information.
        **kwargs
            Model-specific arguments forwarded to :meth:`total_impvar`.

        Returns
        -------
        ndarray or dict
            Result returned by :func:`swap.swap_fukasawa`.
        """
        _reject_conditioning_kwargs(kwargs, "Fukasawa computations")
        validate_positive("n_quad", n_quad)
        validate_interpolation_grid_choice(ks_interp, std)
        if std is not None:
            validate_positive("std", std)
        validate_positive("n_interp", n_interp)

        T = _as_1d_float_array(T)
        if np.any(~np.isfinite(T)) or np.any(T <= 0.0):
            raise ValueError("T must be finite and positive.")

        return swap.swap_fukasawa(
            T,
            lambda k, T: self.total_impvar(k=k, T=T, **kwargs),
            n_quad=n_quad,
            ks_interp=ks_interp,
            opt=opt,
            std=std,
            n_interp=n_interp,
            verbose=verbose,
        )

    @overload
    def swap_magic_strike(
        self,
        T: Any,
        order: Any = None,
        opt: MagicStrikeOpt = "variance",
        tol: float = 1e-8,
        max_iter: int = 100,
        relax: float = 0.8,
        std: float = 5.0,
        n_interp: int = 1001,
        p: float | None = None,
        **kwargs: Any,
    ) -> MagicStrikeResult:
        """Type overload for a single magic-strike contract."""

    @overload
    def swap_magic_strike(
        self,
        T: Any,
        order: Any = None,
        opt: Literal["all"] = "all",
        tol: float = 1e-8,
        max_iter: int = 100,
        relax: float = 0.8,
        std: float = 5.0,
        n_interp: int = 1001,
        p: None = None,
        **kwargs: Any,
    ) -> MagicStrikeAllResult:
        """Type overload for all supported magic-strike contracts."""

    @overload
    def swap_magic_strike(
        self,
        T: Any,
        order: Any = None,
        opt: str = "variance",
        tol: float = 1e-8,
        max_iter: int = 100,
        relax: float = 0.8,
        std: float = 5.0,
        n_interp: int = 1001,
        p: float | None = None,
        **kwargs: Any,
    ) -> MagicStrikeResult | MagicStrikeAllResult:
        """Fallback type overload for a dynamically selected contract."""

    def swap_magic_strike(
        self,
        T,
        order=None,
        opt="variance",
        tol=1e-8,
        max_iter=100,
        relax=0.8,
        std: float = 5.0,
        n_interp: int = 1001,
        p=None,
        **kwargs,
    ):
        """
        Compute magic-strike swap approximations from the model smile.

        Parameters
        ----------
        T : float or array_like
            Positive maturity or maturities.
        order : int, array_like, ``"max"``, or None, optional
            Approximation order or orders. ``None`` and ``"max"`` request all
            supported orders for the selected contract.
        opt : {'variance', 'gamma', 'power',
            'all'},
            default 'variance'
            Swap type to compute. ``'gamma'`` uses the gamma fixed-point method.
            Use ``'all'`` for variance and gamma from the same interpolated smile.
        p : float, optional
            Power-payoff parameter in ``[0, 1]``. Required exactly when
            ``opt='power'``. The returned values are implied power variances.
        tol : float, default 1e-8
            Fixed-point convergence tolerance.
        max_iter : int, default 100
            Maximum number of fixed-point iterations.
        relax : float, default 0.8
            Fixed-point damping factor.
        std : float, default 5
            Half-width of the interpolation grid in ATM total standard
            deviations.
        n_interp : int, default 1001
            Number of interpolation-grid points.
        **kwargs
            Model-specific arguments forwarded to :meth:`total_impvar`.

        Returns
        -------
        dict
            For one contract, maps each requested integer order to a maturity
            array. For ``opt="all"``, first maps each contract name to its
            order dictionary.

        Raises
        ------
        ValueError
            If maturities, orders, interpolation settings, or fixed-point inputs
            are invalid.
        """
        return swap.swap_magic_strike(
            T=T,
            tot_impvar=lambda k, T: self.total_impvar(k=k, T=T, **kwargs),
            order=order,
            opt=opt,
            p=p,
            tol=tol,
            max_iter=max_iter,
            relax=relax,
            std=std,
            n_interp=n_interp,
        )
