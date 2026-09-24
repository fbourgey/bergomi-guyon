from collections.abc import Callable

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import blas
from scipy.special import hyp2f1

from .model import (
    ForwardVarianceModel,
    MonteCarloConfig,
    require_params,
    resolve_mc_config,
    validate_interval,
    validate_positive,
)
from .utils import cholesky_from_svd


class RoughBergomiModel(ForwardVarianceModel):
    """
    Rough Bergomi model.
    """

    def __init__(
        self,
        params: dict,
        xi0: Callable[[np.ndarray], np.ndarray],
        s0: float = 1.0,
    ) -> None:
        """
        Initialize Rough Bergomi model.

        Parameters
        ----------
        xi0 : callable
            Initial forward variance curve function.
        params : dict
            Dictionary containing model parameters:
            - eta: volatility of volatility
            - H: Hurst parameter
            - rho: correlation between Brownian motions
        s0 : float, optional
            Initial stock price, by default 1.0
        """
        self.eta, self.H, self.rho = require_params(params, ("eta", "H", "rho"))

        super().__init__(params=params, xi0=xi0, s0=s0)
        self._check_params()
        self._cholesky_cache = {}
        self._unit_cholesky_cache = {}

    def _check_params(self):
        """Validate Rough Bergomi parameters."""
        validate_positive("eta", self.eta)
        validate_interval("rho", self.rho, -1, 1)
        validate_interval("H", self.H, 0, 1, closed=False)

    def _cached_cholesky_cov_matrix(self, tab_t, conditioning: bool = False):
        chol, scales = self._cached_cholesky_transform(tab_t, conditioning)
        if scales is None:
            return chol

        scaled_chol = scales[:, np.newaxis] * chol
        scaled_chol.setflags(write=False)
        return scaled_chol

    def _cached_cholesky_transform(self, tab_t, conditioning: bool = False):
        tab_t = np.ascontiguousarray(tab_t, dtype=float)
        unit_chol = self._cached_unit_cholesky_cov_matrix(tab_t, conditioning)
        if unit_chol is not None:
            T = float(tab_t[-1])
            n_disc = tab_t.shape[0] - 1
            scales = np.empty(2 * n_disc)
            scales[:n_disc] = T**self.H
            scales[n_disc:] = T**0.5
            return unit_chol, scales

        key = (conditioning, tab_t.dtype.str, tab_t.shape, tab_t.tobytes())
        chol = self._cholesky_cache.get(key)
        if chol is None:
            chol = self.cholesky_cov_matrix(tab_t, conditioning=conditioning)
            chol.setflags(write=False)
            self._cholesky_cache[key] = chol
        return chol, None

    def _cached_unit_cholesky_cov_matrix(self, tab_t, conditioning: bool):
        T = float(tab_t[-1])
        n_disc = tab_t.shape[0] - 1
        if T <= 0.0 or n_disc <= 0:
            return None

        unit_grid = np.linspace(0.0, 1.0, n_disc + 1)
        if not np.allclose(tab_t / T, unit_grid, rtol=1e-12, atol=1e-14):
            return None

        key = (conditioning, n_disc)
        chol = self._unit_cholesky_cache.get(key)
        if chol is None:
            chol = self.cholesky_cov_matrix(unit_grid, conditioning=conditioning)
            chol.setflags(write=False)
            self._unit_cholesky_cache[key] = chol
        return chol

    def kernel(self, u, t):
        return self.eta * np.sqrt(2.0 * self.H) * (u - t) ** (self.H - 0.5)

    def cov_levy_fbm(self, u, v):
        r"""
        Compute the covariance matrix of Levy's fractional Brownian motion.

        It corresponds to:

        Cov(W_u^H, W_v^H) = \int_0^{min(u,v)} (u-s)^{H-1/2} (v-s)^{H-1/2} ds

        where:

        W_u^H = \int_0^u (u-s)^{H-1/2} dW_s

        Parameters
        ----------
        u : np.ndarray or float
            First set of time points.
        v : np.ndarray or float
            Second set of time points.

        Returns
        -------
        np.ndarray
            Covariance matrix evaluated at (u, v).
        """
        u_max_v = np.maximum(u, v)
        u_min_v = np.minimum(u, v)
        cov = (
            u_min_v ** (self.H + 0.5)
            * u_max_v ** (self.H - 0.5)
            * hyp2f1(1.0, 0.5 - self.H, 1.5 + self.H, u_min_v / u_max_v)
        )
        return cov / (self.H + 0.5)

    def cholesky_cov_matrix(
        self, tab_t, return_cov: bool = False, conditioning: bool = False
    ):
        r"""
        Compute the lower-triangular Cholesky factor
        of the covariance matrix of the Gaussian vector (Y_{t_i}, W_{t_i})
        for 1 <= i <= n, where t_i are the timesteps in tab_t.

        Here, W is a standard Brownian motion and
        Y_t = \sqrt{2H} \int_0^t (t-s)^{H-1/2} dW_s.

        Parameters
        ----------
        tab_t : np.ndarray
            Array of time grid points (shape: n_steps + 1,).
        return_cov : bool, optional
            If True, return the full covariance matrix instead of its Cholesky factor.
            Default is False.
        conditioning : bool, optional
            If True, compute the conditional covariance (see Bergomi's book, Chapter
            8, Appendix A).
            Default is False.

        Returns
        -------
        np.ndarray
            Lower-triangular Cholesky factor of the covariance matrix, or the covariance
            matrix itself if `return_cov` is True.
        """
        n_disc = tab_t.shape[0] - 1
        # repeat tab_t[1:] n_disc times as columns (shape: n_disc x n_disc)
        u = np.tile(tab_t[1:], (n_disc, 1)).T
        cov_y = 2.0 * self.H * self.cov_levy_fbm(u, u.T)
        cov_w = np.minimum(u, u.T)
        cov_yw = u ** (self.H + 0.5) - (u - cov_w) ** (self.H + 0.5)
        cov_yw *= np.sqrt(2.0 * self.H) / (self.H + 0.5)
        if not conditioning:
            cov_yw *= self.rho
        cov = np.block(
            [
                [cov_y, cov_yw],
                [cov_yw.T, cov_w],
            ]
        )
        if return_cov:
            return cov
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            chol = cholesky_from_svd(cov)
        except Exception as e:
            print(f"Error in Cholesky decomposition: {e}")
            raise

        return chol

    def simulate_mc(
        self,
        tab_t: np.ndarray,
        n_mc: int,
        n_loop: int = 1,
        seed: int | None = None,
        antithetic: bool = False,
    ) -> dict:
        rng = np.random.default_rng(seed)

        n_mc_loop, remainder = divmod(n_mc, n_loop)
        if remainder != 0:
            raise ValueError("n_mc must be divisible by n_loop")
        if antithetic and n_mc_loop % 2 != 0:
            raise ValueError("n_mc // n_loop must be even when antithetic=True")

        h_half = self.H == 0.5
        n_disc = tab_t.shape[0] - 1
        dt = tab_t[1] - tab_t[0]

        int_v_dt = np.zeros(n_mc)
        int_sqrt_v_dw = np.zeros(n_mc)

        # Precompute deterministic quantities once
        tab_t_col = tab_t[:, np.newaxis]
        xi0_t = self.xi0(tab_t_col)
        xi0_prev = xi0_t[:-1, :]
        xi0_next = xi0_t[1:, :]
        drift_t = -0.5 * self.eta**2 * tab_t_col ** (2.0 * self.H)

        # Precompute Cholesky once (expensive)
        if not h_half:
            chol, chol_scales = self._cached_cholesky_transform(tab_t)

        sqrt_dt = np.sqrt(dt)
        rho_perp = np.sqrt(1.0 - self.rho**2)
        n_draw = n_mc_loop // 2 if antithetic else n_mc_loop

        def _normal_antithetic(shape):
            shocks = rng.normal(0.0, 1.0, shape)
            if antithetic:
                shocks = np.concatenate((shocks, -shocks), axis=1)
            return shocks

        for i in range(n_loop):
            sl = slice(i * n_mc_loop, (i + 1) * n_mc_loop)

            if h_half:
                dy = sqrt_dt * _normal_antithetic((n_disc, n_draw))
                y = np.empty((n_disc + 1, n_mc_loop))
                y[0, :] = 0.0
                np.cumsum(dy, axis=0, out=y[1:, :])

                if rho_perp == 0.0:
                    dw = self.rho * dy
                else:
                    dz = sqrt_dt * _normal_antithetic((n_disc, n_draw))
                    dw = self.rho * dy + rho_perp * dz
            else:
                normal = _normal_antithetic((2 * n_disc, n_draw))
                normal = blas.dtrmm(
                    1.0,
                    chol,
                    normal,
                    side=0,
                    lower=1,
                    trans_a=0,
                    diag=0,
                )
                if chol_scales is not None:
                    normal *= chol_scales[:, np.newaxis]
                y = np.empty((n_disc + 1, n_mc_loop))
                y[0, :] = 0.0
                y[1:, :] = normal[:n_disc, :]
                w = normal[n_disc:, :]
                dw = np.empty_like(w)
                dw[0, :] = w[0, :]
                dw[1:, :] = w[1:, :] - w[:-1, :]
                normal = None  # free memory

            y *= self.eta
            y += drift_t
            np.exp(y, out=y)
            exp_term = y

            exp_prev = exp_term[:-1, :]
            exp_next = exp_term[1:, :]
            v_prev = xi0_prev * exp_prev
            v_next = xi0_next * exp_next
            sqrt_v_prev = np.sqrt(v_prev)
            int_sqrt_v_dw[sl] = np.sum(sqrt_v_prev * dw, axis=0)
            int_v_dt[sl] = 0.5 * dt * np.sum(v_prev + v_next, axis=0)

        out = {
            "int_v_dt": int_v_dt,
            "int_sqrt_v_dw": int_sqrt_v_dw,
        }

        return out

    def impvol(
        self,
        k,
        T: float,
        config: MonteCarloConfig | None = None,
        *,
        return_skew: bool = False,
        **mc_kwargs,
    ):
        config = resolve_mc_config(config, mc_kwargs)
        return self.impvol_mc(
            k=k,
            T=T,
            n_mc=config.n_mc,
            n_disc=config.n_disc,
            n_loop=config.n_loop,
            seed=config.seed,
            antithetic=config.antithetic,
            return_skew=return_skew,
        )


def get_params_rough_bergomi(
    id: int,
) -> tuple[dict, Callable[[np.ndarray], np.ndarray]]:
    """Get Rough Bergomi parameters for a given id."""
    if id not in [1, 2]:
        raise ValueError("Invalid id. Please choose a valid id.")

    _mats = (7 / 365, 1 / 12, 3 / 12, 6 / 12, 9 / 12, 1, 1.5, 2, 2.5, 3)

    if id == 1:
        params = {"eta": 1.5, "H": 0.23, "rho": -0.62}
        _xi0_mats = (
            0.0101,
            0.0164,
            0.0197,
            0.0277,
            0.0319,
            0.0411,
            0.0450,
            0.0587,
            0.0679,
            0.0830,
        )
    if id == 2:
        params = {"eta": 2.39, "H": 0.11, "rho": -0.86}
        _xi0_mats = (
            0.1396,
            0.1003,
            0.0801,
            0.0665,
            0.0551,
            0.0605,
            0.0623,
            0.0566,
            0.0797,
            0.0800,
        )

    _mats = np.asarray(_mats, dtype=float)
    _xi0_mats = np.asarray(_xi0_mats, dtype=float)
    xi0 = interp1d(
        _mats,
        _xi0_mats,
        kind="cubic",
        bounds_error=False,
        fill_value=(_xi0_mats[0], _xi0_mats[-1]),  # type: ignore
    )
    return params, xi0
