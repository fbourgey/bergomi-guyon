import numpy as np
from scipy import integrate

from .model import (
    ForwardVarianceModel,
    require_params,
    validate_interval,
    validate_nonnegative,
    validate_positive,
)
from .utils import _xi0_heston, black_impvol, lewis_formula_otm_price


class HestonModel(ForwardVarianceModel):
    """Classical Heston forward-variance model."""

    def __init__(self, params, s0=1.0):
        """
        Initialize a Heston model.

        Parameters
        ----------
        params : dict
            Dictionary containing ``lbd``, ``rho``, ``nu``, ``vbar``, and ``v``.
        s0 : float, optional
            Initial spot price.
        """
        super().__init__(params=params, s0=s0)
        self.lbd, self.rho, self.nu, self.vbar, self.v = require_params(
            params, ("lbd", "rho", "nu", "vbar", "v")
        )
        self.xi0 = lambda t: _xi0_heston(t=t, lbd=self.lbd, vbar=self.vbar, v=self.v)
        self._check_params()

    def _check_params(self):
        """Check that parameters satisfy necessary conditions for the Heston model."""
        validate_positive("nu", self.nu)
        validate_nonnegative("lbd", self.lbd)
        validate_interval("rho", self.rho, -1.0, 1.0)
        validate_nonnegative("vbar", self.vbar)
        validate_nonnegative("v", self.v)

    def _clone_init_kwargs(self, params: dict) -> dict:
        """Heston reconstructs xi0 from params, so do not pass xi0 explicitly."""
        return {"params": params, "s0": self.s0}

    def kernel(self, u, t):
        """Exponential Heston kernel."""
        lbd = self.params["lbd"]
        nu = self.params["nu"]
        return nu * np.exp(-lbd * (u - t))

    def charfunc(self, u, T):
        """
        Compute the characteristic function of the Heston model E[exp(i * u * X_T)]
        where X_T = log(S_T/S_0) at u for time T.

        Parameters
        ----------
        u : float or array_like
            Points at which to evaluate the characteristic function
        T : float
            Maturity


        Returns
        -------
        complex or array_like
            Value of the characteristic function
        """
        al = -u * u / 2 - 1j * u / 2
        bet = self.lbd - self.rho * self.nu * 1j * u
        gam = self.nu**2 / 2
        d = np.sqrt(bet * bet - 4 * al * gam)
        rp = (bet + d) / (2 * gam)
        rm = (bet - d) / (2 * gam)
        g = rm / rp
        D = rm * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
        C = self.lbd * (
            rm * T - 2 / self.nu**2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
        )
        return np.exp(C * self.vbar + D * self.v)

    def impvol(self, k, T):
        """
        Calculate implied volatility in the Heston model using the characteristic
        function.

        Parameters
        ----------
        k : array_like
            Log strike k = log(K/F)
        T : float
            Maturity

        Returns
        -------
        array_like
            Black implied volatility
        """
        k = np.atleast_1d(np.asarray(k))
        otm_price = lewis_formula_otm_price(
            phi=lambda u, T: self.charfunc(u=u, T=T), k=k, T=T
        )
        opttype = 2 * (k > 0) - 1  # OTM options
        return black_impvol(K=np.exp(k), T=T, F=1, value=otm_price, opttype=opttype)

    def var_swap(self, T):
        """
        Compute the fair strike of a total variance swap.

        It is given by E[int_0^T v_t dt] where v_t is the variance process.

        Parameters
        ----------
        T : array_like
            Time to maturity

        Returns
        -------
        array_like
            Fair strike of the variance swap
        """
        T = np.asarray(T)

        if self.lbd == 0.0:
            return self.v * T

        return (
            self.vbar * T
            + (self.v - self.vbar) * (1.0 - np.exp(-self.lbd * T)) / self.lbd
        )

    def gamma_swap(self, T):
        """Compute the fair strike of a total gamma swap."""
        T = np.asarray(T)
        lbd_prime = self.lbd - self.rho * self.nu
        vbar_prime = self.lbd * self.vbar / lbd_prime

        if lbd_prime == 0.0:
            return vbar_prime * T

        return (
            vbar_prime * T
            + (self.v - vbar_prime) * (1.0 - np.exp(-lbd_prime * T)) / lbd_prime
        )

    def implied_power_variance(self, T, p):
        """Return the Heston implied power variance ``W_p(T)``.

        For ``p`` in ``[0, 1]``, ``W_p`` is defined by

        ``E[exp(p * X_T)] = exp(0.5 * p * (p - 1) * W_p(T))``.

        The endpoint values are understood by continuity: ``W_0`` is the total
        variance contract and ``W_1`` is the total gamma contract.

        Parameters
        ----------
        T : float or array_like
            Nonnegative maturity or maturities.
        p : float
            Scalar power in ``[0, 1]``.

        Returns
        -------
        float or ndarray
            Implied power variance with shape compatible with ``T``.

        Raises
        ------
        ValueError
            If ``T`` is invalid or ``p`` is not a finite scalar in ``[0, 1]``.
        """
        T = np.asarray(T, dtype=float)
        if np.any(~np.isfinite(T)) or np.any(T < 0.0):
            raise ValueError("T must be finite and nonnegative.")

        p_array = np.asarray(p, dtype=float)
        if p_array.ndim != 0:
            raise ValueError("p must be a scalar.")
        p = float(p_array)
        if not np.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError("p must be finite and between 0 and 1.")

        if p == 0.0:
            return self.var_swap(T)
        if p == 1.0:
            return self.gamma_swap(T)

        lambda_p = 0.5 * p * (p - 1.0)
        mgf = np.asarray(self.charfunc(u=-1j * p, T=T))
        if np.any(~np.isfinite(mgf)) or np.any(mgf.real <= 0.0):
            raise ValueError("The Heston power moment is not finite and positive.")
        if np.any(np.abs(mgf.imag) > 1e-10 * np.maximum(1.0, mgf.real)):
            raise ValueError("The Heston power moment must be real.")

        log_mgf = np.log1p(mgf.real - 1.0)
        return log_mgf / lambda_p

    def _variance_log_spot_single(self, T):
        """
        Helper function to compute the variance of the log spot price at a single
        maturity T.

        Variance from characteristic function computed from
            Var[X] = -phi''(0) + (phi'(0))^2 where phi(u)=E[exp(i*u*X)].
        """
        _, cf_du, cf_du2 = self.charfunc_derivs(u=0.0, T=T)
        out = -cf_du2 + cf_du**2
        return out.real

    def charfunc_derivs(self, u, T):
        """
        Compute the characteristic function and its first and second derivatives with
        respect to u.

        Parameters
        ----------
        u : float or array_like
            Points at which to evaluate the characteristic function
        T : float
            Maturity

        Returns
        -------
        complex or array_like
            Value of the characteristic function
        complex or array_like
            First derivative of the characteristic function
        complex or array_like
            Second derivative of the characteristic function
        """
        nu = self.nu
        rho = self.rho
        lbd = self.lbd
        vbar = self.vbar
        v = self.v

        alpha = -u * u / 2 - 0.5j * u
        beta = lbd - rho * nu * 1j * u
        gamma = nu**2 / 2

        Delta = beta * beta - 4 * alpha * gamma
        d = np.sqrt(Delta)

        alpha_u = -u - 0.5j
        alpha_uu = -1.0
        beta_u = -1j * rho * nu

        Delta_u = 2 * beta * beta_u - 4 * gamma * alpha_u
        Delta_uu = 2 * beta_u * beta_u - 4 * gamma * alpha_uu

        d_u = Delta_u / (2 * d)
        d_uu = Delta_uu / (2 * d) - Delta_u**2 / (4 * d**3)

        rp = (beta + d) / (2 * gamma)
        rm = (beta - d) / (2 * gamma)

        rp_u = (beta_u + d_u) / (2 * gamma)
        rm_u = (beta_u - d_u) / (2 * gamma)

        rp_uu = d_uu / (2 * gamma)
        rm_uu = -d_uu / (2 * gamma)

        g = rm / rp
        g_u = (rm_u * rp - rm * rp_u) / rp**2
        g_uu = (
            rm_uu / rp
            - 2 * rm_u * rp_u / rp**2
            - rm * rp_uu / rp**2
            + 2 * rm * rp_u**2 / rp**3
        )

        E = np.exp(-d * T)

        N = rm * (1 - E)
        N_u = rm_u * (1 - E) + rm * T * d_u * E
        N_uu = (
            rm_uu * (1 - E)
            + 2 * rm_u * T * d_u * E
            + rm * E * (T * d_uu - T**2 * d_u**2)
        )

        Q = 1 - g * E
        Q_u = E * (T * g * d_u - g_u)
        Q_uu = E * (2 * T * g_u * d_u + T * g * d_uu - g_uu - T**2 * g * d_u**2)

        D = N / Q
        D_u = (N_u * Q - N * Q_u) / Q**2
        D_uu = N_uu / Q - N * Q_uu / Q**2 - 2 * N_u * Q_u / Q**2 + 2 * N * Q_u**2 / Q**3

        L_u = Q_u / Q + g_u / (1 - g)
        L_uu = Q_uu / Q - (Q_u / Q) ** 2 + g_uu / (1 - g) + g_u**2 / (1 - g) ** 2

        C = lbd * (rm * T - 2 / nu**2 * np.log(Q / (1 - g)))
        C_u = lbd * (T * rm_u - 2 / nu**2 * L_u)
        C_uu = lbd * (T * rm_uu - 2 / nu**2 * L_uu)

        A = vbar * C + v * D
        A_u = vbar * C_u + v * D_u
        A_uu = vbar * C_uu + v * D_uu

        phi = np.exp(A)
        phi_u = phi * A_u
        phi_uu = phi * (A_uu + A_u**2)

        return phi, phi_u, phi_uu

    def variance_log_spot(self, T):
        """
        Compute the variance of log(S_T/S_0) using the characteristic function.

        Parameters
        ----------
        T : array_like
            Time to maturity

        Returns
        -------
        array_like
            Variance of log(S_T/S_0)
        """
        return np.array([self._variance_log_spot_single(Ti) for Ti in np.atleast_1d(T)])

    def vol_swap_old(self, T, upper_lim=np.inf):
        """
        Compute the fair strike of a total volatility swap in the Heston model.

        Uses Gatheral's formula (The Volatility Surface, Equation 11.6) based on
        Laplace transforms of the variance process.

        It computes E[sqrt(int_0^T v_t dt)] where v_t is the variance process.

        Parameters
        ----------
        T : float or array_like
            Time to maturity
        upper_lim : float, optional
            Upper limit for numerical integration. Default is np.inf.

        Returns
        -------
        array_like
            Fair strike of the volatility swap

        Notes
        -----
        Numerical integration may be unstable for certain parameter combinations.
        """
        # TODO: improve code. numerical integration is unstable.
        T = np.atleast_1d(T)

        def integrand(psi, Ti):
            phi = np.sqrt(self.lbd**2 + 2 * psi * self.nu**2)
            denom = (phi + self.lbd) * (np.exp(phi * Ti) - 1) + 2 * phi
            A = 2 * phi * np.exp(0.5 * (phi + self.lbd) * Ti) / denom
            B = 2 * (np.exp(phi * Ti) - 1) / denom
            return (
                -np.expm1(
                    (2 * self.lbd * self.vbar / self.nu**2) * np.log(A)
                    - psi * self.v * B
                )
                / psi**1.5
            )

        integrals = np.array(
            [
                integrate.quad(lambda psi, Ti=Ti: integrand(psi, Ti), 0.0, upper_lim)[0]
                for Ti in T
            ]
        )
        return integrals / (2.0 * np.sqrt(np.pi))

    def vol_swap(self, T, *, upper_lim=np.inf, epsabs=1e-10, epsrel=1e-10, limit=200):
        """
        Compute the fair strike of a total volatility swap in the Heston model.

        Parameters
        ----------
        T : float or array_like
            Time to maturity.
        upper_lim : float, optional
            Upper truncation level in the original ``psi`` integration variable.
            Use ``np.inf`` for the full semi-infinite integral.
        epsabs : float, optional
            Absolute tolerance for numerical integration.
        epsrel : float, optional
            Relative tolerance for numerical integration.
        limit : int, optional
            Maximum number of QUADPACK subintervals.

        Returns
        -------
        float or ndarray
            Fair strike of the total volatility swap.

        Notes
        -----
        Reference: Gatheral, *The Volatility Surface*, Equation 11.6.
        The implementation removes the ``psi = 0`` integrable singularity via
        ``psi = (u / (1 - u))^2`` and uses overflow-safe formulas for large
        ``phi T``.
        """
        # TODO: check code

        T_arr = np.atleast_1d(np.asarray(T, dtype=float))
        alpha = 2.0 * self.lbd * self.vbar / (self.nu * self.nu)

        # --- stable helpers ---------------------------------------------------------
        def _logA_and_B(psi, Ti):
            # phi = sqrt(lbd^2 + 2*psi*nu^2)
            phi = np.sqrt(self.lbd * self.lbd + 2.0 * psi * (self.nu * self.nu))
            x = phi * Ti  # may be large

            # denom = (phi+lbd) * (exp(x)-1) + 2*phi
            # A = 2*phi*exp(0.5*(phi+lbd)*T) / denom
            # B = 2*(exp(x)-1) / denom
            #
            # We compute logA and B with a large-x branch to avoid overflow.

            if x < 50.0:
                em1 = np.expm1(x)
                denom = (phi + self.lbd) * em1 + 2.0 * phi

                # logA is more stable than A itself
                logA = np.log(2.0 * phi) + 0.5 * (phi + self.lbd) * Ti - np.log(denom)
                B = 2.0 * em1 / denom
                return logA, B

            # Large x: expm1(x) ~ exp(x) and denom ~ (phi+lbd)*exp(x)
            # => log denom ~ log(phi+lbd) + x
            # => logA ~ log(2phi) + 0.5*(phi+lbd)T - (log(phi+lbd)+x)
            log_denom = np.log(phi + self.lbd) + x
            logA = np.log(2.0 * phi) + 0.5 * (phi + self.lbd) * Ti - log_denom

            # B = 2*expm1(x)/denom ~ 2*exp(x)/((phi+lbd)exp(x)) = 2/(phi+lbd)
            B = 2.0 / (phi + self.lbd)
            return logA, B

        def _integrand_psi(psi, Ti):
            # original integrand in ψ:
            #   -expm1(alpha*log(A) - psi*v0*B) / psi^(3/2)
            logA, B = _logA_and_B(psi, Ti)
            expo = alpha * logA - psi * self.v * B
            return -np.expm1(expo) / (psi**1.5)

        # --- stable integration via change of variables -----------------------------
        # Use ψ(u) = (u/(1-u))^2 for (0,∞) and also removes the ψ=0 sqrt-singularity.
        #
        # dψ/du = 2u/(1-u)^3
        #
        # Integral_ψ=0..∞ f(ψ) dψ = Integral_u=0..1 f(ψ(u)) * dψ/du du
        #
        # For finite upper_lim, integrate u in [0, u_max] where ψ(u_max)=upper_lim.

        def _psi_from_u(u):
            t = u / (1.0 - u)
            return t * t

        def _dpsi_du(u):
            return 2.0 * u / ((1.0 - u) ** 3)

        def _integrand_u(u, Ti):
            psi = _psi_from_u(u)
            return _integrand_psi(psi, Ti) * _dpsi_du(u)

        def _u_max_from_upper(upper):
            # Solve upper = (u/(1-u))^2 => sqrt(upper)=u/(1-u) => u = s/(1+s)
            s = np.sqrt(upper)
            return s / (1.0 + s)

        out = np.empty_like(T_arr)

        for i, Ti in enumerate(T_arr):
            if Ti <= 0:
                out[i] = 0.0
                continue

            if np.isinf(upper_lim):
                a, b = 0.0, 1.0
                # avoid evaluating exactly at u=1
                b = np.nextafter(1.0, 0.0)
            else:
                if upper_lim <= 0:
                    out[i] = 0.0
                    continue
                a, b = 0.0, _u_max_from_upper(float(upper_lim))

            val, _ = integrate.quad(
                lambda u, Ti=Ti: _integrand_u(u, Ti),
                a,
                b,
                epsabs=epsabs,
                epsrel=epsrel,
                limit=limit,
                points=[0.0],  # tells QUADPACK about the endpoint behavior
            )
            out[i] = val / (2.0 * np.sqrt(np.pi))

        return out[0] if np.ndim(T) == 0 else out

    def var_integrated_variance(self, T):
        """
        Compute the variance of the integrated variance ``int_0^T v_t dt``.
        """
        # TODO: to be checked
        expT = np.exp(-self.lbd * T)
        exp2T = np.exp(-2 * self.lbd * T)

        out1 = self.vbar * (self.lbd * T - 3 / 2 + 2 * expT - exp2T / 2)
        out2 = (self.v - self.vbar) * (1 - exp2T - 2 * self.lbd * T * expT)

        return (self.nu**2 / (self.lbd**3)) * (out1 + out2)

    def var_integrated_variance_mc(self, T, n_disc, n_paths, seed=None):
        """
        Compute the variance of the integrated variance using Monte Carlo simulation.
        """

        # TODO: to be checked
        def _mc_single(Ti):
            _, _, int_v_dt = self.simulate_paths_qe_scheme(
                T=Ti, n_disc=n_disc, n_paths=n_paths, seed=seed
            )
            return np.var(np.sum(int_v_dt, axis=0))

        return np.array([_mc_single(Ti) for Ti in np.atleast_1d(T)])

    def simulate_paths_qe_scheme(
        self, T, n_disc, n_paths, psi_c=1.5, seed=None, eps=1e-14
    ):
        """
        Simulate Heston model paths using Andersen's QE discretization scheme.

        Parameters
        ----------
        T : float
            Time to maturity.
        n_disc : int
            Number of discretization steps.
        n_paths : int
            Number of Monte Carlo paths.
        psi_c : float, default 1.5
            Critical value for quadratic/exponential scheme switching.
        seed : int, optional
            Random seed for reproducibility.
        eps : float, default 1e-14
            Floor to keep conditional mean/variance and v_t non-negative.

        Returns
        -------
        S : ndarray, shape (n_disc + 1, n_paths)
            Stock price paths.
        v : ndarray, shape (n_disc + 1, n_paths)
            Variance paths.
        int_v_dt : ndarray, shape (n_disc, n_paths)
            Integrated variance over each time step.
        """
        if seed is not None:
            np.random.seed(seed)

        logS_qe = np.zeros((n_disc + 1, n_paths), dtype=float)
        v_qe = np.zeros((n_disc + 1, n_paths), dtype=float)
        int_v_dt = np.zeros((n_disc, n_paths), dtype=float)
        logS_qe[0, :] = np.log(self.s0)
        v_qe[0, :] = self.v

        ts = np.linspace(0.0, T, n_disc + 1)
        dt = ts[1] - ts[0]
        edt = np.exp(-self.lbd * dt)

        for i in range(n_disc):
            # conditional mean and variance
            m = (v_qe[i, :] - self.vbar) * edt + self.vbar
            m = np.maximum(m, eps)  # ensure positivity
            s2 = (self.nu**2 / self.lbd) * (
                edt * (1 - edt) * (v_qe[i, :] - self.vbar)
                + self.vbar / 2 * (1 - edt**2)
            )
            # compute relative variance
            psi = s2 / m**2

            # Regime 1: Quadratic form
            mask_quad = psi <= psi_c
            if np.any(mask_quad):
                psi_quad = psi[mask_quad]
                Z_quad = np.random.normal(size=mask_quad.sum())
                b2 = (2.0 + 2.0 * np.sqrt(1.0 - psi_quad / 2.0) - psi_quad) / psi_quad
                a = m[mask_quad] / (1.0 + b2)
                v_qe[i + 1, mask_quad] = a * (b2**0.5 + Z_quad) ** 2

            # Regime 2: Exponential form
            mask_exp = ~mask_quad
            if np.any(mask_exp):
                psi_exp = psi[mask_exp]
                # clip p to [0, 1)
                p = np.clip((psi_exp - 1.0) / (psi_exp + 1.0), 0.0, 1.0 - 1e-15)
                beta = m[mask_exp] * (psi_exp + 1.0) / 2.0

                U_exp = np.random.uniform(0.0, 1.0, size=mask_exp.sum())
                alive = U_exp > p  # if False -> atom at zero

                # start at zero
                v_qe[i + 1, mask_exp] = 0.0
                if np.any(alive):
                    U_exp_new = np.random.uniform(0.0, 1.0, size=alive.sum())
                    U_exp_new = np.clip(U_exp_new, eps, 1.0)
                    idx_exp = np.where(mask_exp)[0]
                    alive_idx = idx_exp[alive]
                    v_qe[i + 1, alive_idx] = -beta[alive] * np.log(U_exp_new)

            v_qe[i + 1, :] = np.maximum(v_qe[i + 1, :], eps)  # ensure positivity

            # trapezoidal rule for integrated variance
            int_v_trap_i = 0.5 * (v_qe[i, :] + v_qe[i + 1, :]) * dt
            int_v_trap_i = np.maximum(int_v_trap_i, 0.0)
            int_v_dt[i, :] = int_v_trap_i

            logS_qe[i + 1, :] = (
                logS_qe[i, :]
                - 0.5 * int_v_trap_i
                + self.rho
                * (
                    v_qe[i + 1, :]
                    - v_qe[i, :]
                    - self.lbd * self.vbar * dt
                    + self.lbd * int_v_trap_i
                )
                / self.nu
                + np.sqrt(np.maximum((1.0 - self.rho**2) * int_v_trap_i, 0.0))
                * np.random.normal(size=n_paths)
            )

        return np.exp(logS_qe), v_qe, int_v_dt


def get_params_heston(id: int) -> dict:
    """Return one of the predefined Heston parameter sets."""
    if id == 1:
        # Bergomi (2016), Stochastic Volatility Modeling, Chapter 6, page 211
        # Initial variance v in [0.01, 0.04, 0.16]
        return {"v": 0.16, "lbd": 1.0, "vbar": 0.04, "nu": 0.6, "rho": -0.8}
    elif id == 2:
        # Bourgey et al. (2025), "Smile Dynamics and Rough Volatility"
        # Calibration on 5 August 2024,
        return {"v": 0.117, "lbd": 3.37, "vbar": 0.048, "nu": 1.99, "rho": -0.68}
    else:
        raise ValueError("Invalid id. Please choose a valid id.")
