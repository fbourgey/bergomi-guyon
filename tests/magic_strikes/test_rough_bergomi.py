import numpy as np
import pytest

from magic_strikes.rough_bergomi import RoughBergomiModel


def _flat_xi0(t):
    return 0.04 * np.ones_like(t)


def test_rough_bergomi_simulation_is_seed_reproducible():
    model = RoughBergomiModel(
        params={"eta": 1.0, "H": 0.3, "rho": -0.5},
        xi0=_flat_xi0,
    )
    tab_t = np.linspace(0.0, 0.5, 6)

    out_1 = model.simulate_mc(tab_t=tab_t, n_mc=8, n_loop=2, seed=123)
    out_2 = model.simulate_mc(tab_t=tab_t, n_mc=8, n_loop=2, seed=123)

    assert np.allclose(out_1["int_v_dt"], out_2["int_v_dt"])
    assert np.allclose(out_1["int_sqrt_v_dw"], out_2["int_sqrt_v_dw"])


def test_rough_bergomi_antithetic_simulation_is_seed_reproducible_and_shaped():
    model = RoughBergomiModel(
        params={"eta": 1.0, "H": 0.3, "rho": -0.5},
        xi0=_flat_xi0,
    )
    tab_t = np.linspace(0.0, 0.5, 6)

    out_1 = model.simulate_mc(tab_t=tab_t, n_mc=8, n_loop=2, seed=123, antithetic=True)
    out_2 = model.simulate_mc(tab_t=tab_t, n_mc=8, n_loop=2, seed=123, antithetic=True)

    assert out_1["int_v_dt"].shape == (8,)
    assert out_1["int_sqrt_v_dw"].shape == (8,)
    assert np.allclose(out_1["int_v_dt"], out_2["int_v_dt"])
    assert np.allclose(out_1["int_sqrt_v_dw"], out_2["int_sqrt_v_dw"])


def test_rough_bergomi_antithetic_requires_even_paths_per_loop():
    model = RoughBergomiModel(
        params={"eta": 1.0, "H": 0.3, "rho": -0.5},
        xi0=_flat_xi0,
    )

    with pytest.raises(ValueError, match="n_mc // n_loop must be even"):
        model.simulate_mc(
            tab_t=np.linspace(0.0, 0.5, 6),
            n_mc=6,
            n_loop=2,
            seed=123,
            antithetic=True,
        )


def test_rough_bergomi_impvol_forwards_antithetic_config(monkeypatch):
    model = RoughBergomiModel(
        params={"eta": 1.0, "H": 0.3, "rho": -0.5},
        xi0=_flat_xi0,
    )
    seen = []

    def fake_impvol_mc(**kwargs):
        seen.append(kwargs)
        return 0.2

    monkeypatch.setattr(model, "impvol_mc", fake_impvol_mc)

    model.impvol(k=0.0, T=1.0, n_mc=8, n_disc=4, antithetic=True)

    assert seen[0]["antithetic"] is True


def test_rough_bergomi_impvol_rejects_conditioning_config_value():
    model = RoughBergomiModel(
        params={"eta": 1.0, "H": 0.3, "rho": -0.5},
        xi0=_flat_xi0,
    )

    with pytest.raises(TypeError, match="Unexpected Monte Carlo config values"):
        model.impvol(k=0.0, T=1.0, n_mc=8, n_disc=4, conditioning=True)


def test_rough_bergomi_simulation_uses_trapezoidal_variance_integral():
    eta = 0.7
    model = RoughBergomiModel(
        params={"eta": eta, "H": 0.5, "rho": 1.0},
        xi0=_flat_xi0,
    )
    tab_t = np.linspace(0.0, 0.5, 6)
    n_mc = 4
    seed = 123

    out = model.simulate_mc(tab_t=tab_t, n_mc=n_mc, n_loop=1, seed=seed)

    n_disc = tab_t.shape[0] - 1
    dt = tab_t[1] - tab_t[0]
    dy = np.random.default_rng(seed).normal(0.0, np.sqrt(dt), (n_disc, n_mc))
    y = np.empty((n_disc + 1, n_mc))
    y[0, :] = 0.0
    np.cumsum(dy, axis=0, out=y[1:, :])
    y *= eta
    y += -0.5 * eta**2 * tab_t[:, np.newaxis]
    v_prev = _flat_xi0(tab_t[:-1, np.newaxis]) * np.exp(y[:-1, :])
    v_next = _flat_xi0(tab_t[1:, np.newaxis]) * np.exp(y[1:, :])

    assert np.allclose(out["int_v_dt"], 0.5 * dt * np.sum(v_prev + v_next, axis=0))


def test_rough_bergomi_reuses_unit_cholesky_for_uniform_grids(monkeypatch):
    model = RoughBergomiModel(
        params={"eta": 1.0, "H": 0.3, "rho": -0.5},
        xi0=_flat_xi0,
    )
    calls = 0
    original = model.cholesky_cov_matrix

    def wrapped_cholesky_cov_matrix(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "cholesky_cov_matrix", wrapped_cholesky_cov_matrix)

    model.simulate_mc(tab_t=np.linspace(0.0, 0.5, 6), n_mc=8, n_loop=2, seed=1)
    model.simulate_mc(tab_t=np.linspace(0.0, 1.2, 6), n_mc=8, n_loop=2, seed=2)

    assert calls == 1


def test_rough_bergomi_scaled_unit_cholesky_matches_full_covariance():
    model = RoughBergomiModel(
        params={"eta": 1.0, "H": 0.3, "rho": -0.5},
        xi0=_flat_xi0,
    )
    tab_t = np.linspace(0.0, 0.7, 6)

    chol = model._cached_cholesky_cov_matrix(tab_t, conditioning=True)
    cov = model.cholesky_cov_matrix(tab_t, conditioning=True, return_cov=True)

    assert np.allclose(chol @ chol.T, cov)


def test_rough_bergomi_swap_mc_estimators_returns_finite_smoke():
    model = RoughBergomiModel(
        params={"eta": 0.3, "H": 0.5, "rho": -0.2},
        xi0=_flat_xi0,
    )

    out = model.swap_mc_estimators(
        T=0.5,
        n_mc=2000,
        n_disc=4,
        n_loop=2,
        seed=1,
        magic_order=1,
        magic_std=2.0,
        magic_n_interp=41,
        magic_extrapolation="flat",
        fukasawa_n_quad=1,
        fukasawa_std=2.0,
        fukasawa_n_interp=41,
    )

    assert set(out) == {"mc", "magic_strikes", "fukasawa"}
    for value in out["mc"].values():
        assert np.all(np.isfinite(value))
    for contract in out["magic_strikes"].values():
        for value in contract.values():
            assert np.all(np.isfinite(value))
    for value in out["fukasawa"].values():
        assert np.all(np.isfinite(value))
