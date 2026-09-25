import numpy as np
import pytest

from magic_strikes.heston import HestonModel, get_params_heston


@pytest.mark.parametrize(
    "Ts,v,lbd,vbar,nu,rho",
    [
        (np.linspace(0.1, 1.0, 20), 0.04, 1.0, 0.04, 0.6, -0.8),
        (np.array([0.1, 0.5, 1.0]), 0.06, 0.0, 0.04, 0.5, -0.2),
    ],
)
def test_heston_varswap(Ts, v, lbd, vbar, nu, rho):
    params = {"v": v, "lbd": lbd, "vbar": vbar, "nu": nu, "rho": rho}
    heston = HestonModel(params=params)
    varswaps = heston.var_swap(T=Ts)
    if lbd == 0.0:
        assert np.allclose(varswaps, v * Ts)
    else:
        varswaps_quad = heston.var_swap_fukasawa(T=Ts, n_quad=5, std=10)
        assert np.allclose(varswaps, varswaps_quad, atol=1e-3)


def test_heston_gammaswap_matches_fukasawa_quadrature():
    params = {"v": 0.04, "lbd": 1.0, "vbar": 0.04, "nu": 0.6, "rho": -0.8}
    heston = HestonModel(params=params)
    Ts = np.linspace(0.1, 1.0, 10)

    gamma_swaps = heston.gamma_swap(T=Ts)
    gamma_swaps_quad = heston.gamma_swap_fukasawa(T=Ts, n_quad=7, std=10)

    assert np.allclose(gamma_swaps, gamma_swaps_quad, atol=1e-3)


def test_heston_gammaswap_zero_share_measure_reversion():
    # lbd - rho * nu = 0: under the share measure v_t = v + lbd * vbar * t.
    params = {"v": 0.04, "lbd": 0.3, "vbar": 0.05, "nu": 0.6, "rho": 0.5}
    heston = HestonModel(params=params)
    Ts = np.array([0.1, 0.5, 1.0])

    gamma_swaps = heston.gamma_swap(T=Ts)

    assert np.allclose(gamma_swaps, 0.04 * Ts + 0.5 * 0.3 * 0.05 * Ts**2)
    # Continuity with a nearby nonzero share-measure reversion.
    near = HestonModel(params={**params, "rho": 0.5 - 1e-4}).gamma_swap(T=Ts)
    assert np.allclose(gamma_swaps, near, rtol=1e-4)


def test_heston_fukasawa_methods_run_with_default_grid():
    params = {"v": 0.04, "lbd": 1.0, "vbar": 0.04, "nu": 0.6, "rho": -0.8}
    heston = HestonModel(params=params)
    T = np.array([0.5])

    gamma_swaps = heston.gamma_swap_fukasawa(T=T)
    swaps = heston.swap_fukasawa(T=T, opt="gamma")

    assert np.allclose(gamma_swaps, heston.gamma_swap(T=T), atol=1e-3)
    assert np.allclose(swaps, gamma_swaps)


def test_heston_implied_power_variance_reconstructs_power_moment():
    params = {"v": 0.117, "lbd": 3.37, "vbar": 0.048, "nu": 1.99, "rho": -0.68}
    heston = HestonModel(params=params)
    Ts = np.array([0.0, 0.1, 0.5, 1.0])
    p = 0.3

    observed = heston.implied_power_variance(T=Ts, p=p)
    lambda_p = 0.5 * p * (p - 1.0)
    reconstructed = np.exp(lambda_p * observed)
    expected = heston.charfunc(u=-1j * p, T=Ts).real

    assert observed.shape == Ts.shape
    assert np.all(np.isfinite(observed))
    assert np.all(observed >= 0.0)
    assert np.allclose(reconstructed, expected, rtol=1e-12, atol=1e-12)


def test_heston_implied_power_variance_endpoints_match_variance_and_gamma():
    params = {"v": 0.04, "lbd": 1.0, "vbar": 0.04, "nu": 0.6, "rho": -0.8}
    heston = HestonModel(params=params)
    Ts = np.array([0.0, 0.1, 0.5, 1.0])

    assert np.allclose(heston.implied_power_variance(Ts, p=0.0), heston.var_swap(Ts))
    assert np.allclose(heston.implied_power_variance(Ts, p=1.0), heston.gamma_swap(Ts))


@pytest.mark.parametrize("p", [None, -0.1, 1.1, np.nan, [0.5]])
def test_heston_implied_power_variance_rejects_invalid_power(p):
    heston = HestonModel(get_params_heston(1))
    match = "scalar" if isinstance(p, list) else "finite and between"

    with pytest.raises(ValueError, match=match):
        heston.implied_power_variance(T=0.5, p=p)


@pytest.mark.parametrize("T", [-0.1, np.nan, np.inf])
def test_heston_implied_power_variance_rejects_invalid_maturity(T):
    heston = HestonModel(get_params_heston(1))

    with pytest.raises(ValueError, match="finite and nonnegative"):
        heston.implied_power_variance(T=T, p=0.3)


def test_heston_varswap_fukasawa_rejects_nonpositive_maturities():
    params = {"v": 0.04, "lbd": 1.0, "vbar": 0.04, "nu": 0.6, "rho": -0.8}
    heston = HestonModel(params=params)

    with pytest.raises(ValueError, match="T must be finite and positive"):
        heston.var_swap_fukasawa(T=np.array([0.0, 0.5]))


def test_heston_fukasawa_rejects_both_ks_interp_and_std():
    params = {"v": 0.04, "lbd": 1.0, "vbar": 0.04, "nu": 0.6, "rho": -0.8}
    heston = HestonModel(params=params)
    ks_interp = np.array([-0.2, 0.0, 0.2])

    with pytest.raises(
        ValueError,
        match="Cannot specify both ks_interp and std; they both define "
        "the interpolation grid.",
    ):
        heston.var_swap_fukasawa(T=0.5, ks_interp=ks_interp)


def test_heston_impvol_is_finite_on_representative_slice():
    params = {"v": 0.04, "lbd": 1.0, "vbar": 0.04, "nu": 0.6, "rho": -0.8}
    heston = HestonModel(params=params)

    impvol = heston.impvol(k=np.array([-0.2, 0.0, 0.2]), T=1.0)

    assert impvol.shape == (3,)
    assert np.all(np.isfinite(impvol))


def test_heston_magic_strike_outputs_are_finite_and_ordered_by_request():
    params = {"v": 0.04, "lbd": 1.0, "vbar": 0.04, "nu": 0.6, "rho": -0.8}
    heston = HestonModel(params=params)

    result = heston.swap_magic_strike(
        T=np.array([0.5, 1.0]),
        order=[1, 2, 3, 4],
        opt="variance",
        std=8,
        n_interp=801,
    )

    assert list(result) == [1, 2, 3, 4]
    for values in result.values():
        assert values.shape == (2,)
        assert np.all(np.isfinite(values))


def test_heston_magic_strike_variance_is_accurate_for_short_maturities():
    params = {"v": 0.04, "lbd": 1.0, "vbar": 0.04, "nu": 0.6, "rho": -0.8}
    heston = HestonModel(params=params)
    Ts = np.array([1e-4, 1e-3, 1e-2])

    result = heston.swap_magic_strike(
        T=Ts,
        order=[2, 3, 4],
        opt="variance",
        std=8,
        n_interp=2001,
    )
    reference = heston.var_swap(Ts)

    for order in (2, 3, 4):
        rel_err = np.abs(result[order] - reference) / reference
        assert np.all(rel_err < 1e-2)


def test_heston_magic_strike_gamma_is_accurate_for_short_maturities():
    params = {"v": 0.04, "lbd": 1.0, "vbar": 0.04, "nu": 0.6, "rho": -0.8}
    heston = HestonModel(params=params)
    Ts = np.array([1e-4, 1e-3, 1e-2])

    result = heston.swap_magic_strike(
        T=Ts,
        order=[2, 3],
        opt="gamma",
        std=8,
        n_interp=2001,
    )
    reference = heston.gamma_swap(Ts)

    for order in (2, 3):
        rel_err = np.abs(result[order] - reference) / reference
        assert np.all(rel_err < 5e-3)


def test_get_params_heston_returns_model_ready_keys_for_first_three_sets():
    for set_id in (1, 2):
        params = get_params_heston(set_id)
        assert set(params) == {"v", "lbd", "vbar", "nu", "rho"}
