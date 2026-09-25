import warnings

import numpy as np
import pandas as pd
import pytest

from magic_strikes import magic_strike, swap


def finite_difference_values(f, *, center, spacing, max_order=5):
    return magic_strike._finite_differences(
        f,
        center=center,
        spacing=spacing,
        max_order=max_order,
    )


def flat_tot_impvar(k, T, level=0.04):
    return level * np.ones_like(np.atleast_1d(k), dtype=float)


def test_black_d_matches_closed_form_for_flat_smile():
    k = np.array([-0.1, 0.0, 0.2])
    T = 1.0
    total_variance = 0.04
    total_vol = total_variance**0.5

    d_plus = swap.black_d(k, T, lambda k, T: flat_tot_impvar(k, T, total_variance))
    d_minus = swap.black_d(
        k, T, lambda k, T: flat_tot_impvar(k, T, total_variance), opt="minus"
    )

    assert np.allclose(d_plus, -k / total_vol + 0.5 * total_vol)
    assert np.allclose(d_minus, -k / total_vol - 0.5 * total_vol)


def test_func_g_inverts_black_d_for_flat_smile():
    total_variance = 0.09
    g_minus = swap.func_g(
        T=1.0,
        tot_impvar=lambda k, T: flat_tot_impvar(k, T, total_variance),
        opt="minus",
        std=3,
        n_interp=401,
    )
    ks = np.linspace(-0.4, 0.4, 11)
    d_vals = swap.black_d(
        ks, 1.0, lambda k, T: flat_tot_impvar(k, T, total_variance), opt="minus"
    )

    assert np.allclose(g_minus(d_vals), ks, atol=1e-6)


def test_func_g_raises_outside_fitted_d_domain():
    total_variance = 0.09
    g_minus = swap.func_g(
        T=1.0,
        tot_impvar=lambda k, T: flat_tot_impvar(k, T, total_variance),
        opt="minus",
        std=3,
        n_interp=401,
    )
    d_min = swap.black_d(
        np.array([0.9]),
        1.0,
        lambda k, T: flat_tot_impvar(k, T, total_variance),
        opt="minus",
    )[0]

    with pytest.raises(ValueError, match="Interpolation input must be in"):
        g_minus(np.array([d_min - 1e-6]))


def test_swap_fukasawa_returns_constant_total_variance_for_flat_smile():
    total_variance = 0.04
    variance_swap = swap.swap_fukasawa(
        T=np.array([0.5, 1.0]),
        tot_impvar=lambda k, T: flat_tot_impvar(k, T, total_variance),
        n_quad=5,
        opt="variance",
    )
    gamma_swap = swap.swap_fukasawa(
        T=np.array([0.5, 1.0]),
        tot_impvar=lambda k, T: flat_tot_impvar(k, T, total_variance),
        n_quad=5,
        opt="gamma",
    )

    assert np.allclose(variance_swap, total_variance)
    assert np.allclose(gamma_swap, total_variance)


def test_swap_fukasawa_verbose_prints_current_maturity(capsys):
    total_variance = 0.04

    result = swap.swap_fukasawa(
        T=np.array([0.5, 1.0]),
        tot_impvar=lambda k, T: flat_tot_impvar(k, T, total_variance),
        n_quad=5,
        opt="all",
        verbose=True,
    )

    assert set(result) == {"variance", "gamma"}
    captured = capsys.readouterr().out
    assert "Computing Fukasawa all swap for T=0.5000... (1/2)" in captured
    assert "Computing Fukasawa all swap for T=1.0000... (2/2)" in captured


def test_swap_magic_strike_returns_constant_total_variance_for_flat_smile():
    total_variance = 0.04

    result = swap.swap_magic_strike(
        T=np.array([0.5, 1.0]),
        tot_impvar=lambda k, T: flat_tot_impvar(k, T, total_variance),
        order=[1, 2, 3, 4, 5],
        opt="variance",
        n_interp=401,
    )

    for values in result.values():
        assert np.allclose(values, total_variance)


def test_swap_magic_strike_power_returns_implied_power_variance():
    result = swap.swap_magic_strike(
        T=np.array([0.5, 1.0]),
        tot_impvar=lambda k, T: flat_tot_impvar(k, T, 0.04),
        order=[1, 5],
        opt="power",
        p=0.3,
        n_interp=401,
    )

    assert set(result) == {1, 5}
    assert np.allclose(result[1], 0.04)
    assert np.allclose(result[5], 0.04)


def test_swap_magic_strike_all_rejects_power_parameter():
    with pytest.raises(ValueError, match="only be specified"):
        swap.swap_magic_strike(
            T=0.5,
            tot_impvar=lambda k, T: flat_tot_impvar(k, T, 0.04),
            order=1,
            opt="all",
            p=0.3,
        )


def test_swap_magic_strike_forwards_relax_to_fixed_point_layer(monkeypatch):
    calls = []

    def fake_swap_many(
        tot_impvar, orders_by_opt, p=None, tol=1e-8, max_iter=100, relax=0.8
    ):
        calls.append(relax)
        return {"variance": {1: float(tot_impvar(0.0))}}

    monkeypatch.setattr(magic_strike, "swap_many", fake_swap_many)

    result = swap.swap_magic_strike(
        T=0.5,
        tot_impvar=lambda k, T: flat_tot_impvar(k, T, 0.04),
        order=1,
        opt="variance",
        relax=0.35,
        n_interp=101,
    )

    assert calls == [0.35]
    assert result[1] == pytest.approx([0.04])


def test_swap_magic_strike_all_builds_smile_once_per_maturity():
    calls = 0

    def counted_tot_impvar(k, T):
        nonlocal calls
        calls += 1
        return flat_tot_impvar(k, T, 0.04)

    result = swap.swap_magic_strike(
        T=1.0,
        tot_impvar=counted_tot_impvar,
        order=[1, 2],
        opt="all",
        n_interp=401,
    )

    assert set(result) == {"variance", "gamma"}
    assert calls == 2


def test_swap_magic_strike_raises_when_grid_is_too_narrow():
    ks_interp = np.array([-0.2, 0.2])
    total_variance = np.array([0.04, 0.04])

    with pytest.raises(
        ValueError, match="Need at least three finite interpolation points"
    ):
        swap.swap_magic_strike(
            T=1.0,
            tot_impvar=lambda k, T: np.interp(k, ks_interp, total_variance),
            order=[1, 2],
            opt="variance",
            ks_interp=ks_interp,
            std=None,
        )


def test_swap_magic_strike_raises_when_sanitized_grid_is_too_narrow():
    ks_interp = np.array([-0.2, -0.2, 0.0, 0.2, 0.3])
    smile_values = np.array([0.04, 0.04, np.nan, 0.05, np.inf])

    def tot_impvar(k, T):
        return np.interp(
            k, ks_interp, np.nan_to_num(smile_values, nan=0.04, posinf=0.05)
        )

    with pytest.raises(ValueError, match="Interpolation input must be in"):
        swap.swap_magic_strike(
            T=1.0,
            tot_impvar=tot_impvar,
            order=[2],
            opt="variance",
            ks_interp=ks_interp,
            std=None,
        )


def test_swap_magic_strike_flat_extrapolation_handles_narrow_grid():
    ks_interp = np.array([-0.05, 0.0, 0.05])
    total_variance = 0.04 * np.ones_like(ks_interp)

    result = swap.swap_magic_strike(
        T=1.0,
        tot_impvar=lambda k, T: np.interp(k, ks_interp, total_variance),
        order=[3],
        opt="variance",
        ks_interp=ks_interp,
        std=None,
        extrapolation="flat",
    )

    assert np.allclose(result[3], [0.04])


def test_build_market_tot_impvar_rejects_fewer_than_three_valid_points():
    with pytest.raises(
        ValueError, match="Need at least three finite interpolation points"
    ):
        swap._build_market_tot_impvar(
            log_moneyness=np.array([-0.2, 0.2]),
            vol=np.array([0.2, 0.2]),
            T=1.0,
        )


def test_swap_magic_strike_warns_and_returns_finite_value_on_nonconvergence():
    with pytest.warns(RuntimeWarning, match="reached max_iter"):
        result = swap.swap_magic_strike(
            T=1.0,
            tot_impvar=lambda k, T: flat_tot_impvar(k, T, 0.04),
            order=[1],
            opt="variance",
            max_iter=0,
        )

    assert np.all(np.isfinite(result[1]))


def test_relaxed_fixed_point_warns_when_max_iter_is_reached():
    with pytest.warns(RuntimeWarning, match="reached max_iter"):
        result = magic_strike._relaxed_fixed_point(
            lambda x: x + 1.0,
            x0=1.0,
            tol=1e-12,
            max_iter=2,
            relax=0.5,
        )

    assert np.isfinite(result)


def test_variance_swap_rejects_unsupported_orders():
    with pytest.raises(ValueError, match="order must be either 1, 2, 3, 4, or 5"):
        magic_strike._variance_swap(lambda k: 0.04, order=0)

    with pytest.raises(ValueError, match="order must be either 1, 2, 3, 4, or 5"):
        magic_strike._variance_swap(lambda k: 0.04, order=6)


def test_variance_swap_uses_paper_stencil():
    sampled_strikes = []

    def flat_smile(k):
        sampled_strikes.append(k)
        return 0.04

    result = magic_strike._variance_swap(flat_smile, order=3)

    assert np.allclose(result, 0.04)
    assert set(np.round(sampled_strikes, 12)) == {-0.22, -0.02, 0.18, 0.0}


def test_finite_differences_reject_invalid_inputs():
    with pytest.raises(ValueError, match="spacing must be finite and positive"):
        finite_difference_values(
            lambda k: 0.04,
            center=0.0,
            spacing=0.0,
            max_order=1,
        )


def test_finite_differences_return_paper_normalized_values():
    def cubic_smile(k):
        return 1.0 + 2.0 * k + 3.0 * k**2 + 4.0 * k**3

    spacing = 0.2
    d0, d1, d2, d3, d4, _ = finite_difference_values(
        cubic_smile,
        center=0.0,
        spacing=spacing,
        max_order=4,
    )

    assert d0 == pytest.approx(1.0)
    assert d1 == pytest.approx(2.0 + 4.0 * spacing**2)
    assert d2 == pytest.approx(3.0)
    assert d3 == pytest.approx(4.0)
    assert d4 == pytest.approx(0.0)


def test_gamma_swap_flat_smile_returns_total_variance_for_all_orders():
    for order in [1, 2, 3, 4, 5]:
        assert magic_strike._gamma_swap(lambda k: 0.04, order=order) == pytest.approx(
            0.04
        )


def test_magic_strike_grid_uses_the_contract_center():
    variance = magic_strike._magic_strike_grid(0.04, order=3, opt="variance")
    gamma = magic_strike._magic_strike_grid(0.04, order=3, opt="gamma")

    assert np.allclose(variance, [-0.22, -0.02, 0.18])
    assert np.allclose(gamma, [-0.18, 0.02, 0.22])


def test_power_swap_endpoints_match_variance_and_gamma():
    def smooth_smile(k):
        return 0.04 + 0.002 * k + 0.0015 * k**2 + 0.0003 * k**3 + 0.00005 * k**4

    for order in [1, 2, 3, 4, 5]:
        variance = magic_strike._variance_swap(smooth_smile, order=order)
        gamma = magic_strike._gamma_swap(smooth_smile, order=order)
        power_zero = magic_strike._power_swap(smooth_smile, order=order, p=0.0)
        power_one = magic_strike._power_swap(smooth_smile, order=order, p=1.0)
        assert power_zero == pytest.approx(variance, abs=2e-13)
        assert power_one == pytest.approx(gamma, abs=2e-13)


def test_power_swap_flat_smile_returns_total_variance_for_all_orders_and_powers():
    for p in [0.0, 0.3, 0.5, 1.0]:
        for order in [1, 2, 3, 4, 5]:
            assert magic_strike._power_swap(
                lambda k: 0.04, order=order, p=p
            ) == pytest.approx(0.04)


@pytest.mark.parametrize("p", [None, -0.1, 1.1, np.nan, [0.5]])
def test_power_swap_rejects_invalid_power_parameter(p):
    match = "required" if p is None else "scalar|finite and between"
    with pytest.raises(ValueError, match=match):
        magic_strike.swap(lambda k: 0.04, order=1, opt="power", p=p)


def test_non_power_contract_rejects_power_parameter():
    with pytest.raises(ValueError, match="only be specified"):
        magic_strike.swap(lambda k: 0.04, order=1, opt="variance", p=0.5)


def test_swap_many_uses_power_order_continuation(monkeypatch):
    calls = []

    def fake_power_swap(f, order, p, tol=1e-8, max_iter=100, relax=0.8, x0=None):
        calls.append((order, p, x0))
        return 0.1 * order

    monkeypatch.setattr(magic_strike, "_power_swap", fake_power_swap)

    result = magic_strike.swap_many(
        tot_impvar=lambda k: 0.04,
        orders_by_opt={"power": [5]},
        p=0.3,
    )

    assert result["power"] == {5: pytest.approx(0.5)}
    assert calls == [
        (1, 0.3, None),
        (2, 0.3, pytest.approx(0.1)),
        (3, 0.3, pytest.approx(0.2)),
        (4, 0.3, pytest.approx(0.3)),
        (5, 0.3, pytest.approx(0.4)),
    ]


def test_gamma_swap_rejects_invalid_initial_guess():
    with pytest.raises(ValueError, match="initial guess must be finite and positive"):
        magic_strike._gamma_swap(lambda k: 0.04, order=1, x0=0.0)


def test_swap_many_uses_gamma_order_continuation(monkeypatch):
    calls = []

    def fake_gamma_swap(f, order, tol=1e-8, max_iter=100, relax=0.8, x0=None):
        calls.append((order, x0))
        return 0.1 * order

    monkeypatch.setattr(magic_strike, "_gamma_swap", fake_gamma_swap)

    result = magic_strike.swap_many(
        tot_impvar=lambda k: 0.04,
        orders_by_opt={"gamma": [5]},
    )

    assert result["gamma"] == {5: pytest.approx(0.5)}
    assert calls == [
        (1, None),
        (2, pytest.approx(0.1)),
        (3, pytest.approx(0.2)),
        (4, pytest.approx(0.3)),
        (5, pytest.approx(0.4)),
    ]


def test_swap_market_handles_duplicate_strikes_and_missing_quotes():
    ivol_data = pd.DataFrame(
        {
            "Texp": [0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
            "Bid": [0.16, 0.18, 0.20, np.nan, 0.22, 0.17, 0.19, 0.21, 0.23],
            "Ask": [0.18, 0.20, 0.22, np.nan, 0.24, 0.19, 0.21, 0.23, 0.25],
            "Fwd": [1.0, 1.0, 1.0, 1.0, 1.0, 1.05, 1.05, 1.05, 1.05],
            "Strike": [0.7, 0.9, 1.0, 1.0, 1.3, 0.6, 0.95, 1.05, 1.5],
        }
    )

    result = swap.swap_market(ivol_data, n_quad=2, extrapolation="flat")

    assert np.allclose(result["expiries"], [0.5, 1.0])
    for key in ("vs_mid", "vs_bid", "vs_ask"):
        assert result[key].shape == (2,)
        assert np.all(np.isfinite(result[key]))
    for key in ("vs_mid_magic", "vs_bid_magic", "vs_ask_magic"):
        assert result[key].shape == (5, 2)
        assert np.all(np.isfinite(result[key]))
    assert np.all(result["n_quad_used"] == 2)


def test_swap_market_gamma_computes_magic_orders_through_five(monkeypatch):
    ivol_data = pd.DataFrame(
        {
            "Texp": np.ones(5),
            "Bid": 0.19 * np.ones(5),
            "Ask": 0.21 * np.ones(5),
            "Fwd": np.ones(5),
            "Strike": np.exp(np.linspace(-1.0, 1.0, 5)),
        }
    )
    seen_orders = []

    def fake_swap_fukasawa(**kwargs):
        return np.array([0.04])

    def fake_swap_magic_strike(**kwargs):
        orders = tuple(kwargs["order"])
        seen_orders.append((kwargs["opt"], orders))
        return {order: np.array([0.04 + 0.001 * order]) for order in orders}

    monkeypatch.setattr(swap, "swap_fukasawa", fake_swap_fukasawa)
    monkeypatch.setattr(swap, "swap_magic_strike", fake_swap_magic_strike)

    result = swap.swap_market(ivol_data, opt="gamma", n_quad=3)

    assert result["vs_mid_magic"].shape == (5, 1)
    assert seen_orders == [("gamma", (1, 2, 3, 4, 5))] * 3
    assert np.allclose(
        result["vs_mid_magic"][:, 0],
        [0.041, 0.042, 0.043, 0.044, 0.045],
    )


def test_swap_market_forwards_relax_to_magic_strike(monkeypatch):
    ivol_data = pd.DataFrame(
        {
            "Texp": [0.5, 0.5, 0.5],
            "Bid": [0.18, 0.20, 0.22],
            "Ask": [0.20, 0.22, 0.24],
            "Fwd": [1.0, 1.0, 1.0],
            "Strike": [0.8, 1.0, 1.2],
        }
    )
    seen_relax = []

    def fake_swap_fukasawa(**kwargs):
        return np.array([0.04])

    def fake_swap_magic_strike(**kwargs):
        orders = tuple(kwargs["order"])
        seen_relax.append(kwargs["relax"])
        return {order: np.array([0.04]) for order in orders}

    monkeypatch.setattr(swap, "swap_fukasawa", fake_swap_fukasawa)
    monkeypatch.setattr(swap, "swap_magic_strike", fake_swap_magic_strike)

    swap.swap_market(ivol_data, n_quad=2, relax=0.35)

    assert seen_relax == [0.35, 0.35, 0.35]


def test_swap_market_auto_n_quad_returns_used_orders():
    ivol_data = pd.DataFrame(
        {
            "Texp": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
            "Bid": [0.16, 0.18, 0.22, 0.17, 0.19, 0.23],
            "Ask": [0.18, 0.20, 0.24, 0.19, 0.21, 0.25],
            "Fwd": [1.0, 1.0, 1.0, 1.05, 1.05, 1.05],
            "Strike": [0.7, 0.9, 1.3, 0.6, 0.95, 1.5],
        }
    )

    result = swap.swap_market(ivol_data, n_quad=None, n_quad_auto_max=5)

    assert result["n_quad_used"].shape == result["expiries"].shape
    assert np.all((1 <= result["n_quad_used"]) & (result["n_quad_used"] <= 5))
    assert np.all(np.isfinite(result["vs_mid"]))
    assert np.all(np.isfinite(result["vs_bid"]))
    assert np.all(np.isfinite(result["vs_ask"]))


def test_swap_market_auto_n_quad_selects_lower_in_bounds_order():
    ivol_data = pd.DataFrame(
        {
            "Texp": [1.0, 1.0, 1.0],
            "Bid": [0.20, 0.20, 0.20],
            "Ask": [0.22, 0.22, 0.22],
            "Fwd": [1.0, 1.0, 1.0],
            "Strike": np.exp([-0.05, 0.0, 0.05]),
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        explicit = swap.swap_market(ivol_data, n_quad=2)
        automatic = swap.swap_market(ivol_data, n_quad=None, n_quad_auto_max=2)

    assert np.isnan(explicit["vs_mid"][0])
    assert automatic["n_quad_used"][0] == 1
    assert np.isfinite(automatic["vs_mid"][0])


def test_swap_market_auto_n_quad_warns_when_no_order_is_in_bounds(monkeypatch):
    ivol_data = pd.DataFrame(
        {
            "Texp": [0.5, 0.5, 0.5],
            "Bid": [0.18, 0.20, 0.22],
            "Ask": [0.20, 0.22, 0.24],
            "Fwd": [1.0, 1.0, 1.0],
            "Strike": [0.8, 1.0, 1.2],
        }
    )

    def fake_largest_in_bounds_n_quad(**kwargs):
        return None

    def fake_swap_magic_strike(**kwargs):
        return {order: np.array([0.04]) for order in kwargs["order"]}

    monkeypatch.setattr(
        swap, "_largest_in_bounds_n_quad", fake_largest_in_bounds_n_quad
    )
    monkeypatch.setattr(swap, "swap_magic_strike", fake_swap_magic_strike)

    with pytest.warns(
        RuntimeWarning,
        match="No in-domain automatic Fukasawa quadrature order is available",
    ):
        result = swap.swap_market(ivol_data, n_quad=None, n_quad_auto_max=5)

    assert np.isnan(result["n_quad_used"][0])
    assert np.isnan(result["vs_mid"][0])
    assert np.isnan(result["vs_bid"][0])
    assert np.isnan(result["vs_ask"][0])


def test_swap_market_auto_n_quad_uses_shared_minimum_order(monkeypatch):
    ivol_data = pd.DataFrame(
        {
            "Texp": [0.5, 0.5, 0.5],
            "Bid": [0.18, 0.20, 0.22],
            "Ask": [0.20, 0.22, 0.24],
            "Fwd": [1.0, 1.0, 1.0],
            "Strike": [0.8, 1.0, 1.2],
        }
    )
    helper_orders = iter([5, 3, 4])
    seen_n_quad = []

    def fake_largest_in_bounds_n_quad(**kwargs):
        return next(helper_orders)

    def fake_swap_fukasawa(**kwargs):
        seen_n_quad.append(kwargs["n_quad"])
        return np.array([0.04])

    def fake_swap_magic_strike(**kwargs):
        return {order: np.array([0.04]) for order in kwargs["order"]}

    monkeypatch.setattr(
        swap, "_largest_in_bounds_n_quad", fake_largest_in_bounds_n_quad
    )
    monkeypatch.setattr(swap, "swap_fukasawa", fake_swap_fukasawa)
    monkeypatch.setattr(swap, "swap_magic_strike", fake_swap_magic_strike)

    result = swap.swap_market(ivol_data, n_quad=None, n_quad_auto_max=5)

    assert result["n_quad_used"][0] == 3
    assert seen_n_quad == [3, 3, 3]


def test_swap_market_forwards_magic_strike_parameters(monkeypatch):
    ivol_data = pd.DataFrame(
        {
            "Texp": [0.5, 0.5, 0.5],
            "Bid": [0.18, 0.20, 0.22],
            "Ask": [0.20, 0.22, 0.24],
            "Fwd": [1.0, 1.0, 1.0],
            "Strike": [0.8, 1.0, 1.2],
        }
    )
    seen = []

    def fake_swap_fukasawa(**kwargs):
        return np.array([0.04])

    def fake_swap_magic_strike(**kwargs):
        seen.append(
            (
                tuple(kwargs["order"]),
                kwargs["tol"],
                kwargs["max_iter"],
            )
        )
        return {order: np.array([0.04]) for order in kwargs["order"]}

    monkeypatch.setattr(swap, "swap_fukasawa", fake_swap_fukasawa)
    monkeypatch.setattr(swap, "swap_magic_strike", fake_swap_magic_strike)

    result = swap.swap_market(
        ivol_data,
        opt="gamma",
        n_quad=2,
        magic_orders=[1, 2],
        magic_tol=1e-6,
        magic_max_iter=17,
    )

    assert result["vs_mid_magic"].shape == (2, 1)
    # Gamma receives the controls for each quote side.
    assert seen == [((1, 2), 1e-6, 17)] * 3


def test_magic_strike_apis_reject_removed_stencil_argument():
    ivol_data = pd.DataFrame(
        {
            "Texp": [0.5, 0.5, 0.5],
            "Bid": [0.18, 0.20, 0.22],
            "Ask": [0.20, 0.22, 0.24],
            "Fwd": [1.0, 1.0, 1.0],
            "Strike": [0.8, 1.0, 1.2],
        }
    )

    with pytest.raises(TypeError, match="unexpected keyword argument 'z'"):
        swap.swap_market(ivol_data, n_quad=2, z=1.0)

    with pytest.raises(TypeError, match="unexpected keyword argument 'z'"):
        magic_strike.swap(lambda k: 0.04, order=1, z=1.0)


@pytest.mark.parametrize("opt", ["unsupported", "leverage"])
def test_swap_market_rejects_invalid_contract_opt(opt):
    ivol_data = pd.DataFrame(
        {
            "Texp": [0.5, 0.5, 0.5],
            "Bid": [0.18, 0.20, 0.22],
            "Ask": [0.20, 0.22, 0.24],
            "Fwd": [1.0, 1.0, 1.0],
            "Strike": [0.8, 1.0, 1.2],
        }
    )

    with pytest.raises(ValueError, match="opt must be either"):
        swap.swap_market(ivol_data, opt=opt)


def test_swap_market_rejects_invalid_magic_strike_parameters():
    ivol_data = pd.DataFrame(
        {
            "Texp": [0.5, 0.5, 0.5],
            "Bid": [0.18, 0.20, 0.22],
            "Ask": [0.20, 0.22, 0.24],
            "Fwd": [1.0, 1.0, 1.0],
            "Strike": [0.8, 1.0, 1.2],
        }
    )

    with pytest.raises(ValueError, match="unsupported orders"):
        swap.swap_market(ivol_data, opt="gamma", magic_orders=[1, 6])

    with pytest.raises(ValueError, match="magic_tol must be finite and positive"):
        swap.swap_market(ivol_data, magic_tol=0.0)

    with pytest.raises(ValueError, match="magic_max_iter must be positive"):
        swap.swap_market(ivol_data, magic_max_iter=0)


def test_swap_market_scalar_slice_selects_single_expiry():
    ivol_data = pd.DataFrame(
        {
            "Texp": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
            "Bid": [0.16, 0.18, 0.22, 0.17, 0.19, 0.23],
            "Ask": [0.18, 0.20, 0.24, 0.19, 0.21, 0.25],
            "Fwd": [1.0, 1.0, 1.0, 1.05, 1.05, 1.05],
            "Strike": [0.7, 0.9, 1.3, 0.6, 0.95, 1.5],
        }
    )

    result = swap.swap_market(ivol_data, slices=0, n_quad=2)

    assert np.allclose(result["expiries"], [0.5])
    assert result["vs_mid"].shape == (1,)
    assert result["vs_mid_magic"].shape == (5, 1)


def test_build_market_tot_impvar_raises_outside_strike_domain():
    tot_impvar, _ = swap._build_market_tot_impvar(
        log_moneyness=np.array([-0.2, 0.0, 0.2]),
        vol=np.array([0.2, 0.2, 0.2]),
        T=1.0,
    )

    with pytest.raises(ValueError, match="Interpolation input must be in"):
        tot_impvar(np.array([0.3]), 1.0)


def test_build_market_tot_impvar_flat_extrapolates_to_endpoint_values():
    tot_impvar, _ = swap._build_market_tot_impvar(
        log_moneyness=np.array([-0.2, 0.0, 0.2]),
        vol=np.array([0.1, 0.2, 0.3]),
        T=2.0,
        extrapolation="flat",
    )

    left, center, right = tot_impvar(np.array([-0.4, 0.0, 0.4]), 2.0)

    assert np.isclose(left, 0.1**2 * 2.0)
    assert np.isclose(center, 0.2**2 * 2.0)
    assert np.isclose(right, 0.3**2 * 2.0)


def test_market_smile_helpers_reject_invalid_extrapolation():
    ivol_data = pd.DataFrame(
        {
            "Texp": [0.5, 0.5, 0.5],
            "Bid": [0.18, 0.20, 0.22],
            "Ask": [0.20, 0.22, 0.24],
            "Fwd": [1.0, 1.0, 1.0],
            "Strike": [0.9, 1.0, 1.1],
        }
    )

    with pytest.raises(ValueError, match="extrapolation must be either"):
        swap._build_market_tot_impvar(
            log_moneyness=np.array([-0.2, 0.0, 0.2]),
            vol=np.array([0.1, 0.2, 0.3]),
            T=1.0,
            extrapolation="linear",
        )

    with pytest.raises(ValueError, match="extrapolation must be either"):
        swap.swap_magic_strike(
            T=1.0,
            tot_impvar=lambda k, T: flat_tot_impvar(k, T, 0.04),
            order=[1],
            extrapolation="linear",
        )

    with pytest.raises(ValueError, match="extrapolation must be either"):
        swap.swap_market(ivol_data, extrapolation="linear")


def test_swap_market_drops_rows_with_missing_expiry_instead_of_crashing():
    ivol_data = pd.DataFrame(
        {
            "Texp": [0.5, 0.5, 0.5, np.nan, 1.0, 1.0, 1.0],
            "Bid": [0.18, 0.20, 0.22, 0.21, 0.17, 0.19, 0.23],
            "Ask": [0.20, 0.22, 0.24, 0.23, 0.19, 0.21, 0.25],
            "Fwd": [1.0, 1.0, 1.0, 1.0, 1.05, 1.05, 1.05],
            "Strike": [0.7, 1.0, 1.3, 1.0, 0.6, 0.95, 1.5],
        }
    )

    result = swap.swap_market(ivol_data, n_quad=2)

    assert np.allclose(result["expiries"], [0.5, 1.0])
    assert np.all(np.isfinite(result["vs_mid"]))


def test_swap_market_rejects_inconsistent_forward_within_expiry():
    ivol_data = pd.DataFrame(
        {
            "Texp": [0.5, 0.5, 0.5],
            "Bid": [0.18, 0.20, 0.22],
            "Ask": [0.20, 0.22, 0.24],
            "Fwd": [1.0, 1.01, 1.0],
            "Strike": [0.9, 1.0, 1.1],
        }
    )

    with pytest.raises(
        ValueError, match="Forward prices must be identical within each expiry."
    ):
        swap.swap_market(ivol_data, n_quad=2)


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("Bid", 0.0, "Bid implied volatilities must be finite and positive."),
        ("Bid", -0.1, "Bid implied volatilities must be finite and positive."),
        ("Bid", np.inf, "Bid implied volatilities must be finite and positive."),
        ("Ask", 0.0, "Ask implied volatilities must be finite and positive."),
        ("Ask", -0.1, "Ask implied volatilities must be finite and positive."),
        ("Ask", np.inf, "Ask implied volatilities must be finite and positive."),
    ],
)
def test_swap_market_rejects_invalid_bid_ask_vols(column, bad_value, message):
    ivol_data = pd.DataFrame(
        {
            "Texp": [0.5, 0.5, 0.5],
            "Bid": [0.18, 0.20, 0.22],
            "Ask": [0.20, 0.22, 0.24],
            "Fwd": [1.0, 1.0, 1.0],
            "Strike": [0.9, 1.0, 1.1],
        }
    )
    ivol_data.loc[1, column] = bad_value

    with pytest.raises(ValueError, match=message):
        swap.swap_market(ivol_data, n_quad=2)


def test_swap_market_keeps_only_rows_with_ask_above_bid():
    ivol_data = pd.DataFrame(
        {
            "Texp": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "Bid": [0.16, 0.18, 0.20, 0.21, 0.22, 0.24],
            "Ask": [0.18, 0.20, 0.19, 0.21, 0.24, 0.26],
            "Fwd": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "Strike": [0.7, 0.9, 1.0, 1.05, 1.1, 1.3],
        }
    )

    result = swap.swap_market(ivol_data, n_quad=2, extrapolation="flat")

    assert np.allclose(result["expiries"], [0.5])
    assert np.all(np.isfinite(result["vs_mid"]))
    assert np.all(np.isfinite(result["vs_bid"]))
    assert np.all(np.isfinite(result["vs_ask"]))
    assert np.all(np.isfinite(result["vs_mid_magic"]))
    assert np.all(np.isfinite(result["vs_bid_magic"]))
    assert np.all(np.isfinite(result["vs_ask_magic"]))


def test_swap_market_flat_extrapolation_handles_explicit_order_on_narrow_smile():
    ivol_data = pd.DataFrame(
        {
            "Texp": [1.0, 1.0, 1.0],
            "Bid": [0.20, 0.20, 0.20],
            "Ask": [0.22, 0.22, 0.22],
            "Fwd": [1.0, 1.0, 1.0],
            "Strike": np.exp([-0.05, 0.0, 0.05]),
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bounded = swap.swap_market(ivol_data, n_quad=2)
        flat = swap.swap_market(ivol_data, n_quad=2, extrapolation="flat")

    assert np.isnan(bounded["vs_mid"][0])
    assert np.isfinite(flat["vs_mid"][0])
    assert np.all(np.isfinite(flat["vs_mid_magic"][:, 0]))


def test_swap_market_warns_with_side_context_on_fukasawa_failure(monkeypatch):
    ivol_data = pd.DataFrame(
        {
            "Texp": [0.5, 0.5, 0.5],
            "Bid": [0.18, 0.20, 0.22],
            "Ask": [0.20, 0.22, 0.24],
            "Fwd": [1.0, 1.0, 1.0],
            "Strike": [0.8, 1.0, 1.2],
        }
    )

    def fake_swap_fukasawa(**kwargs):
        raise ValueError("forced failure")

    def fake_swap_magic_strike(**kwargs):
        return {order: np.array([0.04]) for order in kwargs["order"]}

    monkeypatch.setattr(swap, "swap_fukasawa", fake_swap_fukasawa)
    monkeypatch.setattr(swap, "swap_magic_strike", fake_swap_magic_strike)

    with pytest.warns(RuntimeWarning) as record:
        result = swap.swap_market(ivol_data, n_quad=2)

    messages = [str(w.message) for w in record]
    assert any("bid quotes" in message for message in messages)
    assert any("ask quotes" in message for message in messages)
    assert any("mid quotes" in message for message in messages)
    assert all("forced failure" in message for message in messages)
    assert np.isnan(result["vs_bid"][0])
    assert np.isnan(result["vs_ask"][0])
    assert np.isnan(result["vs_mid"][0])
    assert np.all(np.isfinite(result["vs_bid_magic"]))


def test_swap_market_accepts_extra_nonnumeric_columns():
    ivol_data = pd.DataFrame(
        {
            "Expiry": [
                "2025-07-03",
                "2025-07-03",
                "2025-07-03",
                "2025-08-01",
                "2025-08-01",
                "2025-08-01",
            ],
            "Texp": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
            "Bid": [0.16, 0.18, 0.22, 0.17, 0.19, 0.23],
            "Ask": [0.18, 0.20, 0.24, 0.19, 0.21, 0.25],
            "Fwd": [1.0, 1.0, 1.0, 1.05, 1.05, 1.05],
            "Strike": [0.7, 0.9, 1.3, 0.6, 0.95, 1.5],
        }
    )

    result = swap.swap_market(ivol_data, n_quad=2)

    assert np.allclose(result["expiries"], [0.5, 1.0])
    assert np.all(np.isfinite(result["vs_mid"]))


def test_swap_market_can_record_estimator_compute_times(monkeypatch):
    ivol_data = pd.DataFrame(
        {
            "Texp": [0.5, 0.5, 0.5],
            "Bid": [0.16, 0.18, 0.22],
            "Ask": [0.18, 0.20, 0.24],
            "Fwd": [1.0, 1.0, 1.0],
            "Strike": [0.7, 0.9, 1.3],
        }
    )
    durations = [
        0.10,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.20,
        0.11,
        0.12,
        0.13,
        0.14,
        0.15,
        0.30,
        0.21,
        0.22,
        0.23,
        0.24,
        0.25,
    ]
    clock_values = []
    current = 0.0
    for duration in durations:
        clock_values.extend([current, current + duration])
        current += duration + 1.0
    clock_values = iter(clock_values)

    def fake_swap_fukasawa(**kwargs):
        return np.array([0.04])

    def fake_swap_magic_strike(**kwargs):
        return {order: np.array([0.04]) for order in kwargs["order"]}

    monkeypatch.setattr(swap.time, "perf_counter", lambda: next(clock_values))
    monkeypatch.setattr(swap, "swap_fukasawa", fake_swap_fukasawa)
    monkeypatch.setattr(swap, "swap_magic_strike", fake_swap_magic_strike)

    result = swap.swap_market(ivol_data, n_quad=2, record_compute_time=True)

    assert set(result["compute_time"]) == {"fukasawa", "magic_strike"}
    assert np.allclose(result["compute_time"]["fukasawa"], 0.6)
    assert np.allclose(
        result["compute_time"]["magic_strike"],
        [0.33, 0.36, 0.39, 0.42, 0.45],
    )
