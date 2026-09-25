import numpy as np
import pytest

from magic_strikes.utils import (
    black_impvol,
    black_price,
    gauss_legendre,
    lewis_formula_otm_price,
    mittag_leffler_two,
)


def test_black_impvol_recovers_vector_vols():
    K = np.array([0.8, 1.0, 1.2])
    T = 1.5
    F = 1.0
    vol = np.array([0.15, 0.2, 0.35])
    opttype = np.array([-1.0, 1.0, 1.0])

    value = black_price(K=K, T=T, F=F, vol=vol, opttype=opttype)
    impvol = black_impvol(K=K, T=T, F=F, value=value, opttype=opttype, TOL=1e-10)

    assert np.allclose(impvol, vol, atol=1e-8)


def test_black_price_handles_zero_time_and_invalid_inputs():
    price = black_price(
        K=np.array([0.9, 1.1, -1.0]),
        T=0.0,
        F=1.0,
        vol=0.2,
        opttype=np.array([1.0, -1.0, 1.0]),
    )

    assert np.allclose(price[:2], [0.1, 0.1])
    assert np.isnan(price[2])

    with pytest.raises(ValueError, match="opttype"):
        black_price(K=1.0, T=1.0, F=1.0, vol=0.2, opttype=0.0)


def test_black_impvol_rejects_bad_opttype_shape():
    K = np.array([0.9, 1.1])
    value = np.array([0.1, 0.1])

    with pytest.raises(ValueError, match="opttype"):
        black_impvol(K=K, T=1.0, F=1.0, value=value, opttype=np.ones(3))


def test_black_impvol_returns_nan_for_price_above_max_vol_bound():
    K = np.array([1.0])

    impvol = black_impvol(K=K, T=1.0, F=1.0, value=np.array([2.0]))

    assert np.isnan(impvol[0])


def test_lewis_formula_rejects_nonpositive_maturity():
    with pytest.raises(ValueError, match="T must be positive"):
        lewis_formula_otm_price(phi=lambda u, T: np.ones_like(T), k=0.0, T=0.0)


def test_mittag_leffler_two_rejects_nonpositive_parameters():
    with pytest.raises(ValueError, match="alpha and beta"):
        mittag_leffler_two(0.0, alpha=0.0, beta=1.0)


def test_gauss_legendre_matches_interval_scaling_and_is_cache_safe():
    knots, weights = gauss_legendre(2.0, 5.0, 4)
    raw_knots, raw_weights = np.polynomial.legendre.leggauss(4)

    assert np.allclose(knots, 1.5 * raw_knots + 3.5)
    assert np.allclose(weights, 1.5 * raw_weights)

    knots[0] = 999.0
    weights[0] = 999.0
    fresh_knots, fresh_weights = gauss_legendre(2.0, 5.0, 4)

    assert fresh_knots[0] != 999.0
    assert fresh_weights[0] != 999.0
