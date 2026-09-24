"""Contract identities and benchmarks from the companion paper."""

import numpy as np
import pytest

from magic_strikes import magic_strike, swap
from magic_strikes.heston import HestonModel, get_params_heston
from magic_strikes.rough_bergomi import RoughBergomiModel, get_params_rough_bergomi


def smile(k):
    return 0.06 * (1 - 0.3 * np.tanh(k) + 0.1 * np.asarray(k) ** 2)


def test_reflection_and_power_endpoints():
    result = magic_strike.swap(smile, opt="all", tol=1e-12)
    reflected = magic_strike.swap(lambda k: smile(-k), tol=1e-12)
    p0 = magic_strike.swap(smile, opt="power", p=0, tol=1e-12)
    p1 = magic_strike.swap(smile, opt="power", p=1, tol=1e-12)
    assert set(result) == {"variance", "gamma"}
    for order in range(1, 6):
        assert result["gamma"][order] == pytest.approx(reflected[order], abs=2e-11)
        assert p0[order] == pytest.approx(result["variance"][order], abs=2e-11)
        assert p1[order] == pytest.approx(result["gamma"][order], abs=2e-11)


@pytest.mark.parametrize("set_id", [1, 2])
def test_paper_rough_bergomi_sets_compute_supported_contracts(set_id):
    params, xi0 = get_params_rough_bergomi(set_id)
    model = RoughBergomiModel(params, xi0)
    result = model.swap_mc_estimators(
        T=[0.25],
        n_mc=2000,
        n_disc=16,
        seed=1234,
        antithetic=True,
        magic_n_interp=101,
        fukasawa_n_interp=101,
    )
    assert set(result) == {"mc", "magic_strikes", "fukasawa"}
    for method in ("mc", "fukasawa"):
        values = result[method]
        assert set(values) == {"variance", "gamma"}
        assert np.all(np.isfinite(list(values.values())))
    assert result["mc"]["variance"] == pytest.approx(model.var_swap_quad([0.25]))
    assert set(result["magic_strikes"]) == {"variance", "gamma"}
    for order in range(1, 6):
        values = result["magic_strikes"]
        assert np.all(np.isfinite([contract[order] for contract in values.values()]))
