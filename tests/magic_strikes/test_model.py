import numpy as np
import pytest

from magic_strikes import swap
from magic_strikes.heston import HestonModel
from magic_strikes.model import ForwardVarianceModel, MonteCarloConfig


class DummyModel(ForwardVarianceModel):
    def kernel(self, u, t):
        return np.zeros_like(np.atleast_1d(u), dtype=float)

    def simulate_mc(
        self,
        tab_t,
        n_mc,
        n_loop=1,
        seed=None,
        antithetic=False,
    ):
        return {
            "int_v_dt": np.full(n_mc, float(tab_t[-1])),
            "int_sqrt_v_dw": np.full(n_mc, 0.1),
        }

    def impvol(self, k, T, **kwargs):
        return 0.2 * np.ones_like(np.atleast_1d(k), dtype=float)


def _control_variate_mean(payoff, control, control_mean):
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


def test_forward_variance_model_accepts_scalar_returning_xi0():
    model = DummyModel(params={}, xi0=lambda t: 0.04)

    assert np.isclose(model.xi0_0, 0.04)
    assert model.xi0_flat


def test_heston_clone_with_params_is_subclass_safe():
    model = HestonModel(
        params={"v": 0.04, "lbd": 1.0, "vbar": 0.04, "nu": 0.6, "rho": -0.8}
    )

    cloned = model._clone_with_params(v=0.09)

    assert isinstance(cloned, HestonModel)
    assert np.isclose(cloned.v, 0.09)
    assert np.isclose(cloned.s0, model.s0)


def test_swap_mc_all_variance_uses_quadrature_not_path_integral(monkeypatch):
    model = DummyModel(params={}, xi0=lambda t: 0.04)

    def fake_simulate_mc(**kwargs):
        return {
            "int_v_dt": np.full(4, 100.0),
            "int_sqrt_v_dw": np.full(4, 0.1),
        }

    monkeypatch.setattr(model, "simulate_mc", fake_simulate_mc)

    out = model.swap_mc_all(T=np.array([0.5, 1.0]), n_mc=4, n_disc=2, n_loop=2)

    assert set(out) == {"variance", "gamma"}
    assert np.allclose(out["variance"], [0.02, 0.04])


def test_var_swap_trapezoidal_grid_uses_trapezoidal_rule():
    model = DummyModel(params={}, xi0=lambda t: 0.04 + 0.02 * np.asarray(t))

    assert np.isclose(model._var_swap_trapezoidal_grid(np.array([0.0, 0.5, 1.0])), 0.05)
    assert np.allclose(model.var_swap_quad(1.0), [0.05])


def test_impvol_mc_forwards_antithetic_to_simulate_mc(monkeypatch):
    model = DummyModel(params={}, xi0=lambda t: 0.04)
    seen = []

    def fake_simulate_mc(**kwargs):
        seen.append(kwargs)
        return {
            "int_v_dt": np.full(6, 0.04),
            "int_sqrt_v_dw": np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]),
        }

    monkeypatch.setattr(model, "simulate_mc", fake_simulate_mc)

    model.impvol_mc(k=0.0, T=1.0, n_mc=6, n_disc=2, antithetic=True)

    assert seen[0]["antithetic"] is True


def test_swap_mc_all_forwards_antithetic_to_simulate_mc(monkeypatch):
    model = DummyModel(params={}, xi0=lambda t: 0.04)
    seen = []

    def fake_simulate_mc(**kwargs):
        seen.append(kwargs)
        return {
            "int_v_dt": np.full(4, 0.04),
            "int_sqrt_v_dw": np.array([-0.1, 0.0, 0.1, 0.2]),
        }

    monkeypatch.setattr(model, "simulate_mc", fake_simulate_mc)

    model.swap_mc_all(T=np.array([0.5, 1.0]), n_mc=4, n_disc=2, antithetic=True)

    assert [kwargs["antithetic"] for kwargs in seen] == [True, True]


@pytest.mark.parametrize("n_batch", [0, -1])
def test_swap_mc_all_rejects_invalid_n_batch(n_batch):
    model = DummyModel(params={}, xi0=lambda t: 0.04)

    with pytest.raises(ValueError, match="n_batch must be a positive integer"):
        model.swap_mc_all(T=1.0, n_mc=4, n_disc=2, n_batch=n_batch)


@pytest.mark.parametrize("n_loop", [0, -1])
def test_swap_mc_all_rejects_invalid_n_loop(n_loop):
    model = DummyModel(params={}, xi0=lambda t: 0.04)

    with pytest.raises(ValueError, match="n_loop must be a positive integer"):
        model.swap_mc_all(T=1.0, n_mc=4, n_disc=2, n_loop=n_loop)


def test_swap_mc_all_rejects_non_divisible_n_mc():
    model = DummyModel(params={}, xi0=lambda t: 0.04)

    with pytest.raises(ValueError, match="n_mc must be divisible by n_loop"):
        model.swap_mc_all(T=1.0, n_mc=5, n_disc=2, n_loop=2)


def test_swap_mc_all_rejects_conditioning_config_value():
    model = DummyModel(params={}, xi0=lambda t: 0.04)

    with pytest.raises(TypeError, match="Unexpected Monte Carlo config values"):
        model.swap_mc_all(T=1.0, n_mc=4, n_disc=2, conditioning=True)


def test_swap_mc_all_verbose_prints_progress(capsys):
    model = DummyModel(params={}, xi0=lambda t: 0.04)

    result = model.swap_mc_all(
        T=np.array([0.5, 1.0]),
        n_mc=4,
        n_disc=2,
        n_loop=2,
        seed=123,
        n_batch=2,
        verbose=True,
    )

    assert set(result) == {
        f"{contract}{suffix}"
        for contract in ("variance", "gamma")
        for suffix in ("", "_low", "_high")
    }
    captured = capsys.readouterr().out
    assert "Computing Monte Carlo swaps:" in captured
    assert "Model: DummyModel" in captured
    assert "Monte Carlo paths: 4" in captured
    assert "Number of independent batches: 2" in captured
    assert "Running batch 1/2 with seed" in captured
    assert "Running batch 2/2 with seed" in captured
    assert "1/2: expiry 0.5000" in captured
    assert "2/2: expiry 1.0000" in captured


def test_swap_mc_estimators_rejects_conditioning():
    model = DummyModel(params={"rho": -0.5}, xi0=lambda t: 0.04)

    with pytest.raises(TypeError, match="Unexpected Monte Carlo config values"):
        model.swap_mc_estimators(T=1.0, n_mc=4, n_disc=2, conditioning=True)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        MonteCarloConfig(n_mc=4, n_disc=2, conditioning=True)


def test_forward_variance_model_swap_magic_strike_power():
    model = DummyModel(params={}, xi0=lambda t: 0.04)

    out = model.swap_magic_strike(
        T=np.array([0.5, 1.0]), order=[1, 5], opt="power", p=0.3
    )

    assert set(out) == {1, 5}
    assert np.allclose(out[1], [0.02, 0.04])
    assert np.allclose(out[5], [0.02, 0.04])


def test_forward_variance_model_swap_magic_strike_forwards_relax(monkeypatch):
    model = DummyModel(params={}, xi0=lambda t: 0.04)
    calls = []

    def fake_swap_magic_strike(**kwargs):
        calls.append(kwargs)
        return {1: np.array([0.04])}

    monkeypatch.setattr(swap, "swap_magic_strike", fake_swap_magic_strike)

    out = model.swap_magic_strike(T=0.5, order=1, opt="variance", relax=0.35)

    assert out[1] == pytest.approx([0.04])
    assert calls[0]["relax"] == 0.35


def test_forward_variance_model_swap_magic_strike_forwards_power(monkeypatch):
    model = DummyModel(params={}, xi0=lambda t: 0.04)
    calls = []

    def fake_swap_magic_strike(**kwargs):
        calls.append(kwargs)
        return {1: np.array([0.04])}

    monkeypatch.setattr(swap, "swap_magic_strike", fake_swap_magic_strike)

    model.swap_magic_strike(T=0.5, order=1, opt="power", p=0.3)

    assert calls[0]["p"] == 0.3


def test_forward_variance_model_swap_fukasawa_verbose(capsys):
    model = DummyModel(params={}, xi0=lambda t: 0.04)

    model.swap_fukasawa(T=np.array([0.5, 1.0]), opt="variance", std=5, verbose=True)

    captured = capsys.readouterr().out
    assert "Computing Fukasawa variance swap for T=0.5000... (1/2)" in captured
    assert "Computing Fukasawa variance swap for T=1.0000... (2/2)" in captured
