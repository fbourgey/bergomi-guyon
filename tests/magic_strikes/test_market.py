"""Quoted-strike filtering and real SPX data regressions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from magic_strikes.market import market_comparison

DATA = Path(__file__).resolve().parents[2] / "data"


def test_coverage_distinguishes_narrow_and_wide_flat_smiles():
    frames = []
    for maturity, width in [(0.5, 1.0), (1.0, 0.1)]:
        frames.append(
            pd.DataFrame(
                {
                    "Texp": maturity,
                    "Strike": 100 * np.exp(np.linspace(-width, width, 31)),
                    "Fwd": 100.0,
                    "Bid": 0.19,
                    "Ask": 0.21,
                }
            )
        )
    result = market_comparison(pd.concat(frames))
    assert result["included"].tolist() == [True, False]
    failures = result["exclusions"]
    assert set(failures["Texp"]) == {1.0}
    assert set(failures["contract"]) == {"variance", "gamma"}
    assert set(failures["side"]) == {"bid", "ask", "mid"}
    assert set(failures["reason"]) == {"magic strikes outside quoted range"}
    # Excluded values remain inspectable; a flat smile is still priced exactly.
    variance = result["estimates"]["variance"]
    assert np.allclose(variance["vs_mid_magic"], [[0.02, 0.04]] * 5)


@pytest.mark.parametrize("date", ["20250210", "20250404", "20250407", "20250702"])
def test_four_spx_dates_have_in_bounds_estimates_and_report_exclusions(date):
    data = pd.read_csv(DATA / f"{date}_spx_vol.csv", index_col=0)
    result = market_comparison(data)
    keep = result["included"]
    estimates = result["estimates"]
    maturities = estimates["variance"]["expiries"]
    assert keep.sum() > 20
    assert set(result["exclusions"]["Texp"]) == set(maturities[~keep])
    assert set(estimates) == {"variance", "gamma"}
    quotes = data.dropna(subset=["Bid", "Ask", "Texp", "Strike", "Fwd"])
    quotes = quotes.loc[quotes["Ask"] > quotes["Bid"]]
    for opt, sign in [("variance", -1), ("gamma", 1)]:
        for side in ("bid", "ask", "mid"):
            values = estimates[opt][f"vs_{side}_magic"][:, keep]
            assert values.shape == (5, keep.sum())
            assert np.all(np.isfinite(values) & (values > 0))
            # Independently verify the outermost strikes at each order.
            for j, maturity in enumerate(maturities[keep]):
                q = quotes.loc[quotes["Texp"] == maturity]
                bounds = np.log(q["Strike"] / q["Fwd"])
                levels = values[:, j]
                center = sign * levels / 2
                radius = np.array([0, 1, 1, 2, 2]) * np.sqrt(levels)
                assert np.all(center - radius >= bounds.min())
                assert np.all(center + radius <= bounds.max())
