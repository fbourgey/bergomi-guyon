"""SPX comparisons with an explicit quoted-strike coverage report."""

import numpy as np
import pandas as pd

from .magic_strike import _magic_strike_grid, _normalize_orders
from .swap import swap_market


def market_comparison(ivol_data, orders=(1, 2, 3, 4, 5)):
    """Evaluate the paper's market experiment and flag unsupported expiries.

    Use PCHIP interpolation of total implied variance, flat extrapolation, and
    the ten-node Fukasawa benchmark from Section 5.3. An expiry is included only
    if every requested variance and gamma approximation, on bid, ask, and mid
    smiles, is finite and has its final magic strikes inside the quoted range.
    Nonfinite benchmarks also exclude an expiry. Solver warnings retain the
    original implementation's behavior and must be inspected separately.

    Returns a dictionary with ``estimates`` (variance, gamma), an
    ``included`` mask aligned with each estimate's ``expiries``, and an
    ``exclusions`` DataFrame explaining each failed check. Estimates retain all
    expiries so excluded values remain available for inspection. Plot only the
    included subset for the paper comparison.

    All contract values are total values; divide by maturity to reproduce
    the annualized plots.
    """
    orders = _normalize_orders("variance", orders)
    estimates = {
        opt: swap_market(
            ivol_data,
            opt=opt,
            n_quad=10,
            extrapolation="flat",
            magic_orders=orders,
        )
        for opt in ("variance", "gamma")
    }
    variance = estimates["variance"]
    expiries = variance["expiries"]
    required = ["Bid", "Ask", "Texp", "Strike", "Fwd"]
    data = ivol_data.dropna(subset=required).copy()
    data[required] = data[required].astype(float)
    data = data.loc[data["Ask"] > data["Bid"]]
    included = np.ones(expiries.size, dtype=bool)
    failures = []
    for i, maturity in enumerate(expiries):
        quotes = data.loc[data["Texp"] == maturity]
        ks = np.log(quotes["Strike"].to_numpy() / quotes["Fwd"].to_numpy())
        lower, upper = float(ks.min()), float(ks.max())
        for opt in ("variance", "gamma"):
            result = estimates[opt]
            for side in ("bid", "ask", "mid"):
                benchmark = result[f"vs_{side}"][i]
                checks = [(None, benchmark)] + list(
                    zip(orders, result[f"vs_{side}_magic"][:, i], strict=True)
                )
                for order, value in checks:
                    strike_min = strike_max = np.nan
                    if not np.isfinite(value) or value <= 0:
                        reason = "nonfinite or nonpositive estimate"
                    elif order is None:
                        continue
                    else:
                        strikes = _magic_strike_grid(value, order, opt)
                        strike_min, strike_max = strikes[0], strikes[-1]
                        if lower <= strike_min and strike_max <= upper:
                            continue
                        reason = "magic strikes outside quoted range"
                    included[i] = False
                    failures.append(
                        (
                            maturity,
                            opt,
                            side,
                            order,
                            lower,
                            upper,
                            strike_min,
                            strike_max,
                            reason,
                        )
                    )
    exclusions = pd.DataFrame(
        failures,
        columns=[
            "Texp",
            "contract",
            "side",
            "order",
            "quote_min",
            "quote_max",
            "strike_min",
            "strike_max",
            "reason",
        ],
    )
    return {"estimates": estimates, "included": included, "exclusions": exclusions}
