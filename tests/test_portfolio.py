import math
import pandas as pd
import numpy as np

from portfolio import compute_max_sharpe


def make_prices(days=200, assets=3, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=days)
    rets = rng.normal(loc=0.0005, scale=0.01, size=(days, assets))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    cols = [f"00000{i+1}" for i in range(assets)]
    return pd.DataFrame(prices, index=dates, columns=cols)


def test_compute_max_sharpe_basic():
    prices = make_prices()
    weights, perf = compute_max_sharpe(prices, rf=0.01, max_weight=0.8)

    # weights are valid
    w_vals = list(weights.values())
    assert all(0 <= w <= 0.80001 for w in w_vals)
    assert math.isclose(sum(w_vals), 1.0, rel_tol=1e-4, abs_tol=1e-4)

    # performance is tuple of 3 finite numbers
    assert isinstance(perf, tuple) and len(perf) == 3
    assert all(np.isfinite(x) for x in perf)


def test_compute_max_sharpe_raises_on_empty():
    with np.testing.assert_raises(ValueError):
        compute_max_sharpe(pd.DataFrame(), rf=0.01, max_weight=0.8)
