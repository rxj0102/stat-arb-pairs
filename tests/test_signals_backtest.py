"""Point-in-time and accounting tests for the z-score signals and the backtest."""

import numpy as np
import pandas as pd
import pytest

from backtest.engine import run_backtest
from signals.zscore_signals import compute_zscore, generate_signals


def _spread(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(np.cumsum(rng.normal(0, 1, n)) * 0.2 + rng.normal(0, 1, n), index=idx)


def test_zscore_is_point_in_time():
    """z at date t must not change if later observations are removed."""
    s = _spread()
    full = compute_zscore(s, window=60)
    for cut in (120, 200, 333):
        trunc = compute_zscore(s.iloc[:cut], window=60)
        pd.testing.assert_series_equal(trunc, full.iloc[:cut], check_names=False)


def test_zscore_nan_until_window_filled():
    z = compute_zscore(_spread(), window=60)
    assert z.iloc[:59].isna().all()
    assert z.iloc[59:].notna().all()


def test_signals_only_take_allowed_values_and_start_flat():
    sig = generate_signals(compute_zscore(_spread(), window=60))
    assert set(np.unique(sig.values)).issubset({-1, 0, 1})
    assert sig.iloc[0] == 0


def test_signals_are_causal():
    """Signal at t is a function of z up to t only (state machine over the past)."""
    z = compute_zscore(_spread(), window=60)
    full = generate_signals(z)
    for cut in (150, 250):
        trunc = generate_signals(z.iloc[:cut])
        np.testing.assert_array_equal(trunc.values, full.values[:cut])


def test_entry_exit_and_stop_rules():
    z = pd.Series([0.0, -2.0, -1.0, -0.2, 2.0, 3.5, 0.0])
    sig = generate_signals(z, entry_z=1.5, exit_z=0.5, stop_z=3.0)
    # long on z < -1.5, hold while |z| >= exit, flat once |z| < exit,
    # short on z > 1.5, stop-loss to flat when |z| > stop
    assert list(sig.values) == [0, 1, 1, 0, -1, 0, 0]


def _prices(n=300, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    s1 = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    s2 = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    beta = pd.Series(1.0, index=idx)
    return s1, s2, beta


def test_backtest_pnl_uses_lagged_position():
    """P&L on day t may only depend on the position held at the close of t-1."""
    s1, s2, beta = _prices()
    pos = pd.Series(0.0, index=s1.index)
    pos.iloc[50:120] = 1.0
    base = run_backtest(s1, s2, beta, pos)

    changed = pos.copy()
    changed.iloc[200:] = -1.0  # alter only the future
    alt = run_backtest(s1, s2, beta, changed)
    pd.testing.assert_series_equal(base.pnl.iloc[:200], alt.pnl.iloc[:200])


def test_flat_position_earns_nothing_and_pays_no_costs():
    s1, s2, beta = _prices()
    res = run_backtest(s1, s2, beta, pd.Series(0.0, index=s1.index))
    assert (res.pnl == 0).all()
    assert res.total_trades == 0


def test_costs_reduce_pnl():
    s1, s2, beta = _prices()
    pos = pd.Series(0.0, index=s1.index)
    pos.iloc[10:20] = 1.0
    res = run_backtest(s1, s2, beta, pos)
    gross = (pos.shift(1).fillna(0) * (s1.pct_change().fillna(0) - beta * s2.pct_change().fillna(0)))
    assert res.pnl.sum() < gross.sum()
