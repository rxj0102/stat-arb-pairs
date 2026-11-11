"""Break-momentum study: timing and sign conventions."""
import numpy as np
import pandas as pd

from analysis import break_momentum as bm


def _setup(n=120):
    idx = pd.bdate_range("2022-01-03", periods=n)
    s2 = pd.Series(100.0, index=idx)
    s1 = pd.Series(100.0, index=idx)
    beta = pd.Series(1.0, index=idx)
    z = pd.Series(0.0, index=idx)
    return s1, s2, beta, z


def test_trade_follows_deviation_and_starts_next_day():
    s1, s2, beta, z = _setup()
    # Spread drifts up 1 percent a day from day 61 on; z was positive at day 59.
    s1.iloc[61:] = 100 * np.cumprod(np.full(len(s1) - 61, 1.01))
    z.iloc[59] = 2.0
    out = bm.forward_trend_returns(s1, s2, beta, z, events=[60], horizons=(5,), cost_bps=0)
    assert out["direction"].iloc[0] == 1.0
    assert out["ret_5d"].iloc[0] > 0.04
    # Moves on the break day itself are not earned.
    s1b = s1.copy()
    s1b.iloc[60] = 150.0
    s1b.iloc[61:] = 150.0
    out_b = bm.forward_trend_returns(s1b, s2, beta, z, events=[60], horizons=(5,), cost_bps=0)
    assert abs(out_b["ret_5d"].iloc[0]) < 1e-12


def test_costs_are_charged():
    s1, s2, beta, z = _setup()
    z.iloc[59] = -1.0
    out = bm.forward_trend_returns(s1, s2, beta, z, events=[60], horizons=(5,), cost_bps=5)
    assert np.isclose(out["ret_5d"].iloc[0], -4 * 5 / 1e4)


def test_break_dates_need_a_prior_pass(monkeypatch):
    s1, s2, _, _ = _setup()
    seq = iter([0.01, 0.5, 0.5, 0.01, 0.5] + [0.01] * 50)
    monkeypatch.setattr(bm, "coint", lambda a, b: (0.0, next(seq), None))
    assert bm.break_dates(s1, s2, window=60, recheck_freq=5) == [65, 80]


def test_summary_counts_and_tstat():
    ev = pd.DataFrame({"ret_5d": [0.01, 0.02, -0.005, 0.015]})
    s = bm.summarise(ev, horizons=(5,))
    assert s["n_events"].iloc[0] == 4
    assert s["hit_rate"].iloc[0] == 0.75
