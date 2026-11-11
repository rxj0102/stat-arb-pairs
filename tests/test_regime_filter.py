"""The cointegration regime overlay must hold a pause until the next passing test."""
import numpy as np
import pandas as pd

from risk import regime_filter as rf


def _series(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    s1 = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx)
    s2 = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx)
    return s1, s2


def test_pause_holds_between_checks(monkeypatch):
    """Once a check fails, every day until the next passing check must be flat."""
    s1, s2 = _series()
    pos = pd.Series(1.0, index=s1.index)
    # Fail every check from day 100 on; pass before that.
    calls = {"i": 0}

    def fake_coint(w1, w2):
        end = s1.index.get_loc(w1.index[-1]) + 1
        return 0.0, (0.5 if end >= 100 else 0.01), None

    monkeypatch.setattr(rf, "coint", fake_coint)
    out = rf.check_regime(s1, s2, pos, window=60, recheck_freq=5)
    first_fail = next(i for i in range(60, 200) if i % 5 == 0 and i >= 100)
    assert (out.iloc[first_fail:] == 0).all()
    assert (out.iloc[:first_fail] == 1).all()


def test_pause_lifts_after_passing_check(monkeypatch):
    s1, s2 = _series()
    pos = pd.Series(1.0, index=s1.index)

    def fake_coint(w1, w2):
        end = s1.index.get_loc(w1.index[-1]) + 1
        return 0.0, (0.5 if 100 <= end < 150 else 0.01), None

    monkeypatch.setattr(rf, "coint", fake_coint)
    out = rf.check_regime(s1, s2, pos, window=60, recheck_freq=5)
    assert (out.iloc[100:150] == 0).all()
    assert (out.iloc[150:] == 1).all()
