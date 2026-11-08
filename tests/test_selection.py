"""Pair selection must not use out-of-sample data."""
import numpy as np
import pandas as pd

from universe import cointegration_screen as cs


def _prices(n=900, seed=3, scramble_oos=False):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    a = np.cumsum(rng.normal(0, 1, n)) + 100
    b = np.cumsum(rng.normal(0, 1, n)) + 100
    c = a + rng.normal(0, 0.5, n)          # cointegrated with A throughout
    df = pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)
    if scramble_oos:                        # destroy the relationship out of sample
        split = int(n * 0.6)
        rng2 = np.random.default_rng(99)
        df.iloc[split:, 2] = 100 + np.cumsum(rng2.normal(0, 1, n - split))
    return df


def _run(monkeypatch, tmp_path, prices, **kw):
    monkeypatch.setattr(cs, "SECTOR_TICKERS", {"x": ["A", "B", "C"]})
    monkeypatch.setattr(cs, "download_prices", lambda *a, **k: prices)
    monkeypatch.setattr(cs, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cs, "rolling_coint_stability", lambda *a, **k: 0.0)
    df = cs.screen_pairs(halflife_min=0, halflife_max=10_000, **kw)
    return set(zip(df.ticker1, df.ticker2))


def test_selection_ignores_out_of_sample_data(monkeypatch, tmp_path):
    clean = _run(monkeypatch, tmp_path, _prices())
    scrambled = _run(monkeypatch, tmp_path, _prices(scramble_oos=True))
    assert ("A", "C") in clean
    assert clean == scrambled


def test_full_sample_screen_does_depend_on_oos(monkeypatch, tmp_path):
    """The old behaviour (train_frac=None) lets OOS data change the selection."""
    clean = _run(monkeypatch, tmp_path, _prices(), train_frac=None)
    scrambled = _run(monkeypatch, tmp_path, _prices(scramble_oos=True), train_frac=None)
    assert clean != scrambled


def test_empty_screen_does_not_crash(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2019-01-01", periods=600)
    px = pd.DataFrame({k: 100 + np.cumsum(rng.normal(0, 1, 600)) for k in "ABC"}, index=idx)
    assert _run(monkeypatch, tmp_path, px, eg_pval_threshold=1e-12) == set()


def test_empty_download_raises(monkeypatch, tmp_path):
    import pytest
    monkeypatch.setattr(cs, "SECTOR_TICKERS", {"x": ["A", "B"]})
    monkeypatch.setattr(cs, "download_prices", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(cs, "DATA_DIR", tmp_path)
    with pytest.raises(RuntimeError):
        cs.screen_pairs()
