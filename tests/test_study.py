"""Checks on the study: no look-ahead, the trading rule, and the accounting."""
import numpy as np
import pandas as pd
import pytest

import study
from study import Config, formation_stats, newey_west_t, run, sector_pairs, trade_pair
from universe.sector_map import SECTOR_TICKERS

CFG = Config()


def fake_returns(n=1400, seed=0):
    rng = np.random.default_rng(seed)
    names = SECTOR_TICKERS[next(iter(SECTOR_TICKERS))][:5]
    common = rng.normal(0, 0.008, n)
    data = {k: common + rng.normal(0, 0.006, n) for k in names}
    return pd.DataFrame(data, index=pd.bdate_range("2000-01-03", periods=n))


@pytest.fixture(scope="module")
def rets():
    return fake_returns()


@pytest.fixture(scope="module")
def res(rets):
    return run(rets, CFG)


# ---------- universe ----------

def test_pairs_are_within_sector_only():
    cols = [n for names in SECTOR_TICKERS.values() for n in names]
    sector_of = {n: s for s, names in SECTOR_TICKERS.items() for n in names}
    pairs = sector_pairs(cols)
    assert len(pairs) == 142
    assert all(sector_of[a] == sector_of[b] for a, b in pairs)


def test_snapshot_loads():
    df = study.load()
    assert df.shape[1] == 49 and df.index.is_monotonic_increasing
    assert df.abs().max().max() < 1.0


# ---------- trading rule ----------

def flat(n):
    return np.zeros(n)


def test_no_trade_inside_the_band():
    z = np.array([0.5, 1.9, -1.9, 0.0, 1.0])
    out = trade_pair(z, flat(5), flat(5), 1.0, CFG)
    assert out["opened"] == 0 and not out["ret"].any()


def test_signal_is_traded_the_next_day():
    z = np.array([0.0, 2.5, 2.5, 2.5, 2.5])
    r_y = np.array([0.0, 0.10, 0.01, 0.0, 0.0])      # a big move on the signal day itself
    out = trade_pair(z, r_y, flat(5), 1.0, CFG)
    assert out["gross"][1] == 0.0                     # not captured
    assert out["gross"][2] == pytest.approx(-0.01 / 2)  # short the spread from the next day


def test_short_when_spread_is_high_and_long_when_low():
    r_y = np.full(4, 0.01)
    assert trade_pair(np.array([3.0, 3, 3, 3]), r_y, flat(4), 1.0, CFG)["gross"][1] < 0
    assert trade_pair(np.array([-3.0, -3, -3, -3]), r_y, flat(4), 1.0, CFG)["gross"][1] > 0


def test_close_when_spread_crosses_zero():
    z = np.array([2.5, 1.0, -0.1, -0.5, -0.5, 0.0])
    out = trade_pair(z, np.full(6, 0.01), flat(6), 1.0, CFG)
    assert out["opened"] == 1 and out["converged"] == 1
    assert out["gross"][1] != 0 and out["gross"][2] != 0   # held while z is still positive
    assert not out["gross"][3:].any()                      # flat after the cross


def test_costs_charged_on_open_close_and_forced_close():
    cfg = Config(cost=0.001)
    closed = trade_pair(np.array([2.5, 1.0, -0.1, 0.0, 0.0]), flat(5), flat(5), 1.0, cfg)
    assert closed["ret"].sum() == pytest.approx(-0.002)    # one open and one close
    forced = trade_pair(np.array([2.5, 2.5, 2.5, 2.5, 2.5]), flat(5), flat(5), 1.0, cfg)
    assert forced["converged"] == 0
    assert forced["ret"].sum() == pytest.approx(-0.002)    # open, then closed at period end


def test_last_spread_value_cannot_matter():
    rng = np.random.default_rng(1)
    z, r_y, r_x = rng.normal(0, 2, 60), rng.normal(0, 0.01, 60), rng.normal(0, 0.01, 60)
    a = trade_pair(z, r_y, r_x, 0.8, CFG)
    z2 = z.copy()
    z2[-1] = 99.0
    b = trade_pair(z2, r_y, r_x, 0.8, CFG)
    assert np.array_equal(a["ret"], b["ret"])


def test_return_is_on_gross_exposure():
    z = np.full(4, 3.0)
    out = trade_pair(z, flat(4), np.full(4, 0.01), 3.0, CFG)   # short y, long 3 units of x
    assert out["gross"][1] == pytest.approx(3 * 0.01 / 4)


# ---------- formation ----------

def test_formation_recovers_a_known_relation():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.01, 504).cumsum()
    y = 0.1 + 0.7 * x + rng.normal(0, 0.005, 504)
    st = formation_stats(y, x, 1)
    assert st["b"] == pytest.approx(0.7, abs=0.05) and st["a"] == pytest.approx(0.1, abs=0.01)
    assert st["p"] < 0.01


def test_independent_random_walks_rarely_pass():
    rng = np.random.default_rng(3)
    p = [formation_stats(rng.normal(0, 0.01, 504).cumsum(), rng.normal(0, 0.01, 504).cumsum(), 1)["p"]
         for _ in range(200)]
    assert np.mean(np.array(p) < 0.05) < 0.12


# ---------- the rolling run ----------

def test_periods_tile_the_sample_without_overlap(rets, res):
    n = len(res["periods"])
    assert n == (len(rets) - CFG.formation) // CFG.trading
    assert res["net"].index.is_unique
    assert res["net"].index[0] == rets.index[CFG.formation]
    assert len(res["net"]) == n * CFG.trading


def test_selection_and_returns_ignore_the_future(rets, res):
    """Change everything after period k ends. Periods up to k must not move."""
    k = 3
    cut = CFG.formation + (k + 1) * CFG.trading
    changed = rets.copy()
    rng = np.random.default_rng(9)
    changed.iloc[cut:] = rng.normal(0, 0.03, changed.iloc[cut:].shape)
    again = run(changed, CFG)
    day = rets.index[cut - 1]
    pd.testing.assert_frame_equal(res["net"].loc[:day], again["net"].loc[:day])
    for a, b in zip(res["periods"][: k + 1], again["periods"][: k + 1]):
        assert a == b


def test_selection_ignores_the_trading_period(rets, res):
    """Change only the trading window of period k. The pairs chosen for k must not move."""
    k = 2
    s = CFG.formation + k * CFG.trading
    changed = rets.copy()
    changed.iloc[s:s + CFG.trading] = np.random.default_rng(5).normal(0, 0.03, (CFG.trading, rets.shape[1]))
    again = run(changed, CFG)
    for name in ("Cointegration", "Distance"):
        assert res["periods"][k][name]["pairs"] == again["periods"][k][name]["pairs"]
    assert not np.allclose(res["net"].iloc[k * CFG.trading:(k + 1) * CFG.trading],
                           again["net"].iloc[k * CFG.trading:(k + 1) * CFG.trading])


def test_a_gap_in_the_trading_period_does_not_change_selection(rets, res):
    """A missing day that first appears in the trading window of period k is future
    information. The pairs tested and chosen for k must be the same as without it."""
    k = 2
    s = CFG.formation + k * CFG.trading
    holed = rets.copy()
    holed.iloc[s + 40, 0] = np.nan
    again = run(holed, CFG)
    for key in ("n_tested", "n_usable"):
        assert again["periods"][k][key] == res["periods"][k][key]
    for name in ("Cointegration", "Distance", "All pairs"):
        assert again["periods"][k][name]["pairs"] == res["periods"][k][name]["pairs"]
    assert np.isfinite(again["net"].to_numpy()).all()


def test_exit_on_missing_closes_at_last_observed_prices():
    z = np.array([2.5, 2.4, 2.3, 2.2, 2.1, 2.0])
    ry = np.array([0.01, 0.02, 0.03, np.nan, 0.05, 0.06])
    rx = np.zeros(6)
    z2, ry2, rx2 = study.exit_on_missing(z, ry, rx)
    assert np.array_equal(z2, [2.5, 2.4, 0.0, 0.0, 0.0, 0.0])
    assert np.array_equal(ry2, [0.01, 0.02, 0.03, 0.0, 0.0, 0.0])
    assert np.isnan(ry[3])                       # inputs are not modified
    out = trade_pair(z2, ry2, rx2, 1.0, CFG)
    # Short opened at close 0, held on days 1 and 2, closed at close 2.
    assert out["opened"] == 1 and out["converged"] == 1
    assert np.allclose(out["gross"], [0, -0.01, -0.015, 0, 0, 0])


def test_exit_on_missing_leaves_complete_data_alone():
    z, ry, rx = np.ones(5), np.full(5, 0.01), np.full(5, 0.02)
    z2, ry2, rx2 = study.exit_on_missing(z, ry, rx)
    assert z2 is z and ry2 is ry and rx2 is rx


def test_industry_with_missing_data_is_skipped(rets):
    holed = rets.copy()
    first = holed.columns[0]
    holed.iloc[:600, 0] = np.nan
    out = run(holed, CFG)
    assert all(first not in p for p in out["periods"][0]["All pairs"]["pairs"])
    assert any(first in p for p in out["periods"][-1]["All pairs"]["pairs"])


def test_all_pairs_portfolio_is_the_mean_of_pairs(rets):
    one = run(rets, Config(n_pairs=10**6))
    assert np.allclose(one["net"]["Cointegration"], one["net"]["All pairs"])


def test_net_is_never_above_gross(res):
    assert (res["net"].to_numpy() <= res["gross"].to_numpy() + 1e-15).all()


def test_persistence_compares_non_overlapping_windows(res):
    out = study.persistence(res)
    assert out["periods"] == len(res["periods"]) - CFG.formation // CFG.trading
    assert 0 <= out["selected_pass_rate"] <= 1 and 0 <= out["all_pairs_pass_rate"] <= 1


# ---------- statistics ----------

def test_newey_west_matches_plain_t_without_lags():
    d = np.random.default_rng(4).normal(0.001, 0.01, 500)
    plain = d.mean() / (d.std(ddof=0) / np.sqrt(len(d)))
    assert newey_west_t(d, lags=0) == pytest.approx(plain)


def test_sharpe_of_constant_is_nan():
    assert np.isnan(study.sharpe(np.zeros(10)))
