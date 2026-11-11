"""
Regime detection: rolling Engle-Granger cointegration check.
If cointegration breaks (p > 0.10) on a trailing 60-day window,
the pair is paused and positions are closed.
"""

import pandas as pd
from statsmodels.tsa.stattools import coint


COINT_PVAL_BREAK = 0.10
TRAILING_WINDOW = 60
RECHECK_FREQ = 5  # trading days between regime checks


def check_regime(
    s1: pd.Series,
    s2: pd.Series,
    position: pd.Series,
    window: int = TRAILING_WINDOW,
    recheck_freq: int = RECHECK_FREQ,
    pval_break: float = COINT_PVAL_BREAK,
) -> pd.Series:
    """
    Overlay regime filter on position series.
    Every `recheck_freq` days, test trailing `window` days for cointegration.
    If test fails, zero out positions until cointegration is restored.
    Returns filtered position series.
    """
    pos = position.to_numpy(dtype=float, copy=True)
    n = len(pos)
    paused = False
    for i in range(window, n):
        # Re-test on check days only, using data through the previous close.
        if i % recheck_freq == 0:
            w1 = s1.iloc[i - window: i]
            w2 = s2.iloc[i - window: i]
            try:
                _, pval, _ = coint(w1, w2)
            except Exception:
                pval = 1.0
            paused = pval > pval_break
        # The pause holds on every day until a later check passes.
        if paused:
            pos[i] = 0
    return pd.Series(pos, index=position.index, name="position_filtered")


def apply_position_cap(
    pnl_dict: dict, max_weight: float = 0.10, vol_window: int = 60
) -> dict:
    """
    Inverse-volatility pair weights, each capped at ``max_weight``.

    Weights on day t use only P&L observed through t-1 (trailing
    ``vol_window`` standard deviation, shifted one day), so sizing has no
    look-ahead. Weights are not renormalised after capping: when the cap
    binds, the unallocated share stays in cash, so no pair ever exceeds
    the cap. Returns the scaled daily P&L of each pair.
    """
    if not pnl_dict:
        return {}
    df = pd.DataFrame(pnl_dict).fillna(0.0)
    vol = df.rolling(vol_window, min_periods=20).std().shift(1)
    inv = 1.0 / vol.where(vol > 0)
    weights = inv.div(inv.sum(axis=1), axis=0).clip(upper=max_weight).fillna(0.0)
    scaled = df * weights
    return {pair: scaled[pair] for pair in df.columns}
