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
    pnl_dict: dict, max_weight: float = 0.10
) -> dict:
    """
    Cap any single pair at max_weight fraction of total portfolio.
    Scales daily P&L series so no pair exceeds the cap.
    """
    scaled = {}
    total_vol = sum(p.std() for p in pnl_dict.values() if p.std() > 0)
    for pair, pnl in pnl_dict.items():
        pair_vol = pnl.std()
        weight = (pair_vol / total_vol) if total_vol > 0 else max_weight
        if weight > max_weight:
            scale = max_weight / weight
        else:
            scale = 1.0
        scaled[pair] = pnl * scale
    return scaled
