"""
Does a broken pair keep moving the way it broke?

The Oil Money study (je-suis-tm) found that when a fitted relationship
breaks, the deviation tends to keep going for a while, so the break is an
entry signal for trend following rather than an exit only. This module
tests the same idea on the pairs book, out of sample only.

Definition of a break: on a regime check day (every ``recheck_freq`` days),
the trailing ``window`` day Engle-Granger p-value exceeds ``pval_break``
after the previous check passed. This is exactly the rule that pauses
mean reversion trading in ``risk.regime_filter.check_regime``.

At each break the trade direction is the sign of the spread z-score at
the close before the break check: if the spread is above its rolling mean,
go long the spread (bet the deviation continues), and vice versa. The
spread is hedged with the beta known at that close and held for h days.
Entry is the close of the check day, so the first return earned is the
next day's. A round trip costs 4 legs x ``cost_bps`` (default 5 bps).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def break_dates(
    s1: pd.Series,
    s2: pd.Series,
    window: int = 60,
    recheck_freq: int = 5,
    pval_break: float = 0.10,
) -> list[int]:
    """Integer positions of check days where cointegration newly fails."""
    out: list[int] = []
    prev_ok: bool | None = None
    for i in range(window, len(s1)):
        if i % recheck_freq != 0:
            continue
        try:
            _, pval, _ = coint(s1.iloc[i - window: i], s2.iloc[i - window: i])
        except Exception:
            pval = 1.0
        ok = pval <= pval_break
        if prev_ok is True and not ok:
            out.append(i)
        prev_ok = ok
    return out


def forward_trend_returns(
    s1: pd.Series,
    s2: pd.Series,
    beta: pd.Series,
    zscore: pd.Series,
    events: list[int],
    horizons: tuple[int, ...] = (5, 10, 20),
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Net return of a trend-following spread trade after each break."""
    r1 = s1.pct_change()
    r2 = s2.pct_change()
    rows = []
    cost = 4 * cost_bps / 1e4
    for i in events:
        z_prev = zscore.iloc[i - 1]
        if not np.isfinite(z_prev) or z_prev == 0:
            continue
        direction = float(np.sign(z_prev))
        b = float(beta.iloc[i])  # hedge set at the check day's close
        row = {"date": s1.index[i], "z_at_break": float(z_prev), "direction": direction}
        for h in horizons:
            if i + h >= len(s1):
                row[f"ret_{h}d"] = np.nan
                continue
            leg = r1.iloc[i + 1: i + 1 + h] - b * r2.iloc[i + 1: i + 1 + h]
            row[f"ret_{h}d"] = direction * float(leg.sum()) - cost
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(events: pd.DataFrame, horizons: tuple[int, ...] = (5, 10, 20)) -> pd.DataFrame:
    """Mean, t-stat and hit rate per horizon across all break events."""
    out = []
    for h in horizons:
        x = events[f"ret_{h}d"].dropna() if len(events) else pd.Series(dtype=float)
        n = int(len(x))
        mean = float(x.mean()) if n else np.nan
        sd = float(x.std(ddof=1)) if n > 1 else np.nan
        t = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
        out.append({
            "horizon_days": h,
            "n_events": n,
            "mean_net_return": round(mean, 5) if n else np.nan,
            "t_stat": round(t, 2) if np.isfinite(t) else np.nan,
            "hit_rate": round(float((x > 0).mean()), 3) if n else np.nan,
        })
    return pd.DataFrame(out)
