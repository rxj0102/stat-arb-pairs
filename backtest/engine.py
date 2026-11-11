"""
Vectorized backtesting engine for stat-arb pairs.

P&L is computed on the dollar spread:
  long spread  = long S1, short β*S2  (normalized to 1-dollar notional on S1)
  short spread = short S1, long β*S2
Transaction cost: 5bps one-way per leg (2 legs = 10bps round-trip per trade).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


TC_BPS = 5e-4   # 5 basis points one-way


@dataclass
class BacktestResult:
    ticker1: str
    ticker2: str
    pnl: pd.Series          # daily P&L (dollar)
    cum_returns: pd.Series  # cumulative return
    sharpe: float
    sortino: float
    max_drawdown: float
    hit_rate: float
    total_trades: int
    avg_pnl_per_trade: float


def _trade_counts(position: pd.Series) -> pd.Series:
    """Number of times position changes (new trade opened)."""
    return (position.diff().fillna(0) != 0) & (position != 0)


def run_backtest(
    s1: pd.Series,
    s2: pd.Series,
    beta: pd.Series,
    position: pd.Series,
    ticker1: str = "S1",
    ticker2: str = "S2",
) -> BacktestResult:
    """
    Parameters
    ----------
    s1, s2   : price series
    beta     : hedge ratio (Kalman)
    position : {-1, 0, 1} signal series
    """
    # Align
    common = s1.index.intersection(s2.index).intersection(position.index)
    s1 = s1.loc[common]
    s2 = s2.loc[common]
    beta = beta.loc[common]
    pos = position.loc[common]

    # Daily returns
    r1 = s1.pct_change().fillna(0)
    r2 = s2.pct_change().fillna(0)

    # Spread return: long β*S2 → hedge ratio positions
    # Position = +1: long S1, short β*S2
    # P&L per unit notional
    # The hedge held over day t is the beta known at the close of t-1.
    # beta[t] is updated with day t prices, so using it here would leak
    # the same-day observation into the P&L.
    beta_held = beta.shift(1).bfill()
    spread_ret = r1 - beta_held * r2

    # Gross P&L
    gross_pnl = pos.shift(1).fillna(0) * spread_ret

    # Transaction costs: charged when position changes
    pos_change = pos.diff().fillna(0).abs()
    # Each leg costs TC_BPS; 2 legs (S1 and S2)
    tc = pos_change * 2 * TC_BPS
    net_pnl = gross_pnl - tc

    cum_ret = (1 + net_pnl).cumprod() - 1

    # Risk metrics
    ann = np.sqrt(252)
    mean_r = net_pnl.mean()
    std_r = net_pnl.std()
    sharpe = (mean_r / std_r * ann) if std_r > 0 else 0.0

    downside = net_pnl[net_pnl < 0].std()
    sortino = (mean_r / downside * ann) if downside > 0 else 0.0

    cum_wealth = (1 + net_pnl).cumprod()
    running_max = cum_wealth.cummax()
    drawdown = (cum_wealth - running_max) / running_max
    max_dd = drawdown.min()

    # Trade-level stats
    trade_start = (pos.diff().fillna(0) != 0) & (pos != 0)
    trade_end = (pos.diff().fillna(0) != 0) & (pos.shift(1).fillna(0) != 0)

    # Simple: group by trade id
    trade_id = (pos.diff().fillna(0) != 0).cumsum()
    trade_pnl = net_pnl.groupby(trade_id).sum()
    # Only count trades where we had a non-zero position
    active_mask = pos.groupby(trade_id).max().abs() > 0
    trade_pnl = trade_pnl[active_mask]

    total_trades = int(trade_start.sum())
    hit_rate = float((trade_pnl > 0).mean()) if len(trade_pnl) > 0 else 0.0
    avg_pnl = float(trade_pnl.mean()) if len(trade_pnl) > 0 else 0.0

    return BacktestResult(
        ticker1=ticker1,
        ticker2=ticker2,
        pnl=net_pnl,
        cum_returns=cum_ret,
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        max_drawdown=round(max_dd, 4),
        hit_rate=round(hit_rate, 3),
        total_trades=total_trades,
        avg_pnl_per_trade=round(avg_pnl, 6),
    )


def split_insample_oos(series: pd.Series, train_frac: float = 0.60):
    """Split series into in-sample and out-of-sample portions."""
    split = int(len(series) * train_frac)
    return series.iloc[:split], series.iloc[split:]
