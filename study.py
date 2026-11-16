"""Does a cointegration screen find pairs that keep reverting?

Rolling design in the spirit of Gatev, Goetzmann and Rouwenhorst (2006), fixed before the run:

  universe    the 142 within-sector pairs of the Ken French 49 industry portfolios
  formation   24 months (504 trading days). For each pair: OLS of one log price on the other,
              the Engle-Granger p-value, and the squared distance between the two price paths.
  selection   three portfolios per period
                cointegration   the 10 pairs with the lowest Engle-Granger p-value
                distance        the 10 pairs with the smallest squared distance
                all pairs       every pair, the benchmark a screen has to beat
  trading     the next 6 months (126 days), then everything is re-formed. The spread is measured
              in formation standard deviations around the formation mean. Open when it is more
              than 2 away, close when it crosses zero, close everything at the end of the period.
  execution   a signal at the close of day t is held from day t+1. 5 bps per unit traded per leg.
  return      per pair on gross exposure; a portfolio is the plain mean over its pairs, counting
              the idle ones (return on committed capital).

Nothing in the trading period is used to pick pairs, hedge ratios or thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from universe.sector_map import SECTOR_TICKERS

SNAPSHOT = Path(__file__).resolve().parent / "data" / "snapshot" / "industry_returns.csv.gz"
PORTFOLIOS = ["Cointegration", "Distance", "All pairs"]


@dataclass(frozen=True)
class Config:
    formation: int = 504
    trading: int = 126
    n_pairs: int = 10
    entry: float = 2.0
    cost: float = 0.0005
    eg_lags: int = 1


def load() -> pd.DataFrame:
    return pd.read_csv(SNAPSHOT, index_col="date", parse_dates=True)


def sector_pairs(columns) -> list[tuple[str, str]]:
    out = []
    for names in SECTOR_TICKERS.values():
        out += list(combinations([n for n in names if n in columns], 2))
    return out


def formation_stats(y: np.ndarray, x: np.ndarray, lags: int) -> dict:
    """Hedge regression, spread scale, Engle-Granger p-value and price distance on one window.

    y and x are log prices rebased to zero at the start of the window.
    """
    b, a = np.polyfit(x, y, 1)
    e = y - a - b * x
    p = float(coint(y, x, trend="c", maxlag=lags, autolag=None)[1])
    return {"a": float(a), "b": float(b), "sd": float(e.std(ddof=1)), "p": p,
            "ssd": float(((y - x) ** 2).sum())}


def trade_pair(z: np.ndarray, r_y: np.ndarray, r_x: np.ndarray, b: float, cfg: Config) -> dict:
    """Trade one pair through one period. z[t] is the spread at the close of day t.

    Position decided at close t is held on day t+1, so z[-1] cannot affect any return here.
    Returns daily net returns on gross exposure and counts of trades opened and converged.
    """
    n = len(z)
    pos = np.zeros(n)          # position held during day t
    cur, opened, converged = 0.0, 0, 0
    for t in range(n - 1):
        if cur == 0.0 and abs(z[t]) > cfg.entry:
            cur = -np.sign(z[t])
            opened += 1
        elif cur != 0.0 and z[t] * cur >= 0:      # spread crossed its formation mean
            cur = 0.0
            converged += 1
        pos[t + 1] = cur
    gross = pos * (r_y - b * r_x) / (1.0 + abs(b))
    turn = np.abs(np.diff(pos, prepend=0.0))
    turn[-1] += abs(pos[-1])                       # forced close at the end of the period
    return {"ret": gross - cfg.cost * turn, "gross": gross, "opened": opened, "converged": converged}


def run(rets: pd.DataFrame, cfg: Config = Config(), progress: bool = False) -> dict:
    """Walk through every trading period. Returns daily portfolio returns and diagnostics."""
    logp = np.log1p(rets).cumsum()
    pairs = sector_pairs(rets.columns)
    starts = list(range(cfg.formation, len(rets) - cfg.trading + 1, cfg.trading))
    daily = {p: [] for p in PORTFOLIOS}
    gross = {p: [] for p in PORTFOLIOS}
    periods, pvals = [], []
    for k, s in enumerate(starts):
        f0, t1 = s - cfg.formation, s + cfg.trading
        window = rets.iloc[f0:t1]
        rows = []
        for a_name, b_name in pairs:
            if window[[a_name, b_name]].isna().any().any():
                continue
            y = (logp[a_name].iloc[f0:t1] - logp[a_name].iloc[f0]).to_numpy()
            x = (logp[b_name].iloc[f0:t1] - logp[b_name].iloc[f0]).to_numpy()
            st = formation_stats(y[: cfg.formation], x[: cfg.formation], cfg.eg_lags)
            st.update({"pair": f"{a_name}/{b_name}", "y": y, "x": x, "ra": window[a_name].to_numpy()[cfg.formation:],
                       "rb": window[b_name].to_numpy()[cfg.formation:]})
            rows.append(st)
        usable = [r for r in rows if r["b"] > 0 and r["sd"] > 0]
        for r in usable:
            z = (r["y"][cfg.formation:] - r["a"] - r["b"] * r["x"][cfg.formation:]) / r["sd"]
            r.update(trade_pair(z, r["ra"], r["rb"], r["b"], cfg))
        chosen = {
            "Cointegration": sorted(usable, key=lambda r: r["p"])[: cfg.n_pairs],
            "Distance": sorted(usable, key=lambda r: r["ssd"])[: cfg.n_pairs],
            "All pairs": usable,
        }
        idx = rets.index[s:t1]
        info = {"start": str(idx[0].date()), "n_usable": len(usable),
                "n_pass_5pct": int(sum(r["p"] < 0.05 for r in rows)), "n_tested": len(rows)}
        for name, sel in chosen.items():
            daily[name].append(pd.Series(np.mean([r["ret"] for r in sel], axis=0), index=idx))
            gross[name].append(pd.Series(np.mean([r["gross"] for r in sel], axis=0), index=idx))
            info[name] = {"pairs": [r["pair"] for r in sel], "opened": int(sum(r["opened"] for r in sel)),
                          "converged": int(sum(r["converged"] for r in sel)),
                          "traded_pairs": int(sum(r["opened"] > 0 for r in sel))}
        periods.append(info)
        pvals.append({r["pair"]: r["p"] for r in rows})
        if progress and k % 20 == 0:
            print(f"  period {k + 1}/{len(starts)} from {idx[0].date()}", flush=True)
    return {"net": pd.DataFrame({p: pd.concat(v) for p, v in daily.items()}),
            "gross": pd.DataFrame({p: pd.concat(v) for p, v in gross.items()}),
            "periods": periods, "pvals": pd.DataFrame(pvals), "cfg": cfg}


def persistence(res: dict, level: float = 0.05) -> dict:
    """Is a pair picked by the screen still cointegrated in the next, non-overlapping window?

    Compares the pass rate of the selected pairs with the pass rate of all pairs, both measured
    on the window that starts when the formation window ends.
    """
    cfg, pv = res["cfg"], res["pvals"]
    gap = cfg.formation // cfg.trading
    hit, base, n = [], [], 0
    for k in range(len(pv) - gap):
        later = pv.iloc[k + gap]
        sel = [p for p in res["periods"][k]["Cointegration"]["pairs"] if p in later.index and pd.notna(later[p])]
        if sel:
            hit.append(float((later[sel] < level).mean()))
            base.append(float((later.dropna() < level).mean()))
            n += 1
    return {"selected_pass_rate": float(np.mean(hit)), "all_pairs_pass_rate": float(np.mean(base)), "periods": n}


def sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(np.sqrt(252) * r.mean() / sd) if sd > 0 else float("nan")


def newey_west_t(d: np.ndarray, lags: int = 21) -> float:
    n = len(d)
    e = d - d.mean()
    v = float(e @ e) / n
    for k in range(1, lags + 1):
        v += 2.0 * (1.0 - k / (lags + 1.0)) * float(e[k:] @ e[:-k]) / n
    return float(d.mean() / np.sqrt(v / n))


def summarize(r: pd.Series) -> dict:
    x = r.to_numpy()
    cum = (1 + r).cumprod()
    return {"ann_return": float(x.mean() * 252), "ann_vol": float(x.std(ddof=1) * np.sqrt(252)),
            "sharpe": sharpe(x), "t": newey_west_t(x), "max_drawdown": float((cum / cum.cummax() - 1).min())}
