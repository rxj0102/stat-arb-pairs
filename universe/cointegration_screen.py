"""
Pair universe screening via Engle-Granger and Johansen cointegration tests.
Also computes rolling cointegration stability and OU half-life.
"""

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from tqdm import tqdm

from universe.sector_map import SECTOR_TICKERS

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices; drop tickers with >5% missing data."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
    threshold = 0.05 * len(prices)
    prices = prices.dropna(axis=1, thresh=int(len(prices) - threshold))
    prices = prices.ffill().dropna()
    return prices


# ---------------------------------------------------------------------------
# Cointegration tests
# ---------------------------------------------------------------------------

def engle_granger_test(s1: pd.Series, s2: pd.Series) -> dict:
    """Return EG test statistic, p-value, and OLS hedge ratio."""
    score, pvalue, _ = coint(s1, s2)
    # OLS hedge ratio
    beta = np.polyfit(s2, s1, 1)[0]
    return {"eg_stat": score, "eg_pval": pvalue, "ols_beta": beta}


def johansen_test(s1: pd.Series, s2: pd.Series, significance: float = 0.05) -> bool:
    """Return True if Johansen rank test indicates at least one cointegrating vector."""
    data = np.column_stack([s1.values, s2.values])
    try:
        result = coint_johansen(data, det_order=0, k_ar_diff=1)
        # trace statistic vs 5% critical value (index 1)
        return bool(result.lr1[0] > result.cvt[0, 1])
    except Exception:
        return False


# ---------------------------------------------------------------------------
# OU half-life
# ---------------------------------------------------------------------------

def compute_halflife(spread: pd.Series) -> float:
    """Estimate OU half-life via OLS regression of spread on its lag."""
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    idx = spread_diff.index
    spread_lag = spread_lag.loc[idx]
    beta = np.polyfit(spread_lag, spread_diff, 1)[0]
    if beta >= 0:
        return np.inf
    return -np.log(2) / beta


# ---------------------------------------------------------------------------
# Rolling cointegration stability
# ---------------------------------------------------------------------------

def rolling_coint_stability(
    s1: pd.Series, s2: pd.Series, window: int = 180, step: int = 20
) -> float:
    """
    Fraction of rolling windows where EG p-value > 0.05 (cointegration fails).
    Returns failure rate in [0, 1].
    """
    n = len(s1)
    fails = 0
    total = 0
    for start in range(0, n - window, step):
        end = start + window
        w1, w2 = s1.iloc[start:end], s2.iloc[start:end]
        try:
            _, pval, _ = coint(w1, w2)
            if pval > 0.05:
                fails += 1
        except Exception:
            fails += 1
        total += 1
    return fails / total if total > 0 else 1.0


# ---------------------------------------------------------------------------
# Main screening pipeline
# ---------------------------------------------------------------------------

def screen_pairs(
    start: str = "2019-01-01",
    end: str = "2024-01-01",
    eg_pval_threshold: float = 0.05,
    halflife_min: int = 5,
    halflife_max: int = 60,
    instability_threshold: float = 0.30,
) -> pd.DataFrame:
    all_tickers = [t for tickers in SECTOR_TICKERS.values() for t in tickers]
    print("Downloading price data...")
    prices = download_prices(all_tickers, start, end)
    available = set(prices.columns)

    records = []

    for sector, tickers in SECTOR_TICKERS.items():
        sector_tickers = [t for t in tickers if t in available]
        pairs = list(itertools.combinations(sector_tickers, 2))
        print(f"\n{sector}: testing {len(pairs)} pairs")

        for t1, t2 in tqdm(pairs, desc=sector[:12], leave=False):
            s1, s2 = prices[t1].dropna(), prices[t2].dropna()
            common = s1.index.intersection(s2.index)
            if len(common) < 252:
                continue
            s1, s2 = s1.loc[common], s2.loc[common]

            eg = engle_granger_test(s1, s2)
            if eg["eg_pval"] >= eg_pval_threshold:
                continue

            joh = johansen_test(s1, s2)

            spread = s1 - eg["ols_beta"] * s2
            hl = compute_halflife(spread)
            if not (halflife_min <= hl <= halflife_max):
                continue

            instability = rolling_coint_stability(s1, s2)

            records.append(
                {
                    "sector": sector,
                    "ticker1": t1,
                    "ticker2": t2,
                    "eg_stat": round(eg["eg_stat"], 4),
                    "eg_pval": round(eg["eg_pval"], 4),
                    "johansen_pass": joh,
                    "ols_beta": round(eg["ols_beta"], 4),
                    "half_life": round(hl, 2),
                    "instability_rate": round(instability, 3),
                    "stable": instability <= instability_threshold,
                }
            )

    df = pd.DataFrame(records).sort_values("eg_pval")
    out_path = DATA_DIR / "cointegrated_pairs.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFound {len(df)} cointegrated pairs → {out_path}")
    return df


if __name__ == "__main__":
    screen_pairs()
