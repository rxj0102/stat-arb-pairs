# Cointegration-Based Statistical Arbitrage Engine

[![CI](https://github.com/rxj0102/stat-arb-pairs/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/stat-arb-pairs/actions/workflows/ci.yml)
### Kalman Filter Hedge Ratio · OU Process · Regime Detection

A pairs trading research system that screens the S&P 500 universe for cointegrated pairs, fits Ornstein-Uhlenbeck dynamics, and generates mean-reversion signals using a dynamic Kalman-filter hedge ratio - with full out-of-sample backtesting and regime-aware risk management.

---

## Architecture

```
stat-arb-pairs/
├── universe/           # Pair selection & cointegration screening
│   ├── sector_map.py         # Static GICS sector → ticker mapping (5 sectors, ~20 tickers each)
│   └── cointegration_screen.py  # Engle-Granger + Johansen tests, rolling stability check
├── models/             # Statistical models
│   ├── ou_model.py           # OU process MLE: κ, θ, σ and half-life
│   └── kalman_hedge.py       # Kalman filter for dynamic β + rolling OLS benchmark
├── signals/            # Trading signal generation
│   └── zscore_signals.py     # Z-score entry/exit/stop-loss rules
├── backtest/           # Vectorized backtesting engine
│   └── engine.py             # Daily P&L, transaction costs, Sharpe/Sortino/MaxDD
├── risk/               # Portfolio risk management
│   └── regime_filter.py      # Rolling cointegration regime check + position caps
├── outputs/            # Generated charts & CSV
├── notebooks/
│   └── stat_arb_walkthrough.ipynb  # End-to-end interactive walkthrough
├── run_pipeline.py     # Main entry point
└── requirements.txt
```

---

## Corrections

- **Selection bias removed.** Cointegrated pairs were screened on the full sample, including the period later reported as out-of-sample. Pairs are now selected on the first 60% of dates only, the same split `run_pipeline.py` uses, so out-of-sample Sharpe ratios are genuinely out of sample.
- **Results regenerated.** The earlier README table showed the five pairs with the highest out-of-sample Sharpe (about 0.65 to 1.0) from the full-sample screen, which was optimistic twice over. It is replaced by statistics across every selected pair from a fresh run.
- **Stale cache.** The pipeline reused `data/cointegrated_pairs.csv` whenever it existed, whatever the dates or selection window, so the old full-sample screen kept being loaded. The cache file name now includes the sample dates and selection window.
- **Hedge ratio timing.** Day-t spread returns used the Kalman hedge ratio estimated from day-t prices. The backtest now uses the previous day's hedge ratio (covered by a test). The effect on results was negligible.
- An empty price download now raises instead of silently writing an empty pair list.

## Methodology

### 1. Universe Construction
- **Sectors**: Information Technology, Financials, Health Care, Consumer Staples, Energy (S&P 500 GICS groupings)
- ~190 tickers across 5 sectors → ~855 within-sector pairs tested
- Data: `yfinance` adjusted close prices, 5-year history (2019–2024)

### 2. Cointegration Screening
Two independent tests must pass:

| Test | Threshold |
|------|-----------|
| **Engle-Granger** ADF on OLS spread residuals | p-value < 0.05 |
| **Johansen** trace statistic rank test | 5% critical value |

**Rolling stability**: Engle-Granger re-run on every 180-day rolling window (step 20 days). Pairs where cointegration fails in >30% of windows are flagged as unstable and excluded.

### 3. Ornstein-Uhlenbeck Process
The spread `X_t = S1_t - β·S2_t` follows:

```
dX_t = κ(θ - X_t)dt + σ dW_t
```

Parameters estimated via **closed-form MLE** on the exact discrete transition density. Half-life filter: only pairs with **5 ≤ HL ≤ 60 days** are traded (fast enough to revert, slow enough to trade).

### 4. Kalman Filter for Dynamic Hedge Ratio

**State space model** (scalar):
```
State:        β_t = β_{t-1} + w_t,      w_t ~ N(0, Q)
Observation:  S1_t = β_t · S2_t + v_t,  v_t ~ N(0, R)
```

The Kalman filter produces a sequential, adaptive estimate of `β_t` that:
- Requires no look-ahead (causal, implementable in real time)
- Adapts faster than rolling OLS to structural breaks in the hedge ratio
- Provides uncertainty estimates (state variance `P_t`)

Delta parameter `δ = 1e-4` controls the ratio `Q/R`, tuned to balance stability vs. responsiveness.

### 5. Signal Generation

```
Spread    = S1_t - β_t^{Kalman} · S2_t
Z-score   = (Spread - μ_{60d}) / σ_{60d}

Entry:     |z| > 1.5  →  long spread if z < -1.5; short if z > 1.5
Exit:      |z| < 0.5
Stop-loss: |z| > 3.0  →  immediate exit
```

### 6. Backtesting
- **Transaction costs**: 5 bps one-way per leg (10 bps round-trip)
- **Split**: 60% in-sample (model fitting only), 40% out-of-sample (live trading simulation)
- **Metrics**: Sharpe, Sortino, max drawdown, hit rate, avg P&L per trade

### 7. Risk Management
- **Regime detection**: Engle-Granger re-tested every 5 trading days on trailing 60 days. If `p > 0.10`, all positions in that pair are closed ("paused") until cointegration resumes.
- **Position cap**: No single pair exceeds 10% of total portfolio risk allocation (volatility-weighted).

---

## Results (out-of-sample, all selected pairs)

<!-- results:start (generated by scripts/render_readme.py, do not edit by hand) -->
Sample 2019-01-01 to 2024-01-01. Pairs screened on the first 60% of dates; out-of-sample trading 2021-12-29 to 2023-12-29 (504 days), 5 bps one-way cost per leg.

| Statistic | Value |
|---|---|
| Pairs passing the in-sample screen | 57 |
| Median OU half-life | 18.4 days |
| Mean OOS Sharpe across pairs | -0.37 |
| Median OOS Sharpe across pairs | -0.38 |
| Pairs with positive OOS Sharpe | 23% |
| Range of pair OOS Sharpe | -1.74 to 1.13 |
| Equal-weight portfolio of all pairs: Sharpe | -1.45 |
| Equal-weight portfolio: annual return / volatility | -4.72% / 3.26% |
| Equal-weight portfolio: max drawdown | -10.32% |
<!-- results:end -->

Every pair that passed the in-sample cointegration and half-life screen is included; none is chosen by its out-of-sample result. On this sample the strategy loses money after costs: most pairs have a negative out-of-sample Sharpe, and so does the equal-weight portfolio. Per-pair results are in `outputs/pair_performance_summary.csv` and daily portfolio returns in `results/portfolio_oos_returns.csv`. The equity-curve plots in `outputs/` show the five pairs with the highest out-of-sample Sharpe, picked after the fact, so they illustrate the mechanics and are not evidence of an edge.

Prices are frozen in `data/snapshot/` so the run is repeatable; delete the file or set `STAT_ARB_REFRESH_DATA=1` to download again. CI checks that this block matches `results/summary.json`.

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (downloads data, screens pairs, backtests, saves outputs)
python run_pipeline.py --start 2019-01-01 --end 2024-01-01

# Or run interactively
jupyter notebook notebooks/stat_arb_walkthrough.ipynb
```

Outputs generated:
- `data/cointegrated_pairs_<start>_<end>_is60.csv` - screened pairs with test stats & half-life
- `results/summary.json` - out-of-sample statistics across all selected pairs
- `outputs/pair_performance_summary.csv` - full performance table
- `outputs/top5_pairs_equity_curves.png` - OOS equity curves of the five best pairs by OOS Sharpe (chosen after the fact)
- `outputs/kalman_vs_ols_beta.png` - Kalman vs rolling OLS comparison

---

## What this project implements

- **Built end-to-end statistical arbitrage engine** in Python screening within-sector S&P 500 pairs using Engle-Granger and Johansen cointegration tests with rolling stability validation
- **Implemented closed-form OU MLE** to estimate mean-reversion speed κ, long-run mean θ, and volatility σ; filtered pairs to half-lives of 5–60 days for tradeable mean-reversion dynamics
- **Developed Kalman filter state-space model** for adaptive hedge ratio estimation, compared against a 60-day rolling OLS hedge ratio; vectorized backtest reports Sharpe, Sortino, drawdown and hit rate per pair
- **Engineered regime-detection risk layer** using rolling Engle-Granger re-tests that pauses pairs during cointegration breakdowns
- **Produced clean IS/OOS validation framework** (60/40 split) with 5bps transaction costs, generating Sharpe, Sortino, max drawdown, and hit rate metrics across full pair universe

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `yfinance` | Market data download |
| `statsmodels` | Engle-Granger, Johansen, ADF tests |
| `numpy / scipy` | Numerical computing, MLE |
| `pandas` | Time series manipulation |
| `matplotlib / seaborn` | Visualisation |
| `pykalman` | Kalman filter reference (manual implementation used) |
| `tqdm` | Progress bars |

## Reproducibility

### Data availability

All inputs come from free, public sources. No paid data or API keys are required unless noted.

- Yahoo Finance via the yfinance package (free, no key)
- Files shipped in the `data/` folder of this repository
- `data/snapshot/close_2019-01-01_2024-01-01.csv`: the adjusted closes behind the Results table, downloaded on 2026-11-10

### Computational requirements

- Python 3.12.3. Every dependency version is pinned in `uv.lock`.
- Checked on Linux x86_64 with 2 CPU cores and 8 GB RAM. No GPU is needed.

```bash
pip install uv     # once
uv sync            # installs the exact pinned environment
uv run pytest -q   # runs the test suite
```

### Runtime

The test suite (13 tests) finishes in about 7 seconds on the machine above. `python run_pipeline.py` takes about 2 minutes on a first run (downloads into the snapshot) and under a minute from the snapshot.

```bash
make reproduce   # run the pipeline, write results/ and outputs/, refresh the README Results block
make check       # confirm the README matches results/summary.json (also run in CI)
```

### Citation

Citation metadata is in `CITATION.cff`. GitHub turns it into a Cite this repository button on the repo page.

