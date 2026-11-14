# Cointegration-Based Statistical Arbitrage Engine

[![CI](https://github.com/rxj0102/stat-arb-pairs/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/stat-arb-pairs/actions/workflows/ci.yml)
### Kalman Filter Hedge Ratio · OU Process · Regime Detection

A pairs trading research system that screens Ken French industry portfolios (free open data) for cointegrated pairs, fits Ornstein-Uhlenbeck dynamics, and generates mean-reversion signals using a dynamic Kalman-filter hedge ratio - with full out-of-sample backtesting and regime-aware risk management.

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

- **Selection bias removed.** Cointegrated pairs were screened on the full sample, including the period later reported as out-of-sample. Pairs are now selected on the first 60% of dates only, the same split `run_pipeline.py` uses, so out-of-sample Sharpe ratios are genuinely out of sample. `data/cointegrated_pairs_2019-01-01_2024-01-01_is60.csv` and `outputs/` have been regenerated with the fix on open data.
- An empty price download now raises instead of silently writing an empty pair list.
- **Same-day hedge ratio.** Daily spread P&L used the Kalman beta estimated from that day's prices. It now uses the prior day's hedge ratio, the one actually held. Covered by tests.
- **Cointegration pause.** A failed rolling cointegration check paused a pair for one step only; the pause now holds until a later check passes.
- **Per-pair cap.** The position cap now uses trailing volatility and is applied across all selected pairs, which is how the portfolio rows in Results are built.
- **Data.** Yahoo Finance was replaced by Ken French 49 industry portfolios (open data), so pairs are industries rather than single stocks. Results above were regenerated on 2026-11-14 after all of these changes.

## Methodology

### 1. Universe Construction
- **Universe**: Ken French 49 industry portfolios (value weighted, daily total returns), grouped into 8 broad sectors in `universe/sector_map.py` ('Other' left out). The earlier version used about 100 single S&P 500 stocks from Yahoo Finance; there is no free open source for daily single-stock prices, so industry portfolios are the closest open substitute.
- 48 industries across 8 sectors, 142 within-sector pairs tested
- Data: Ken French Data Library, 2019-01-02 to 2023-12-29 (1,258 days), loaded by `universe/opendata.py`

### 2. Cointegration Screening
Two independent tests must pass:

| Test | Threshold |
|------|-----------|
| **Benjamini-Hochberg** q-value across all pairs tested | reported; optional filter `screen_pairs(fdr=0.10)` |
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

## Results (Out-of-Sample, open data)

Rendered from `results/summary.json`, `outputs/` and `data/` by `scripts/render_readme.py` after `python run_pipeline.py` and `python -m analysis.sensitivity`; CI checks that the block matches. The pair table lists every selected pair, unranked by any in-sample choice, not a top-5 pick.

<!-- results:start (generated by scripts/render_readme.py, do not edit by hand) -->
Sample 2019-01-01 to 2024-01-01. Pairs screened on the first 60% of dates; out-of-sample trading 2021-12-29 to 2023-12-29 (504 days), 5 bps one-way cost per leg.

Screen: 9 of 142 industry pairs pass Engle-Granger at 5% with a half-life in range (1 also pass Johansen, 0 pass the rolling stability filter). About 7 false positives are expected by chance at that level. Benjamini-Hochberg q-values: 0 pairs below 0.10; the best pair (Cnstr/BldMt, p = 0.0015) has q = 0.21.

| Pair | Sector | Half-life | IS Sharpe | OOS Sharpe | Max DD | Hit rate | OOS trades |
|---|---|---|---|---|---|---|---|
| Mines/Boxes | Materials | 21d | 0.58 | 0.73 | -8.5% | 33% | 15 |
| Ships/FabPr | Industrials | 28d | 0.31 | 0.02 | -7.1% | 67% | 6 |
| Mach/Cnstr | Industrials | 20d | 0.30 | -0.01 | -4.0% | 29% | 7 |
| Beer/Hshld | Consumer Staples | 15d | -0.64 | -0.05 | -2.0% | 50% | 4 |
| Insur/Fin | Financials | 18d | -0.73 | -0.20 | -6.7% | 80% | 5 |
| Softw/LabEq | Information Technology | 16d | -0.96 | -0.25 | -8.5% | 15% | 13 |
| Cnstr/BldMt | Industrials | 13d | -1.02 | -0.55 | -6.0% | 36% | 14 |
| Mach/ElcEq | Industrials | 37d | 0.11 | -0.65 | -6.1% | 11% | 9 |
| PerSv/Meals | Consumer Discretionary | 19d | 0.76 | -0.84 | -4.1% | 11% | 9 |

| Statistic | Value |
|---|---|
| Mean / median OOS Sharpe across pairs | -0.20 / -0.20 |
| Pairs with positive OOS Sharpe | 22% |
| Equal-weight portfolio of all pairs: Sharpe, annual return, max drawdown | -0.07, -0.13%, -1.36% |
| Inverse-vol portfolio capped at 10% per pair: Sharpe, annual return, max drawdown | 0.02, 0.02%, -1.44% |
| Sensitivity grid (12 settings of z-window and entry z): capped-portfolio Sharpe range | -1.01 to 0.37, positive in 6 |
| Break-momentum study: OOS break events | 0 |
<!-- results:end -->

There is no edge here. The screen finds about as many pairs as chance would, the out-of-sample Sharpe ratios centre below zero, and the sensitivity grid splits evenly between positive and negative portfolio Sharpe ratios, with every value small, so no setting stands out as a real effect. Industry portfolios are diversified baskets, and their spreads mean revert less reliably than the single-stock pairs this pipeline was designed for. Half-life in the table is the OU estimate from the backtest; the screening file reports slightly different values from its own OLS spread.

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (downloads Ken French data, screens pairs, backtests, saves outputs)
python run_pipeline.py --start 2019-01-01 --end 2024-01-01

# Or run interactively
jupyter notebook notebooks/stat_arb_walkthrough.ipynb
```

Outputs generated:
- `data/cointegrated_pairs_<start>_<end>_is60.csv` - screened pairs with test stats and half-life
- `data/all_pair_tests.csv` - every pair tested, with Benjamini-Hochberg q-values
- `results/summary.json`, `outputs/portfolio_summary.json` - all-pairs portfolio statistics
- `outputs/sensitivity_grid.csv` - OOS results across z-window and entry settings
- `outputs/pair_performance_summary.csv` - full performance table
- `outputs/top5_pairs_equity_curves.png` - OOS equity curves
- `outputs/kalman_vs_ols_beta.png` - Kalman vs rolling OLS comparison

---

## What this project implements

- **Built end-to-end statistical arbitrage engine** in Python screening within-sector pairs (Ken French 49 industry portfolios on open data) using Engle-Granger and Johansen cointegration tests with rolling stability validation
- **Implemented closed-form OU MLE** to estimate mean-reversion speed κ, long-run mean θ, and volatility σ; filtered pairs to half-lives of 5–60 days for tradeable mean-reversion dynamics
- **Developed Kalman filter state-space model** for adaptive hedge ratio estimation, compared against a 60-day rolling OLS hedge ratio; vectorized backtest reports Sharpe, Sortino, drawdown and hit rate per pair
- **Engineered regime-detection risk layer** using rolling Engle-Granger re-tests that pauses pairs during cointegration breakdowns
- **Produced clean IS/OOS validation framework** (60/40 split) with 5bps transaction costs, generating Sharpe, Sortino, max drawdown, and hit rate metrics across full pair universe

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `universe/opendata.py` | Ken French Data Library download and cache |
| `statsmodels` | Engle-Granger, Johansen, ADF tests |
| `numpy / scipy` | Numerical computing, MLE |
| `pandas` | Time series manipulation |
| `matplotlib / seaborn` | Visualisation |
| `pykalman` | Kalman filter reference (manual implementation used) |
| `tqdm` | Progress bars |

## Data snapshot

`data/open_data_manifest.json` records every open data file behind the results above (SHA-256, rows, first and last date). Ken French and FRED occasionally revise history; after a fresh download, check whether the inputs still match with:

```bash
python -c "from universe.opendata import verify_manifest; print(verify_manifest('data/open_data_manifest.json'))"
```
