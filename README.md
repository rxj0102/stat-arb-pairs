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

- **Selection bias removed.** Pairs are selected on the first 60% of dates only, the same split `run_pipeline.py` uses, so out-of-sample Sharpe ratios are genuinely out of sample.
- **Regime pause fixed.** The cointegration overlay used to zero a position only on the re-test day itself, so a paused pair kept trading on the four days between checks. The pause now holds until a later check passes.
- **Hedge ratio lag.** Daily spread P&L used the Kalman beta updated with that same day's prices. It now uses the beta known at the prior close.
- **Position cap in force.** The 10% per-pair cap was described here but never applied, and it sized pairs with full-sample volatility. It now uses trailing 60 day volatility (shifted one day) and is applied to the combined portfolio.
- **Headline changed from top 5 to all pairs.** The old table ranked pairs by their own OOS Sharpe. The results below cover every selected pair.
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
- **Regime detection**: Engle-Granger re-tested every 5 trading days on trailing 60 days. If `p > 0.10`, all positions in that pair are closed (paused) until cointegration resumes.
- **Position cap**: Pairs are combined with inverse-volatility weights from trailing 60 day P&L (known at the prior close), each capped at 10%. Weights are not renormalised after capping, so any unallocated share stays in cash.

---

## Results (out of sample, all selected pairs)

Produced by `python run_pipeline.py` with the default window (2019-01-01 to 2024-01-01). Pairs are selected on the first 60% of dates; everything below is from the remaining 40% (2021-12-29 to 2023-12-29). Costs are 5 bps one way per leg. Source files: `outputs/portfolio_summary.json`, `outputs/pair_performance_summary.csv`.

| Measure | Value |
|---------|-------|
| Pairs selected in sample | 57 |
| Capped portfolio OOS Sharpe | -0.40 |
| Capped portfolio OOS annual return | -1.55% |
| Capped portfolio OOS max drawdown | -6.27% |
| Mean per-pair OOS Sharpe | -0.28 |
| Median per-pair OOS Sharpe | -0.42 |
| Share of pairs with positive OOS Sharpe | 35.1% |

**Reading this honestly.** Cointegration found in 2019 to 2021 did not translate into profitable mean reversion in 2022 and 2023. The capped portfolio lost money out of sample and about two thirds of pairs had a negative OOS Sharpe. Individual pairs with high OOS Sharpe exist in the CSV, but picking them after the fact is the selection bias this repo now avoids.

### Does a broken pair keep trending?

The Oil Money study by je-suis-tm found that when a fitted FX and crude relationship broke, the deviation kept going, so the break worked as a trend-following entry. `analysis/break_momentum.py` tests the same idea here. At each regime check day where the trailing 60 day Engle-Granger test newly fails (the same rule that pauses trading), the spread is taken in the direction of its prior z-score, entered at that close, and held for h days, net of a 20 bps round trip. OOS segment only. Source: `outputs/break_momentum_summary.csv`.

| Holding period | Break events | Mean net return | t-stat | Hit rate |
|----------------|--------------|-----------------|--------|----------|
| 5 days | 296 | -0.46% | -1.11 | 44.3% |
| 10 days | 288 | -0.66% | -1.02 | 46.9% |
| 20 days | 286 | -0.56% | -0.55 | 45.8% |

No horizon shows momentum. Mean returns are negative and none is significant. The t-stats treat events as independent, and 20 day windows within a pair can overlap, so they overstate significance if anything. On this universe a cointegration break is a reason to stop trading the pair, not an entry signal.

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
- `data/cointegrated_pairs.csv` - screened pairs with test stats & half-life
- `outputs/pair_performance_summary.csv` - full performance table
- `outputs/top5_pairs_equity_curves.png` - OOS equity curves
- `outputs/kalman_vs_ols_beta.png` - Kalman vs rolling OLS comparison
- `outputs/portfolio_summary.json` - capped portfolio and per-pair OOS Sharpe distribution
- `outputs/break_momentum_events.csv`, `outputs/break_momentum_summary.csv` - break-momentum study

---

## What this project implements

- **Built end-to-end statistical arbitrage engine** in Python screening within-sector S&P 500 pairs using Engle-Granger and Johansen cointegration tests with rolling stability validation
- **Implemented closed-form OU MLE** to estimate mean-reversion speed κ, long-run mean θ, and volatility σ; filtered pairs to half-lives of 5–60 days for tradeable mean-reversion dynamics
- **Developed Kalman filter state-space model** for adaptive hedge ratio estimation, compared against a 60-day rolling OLS hedge ratio; vectorized backtest reports Sharpe, Sortino, drawdown and hit rate per pair
- **Engineered regime-detection risk layer** using rolling Engle-Granger re-tests that pauses pairs during cointegration breakdowns
- **Produced clean IS/OOS validation framework** (60/40 split) with 5bps transaction costs, generating Sharpe, Sortino, max drawdown, and hit rate metrics across full pair universe
- **Tested break momentum** out of sample: whether a pair that loses cointegration keeps trending, as in the Oil Money FX study. It does not on this universe

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
