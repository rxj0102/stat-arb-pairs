# Cointegration-Based Statistical Arbitrage Engine

[![CI](https://github.com/rxj0102/stat-arb-pairs/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/stat-arb-pairs/actions/workflows/ci.yml)
### Kalman Filter Hedge Ratio · OU Process · Regime Detection

A production-grade pairs trading system that screens the S&P 500 universe for cointegrated pairs, fits Ornstein-Uhlenbeck dynamics, and generates mean-reversion signals using a dynamic Kalman-filter hedge ratio — with full out-of-sample backtesting and regime-aware risk management.

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

- **Selection bias removed.** Cointegrated pairs were screened on the full sample, including the period later reported as out-of-sample. Pairs are now selected on the first 60% of dates only, the same split `run_pipeline.py` uses, so out-of-sample Sharpe ratios are genuinely out of sample. `data/cointegrated_pairs.csv` and `outputs/` predate this change; delete the CSV and rerun `python run_pipeline.py` to refresh them.
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

## Results — Top 5 Pairs (Out-of-Sample, 2019–2024)

*60% in-sample fit / 40% OOS trading. 5bps one-way transaction costs per leg.*

| Pair | Sector | Half-Life | IS Sharpe | OOS Sharpe | Max DD | Hit Rate |
|------|--------|-----------|-----------|------------|--------|----------|
| JNJ/TMO | Health Care | 26d | -0.16 | **1.01** | -8.8% | 32% |
| PM/MO | Consumer Staples | 33d | -0.51 | **0.99** | -9.3% | 22% |
| PG/COST | Consumer Staples | 24d | -0.68 | **0.90** | -3.3% | 32% |
| MSFT/AMD | Info Technology | 28d | -0.53 | **0.86** | -40.3% | 27% |
| JNJ/HUM | Health Care | 32d | -0.81 | **0.65** | -4.8% | 21% |

*Warning: this table predates the selection fix and is optimistic twice over: pairs were screened on the full sample, and these are the five pairs with the highest OOS Sharpe, i.e. ranked on the test result. Treat it as a pipeline illustration, not evidence of an edge; a fair summary is the average OOS Sharpe across all selected pairs from a fresh run.*

*Note: Negative IS Sharpe is expected — the strategy trains only on cointegration structure, not signal profitability. The OOS forward-test is the true performance measure.*

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
- `data/cointegrated_pairs.csv` — screened pairs with test stats & half-life
- `outputs/pair_performance_summary.csv` — full performance table
- `outputs/top5_pairs_equity_curves.png` — OOS equity curves
- `outputs/kalman_vs_ols_beta.png` — Kalman vs rolling OLS comparison

---

## Resume Bullets

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
