"""
Main pipeline: screen pairs → fit models → backtest → generate outputs.

Usage:
    python run_pipeline.py [--start 2019-01-01] [--end 2024-01-01]
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from universe.cointegration_screen import screen_pairs, download_prices
from models.ou_model import fit_ou_mle
from models.kalman_hedge import kalman_hedge_ratio, rolling_ols_beta
from signals.zscore_signals import compute_zscore, generate_signals
from backtest.engine import run_backtest, split_insample_oos, BacktestResult
from risk.regime_filter import check_regime, apply_position_cap
from analysis.break_momentum import break_dates, forward_trend_returns, summarise

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Full pair pipeline
# ---------------------------------------------------------------------------

def run_pair(
    t1: str, t2: str, prices: pd.DataFrame, train_frac: float = 0.60,
    entry_z: float = 1.5, z_window: int = 60,
) -> dict | None:
    try:
        s1 = prices[t1].dropna()
        s2 = prices[t2].dropna()
        common = s1.index.intersection(s2.index)
        s1, s2 = s1.loc[common], s2.loc[common]
        if len(s1) < 300:
            return None

        # Split IS / OOS on the same calendar date used for pair selection
        split_date = prices.index[int(len(prices) * train_frac)]
        split = int(s1.index.searchsorted(split_date))
        s1_is, s1_oos = s1.iloc[:split], s1.iloc[split:]
        s2_is, s2_oos = s2.iloc[:split], s2.iloc[split:]

        # ------------------------------------------------------------------
        # In-sample: fit Kalman + OU on IS data only
        # OU half-life is computed on the OLS static spread (not Kalman), since
        # the Kalman filter absorbs spread dynamics into β, leaving near-white-noise residuals.
        # ------------------------------------------------------------------
        beta_is, spread_is = kalman_hedge_ratio(s1_is, s2_is)
        ols_beta_is = np.polyfit(s2_is.values, s1_is.values, 1)[0]
        ols_spread_is = s1_is - ols_beta_is * s2_is
        ou_params = fit_ou_mle(ols_spread_is.dropna())
        if not (5 <= ou_params.half_life <= 60):
            return None

        # ------------------------------------------------------------------
        # OOS: use Kalman filter (sequential, no look-ahead)
        # ------------------------------------------------------------------
        beta_oos, spread_oos = kalman_hedge_ratio(s1_oos, s2_oos,
                                                   delta=1e-4, Ve=0.001)

        # Rolling OLS beta for comparison (full series)
        ols_beta_full = rolling_ols_beta(s1, s2, window=60)

        # Signals
        z_is = compute_zscore(spread_is, window=z_window)
        pos_is = generate_signals(z_is, entry_z=entry_z)
        pos_is = check_regime(s1_is, s2_is, pos_is)

        z_oos = compute_zscore(spread_oos, window=z_window)
        pos_oos = generate_signals(z_oos, entry_z=entry_z)
        pos_oos = check_regime(s1_oos, s2_oos, pos_oos)

        # Backtests
        bt_is = run_backtest(s1_is, s2_is, beta_is, pos_is, t1, t2)
        bt_oos = run_backtest(s1_oos, s2_oos, beta_oos, pos_oos, t1, t2)

        # Kalman beta full for plotting
        beta_full, _ = kalman_hedge_ratio(s1, s2)

        return {
            "ticker1": t1,
            "ticker2": t2,
            "half_life": ou_params.half_life,
            "ou_kappa": ou_params.kappa,
            "ou_sigma": ou_params.sigma,
            "IS_sharpe": bt_is.sharpe,
            "OOS_sharpe": bt_oos.sharpe,
            "IS_sortino": bt_is.sortino,
            "OOS_sortino": bt_oos.sortino,
            "max_dd": bt_oos.max_drawdown,
            "hit_rate": bt_oos.hit_rate,
            "total_trades_oos": bt_oos.total_trades,
            "bt_is": bt_is,
            "bt_oos": bt_oos,
            "ols_beta_full": ols_beta_full,
            "kalman_beta_full": beta_full,
            "s1": s1, "s2": s2,
            "split": split,
            "oos_inputs": (s1_oos, s2_oos, beta_oos, z_oos),
        }
    except Exception as e:
        print(f"  [SKIP] {t1}/{t2}: {e}")
        return None


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_equity_curves(top5: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=False)
    fig.suptitle("Top 5 Pairs — OOS Equity Curves", fontsize=14, fontweight="bold")

    for ax, res in zip(axes, top5):
        cum = (1 + res["bt_oos"].pnl).cumprod() - 1
        ax.plot(cum.index, cum.values * 100, color="steelblue", linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_title(
            f"{res['ticker1']}/{res['ticker2']}  "
            f"Sharpe={res['OOS_sharpe']:.2f}  MaxDD={res['max_dd']:.1%}  Hit={res['hit_rate']:.1%}",
            fontsize=9,
        )
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax.set_ylabel("Cum Return (%)", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def plot_kalman_vs_ols(top5: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=False)
    fig.suptitle("Kalman vs Rolling OLS Hedge Ratio", fontsize=14, fontweight="bold")

    for ax, res in zip(axes, top5):
        ols = res["ols_beta_full"].dropna()
        kal = res["kalman_beta_full"]
        split = res["split"]
        split_date = res["s1"].index[split]

        ax.plot(ols.index, ols.values, label="Rolling OLS (60d)", color="orange",
                linewidth=0.9, alpha=0.85)
        ax.plot(kal.index, kal.values, label="Kalman Filter", color="steelblue",
                linewidth=1.0, alpha=0.9)
        ax.axvline(split_date, color="red", linestyle="--", linewidth=0.8,
                   label="IS/OOS split")
        ax.set_title(f"{res['ticker1']}/{res['ticker2']}  β", fontsize=9)
        ax.set_ylabel("Hedge Ratio β", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def portfolio_summary(results: list[dict], max_weight: float = 0.10) -> dict:
    """Out-of-sample summary across ALL selected pairs, not a ranked subset.

    Pairs are combined with trailing inverse-volatility weights capped at
    ``max_weight`` (risk.regime_filter.apply_position_cap), and the
    distribution of per-pair OOS Sharpe ratios is reported alongside.
    """
    oos = {f"{r['ticker1']}/{r['ticker2']}": r["bt_oos"].pnl for r in results}
    capped = apply_position_cap(oos, max_weight=max_weight)
    port = pd.DataFrame(capped).fillna(0.0).sum(axis=1)
    ann = np.sqrt(252)
    sd = port.std()
    wealth = (1 + port).cumprod()
    sharpes = pd.Series([r["OOS_sharpe"] for r in results], dtype=float)
    return {
        "n_pairs": int(len(results)),
        "oos_start": str(port.index.min().date()),
        "oos_end": str(port.index.max().date()),
        "portfolio_oos_sharpe": round(float(port.mean() / sd * ann) if sd > 0 else 0.0, 3),
        "portfolio_oos_ann_return": round(float(port.mean() * 252), 4),
        "portfolio_oos_max_drawdown": round(float((wealth / wealth.cummax() - 1).min()), 4),
        "pair_oos_sharpe_mean": round(float(sharpes.mean()), 3),
        "pair_oos_sharpe_median": round(float(sharpes.median()), 3),
        "pair_oos_sharpe_share_positive": round(float((sharpes > 0).mean()), 3),
        "max_weight_per_pair": max_weight,
        "cost_bps_one_way_per_leg": 5,
    }


def main(start: str = "2019-01-01", end: str = "2024-01-01", top_n: int = 5) -> None:
    # 1. Screen pairs (or load cache)
    coint_path = DATA_DIR / "cointegrated_pairs.csv"
    if coint_path.exists():
        print(f"Loading cached pairs from {coint_path}")
        pairs_df = pd.read_csv(coint_path)
    else:
        pairs_df = screen_pairs(start=start, end=end)

    # Pairs were selected on the in-sample window only (see screen_pairs), so
    # the out-of-sample results are not contaminated by selection bias.
    candidate_pairs = pairs_df.copy()
    print(f"\n{len(candidate_pairs)} cointegrated pairs to backtest")

    # 2. Download prices
    all_tickers = list(set(candidate_pairs["ticker1"].tolist() + candidate_pairs["ticker2"].tolist()))
    if not all_tickers:
        print("No tickers to download. Exiting.")
        return
    print("Downloading full price history...")
    prices = download_prices(all_tickers, start=start, end=end)

    # 3. Run pair pipeline
    results = []
    for _, row in candidate_pairs.iterrows():
        t1, t2 = row["ticker1"], row["ticker2"]
        if t1 not in prices.columns or t2 not in prices.columns:
            continue
        res = run_pair(t1, t2, prices)
        if res:
            results.append(res)

    if not results:
        print("No valid pairs found after full pipeline. Exiting.")
        return

    # 4. Rank by OOS Sharpe
    results.sort(key=lambda r: r["OOS_sharpe"], reverse=True)
    top5 = results[:top_n]

    # 5. Performance summary CSV
    summary_rows = []
    for r in results:
        summary_rows.append({
            "pair": f"{r['ticker1']}/{r['ticker2']}",
            "half_life": r["half_life"],
            "IS_sharpe": r["IS_sharpe"],
            "OOS_sharpe": r["OOS_sharpe"],
            "IS_sortino": r["IS_sortino"],
            "OOS_sortino": r["OOS_sortino"],
            "max_dd": r["max_dd"],
            "hit_rate": r["hit_rate"],
            "total_trades_oos": r["total_trades_oos"],
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("OOS_sharpe", ascending=False)
    summary_path = OUTPUTS / "pair_performance_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nPerformance summary → {summary_path}")
    print(summary_df.head(10).to_string(index=False))

    # 5b. Portfolio of all selected pairs (the headline number)
    summary = portfolio_summary(results)
    with open(OUTPUTS / "portfolio_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nOOS portfolio of all selected pairs:")
    print(json.dumps(summary, indent=2))

    # 5c. Break-momentum study (OOS only): does a broken pair keep trending?
    events = []
    for r in results:
        s1o, s2o, bo, zo = r["oos_inputs"]
        ev = forward_trend_returns(s1o, s2o, bo, zo, break_dates(s1o, s2o))
        if len(ev):
            ev.insert(0, "pair", f"{r['ticker1']}/{r['ticker2']}")
            events.append(ev)
    events_df = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    events_df.to_csv(OUTPUTS / "break_momentum_events.csv", index=False)
    bm_summary = summarise(events_df)
    bm_summary.to_csv(OUTPUTS / "break_momentum_summary.csv", index=False)
    print("\nBreak-momentum study (OOS, net of 20 bps round trip):")
    print(bm_summary.to_string(index=False))

    # 6. Plots
    plot_equity_curves(top5, OUTPUTS / "top5_pairs_equity_curves.png")
    plot_kalman_vs_ols(top5, OUTPUTS / "kalman_vs_ols_beta.png")

    print("\nPipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()
    main(start=args.start, end=args.end, top_n=args.top_n)
