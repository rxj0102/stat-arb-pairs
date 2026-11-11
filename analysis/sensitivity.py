"""
Out-of-sample sensitivity of the pairs book to its two main settings.

Runs the full per-pair pipeline (Kalman hedge, z-score signals, regime
overlay, costs) for every combination of entry threshold and z-score
window, on the pairs selected in sample, and reports the capped portfolio
OOS Sharpe for each. Nothing is chosen from this grid: the defaults stay
entry 1.5 and window 60. The point is to show whether the result depends
on a lucky setting.

Usage:
    python -m analysis.sensitivity
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ENTRY_GRID = (1.0, 1.5, 2.0, 2.5)
WINDOW_GRID = (30, 60, 90)


def run_grid(pairs: pd.DataFrame, prices: pd.DataFrame,
             entry_grid=ENTRY_GRID, window_grid=WINDOW_GRID) -> pd.DataFrame:
    from run_pipeline import portfolio_summary, run_pair

    rows = []
    for w in window_grid:
        for e in entry_grid:
            results = []
            for _, row in pairs.iterrows():
                t1, t2 = row["ticker1"], row["ticker2"]
                if t1 in prices.columns and t2 in prices.columns:
                    res = run_pair(t1, t2, prices, entry_z=e, z_window=w)
                    if res:
                        results.append(res)
            if not results:
                continue
            summ = portfolio_summary(results)
            rows.append({"z_window": w, "entry_z": e, **summ})
            print(f"window={w:>3} entry={e:.1f}  portfolio OOS Sharpe={summ['portfolio_oos_sharpe']:+.2f}")
    return pd.DataFrame(rows)


def main() -> None:
    from universe.cointegration_screen import download_prices

    out = Path("outputs")
    pairs = pd.read_csv("data/cointegrated_pairs.csv")
    tickers = sorted(set(pairs.ticker1) | set(pairs.ticker2))
    prices = download_prices(tickers, start="2019-01-01", end="2024-01-01")
    grid = run_grid(pairs, prices)
    grid.to_csv(out / "sensitivity_grid.csv", index=False)

    piv = grid.pivot(index="z_window", columns="entry_z", values="portfolio_oos_sharpe")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(piv.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [f"{c:.1f}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_xlabel("entry |z|")
    ax.set_ylabel("z-score window (days)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Capped portfolio OOS Sharpe, all selected pairs")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "sensitivity_grid.png", dpi=150)
    print("Saved outputs/sensitivity_grid.csv and outputs/sensitivity_grid.png")


if __name__ == "__main__":
    main()
