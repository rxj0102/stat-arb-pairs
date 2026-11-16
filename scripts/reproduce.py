"""Run the rolling study on the frozen data and write results/.

  results/metrics.json        everything the README reports
  results/daily_returns.csv   daily net return of the three portfolios
  results/periods.csv         pairs chosen and trades in each period
  results/figures/growth.png  growth of 1 in each portfolio, before and after costs
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import study  # noqa: E402

ERAS = [("1965-1979", "1965", "1979"), ("1980-1999", "1980", "1999"), ("2000-2012", "2000", "2012"),
        ("2013-2026", "2013", "2026")]
PAIRS = [("Cointegration", "All pairs"), ("Distance", "All pairs"), ("Cointegration", "Distance")]


def rounded(x):
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(f"{float(x):.6g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def figure(net: pd.DataFrame, gross: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
    colors = {"Cointegration": "#2a78d6", "Distance": "#eb6834", "All pairs": "#8a8980"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "xtick.color": muted, "ytick.color": muted,
                         "text.color": ink, "figure.facecolor": surface, "axes.facecolor": surface,
                         "savefig.facecolor": surface, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (df, title) in zip(axes, ((gross, "Before costs"), (net, "After 5 bps per leg"))):
        cum = (1 + df).cumprod()
        for n, c in colors.items():
            ax.plot(cum.index, cum[n], color=c, linewidth=1.5, label=n)
        ax.axhline(1, color=muted, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_ylim(0.55, 1.45)
        ax.set_yticks([0.6, 0.8, 1, 1.25])
        ax.set_yticklabels(["0.6", "0.8", "1", "1.25"])
        ax.minorticks_off()
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_title(title, loc="left", fontsize=10.5)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Growth of 1 on committed capital (log scale)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "growth.png", dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = study.Config()
    rets = study.load()
    res = study.run(rets, cfg, progress=True)
    net, gross, periods = res["net"], res["gross"], res["periods"]
    portfolios = {}
    for p in study.PORTFOLIOS:
        opened = sum(x[p]["opened"] for x in periods)
        portfolios[p] = {"net": study.summarize(net[p]), "gross": study.summarize(gross[p]),
                         "trades_opened": int(opened),
                         "converged_share": float(sum(x[p]["converged"] for x in periods) / opened),
                         "trades_per_pair_period": float(opened / sum(len(x[p]["pairs"]) for x in periods)),
                         "eras": {lab: {"net": study.sharpe(net[p].loc[a:b].to_numpy()),
                                        "gross": study.sharpe(gross[p].loc[a:b].to_numpy())}
                                  for lab, a, b in ERAS}}
    diffs = []
    for a, b in PAIRS:
        for kind, df in (("gross", gross), ("net", net)):
            d = (df[a] - df[b]).to_numpy()
            diffs.append({"a": a, "b": b, "returns": kind, "ann_diff": float(d.mean() * 252),
                          "t": study.newey_west_t(d)})
    passes = np.array([x["n_pass_5pct"] for x in periods])
    tested = np.array([x["n_tested"] for x in periods])
    payload = rounded({
        "sample": {"data_first": str(rets.index[0].date()), "first_day": str(net.index[0].date()),
                   "last_day": str(net.index[-1].date()), "n_days": int(len(net)), "n_periods": len(periods),
                   "years": float(len(net) / 252), "formation": cfg.formation, "trading": cfg.trading,
                   "n_pairs": cfg.n_pairs, "entry": cfg.entry, "cost_bps": cfg.cost * 1e4,
                   "pairs_tested_mean": float(tested.mean())},
        "screen": {"mean_pass_5pct": float(passes.mean()), "expected_by_chance": float(0.05 * tested.mean()),
                   "share_of_periods_above_chance": float((passes > 0.05 * tested).mean())},
        "persistence": study.persistence(res),
        "portfolios": portfolios, "differences": diffs,
    })
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    net.round(7).to_csv(out / "daily_returns.csv", index_label="date")
    pd.DataFrame([{"start": x["start"], "pass_5pct": x["n_pass_5pct"], "tested": x["n_tested"],
                   **{f"{p}: pairs": " ".join(x[p]["pairs"]) for p in ("Cointegration", "Distance")},
                   **{f"{p}: opened": x[p]["opened"] for p in study.PORTFOLIOS},
                   **{f"{p}: converged": x[p]["converged"] for p in study.PORTFOLIOS}} for x in periods]
                 ).to_csv(out / "periods.csv", index=False)
    figure(net, gross, out / "figures")
    for p in study.PORTFOLIOS:
        print(p, "net", {k: round(v, 3) for k, v in portfolios[p]["net"].items()})
        print(" " * len(p), "gross", {k: round(v, 3) for k, v in portfolios[p]["gross"].items()},
              round(portfolios[p]["converged_share"], 3), portfolios[p]["eras"])
    print(payload["screen"], payload["persistence"])
    for d in diffs:
        print(d)


if __name__ == "__main__":
    main()
