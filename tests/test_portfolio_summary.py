"""The headline summary must cover every selected pair, not a ranked subset."""
import numpy as np
import pandas as pd

import run_pipeline as rp


class _BT:
    def __init__(self, pnl):
        self.pnl = pnl


def test_summary_uses_all_pairs():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2023-01-01", periods=250)
    results = [
        {"ticker1": f"A{j}", "ticker2": f"B{j}", "OOS_sharpe": s,
         "bt_oos": _BT(pd.Series(rng.normal(0, 0.01, 250), index=idx))}
        for j, s in enumerate([1.0, -0.5, 0.2, -1.0])
    ]
    out = rp.portfolio_summary(results)
    assert out["n_pairs"] == 4
    assert out["pair_oos_sharpe_mean"] == round(np.mean([1.0, -0.5, 0.2, -1.0]), 3)
    assert out["pair_oos_sharpe_share_positive"] == 0.5
    assert out["portfolio_oos_max_drawdown"] <= 0
