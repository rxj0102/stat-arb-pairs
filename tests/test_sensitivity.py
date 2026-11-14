"""run_pair must honour the grid settings it is given."""
import numpy as np
import pandas as pd

import run_pipeline as rp


def _prices(n=700, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    common = 100 + np.cumsum(rng.normal(0, 1, n))
    spread = np.zeros(n)
    for t in range(1, n):  # AR(1) spread, half-life about 13 days
        spread[t] = 0.95 * spread[t - 1] + rng.normal(0, 1.0)
    a = common + spread
    b = common
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_entry_threshold_changes_trading():
    px = _prices()
    lo = rp.run_pair("A", "B", px, entry_z=1.0)
    hi = rp.run_pair("A", "B", px, entry_z=2.5)
    assert lo is not None and hi is not None
    assert lo["bt_oos"].total_trades >= hi["bt_oos"].total_trades
