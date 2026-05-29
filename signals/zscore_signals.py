"""
Z-score based entry/exit signal generation on Kalman-filtered spread.
"""

import numpy as np
import pandas as pd


def compute_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score of the spread."""
    mu = spread.rolling(window).mean()
    sigma = spread.rolling(window).std()
    z = (spread - mu) / sigma
    z.name = "zscore"
    return z


def generate_signals(
    zscore: pd.Series,
    entry_z: float = 1.5,
    exit_z: float = 0.5,
    stop_z: float = 3.0,
) -> pd.Series:
    """
    Generate position signals: +1 (long spread), -1 (short spread), 0 (flat).

    Entry  : |z| > entry_z
    Exit   : |z| < exit_z
    Stop   : |z| > stop_z  → immediate exit
    """
    z = zscore.values
    n = len(z)
    pos = np.zeros(n)

    for i in range(1, n):
        if np.isnan(z[i]):
            pos[i] = 0
            continue

        current = pos[i - 1]

        # Stop-loss: exit immediately
        if abs(z[i]) > stop_z:
            pos[i] = 0
            continue

        # Exit condition
        if current != 0 and abs(z[i]) < exit_z:
            pos[i] = 0
            continue

        # Entry
        if current == 0:
            if z[i] < -entry_z:
                pos[i] = 1    # long spread (buy S1, short S2)
            elif z[i] > entry_z:
                pos[i] = -1   # short spread
            else:
                pos[i] = 0
        else:
            pos[i] = current  # hold

    return pd.Series(pos, index=zscore.index, name="position")
