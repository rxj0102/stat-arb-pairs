"""
Kalman Filter for dynamic hedge ratio estimation.

State space:
  x_t = β_t                   (1-D state: hedge ratio)
  y_t = S1_t - β_t * S2_t     (observation: spread as residual)

  State transition:  β_t = β_{t-1} + w_t,   w_t ~ N(0, Q)
  Observation:       S1_t = β_t * S2_t + v_t, v_t ~ N(0, R)
"""

import numpy as np
import pandas as pd


def kalman_hedge_ratio(
    s1: pd.Series,
    s2: pd.Series,
    delta: float = 1e-4,
    Ve: float = 0.001,
) -> tuple[pd.Series, pd.Series]:
    """
    Estimate time-varying hedge ratio β_t via scalar Kalman filter.

    Parameters
    ----------
    s1, s2 : price series (same index)
    delta  : state noise variance (controls how fast β can drift)
    Ve     : observation noise variance

    Returns
    -------
    beta  : pd.Series of hedge ratios
    spread: pd.Series  S1 - beta * S2
    """
    n = len(s1)
    y = s1.values
    x = s2.values

    # Initialise
    beta = np.zeros(n)
    P = np.zeros(n)       # state variance
    P[0] = 1.0
    beta[0] = y[0] / x[0] if x[0] != 0 else 1.0

    Vw = delta / (1 - delta)   # state noise variance (ratio form common in finance)

    for t in range(1, n):
        # Predict
        beta_pred = beta[t - 1]
        P_pred = P[t - 1] + Vw

        # Innovation
        H = x[t]
        innov = y[t] - H * beta_pred
        S = H * P_pred * H + Ve   # innovation variance

        # Kalman gain
        K = P_pred * H / S

        # Update
        beta[t] = beta_pred + K * innov
        P[t] = (1 - K * H) * P_pred

    beta_series = pd.Series(beta, index=s1.index, name="kalman_beta")
    spread_series = s1 - beta_series * s2
    spread_series.name = "spread"
    return beta_series, spread_series


def rolling_ols_beta(
    s1: pd.Series, s2: pd.Series, window: int = 60
) -> pd.Series:
    """Rolling OLS hedge ratio for comparison with Kalman."""
    betas = []
    for i in range(len(s1)):
        if i < window - 1:
            betas.append(np.nan)
        else:
            y = s1.iloc[i - window + 1 : i + 1].values
            x = s2.iloc[i - window + 1 : i + 1].values
            beta = np.polyfit(x, y, 1)[0]
            betas.append(beta)
    return pd.Series(betas, index=s1.index, name="ols_beta")
