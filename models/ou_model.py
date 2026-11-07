"""
Ornstein-Uhlenbeck process fitting via closed-form MLE.

dX_t = κ(θ - X_t)dt + σ dW_t

Closed-form MLE: Brown & Smith (2010) / Leung & Li (2015).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class OUParams:
    kappa: float   # mean reversion speed
    theta: float   # long-run mean
    sigma: float   # volatility
    half_life: float  # ln(2)/kappa in days


def fit_ou_mle(spread: pd.Series) -> OUParams:
    """
    Fit OU parameters to a spread series using closed-form MLE.
    Assumes Δt = 1 (daily observations).
    """
    x = spread.values
    n = len(x) - 1
    x0 = x[:-1]
    x1 = x[1:]

    # Sufficient statistics
    sx = x0.sum()
    sy = x1.sum()
    sxx = (x0 ** 2).sum()
    sxy = (x0 * x1).sum()
    syy = (x1 ** 2).sum()

    # MLE for discrete OU (exact transition density)
    denom = n * sxx - sx ** 2
    if abs(denom) < 1e-12:
        return OUParams(kappa=0.0, theta=float(x.mean()), sigma=float(x.std()), half_life=np.inf)

    mu = (sy * sxx - sx * sxy) / denom
    phi = (n * sxy - sx * sy) / denom   # e^{-κΔt}
    phi = np.clip(phi, 1e-8, 1 - 1e-8)

    sigma2_hat = (syy - 2 * phi * sxy + phi ** 2 * sxx - 2 * mu * (sy - phi * sx) + n * mu ** 2 * (1 - phi) ** 2) / n

    # Convert discrete params to continuous
    kappa = -np.log(phi)   # since phi = e^{-κ}
    theta = mu
    # sigma continuous: sigma^2 = sigma2_hat * 2κ / (1 - e^{-2κ})
    factor = 2 * kappa / (1 - np.exp(-2 * kappa)) if kappa > 1e-8 else 1.0
    sigma = np.sqrt(max(sigma2_hat * factor, 1e-12))

    half_life = np.log(2) / kappa if kappa > 0 else np.inf

    return OUParams(kappa=round(kappa, 6), theta=round(theta, 6),
                    sigma=round(sigma, 6), half_life=round(half_life, 2))


def ou_log_likelihood(spread: pd.Series, params: OUParams) -> float:
    """Log-likelihood of spread under fitted OU params."""
    x = spread.values
    kappa, theta, sigma = params.kappa, params.theta, params.sigma
    dt = 1.0
    phi = np.exp(-kappa * dt)
    mu_cond = theta + (x[:-1] - theta) * phi
    var_cond = sigma ** 2 * (1 - phi ** 2) / (2 * kappa) if kappa > 1e-8 else sigma ** 2 * dt
    ll = -0.5 * np.sum(np.log(2 * np.pi * var_cond) + (x[1:] - mu_cond) ** 2 / var_cond)
    return ll
