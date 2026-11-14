"""Synthetic market data for the tests (no internet needed)."""
import numpy as np
import pandas as pd

N = 500


def random_walk_data(n=N, seed=0):
    """Plain random walks: few or no signals."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))), index=idx)
    gld = pd.Series(50 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n))), index=idx)
    fx = pd.Series(1.3 + np.cumsum(rng.normal(0, 0.002, n)), index=idx)
    history = pd.DataFrame({"SPY": spy, "GLD": gld, "GBPUSD": fx})
    return spy / gld, history


def curvy_data(n=N, seed=0):
    """The log ratio follows a clean sine wave (period 100 bars) plus a little noise,
    so there are clear turning points and the strategy MUST trade."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    log_ratio = 0.3 * np.sin(2 * np.pi * np.arange(n) / 100) + rng.normal(0, 0.003, n)
    gld = pd.Series(50.0 + rng.normal(0, 0.05, n), index=idx)
    ratio = pd.Series(np.exp(log_ratio), index=idx)
    fx = pd.Series(1.3 + np.cumsum(rng.normal(0, 0.002, n)), index=idx)
    history = pd.DataFrame({"SPY": ratio * gld, "GLD": gld, "GBPUSD": fx})
    return ratio, history