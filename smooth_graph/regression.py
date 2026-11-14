"""Quadratic-window regression: the statistics behind the curvature signal.

Everything here is a pure function of numbers in, numbers out. No Portfolio,
no yfinance, no globals, so it is easy to test on its own.

Conventions (match the rest of the codebase):
- Units are BARS, not calendar days.
- Callers should pass the LOG of the ratio, not the raw ratio. Equal
  percentage moves then look equal at any price level.
"""
import math
from dataclasses import dataclass

import numpy as np

# Fewer points than this leaves too few degrees of freedom for a 3-parameter
# fit to give a meaningful standard error.
MIN_WINDOW = 8


@dataclass(frozen=True)
class WindowFit:
    """Result of fitting  L = a*u**2 + b*u + c  to one window.

    u counts bars backwards from the latest bar: u = -(n-1), ..., -1, 0.
    """
    a: float       # curvature coefficient: > 0 cup shape (a minimum), < 0 cap shape
    t_stat: float  # a / standard_error(a): how many standard errors from zero
    vertex: float  # turning point in bars relative to the latest bar
                   # (negative = that many bars ago, positive = in the future)


def fit_window(values) -> WindowFit:
    """Fit a parabola to `values` (oldest first, newest last)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < MIN_WINDOW:
        raise ValueError(f"need at least {MIN_WINDOW} points, got {n}")

    u = np.arange(-(n - 1), 1)  # latest bar is u = 0
    # np.polyfit returns coefficients highest power first: [a, b, c]
    # and, with cov=True, a 3x3 covariance matrix of those coefficients.
    (a, b, _c), cov = np.polyfit(u, values, 2, cov=True)

    se_a = float(np.sqrt(cov[0, 0]))
    if se_a > 0:
        t_stat = float(a / se_a)
    else:  # a perfect fit has zero residual error, so any non-zero `a` is "infinitely" significant
        t_stat = math.copysign(math.inf, a) if a != 0 else 0.0

    vertex = float(-b / (2 * a)) if a != 0 else math.nan  # slope 2a*u + b = 0
    return WindowFit(a=float(a), t_stat=t_stat, vertex=vertex)


def calibrate_threshold(log_steps, window: int, quantile: float = 95.0,
                        n_sims: int = 2000, seed: int = 0) -> float:
    """Return the `quantile`-th percentile of |t_stat| that fit_window produces on
    paths with NO real curvature, built by resampling `log_steps`.

    `log_steps` : 1-D array of daily changes of the log ratio (np.diff of log).
    `window`    : number of bars in each fake path (the same W you trade with).
    `quantile`  : 0-100. 95 means "noise exceeds the returned value 5% of the time".
    `n_sims`    : how many fake paths to simulate.
    `seed`      : makes the result reproducible.
    """
    rng = np.random.default_rng(seed)
    scores = []

    for _ in range(n_sims):
        fake_steps = rng.choice(log_steps, size=window, replace=True)
        fake_path = np.cumsum(fake_steps)
        t = fit_window(fake_path).t_stat

        scores.append(abs(t))

    return float(np.percentile(scores, quantile))


def decide_target(fit: WindowFit, threshold: float, max_age: int):
    if abs(fit.t_stat) < threshold:
        return None

    if fit.vertex < -max_age or fit.vertex > 0 or np.isnan(fit.vertex):
        return None

    if fit.a > 0:
        return "SPY"
    elif fit.a < 0:
        return "GLD"
    else:
        return None