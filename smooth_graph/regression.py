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
from numpy.lib.stride_tricks import sliding_window_view

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


def _fit_windows(windows) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised `fit_window` for a batch of windows.

    `windows` is 2-D, oldest value first in each row. Returns `(t_stat, vertex, a)`
    arrays, one entry per row. This is the single source of truth for the batch
    statistics; `window_stats_all` and the calibration simulations use it, and a
    test pins it against `fit_window`.
    """
    windows = np.asarray(windows, dtype=float)
    if windows.ndim != 2:
        raise ValueError("windows must be a 2-D array")
    n = windows.shape[1]
    if n < MIN_WINDOW:
        raise ValueError(f"need at least {MIN_WINDOW} points, got {n}")

    u = np.arange(-(n - 1), 1, dtype=float)  # latest bar is u = 0
    X = np.vander(u, 3)                      # columns: u**2, u, 1
    xtx_inv = np.linalg.inv(X.T @ X)
    coef = windows @ (xtx_inv @ X.T).T       # rows: a, b, c
    residuals = windows - coef @ X.T
    s2 = (residuals ** 2).sum(axis=1) / (n - 3)
    a, b = coef[:, 0], coef[:, 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = a / np.sqrt(s2 * xtx_inv[0, 0])
        vertex = -b / (2 * a)
    return t_stat, vertex, a


def window_stats_all(values, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """t-statistic, vertex and curvature for EVERY rolling window at once.

    One row per window, in time order, so the last row ends on the last value.
    A vectorised version of `fit_window`.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        raise ValueError(f"need at least {window} values, got {len(values)}")
    return _fit_windows(sliding_window_view(values, window))


def _null_fits(log_steps, window: int, n_sims: int, seed: int):
    """Simulate `n_sims` independent luck-only windows and fit each one."""
    rng = np.random.default_rng(seed)
    steps = rng.choice(np.asarray(log_steps, dtype=float),
                       size=(n_sims, window), replace=True)
    return _fit_windows(np.cumsum(steps, axis=1))


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


def calibrate_joint_threshold(
    log_steps,
    window: int,
    max_age: int,
    target_rate: float = 0.005,
    n_sims: int = 20000,
    seed: int = 0,
) -> float:
    """Threshold on |t_stat| so the FULL decision rule fires at `target_rate`.

    `calibrate_threshold` controls only the marginal rate of large t-statistics.
    The strategy also requires the vertex to be recent, and those two conditions
    are strongly dependent: large |t| selects vertices near the middle of the
    window (median about -window/2), not near the latest bar. Calibrating only
    |t| therefore produces a joint fire rate far below the nominal quantile.

    This function simulates luck-only windows, keeps those whose vertex passes
    `-max_age <= vertex <= 0`, and returns the |t| cut-off that leaves
    `target_rate * n_sims` of the simulated bars passing BOTH gates. In other
    words, `target_rate` is the probability that the strategy fires on a random
    bar if there is no signal. Raises ValueError if `max_age` is so small that
    the target rate is unattainable.
    """
    if not 0.0 < target_rate < 1.0:
        raise ValueError(f"target_rate must be in (0, 1), got {target_rate}")

    t_stat, vertex, _a = _null_fits(log_steps, window, n_sims, seed)
    recent = (vertex >= -max_age) & (vertex <= 0) & np.isfinite(vertex)
    abs_t = np.abs(t_stat[recent])
    abs_t = abs_t[np.isfinite(abs_t)]
    n_recent = len(abs_t)
    n_events = max(1, int(round(target_rate * n_sims)))

    if n_events > n_recent:
        raise ValueError(
            f"target_rate={target_rate} is unattainable at window={window}, "
            f"max_age={max_age}: only {n_recent}/{n_sims} luck-only windows have "
            "a vertex in the required range. Increase max_age or lower the rate."
        )

    # Exactly `n_events` of the recent-vertex windows have |t| >= the cut-off.
    abs_t.sort()
    return float(abs_t[n_recent - n_events])


def noise_joint_rate(
    log_steps,
    window: int,
    max_age: int,
    threshold: float,
    n_sims: int = 10000,
    seed: int = 0,
) -> float:
    """Share of luck-only bars on which the FULL decision rule would fire.

    Use this to state honestly how often the strategy acts compared with pure
    noise. At window 30, threshold ~8.6 and the old max_age=7 it was about
    0.0013 (roughly 1 bar in 750); at the Phase 2 default max_age=15 it is
    about 0.025, because the wider recency gate admits many more luck-only
    vertices. The threshold is unchanged, so this is still "what luck produces".
    """
    t_stat, vertex, _a = _null_fits(log_steps, window, n_sims, seed)
    fires = (
        (np.abs(t_stat) >= threshold)
        & (vertex >= -max_age)
        & (vertex <= 0)
    )
    return float(fires.mean())


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