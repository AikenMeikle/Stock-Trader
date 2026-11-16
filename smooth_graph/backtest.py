"""Backtest driver: runs the quadratic-curvature strategy over the ratio history."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import load_price_data
from .portfolio import Portfolio
from .regression import (
    calibrate_joint_threshold,
    calibrate_threshold,
    fit_window,
    noise_joint_rate,
)
from .strategy import QuadraticStrategy
from .valuation import snapshot_valuations

# --- Strategy parameters (untuned starting values) ---
WINDOW = 30             # bars in each rolling regression window
T_QUANTILE = 95.0       # luck-only t-statistics exceed the threshold 5% of the time
# The recency gate as a share of the window. It is deliberately explicit: the
# sensitivity grid derives each cell's max_age from a fraction, so the default
# must be expressed the same way for the default cell to appear in the grid.
#
# 0.5 (half the window) is a STRUCTURAL default, not a return-maximising one: a
# significant vertex sits near the middle of the window by construction (median
# about -window/2), so window/4 admits only ~2% of significant bars. The default
# was not chosen by ranking grid returns; see DIAGNOSIS.md.
MAX_AGE_FRACTION = 0.5      # half the window: ~window/2
MAX_AGE = max(1, round(WINDOW * MAX_AGE_FRACTION))   # 15 bars
# Phase 3 exit rule: when a held position's turn is no longer recent, rotate
# into this asset rather than holding a stale position forever. SPY is the
# benchmark and the natural neutral holding for a long-only allocator; cash is
# itself a market-timing bet. `None` disables the rule.
FALLBACK = "SPY"
WARMUP_BARS = 250       # bars used ONLY to calibrate the threshold; trading starts after
                        # them (must be >= the window)

# --- Portfolio parameters ---
FX_FEE_RATE = 0.0015
INITIAL_BALANCE = 1000


@dataclass
class BacktestResult:
    """Everything the report needs. Per-bar arrays have one entry per TRADED bar."""
    dates: pd.DatetimeIndex
    strategy_values: np.ndarray     # GBP
    baseline_values: np.ndarray     # GBP
    t_stats: np.ndarray             # t-statistic of the curvature on each bar
    vertices: np.ndarray            # vertex on each bar, in bars relative to that bar
    prices: pd.DataFrame            # SPY, GLD, GBPUSD on the traded bars
    log_ratio: pd.Series            # log(SPY/GLD) for ALL bars, including warm-up
    trades: pd.DataFrame            # index = execution date, column "bought" = "SPY" / "GLD"
    t_threshold: float              # calibrated significance threshold
    noise_joint_rate: float         # luck-only share of bars on which BOTH gates fire
    trade_count: int
    window_bars: int
    t_quantile: float
    max_age: int
    warmup_bars: int
    target_signal_rate: float | None = None   # set when the joint rule was calibrated
    fallback: str | None = None               # Phase 3 stale-signal exit target


def run_backtest(
    window_bars: int = WINDOW,
    t_quantile: float = T_QUANTILE,
    max_age: int = MAX_AGE,
    warmup_bars: int = WARMUP_BARS,
    target_signal_rate: float | None = None,
    fallback: str | None = FALLBACK,
    data=None,
    fx_fee_rate: float = FX_FEE_RATE,
) -> BacktestResult:
    """Run the strategy and the buy-and-hold baseline over the history.

    `data` is an optional (ratio_series, price_history) pair, so a caller that runs
    many backtests (e.g. a sensitivity grid) can download the data once.

    By default the threshold is calibrated on the marginal |t| quantile (the old
    behaviour). Pass `target_signal_rate` to calibrate the FULL decision rule to
    fire on that share of bars under the luck null; `t_quantile` is then ignored.

    `fallback` is the Phase 3 exit policy: a held position whose fitted turn is
    no longer recent is rotated into this ticker. `None` disables the rule.

    `fx_fee_rate` defaults to the unchanged `FX_FEE_RATE`; it is exposed only so
    evaluation can run friction-sensitivity scenarios without altering the
    strategy default.
    """
    assert warmup_bars >= window_bars, "the first window must fit inside the warm-up"
    ratio_series, price_history = data if data is not None else load_price_data()

    # Work in log space. np.log returns a NEW Series, so ratio_series is untouched.
    log_ratio = np.log(ratio_series)

    # Calibrate the threshold from the warm-up bars ONLY, so nothing from the
    # period we trade on leaks into it (no look-ahead).
    warmup_log_steps = np.diff(log_ratio.iloc[:warmup_bars].values)
    if target_signal_rate is None:
        t_threshold = calibrate_threshold(
            warmup_log_steps, window=window_bars, quantile=t_quantile
        )
    else:
        t_threshold = calibrate_joint_threshold(
            warmup_log_steps,
            window=window_bars,
            max_age=max_age,
            target_rate=target_signal_rate,
        )

    # How often luck alone would fire the same rule. Reported next to the real
    # trade count so a small number of trades is not mistaken for selectivity.
    expected_fire_rate = noise_joint_rate(
        warmup_log_steps, window=window_bars, max_age=max_age, threshold=t_threshold
    )

    strategy_portfolio = Portfolio(
        fx_fee_rate=fx_fee_rate, initial_balance=INITIAL_BALANCE
    )
    strategy = QuadraticStrategy(threshold=t_threshold, max_age=max_age, fallback=fallback)

    # The baseline buys SPY on the FIRST TRADED bar, the same day the strategy
    # starts, so both are measured over exactly the same period.
    baseline_portfolio = Portfolio(
        fx_fee_rate=fx_fee_rate, initial_balance=INITIAL_BALANCE
    )
    first_prices = price_history.loc[ratio_series.index[warmup_bars - 1]]
    baseline_portfolio.update_conversion(float(first_prices["GBPUSD"]))
    baseline_portfolio.buy(float(first_prices["SPY"]), "SPY")

    dates, strategy_values, baseline_values, t_stats, vertices, trades = [], [], [], [], [], []
    for cutoff in range(warmup_bars, len(ratio_series) + 1):  # +1 so the final bar is used
        window = log_ratio.iloc[cutoff - window_bars:cutoff]
        fit = fit_window(window.values)

        date = window.index[-1]
        prices = price_history.loc[date]
        strategy_portfolio.update_conversion(float(prices["GBPUSD"]))

        held_before = strategy_portfolio.get_current_stock()
        strategy.update(portfolio=strategy_portfolio, fit=fit, prices=prices)
        held_after = strategy_portfolio.get_current_stock()
        if held_after != held_before:
            trades.append((date, held_after))

        strategy_value, baseline_value = snapshot_valuations(
            strategy_portfolio, baseline_portfolio, prices
        )
        dates.append(date)
        strategy_values.append(strategy_value)
        baseline_values.append(baseline_value)
        t_stats.append(fit.t_stat)
        vertices.append(fit.vertex)

    dates = pd.DatetimeIndex(dates)
    trade_log = pd.DataFrame(trades, columns=["date", "bought"]).set_index("date")
    assert strategy.trade_count == len(trade_log), (
        "trade_count disagrees with the trade log: a queued swap did not change holdings"
    )
    return BacktestResult(
        dates=dates,
        strategy_values=np.asarray(strategy_values, dtype=float),
        baseline_values=np.asarray(baseline_values, dtype=float),
        t_stats=np.asarray(t_stats, dtype=float),
        vertices=np.asarray(vertices, dtype=float),
        prices=price_history.loc[dates, ["SPY", "GLD", "GBPUSD"]],
        log_ratio=log_ratio,
        trades=trade_log,
        t_threshold=t_threshold,
        noise_joint_rate=expected_fire_rate,
        trade_count=strategy.trade_count,
        window_bars=window_bars,
        t_quantile=t_quantile,
        max_age=max_age,
        warmup_bars=warmup_bars,
        target_signal_rate=target_signal_rate,
        fallback=fallback,
    )