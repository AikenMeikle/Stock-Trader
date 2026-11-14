"""Backtest driver: runs the quadratic-curvature strategy over the ratio history."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import load_price_data
from .portfolio import Portfolio
from .regression import calibrate_threshold, fit_window
from .strategy import QuadraticStrategy
from .valuation import snapshot_valuations

# --- Strategy parameters (untuned starting values) ---
WINDOW = 30             # bars in each rolling regression window
T_QUANTILE = 95.0       # luck-only t-statistics exceed the threshold 5% of the time
MAX_AGE = 7             # the turning point must be at most this many bars old (~WINDOW / 4)
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
    prices: pd.DataFrame            # SPY, GLD, GBPUSD on the traded bars
    log_ratio: pd.Series            # log(SPY/GLD) for ALL bars, including warm-up
    trades: pd.DataFrame            # index = execution date, column "bought" = "SPY" / "GLD"
    t_threshold: float              # calibrated significance threshold
    trade_count: int
    window_bars: int
    t_quantile: float
    max_age: int
    warmup_bars: int


def run_backtest(
    window_bars: int = WINDOW,
    t_quantile: float = T_QUANTILE,
    max_age: int = MAX_AGE,
    warmup_bars: int = WARMUP_BARS,
    data=None,
) -> BacktestResult:
    """Run the strategy and the buy-and-hold baseline over the history.

    `data` is an optional (ratio_series, price_history) pair, so a caller that runs
    many backtests (e.g. a sensitivity grid) can download the data once.
    """
    assert warmup_bars >= window_bars, "the first window must fit inside the warm-up"
    ratio_series, price_history = data if data is not None else load_price_data()

    # Work in log space. np.log returns a NEW Series, so ratio_series is untouched.
    log_ratio = np.log(ratio_series)

    # Calibrate the threshold from the warm-up bars ONLY, so nothing from the
    # period we trade on leaks into it (no look-ahead).
    warmup_log_steps = np.diff(log_ratio.iloc[:warmup_bars].values)
    t_threshold = calibrate_threshold(
        warmup_log_steps, window=window_bars, quantile=t_quantile
    )

    strategy_portfolio = Portfolio(
        fx_fee_rate=FX_FEE_RATE, initial_balance=INITIAL_BALANCE
    )
    strategy = QuadraticStrategy(threshold=t_threshold, max_age=max_age)

    # The baseline buys SPY on the FIRST TRADED bar, the same day the strategy
    # starts, so both are measured over exactly the same period.
    baseline_portfolio = Portfolio(
        fx_fee_rate=FX_FEE_RATE, initial_balance=INITIAL_BALANCE
    )
    first_prices = price_history.loc[ratio_series.index[warmup_bars - 1]]
    baseline_portfolio.update_conversion(float(first_prices["GBPUSD"]))
    baseline_portfolio.buy(float(first_prices["SPY"]), "SPY")

    dates, strategy_values, baseline_values, t_stats, trades = [], [], [], [], []
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

    dates = pd.DatetimeIndex(dates)
    return BacktestResult(
        dates=dates,
        strategy_values=np.asarray(strategy_values, dtype=float),
        baseline_values=np.asarray(baseline_values, dtype=float),
        t_stats=np.asarray(t_stats, dtype=float),
        prices=price_history.loc[dates, ["SPY", "GLD", "GBPUSD"]],
        log_ratio=log_ratio,
        trades=pd.DataFrame(trades, columns=["date", "bought"]).set_index("date"),
        t_threshold=t_threshold,
        trade_count=strategy.trade_count,
        window_bars=window_bars,
        t_quantile=t_quantile,
        max_age=max_age,
        warmup_bars=warmup_bars,
    )