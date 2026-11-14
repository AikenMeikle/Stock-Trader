"""Signal detection and order execution for the SPY/GLD ratio strategy."""
import pandas as pd

from .portfolio import Portfolio
from .regression import WindowFit, decide_target


class SwapStrategy:
    """Detects ratio reversals and executes the resulting swaps on the next tick."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.pending_swap: str = ""       # ticker to buy on the next update
        self.trade_count: int = 0

    def update(
        self,
        portfolio: Portfolio,
        gradient: float,
        gradient_rate: float,
        prices: pd.Series,
    ) -> None:
        self._execute_pending_swap(portfolio, prices)
        self._detect_signal(gradient, gradient_rate)

    def _execute_pending_swap(self, portfolio: Portfolio, prices: pd.Series) -> None:
        if not self.pending_swap:
            return

        target = self.pending_swap
        spy_price = float(prices["SPY"])
        gld_price = float(prices["GLD"])

        if target == "SPY":
            portfolio.sell(gld_price, "GLD")
            portfolio.buy(spy_price, "SPY")
        else:  # "GLD"
            portfolio.sell(spy_price, "SPY")
            portfolio.buy(gld_price, "GLD")

        self.pending_swap = ""
        # Count a trade only if the buy actually happened. Portfolio.buy is a
        # silent no-op when the balance cannot cover the commission, so counting
        # the queue would overstate the trade count after a failed execution.
        if target in portfolio.holdings:
            self.trade_count += 1

    def _detect_signal(self, gradient: float, gradient_rate: float) -> None:
        if abs(gradient) > self.threshold:
            return

        # Ratio has bottomed -> switch into SPY; topped -> switch into GLD.
        self.pending_swap = "SPY" if gradient_rate > 0 else "GLD"


class QuadraticStrategy(SwapStrategy):
    """Same execution machinery as SwapStrategy (one-bar delay, trade_count),
    but the signal comes from a quadratic fit of the recent window.

    Inherited and reused unchanged: pending_swap, trade_count, _execute_pending_swap.
    New: update() takes a WindowFit instead of two gradients.

    In THIS class, `threshold` means the t-statistic threshold from
    calibrate_threshold, NOT a gradient threshold.
    """

    def __init__(self, threshold: float, max_age: int) -> None:
        super().__init__(threshold=threshold)
        self.max_age = max_age

    def update(self, portfolio: Portfolio, fit: WindowFit, prices: pd.Series) -> None:
        # ORDER MATTERS: execute yesterday's order first, THEN look at today's signal.
        self._execute_pending_swap(portfolio, prices)
        self._queue_signal(portfolio, fit)

    def _queue_signal(self, portfolio: Portfolio, fit: WindowFit) -> None:
        target = decide_target(fit=fit, threshold=self.threshold, max_age=self.max_age)
        if target is None:
            return

        if target in portfolio.holdings.keys():
            return

        self.pending_swap = target