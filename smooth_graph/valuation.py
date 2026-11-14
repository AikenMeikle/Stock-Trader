"""Portfolio valuation helpers for strategy vs baseline comparison."""
import pandas as pd

from .portfolio import Portfolio


def _valuation_for(portfolio: Portfolio, prices: pd.Series) -> float:
    """GBP value of a portfolio given the current price row."""
    current_stock = portfolio.get_current_stock()
    if current_stock is None:
        return portfolio.get_balance()

    price = float(prices[current_stock])
    return portfolio.get_portfolio_value(price, current_stock)


def snapshot_valuations(
    strategy_portfolio: Portfolio,
    baseline_portfolio: Portfolio,
    prices: pd.Series,
) -> tuple[float, float]:
    """Return (strategy_valuation, baseline_valuation) for the current price row."""
    conversion_rate = prices["GBPUSD"]
    strategy_portfolio.update_conversion(conversion_rate)
    baseline_portfolio.update_conversion(conversion_rate)

    strategy_valuation = _valuation_for(strategy_portfolio, prices)
    baseline_valuation = baseline_portfolio.get_portfolio_value(
        float(prices["SPY"]), "SPY"
    )
    return strategy_valuation, baseline_valuation