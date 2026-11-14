from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Holding:
    shares: float = 0.0
    avg_cost_gbp: float = 0.0

    @property
    def total_cost_gbp(self) -> float:
        return self.shares * self.avg_cost_gbp

    def add(self, shares: float, cost_gbp: float) -> None:
        new_shares = self.shares + shares
        new_total_cost = self.total_cost_gbp + cost_gbp
        self.shares = new_shares
        self.avg_cost_gbp = new_total_cost / new_shares


class Portfolio:
    # Tolerance for floating point comparisons
    EPSILON = 1e-9

    def __init__(
        self,
        fx_fee_rate: float,
        commission: float = 0,
        initial_balance: float = 0,
        usd_conversion: float = 0,
    ) -> None:
        self._balance: float = initial_balance
        self._commission: float = commission
        self._usd_conversion: float = usd_conversion
        self._fx_fee_rate: float = fx_fee_rate
        self.holdings: dict[str, Holding] = {}

    # ----- Configuration -----

    def update_conversion(self, usd_conversion: float) -> None:
        self._usd_conversion = usd_conversion

    def update_commission(self, commission: float) -> None:
        self._commission = commission

    def update_fx_fee(self, fx_fee_rate: float) -> None:
        self._fx_fee_rate = fx_fee_rate

    # ----- Cash balance -----

    def get_balance(self) -> float:
        return self._balance

    def deposit(self, amount: float) -> None:
        self._balance += amount

    def withdraw(self, amount: float) -> bool:
        if self._balance - amount < -self.EPSILON:
            return False
        self._balance = max(0.0, self._balance - amount)
        return True

    # ----- Currency -----

    def usd_to_gbp(self, amount_usd: float) -> float:
        return amount_usd / self._usd_conversion

    # ----- Trading -----

    def buy(self, stock_price_usd: float, stock_name: str) -> None:
        """Purchase as many shares as possible with the current balance."""
        if self._balance <= self._commission:
            return

        net_budget_gbp = self._balance - self._commission
        budget_usd = net_budget_gbp * self._usd_conversion
        budget_usd_after_fx = budget_usd / (1 + self._fx_fee_rate)

        shares_bought = budget_usd_after_fx / stock_price_usd
        gross_cost_usd = shares_bought * stock_price_usd
        gross_cost_gbp = self.usd_to_gbp(gross_cost_usd)
        fx_fee_gbp = gross_cost_gbp * self._fx_fee_rate
        total_cost_gbp = gross_cost_gbp + fx_fee_gbp + self._commission

        if not self.withdraw(total_cost_gbp):
            return

        holding = self.holdings.setdefault(stock_name, Holding())
        holding.add(shares_bought, total_cost_gbp)

    def sell(self, stock_price_usd: float, stock_name: str) -> None:
        """Sell all shares of a stock and credit the net proceeds."""
        holding = self.holdings.get(stock_name)
        if holding is None or holding.shares == 0:
            return

        gross_proceeds_usd = holding.shares * stock_price_usd
        gross_proceeds_gbp = self.usd_to_gbp(gross_proceeds_usd)
        fx_fee_gbp = gross_proceeds_gbp * self._fx_fee_rate
        net_proceeds_gbp = gross_proceeds_gbp - fx_fee_gbp - self._commission

        self.deposit(net_proceeds_gbp)
        del self.holdings[stock_name]

    # ----- Queries -----

    def get_stock_names(self) -> list[str]:
        return list(self.holdings.keys())

    def get_current_stock(self) -> Optional[str]:
        return next(iter(self.holdings), None)

    def get_portfolio_value(self, stock_price_usd: float, stock_name: str) -> float:
        """Total GBP value of a single holding plus cash balance."""
        holding = self.holdings.get(stock_name)
        if holding is None:
            return self._balance

        position_value_usd = holding.shares * stock_price_usd
        return self.usd_to_gbp(position_value_usd) + self._balance