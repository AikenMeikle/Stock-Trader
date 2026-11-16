import pandas as pd
import pytest

from smooth_graph.portfolio import Portfolio
from smooth_graph.regression import WindowFit
from smooth_graph.strategy import QuadraticStrategy

PRICES = pd.Series({"SPY": 500.0, "GLD": 200.0, "GBPUSD": 1.25})

# decide_target(fit, threshold=8.0, max_age=7) gives these answers:
CUP = WindowFit(a=0.01, t_stat=12.0, vertex=-3.0)        # strong recent minimum -> "SPY"
CAP = WindowFit(a=-0.01, t_stat=-12.0, vertex=-3.0)      # strong recent maximum -> "GLD"
NO_SIGNAL = WindowFit(a=0.01, t_stat=1.0, vertex=-3.0)   # too weak -> None
STALE = WindowFit(a=-0.01, t_stat=1.0, vertex=-20.0)     # no signal, turn aged out
FUTURE = WindowFit(a=-0.01, t_stat=1.0, vertex=3.0)      # no signal, turn in the future
UNDEFINED = WindowFit(a=-0.01, t_stat=1.0, vertex=float("nan"))


def make_strategy(fallback=None):
    return QuadraticStrategy(threshold=8.0, max_age=7, fallback=fallback)


def make_portfolio(holding=None):
    """A GBP 1000 portfolio, either all cash or fully invested in `holding`."""
    p = Portfolio(fx_fee_rate=0.0015, initial_balance=1000)
    p.update_conversion(1.25)
    if holding is not None:
        p.buy(PRICES[holding], holding)
    return p


def test_no_signal_queues_nothing():
    p, s = make_portfolio("GLD"), make_strategy()
    s.update(p, NO_SIGNAL, PRICES)
    assert s.pending_swap == ""
    assert s.trade_count == 0


def test_signal_executes_exactly_one_bar_later():
    p, s = make_portfolio("GLD"), make_strategy()

    s.update(p, CUP, PRICES)                    # bar 1: signal appears
    assert p.get_current_stock() == "GLD"       # NOT traded yet
    assert s.pending_swap == "SPY"
    assert s.trade_count == 0                   # queued is not counted

    s.update(p, NO_SIGNAL, PRICES)              # bar 2: execution
    assert p.get_current_stock() == "SPY"
    assert s.pending_swap == ""
    assert s.trade_count == 1


def test_does_not_queue_a_swap_into_what_it_already_holds():
    p, s = make_portfolio("SPY"), make_strategy()
    s.update(p, CUP, PRICES)                    # signal says SPY, we hold SPY
    assert s.pending_swap == ""
    s.update(p, NO_SIGNAL, PRICES)
    assert p.get_current_stock() == "SPY"
    assert s.trade_count == 0


def test_opposite_signal_queues_the_other_ticker():
    p, s = make_portfolio("SPY"), make_strategy()
    s.update(p, CAP, PRICES)
    assert s.pending_swap == "GLD"


def test_from_cash_a_signal_buys_the_target():
    p, s = make_portfolio(None), make_strategy()
    s.update(p, CUP, PRICES)
    assert p.holdings == {}                     # still cash after the signal bar
    s.update(p, NO_SIGNAL, PRICES)
    assert p.get_current_stock() == "SPY"
    assert p.get_balance() < 1.0                # essentially fully invested
    assert s.trade_count == 1


def test_execution_happens_before_detection_on_the_same_bar():
    p, s = make_portfolio("GLD"), make_strategy()
    s.update(p, CUP, PRICES)                    # queue SPY
    s.update(p, CAP, PRICES)                    # bar 2: execute SPY swap, THEN queue GLD
    assert p.get_current_stock() == "SPY"
    assert s.pending_swap == "GLD"
    assert s.trade_count == 1
    s.update(p, NO_SIGNAL, PRICES)              # bar 3: execute GLD swap
    assert p.get_current_stock() == "GLD"
    assert s.trade_count == 2


def test_repeated_signal_does_not_cause_repeated_trades():
    p, s = make_portfolio("GLD"), make_strategy()
    for _ in range(6):
        s.update(p, CUP, PRICES)
    assert p.get_current_stock() == "SPY"
    assert s.trade_count == 1


# ---- Phase 3 exit rule --------------------------------------------------------
def test_stale_vertex_rotates_a_held_position_into_the_fallback():
    p, s = make_portfolio("GLD"), make_strategy(fallback="SPY")
    s.update(p, STALE, PRICES)
    assert s.pending_swap == "SPY"             # queued, not yet traded
    s.update(p, STALE, PRICES)
    assert p.get_current_stock() == "SPY"
    assert s.trade_count == 1


def test_future_vertex_also_counts_as_lost_support():
    p, s = make_portfolio("GLD"), make_strategy(fallback="SPY")
    s.update(p, FUTURE, PRICES)
    assert s.pending_swap == "SPY"


def test_undefined_vertex_counts_as_lost_support():
    p, s = make_portfolio("GLD"), make_strategy(fallback="SPY")
    s.update(p, UNDEFINED, PRICES)
    assert s.pending_swap == "SPY"


def test_insignificant_but_recent_vertex_still_supports_the_position():
    # The exit rule deliberately does NOT re-check the t-statistic: a weak turn
    # that is still inside the recency window is not a reason to sell.
    p, s = make_portfolio("GLD"), make_strategy(fallback="SPY")
    for _ in range(6):
        s.update(p, NO_SIGNAL, PRICES)
    assert p.get_current_stock() == "GLD"
    assert s.pending_swap == ""
    assert s.trade_count == 0


def test_fallback_none_disables_the_exit_rule():
    p, s = make_portfolio("GLD"), make_strategy(fallback=None)
    for _ in range(6):
        s.update(p, STALE, PRICES)
    assert p.get_current_stock() == "GLD"
    assert s.pending_swap == ""
    assert s.trade_count == 0


def test_no_exit_when_the_position_is_already_the_fallback():
    p, s = make_portfolio("SPY"), make_strategy(fallback="SPY")
    for _ in range(6):
        s.update(p, STALE, PRICES)
    assert p.get_current_stock() == "SPY"
    assert s.pending_swap == ""
    assert s.trade_count == 0


def test_invalid_fallback_is_rejected():
    with pytest.raises(ValueError, match="fallback"):
        QuadraticStrategy(threshold=8.0, max_age=7, fallback="CASH")