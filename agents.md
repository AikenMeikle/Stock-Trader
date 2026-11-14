# AGENTS.md

A guide for AI agents (and humans) working on this codebase. Read it before changing anything. It explains what the project is, how the pieces fit together, which rules the code depends on, and how to check that a change is correct.

---

## 1. What this project is

A **backtest harness for a UK-based tactical allocation between two instruments**:

- **SPY**: SPDR S&P 500 ETF (US-listed, priced in USD)
- **GLD**: SPDR Gold Shares ETF (US-listed, priced in USD)

The strategy watches the **log of the SPY/GLD ratio**. Each bar it fits a parabola to the last 30 bars and asks: *has the ratio just turned a corner, and is that bend bigger than luck usually produces?* If yes, it queues a swap that executes on the next bar.

At any moment the portfolio holds **all SPY**, **all GLD**, or **cash** (only before the first trade). The portfolio is accounted in **GBP**, so every trade crosses an FX boundary and pays an FX fee.

This is a **research harness, not a trading system**. Its main product is an honest answer to "does this signal beat luck?", and so far the answer is *no evidence that it does* (section 2).

---

## 2. Status at a glance

Results from the run of 2026-11-14 with the **untuned defaults** (window 30, quantile 95, max age 7, warm-up 250). Data span 2005-01 to 2026-11. Traded 2005-12-28 to 2026-11-13 (5,253 bars). Both portfolios start with GBP 1,000 on the first traded bar.

| | Strategy | Baseline (buy and hold SPY) |
|---|---|---|
| Final value (GBP) | 8,603 | 11,366 |
| Annualised return | 10.9% | 12.4% |
| Max drawdown | -32.4% | -35.0% |
| Return, first half | +164.8% | +148.0% |
| Return, second half | +224.9% | +359.0% |
| Time invested | 89.9% | 100% |

- **The strategy made exactly 2 trades**: it bought SPY from cash on 2008-02-06 and switched to GLD on 2015-03-16. That is one real switch. Every result above is determined by those two decisions, so hit rates and other trade-level statistics are meaningless at this sample size.
- **The sensitivity grid does not support an edge.** Of 18 settings, 16 end below the baseline. Final values of neighbouring settings swing from 3,228 to 19,073 with no smooth pattern, which is what luck looks like. Two settings had zero trades.
- **The signal fires at the rate luck would.** In simulations on random walks (100 histories of 5,253 bars per setting, independent Gaussian steps), the same rules produced trade counts at or below the real data's:

  | Window | Max age | Trades on noise, median (5th-95th pct) | Real data |
  |---|---|---|---|
  | 20 | 5 | 3 (1-6) | 1 |
  | 20 | 10 | 33 (26-44) | 29 |
  | 30 | 8 | 4 (1-6) | 2 |
  | 30 | 15 | 26 (18-33) | 21 |
  | 45 | 11 | 2 (0-4) | 1 |
  | 45 | 22 | 16 (11-22) | 10 |

  `report.py` now prints the equivalent check, built from the real data's own shuffled daily changes.
- **Why it rarely trades** (see section 7): significance and recency pull against each other. On random-walk bars where the bend is significant at window 30, the median turning point is about 14.5 bars back, and only 2.8% are within the last 7 bars.

Treat the strategy's verdict as **inconclusive at best, with no evidence of an edge**.

---

## 3. Quick start

```
source .venv/bin/activate
pip install yfinance pandas numpy matplotlib pytest
pytest                  # 57 tests, no internet needed, about 5 seconds
python report.py        # downloads data; about a minute (the sensitivity grid dominates)
```

`report.py` prints everything and also saves it to `report_output/`:

- `summary.txt`: all printed numbers, including the list of trades
- `1_values.png`: strategy vs baseline value, with drawdowns
- `2_flips.png`: log(SPY/GLD) with every switch marked and the t-statistic below it
- `3_shifted_control.png`: where the strategy sits among shifted-timeline versions of itself

Python 3.12 is in use. The live code needs only `yfinance`, `pandas`, `numpy`, `matplotlib` (and `pytest` for tests). There is no `requirements.txt` yet.

---

## 4. File layout

```
smooth-graph/
├── report.py                  Entry point: runs everything, prints and saves the report
├── pytest.ini                 Test discovery and import path
├── agents.md                  This guide
├── TODO
├── smooth_graph/              Importable package: the backtest harness
│   ├── __init__.py
│   ├── portfolio.py           Portfolio + Holding: cash, positions, FX, fees, buy/sell
│   ├── data.py                Downloads SPY, GLD, GBPUSD from yfinance
│   ├── regression.py          The statistics: fit_window, calibrate_threshold, decide_target
│   ├── strategy.py            SwapStrategy (base class) and QuadraticStrategy (the live strategy)
│   ├── valuation.py           snapshot_valuations: strategy vs baseline in GBP
│   ├── backtest.py            run_backtest() -> BacktestResult
│   └── evaluation.py          Statistics, controls, sensitivity grid, charts
├── tests/                     57 tests (see section 9), no internet needed
│   ├── __init__.py
│   ├── helpers.py             Synthetic market data for the tests
│   └── test_*.py              pytest modules
├── experiments/               Historical prototypes; not used by the live backtest
│   ├── __init__.py
│   ├── quadratic_strategy.py  Earlier notebook-style prototype
│   └── smoothing-spy-to-gld.ipynb
└── report_output/             Generated by `python report.py`
```

---

## 5. Data flow

```
data.load_price_data()
   |-- ratio_series  (SPY/GLD, never modified)
   '-- price_history (SPY, GLD, GBPUSD)
            |
backtest.run_backtest()
   log_ratio = np.log(ratio_series)
   t_threshold = calibrate_threshold(steps of the WARM-UP bars only)
   for each traded bar:
        window = last WINDOW values of log_ratio        (rolling, ends today)
        fit = fit_window(window)                        (a, t_stat, vertex)
        portfolio.update_conversion(today's GBPUSD)
        strategy.update(portfolio, fit, prices)         (1. execute yesterday's order
            |                                            2. queue today's, via decide_target)
            '-- decide_target(fit, threshold, max_age) -> "SPY" | "GLD" | None
        snapshot_valuations(...)                        (GBP values of strategy and baseline)
   -> BacktestResult
            |
evaluation.* / report.py   (statistics, controls, charts)
```

---

## 6. Module reference

### 6.1 `smooth_graph/portfolio.py`: the most important file

Every other module ultimately drives it. **Do not change its semantics casually**; results depend on the exact fee arithmetic.

- `Holding` (dataclass): `shares`, `avg_cost_gbp` (weighted average cost per share **including FX fee and commission**), `total_cost_gbp` property, `add(shares, cost_gbp)`.
- `Portfolio(fx_fee_rate, commission=0, initial_balance=0, usd_conversion=0)`. State: `_balance` (GBP cash), `holdings: dict[str, Holding]` (public), `_commission` (flat GBP per trade), `_usd_conversion`, `_fx_fee_rate`, `EPSILON = 1e-9`.
- **`_usd_conversion` is the GBPUSD rate: USD per 1 GBP.** `usd_to_gbp(x) = x / _usd_conversion`. The default is **0**, so a portfolio whose rate was never set will divide by zero. If you introduce an inverse rate, name it explicitly (`_gbp_per_usd`).
- Methods: `update_conversion(rate)`, `update_commission(fee)`, `update_fx_fee(rate)`, `get_balance()`, `deposit(amount)`, `withdraw(amount) -> bool`, `usd_to_gbp(amount_usd)`, `buy(price_usd, name)`, `sell(price_usd, name)`, `get_stock_names()`, `get_current_stock() -> str | None`, `get_portfolio_value(price_usd, name)`.
- **`buy` is "invest all cash"**, not "buy N shares". With balance `B`, commission `C`, rate `R`, FX fee `f`, price `P`:
  ```
  budget_usd = (B - C) * R / (1 + f)
  shares     = budget_usd / P
  total_cost_gbp = shares*P/R * (1 + f) + C      # then withdraw(total_cost_gbp); no partial fills
  ```
  It is a silent no-op if `B <= C`.
- **`sell` liquidates the whole holding**: `net = shares*P/R * (1 - f) - C`, added to the balance; the holding is deleted. It is a no-op if the ticker is not held.
- `get_current_stock()` returns the first held ticker and **assumes at most one holding**.
- `get_portfolio_value(price_usd, name)` = `shares*price/R + balance`; returns just the balance if the ticker is not held.

### 6.2 `smooth_graph/data.py`

`load_price_data() -> (ratio_series, price_history)`.

- `ratio_series`: `SPY_close / GLD_close`, NaNs dropped.
- `price_history`: DataFrame with columns `SPY`, `GLD`, `GBPUSD`. FX is reindexed to the ratio's index with a forward fill, so there should be no NaNs.
- Every call downloads from 2005-01-01, so call it **once** and pass the result on (`run_backtest(data=...)`). It prints `price_history.head()`.
- yfinance scrapes Yahoo, so downloads can fail or be rate-limited. The code indexes `["Close"][ticker]`, which assumes yfinance's MultiIndex column layout.
- **Prices appear to be dividend-adjusted** (the first printed row has SPY at about 81 on 2005-01-03, far below its roughly USD 120 trading price that day). That suits relative-return comparison but is not the price you would actually pay.

### 6.3 `smooth_graph/regression.py`

Pure functions: numbers in, numbers out. Units are **bars**. Callers pass the **log** ratio.

- `WindowFit` (frozen dataclass): `a` (curvature coefficient, > 0 cup, < 0 cap), `t_stat` (`a` / its standard error), `vertex` (turning point in bars relative to the latest bar: negative = that many bars ago, positive = in the future).
- `fit_window(values) -> WindowFit`: least-squares fit of `L = a*u**2 + b*u + c` with `u = -(n-1)..0` (**today is `u = 0`**, so `b` is today's slope and `c` is today's fitted value). Uses `np.polyfit(u, values, 2, cov=True)`; `vertex = -b/(2a)`. Needs at least `MIN_WINDOW = 8` points. Perfect fits give infinite `t_stat`; `a == 0` gives a NaN vertex.
- `calibrate_threshold(log_steps, window, quantile=95.0, n_sims=2000, seed=0) -> float`: builds fake paths by resampling `log_steps` with replacement, takes the cumulative sum, fits, records `|t_stat|`, and returns the `quantile`-th percentile. It answers "how big a t-statistic does pure luck produce?". A naive "t > 2" fires on pure noise more than half the time, because residuals of a fitted price path are autocorrelated.
- `decide_target(fit, threshold, max_age) -> "SPY" | "GLD" | None`. Fires only if **both** `|t_stat| >= threshold` **and** `-max_age <= vertex <= 0` (inclusive boundaries; a NaN vertex never fires). Direction: `a > 0` means a recent minimum, so buy **SPY**; `a < 0` means a recent maximum, so buy **GLD**. It never looks at the portfolio.

### 6.4 `smooth_graph/strategy.py`

- `SwapStrategy`: base class holding `pending_swap` (`""`, `"SPY"` or `"GLD"`: the ticker to **buy** next bar), `trade_count`, and `_execute_pending_swap(portfolio, prices)`, which sells the other ticker, buys the pending one, clears the note and increments `trade_count`. Its own gradient-based `update()` / `_detect_signal()` belong to an earlier signal and are **not used by the current backtest**.
- `QuadraticStrategy(SwapStrategy)`: `__init__(threshold, max_age)`; here `threshold` is the **t-statistic** threshold from `calibrate_threshold`.
  - `update(portfolio, fit, prices)` does two things **in this exact order**: (1) execute yesterday's pending swap, (2) `_queue_signal`.
  - `_queue_signal` calls `decide_target`; it returns without doing anything if the result is `None` **or the portfolio already holds that ticker**; otherwise it sets `pending_swap`. It **never trades**; it only writes the note.
- **One-bar delay (structural):** a signal computed from the close of bar *t* executes at the close of bar *t+1*. Do not remove it silently.
- `trade_count` counts executed swaps (never queued signals). Because already-held targets are never queued, it equals the number of real trades; a test enforces this.

### 6.5 `smooth_graph/valuation.py`

`snapshot_valuations(strategy_portfolio, baseline_portfolio, prices) -> tuple[float, float]` returns `(strategy_gbp, baseline_gbp)`. It **updates the conversion rate on both portfolios** before valuing. It values whichever ticker `get_current_stock()` reports, or just the cash balance if nothing is held.

### 6.6 `smooth_graph/backtest.py`

Constants (edit these, not magic numbers in the loop): `WINDOW = 30`, `T_QUANTILE = 95.0`, `MAX_AGE = 7`, `WARMUP_BARS = 250`, `FX_FEE_RATE = 0.0015`, `INITIAL_BALANCE = 1000`. All strategy values are **untuned guesses**.

`run_backtest(window_bars=WINDOW, t_quantile=T_QUANTILE, max_age=MAX_AGE, warmup_bars=WARMUP_BARS, data=None) -> BacktestResult`. Pass `data=(ratio_series, price_history)` to avoid re-downloading when running many backtests.

`BacktestResult` fields: `dates`, `strategy_values`, `baseline_values`, `t_stats` (one entry per traded bar); `prices` (SPY/GLD/GBPUSD on traded bars); `log_ratio` (all bars, including warm-up); `trades` (DataFrame indexed by execution date, column `bought`); `t_threshold`; `trade_count`; and the settings used.

Algorithm:
1. `log_ratio = np.log(ratio_series)` (a new Series; `ratio_series` is never modified).
2. Calibrate the threshold from `np.diff(log_ratio.iloc[:warmup_bars])`, i.e. **warm-up bars only**.
3. Create fresh portfolios and a fresh strategy. The **baseline buys SPY on the first traded bar** (`ratio_series.index[warmup_bars - 1]`) and is never touched again.
4. For `cutoff` in `range(warmup_bars, len(ratio_series) + 1)`: window = `log_ratio.iloc[cutoff - window_bars : cutoff]` (**rolling**, fixed length); fit; look up `prices` at `window.index[-1]`; `strategy_portfolio.update_conversion(GBPUSD)`; `strategy.update(...)`; record a trade if the held ticker changed; snapshot valuations.

**Alignment invariants:** the first traded bar is index `warmup_bars - 1` and the last is the final bar. `window.index[-1]` is the date used for both the fit and the pricing. `price_history` and `ratio_series` share an index; anything that filters or resamples one must keep this. `warmup_bars` must be at least `window_bars`.

### 6.7 `smooth_graph/evaluation.py`

- Statistics: `holding_timeline`, `max_drawdown`, `summarise(result)` (strategy vs baseline table; returns are measured against `INITIAL_BALANCE`), `hit_rate(result, horizon=15)`.
- **`shifted_timeline_control(result, n_runs=300)`**: replays the strategy's own holding timeline rotated in time by random amounts (shift 0 excluded). Each run spends exactly as long in cash, SPY and GLD as the real strategy, but at the wrong dates, with identical costs. It asks whether *when* the strategy held each asset added anything.
- **`noise_trade_counts(result, n_sims=200)`**: runs the same rules on histories built by resampling the real daily changes. Compare with `result.trade_count`: if the real count is not clearly above this range, the signal is not firing more than luck would. It preserves fat tails but destroys volatility clustering and autocorrelation. Uses `window_stats_all` (a vectorised `fit_window`) and `_count_trades`.
- `sensitivity_table(data, windows=(20, 30, 45), quantiles=(90, 95, 99), age_fractions=(0.25, 0.5))`: re-runs the backtest across a grid (`max_age = max(1, round(window * fraction))`). **Judge whether a stable pattern exists. Do not pick the best row**; that is overfitting.
- Charts (each returns a Matplotlib `Figure`): `plot_values`, `plot_log_ratio_with_flips`, `plot_control`.

### 6.8 `report.py`

`main()` loads data once, runs the backtest, prints the summary, hit rate, shifted-timeline control, noise check, trade list and (if `RUN_SENSITIVITY`) the sensitivity grid, saves `summary.txt` and the three charts to `report_output/`, then calls `plt.show()`.

---

## 7. The statistics in plain terms

**Why a fitted bend is not evidence by itself.** A parabola fitted to pure random-walk data will find a bend almost every time. So the strategy compares the bend's t-statistic against what luck produces, using `calibrate_threshold`. The threshold depends only on the window length, because t-statistics do not depend on volatility.

**Why the gate rarely fires.** A bend is only measurable with data on **both sides** of its turning point. High significance therefore tends to put the vertex about half a window back, which conflicts with "the turn must be recent". Measured on random walks at window 30: of bars with `|t|` above the threshold, the vertex is within the last 7 bars 2.8% of the time, within 15 bars 55%, and within 22 bars 97%. Raising `max_age` buys more signals at the price of lag.

**Why log space.** Equal percentage moves are equal at any level of the ratio. A fixed threshold on the raw ratio meant different things when the ratio was 0.5 and 2.6.

**Why rolling windows.** The fit describes recent shape, so old data should not dilute it. A rolling window also makes each bar's statistics independent of how much history exists.

**Why the old spline signal was abandoned.** The earlier signal used the second derivative of a smoothing spline at the **last data point**. A smoothing spline has natural boundary conditions, so that value is zero (or about 1e-15 of floating-point noise) there, and the buy/sell direction was effectively random. Do not resurrect a signal that depends on a derivative at the edge of a fitted window.

---

## 8. Invariants and traps

1. **One position, all-in / all-out.** `buy` spends everything; `sell` liquidates everything.
2. **One-bar execution delay.** Signal at the close of *t*, trade at the close of *t+1*.
3. **No look-ahead.** Windows end on the current bar. The threshold is calibrated on warm-up bars only. Never calibrate or tune using data from the period being evaluated.
4. **Set today's FX rate on the strategy portfolio before `strategy.update()`.** `snapshot_valuations` also sets it, but only *after* the trade, so omitting the earlier call would convert trades at the previous bar's rate. (An early draft had exactly this bug.)
5. **GBP accounting.** USD is transient inside `buy`/`sell`/valuation. `_usd_conversion` is GBPUSD; do not invert it.
6. **Units are bars**, not days. `vertex`, `max_age` and `window` are all in bars.
7. **`ratio_series` is never modified.** Take logs into a new Series.
8. **Fresh `Portfolio` and strategy objects for every run.** Both are stateful.
9. **`smooth_graph/backtest.py` imports names at module level on purpose.** The tests monkeypatch `load_price_data`, `fit_window`, `calibrate_threshold`, `QuadraticStrategy` and `Portfolio` in the `backtest` namespace to spy on them. Renaming or re-importing them differently breaks the tests.
10. **`QuadraticStrategy.threshold` is a t-statistic**, not a gradient. Do not mix it up with the old gradient threshold.
11. **Do not change the parameters after looking at results.** The defaults are untuned. If you tune, choose settings on an early slice of history only and evaluate once on the later slice.

---

## 9. Tests

Run `pytest` from the project root (venv activated). Everything runs on **synthetic data**, with no network.

| File | Tests | Covers |
|---|---|---|
| `tests/test_regression.py` | 8 | `fit_window`: recovers known parabolas, vertex sign convention, scale invariance, edge cases |
| `tests/test_calibration.py` | 7 | `calibrate_threshold`: reproducible, monotonic in quantile and window, scale-free, "noise exceeds it about 5% of the time" |
| `tests/test_decision.py` | 9 | `decide_target`: both gates, inclusive boundaries, NaN and infinite inputs |
| `tests/test_strategy.py` | 7 | `QuadraticStrategy`: one-bar delay, never queues what is already held, execute-before-detect ordering |
| `tests/test_backtest.py` | 11 | Rolling log windows, no look-ahead calibration, FX rate current at trade time, trade log |
| `tests/test_evaluation.py` | 15 | Statistics, shifted control, noise check (vectorised rules agree with the real strategy trade for trade), charts |

`tests/helpers.py` provides two datasets:
- `random_walk_data()`: plain random walks, which produce few or no signals.
- `curvy_data()`: a clean sine wave on the log ratio, which **forces trades**.

**Any test of trading behaviour needs `curvy_data()`.** On random walks the strategy barely trades, so a bug in the trading path (a missing FX update, for example) can pass every test unnoticed. When you change behaviour, add a test that fails without your change.

---

## 10. Evaluating results honestly

- **Judge the signal against luck, not just against the baseline.** Use `noise_trade_counts` (is the signal firing more than noise would?) and `shifted_timeline_control` (does the timing add anything over the same time in each asset?).
- **A random-switch-date control is misleading for few-trade strategies.** It sits in cash for a different length of time than the real strategy (about 34% of the history on average, against 10%), which handicaps it. Use the shifted-timeline control instead.
- **Ignore trade-level statistics with few trades.** A hit rate over 2 trades means nothing; wait for dozens.
- **Read the sensitivity grid for stability.** A real effect shows a smooth pattern across neighbouring settings. Results that jump around, or a single good cell, indicate luck.
- **Check both halves of the period.** Winning in one half and losing in the other is luck-like.
- **Compare like with like.** Both portfolios start on the same bar with the same capital. The strategy sits in cash until its first signal, which is a real cost of its timing.

---

## 11. Known limitations

**Model realism:**
- No bid-ask spread or slippage; fractional shares are assumed; cash earns 0%.
- No tax. Fees are an FX fee on both sides plus a flat commission (default 0).
- Execution is at the next bar's close.
- Prices appear dividend-adjusted (section 6.2).
- **UK access:** US-listed ETFs such as SPY and GLD are generally not available to UK retail investors under PRIIPs/KID rules; a real implementation would use UCITS equivalents.

**Engineering:**
- No `requirements.txt` / `pyproject.toml`; no logging framework (diagnostics are prints).
- No caching of yfinance downloads.
- `get_current_stock()` assumes at most one holding.
- `SwapStrategy._execute_pending_swap` assumes an "execute everything at the close" model, so do not reuse it verbatim for live trading.

---

## 12. Design history (why the code is the way it is)

1. The original signal used a smoothing spline's first derivative near zero and the sign of its second derivative at the window's last point. Experiments showed that second derivative is numerically zero there, so the buy/sell direction was noise. It was replaced.
2. The replacement fits a parabola to a rolling window, because the curvature uses every point in the window, not a single boundary value.
3. The significance threshold is calibrated by simulation, because naive t-statistics are badly overstated on price paths.
4. The loop originally stopped one bar early (the final bar was never used). It now runs to `len(ratio_series) + 1`, and a test pins it.
5. The baseline originally bought at row 0 of the data while the strategy started years later. It now buys on the first traded bar.
6. A first random-timing control was replaced by the shifted-timeline control and the noise check, because it was handicapped by cash drag.

---

## 13. Extending safely

| Change | Where | Notes |
|---|---|---|
| Tune window / quantile / max age | `smooth_graph/backtest.py` constants or `run_backtest` arguments | Follow rule 11 in section 8. |
| New signal | `smooth_graph/regression.py` (statistic and decision), `smooth_graph/strategy.py` (a new `SwapStrategy` subclass that overrides `update`) | Keep `pending_swap` meaning "ticker to buy". Keep the one-bar delay. Add tests using `curvy_data()`. |
| New instrument | `smooth_graph/data.py`, `smooth_graph/strategy.py`, `smooth_graph/valuation.py` | The one-holding assumption becomes a pressure point. |
| Partial sizing | `smooth_graph/portfolio.py`: add alongside `buy`, do not change it | That is a strategy change. |
| New statistics or charts | `smooth_graph/evaluation.py` | Keep functions pure and covered by tests on hand-built `BacktestResult` objects. |
| Cache downloads | `smooth_graph/data.py` | Parquet keyed by date range. |

---

## 14. Quick mental model

1. Load SPY, GLD and GBPUSD; compute the log of SPY/GLD.
2. Learn from the warm-up bars how big a bend luck can produce (the t threshold).
3. For each later bar: fit a parabola to the last 30 log-ratio values.
4. Execute yesterday's queued swap, if any.
5. If the bend is statistically significant and its turning point is at most 7 bars old: queue SPY for a cup, GLD for a cap, unless you already hold it.
6. Value the strategy and the buy-and-hold baseline in GBP.
7. Judge the result against luck (noise check, shifted-timeline control), not just against the baseline.

---

## 15. If something looks wrong

- **Very few trades** → expected with the default settings (section 7). Check the t-statistic panel in `2_flips.png` and `max_age`.
- **`ZeroDivisionError` in `usd_to_gbp`** → a portfolio's FX rate was never set (default 0).
- **Valuations off by a constant factor** → GBPUSD used the wrong way round.
- **Balance unexpectedly non-zero for long stretches** → a `buy` returned early because `balance <= commission`, or `withdraw` refused a cost slightly above the balance.
- **Tests pass but a trading bug slips through** → the test data produced no signals; use `curvy_data()`.
- **`KeyError` on `["Close"][ticker]`** → yfinance changed its column layout; inspect the downloaded frame.
- **A download returns an empty frame** → Yahoo rate-limited the request; retry.