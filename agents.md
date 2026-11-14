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

Results from the run of 2026-11-14 with the **Phase 2 defaults** (window 30, quantile 95, max age 15 = window/2, warm-up 250). Data span 2005-01 to 2026-11. Traded 2005-12-28 to 2026-11-13 (5,253 bars). Both portfolios start with GBP 1,000 on the first traded bar.

| | Strategy | Baseline (buy and hold SPY) |
|---|---|---|
| Final value (GBP) | 14,326 | 11,366 |
| Annualised return | 13.6% | 12.4% |
| Max drawdown | -23.0% | -35.0% |
| Return, first half | +419.3% | +148.0% |
| Return, second half | +175.9% | +359.0% |
| Time invested | 99.3% | 100% |
| In-sample, 2005-12-28..2014-12-31 | +341.7% | +117.2% |
| Out-of-sample, 2015-01-02..2026-11-13 | **+224.4%** | **+424.1%** |

- **The default now makes 21 trades**, up from 2. `MAX_AGE_FRACTION = 0.5` is a *structural* choice: a significant vertex sits near the middle of the window (median -14.8 bars), so `window/4` admitted only ~2% of significant bars while `window/2` admits ~53%. It was not selected by maximising returns.
- **The 21 trades are at the luck rate, not above it.** The same rules on shuffled versions of the real daily changes make a median of 25 trades (5th-95th percentile 17-33). More trades made the harness informative; they did not create evidence of an edge.
- **The out-of-sample result is worse than the baseline.** In-sample the strategy returns +341.7% vs +117.2%; out-of-sample it returns +224.4% vs the baseline's +424.1%. The full-sample win is an in-sample phenomenon.
- **The honesty check, not the return, is the headline.** `SPLIT_DATE = 2014-12-31` was pre-registered. Across 27 grid cells (windows 20/30/45 x q90/95/99 x age fractions 0.35/0.50/0.65), **0 beat the out-of-sample baseline**. The raw best in-sample cell (45/90/22) did +310.6% OOS; the neighbour-robust in-sample pick (20/90/13) did +326.4% OOS. See `report_output/summary.txt` and `DIAGNOSIS.md`.
- **Historical default (`max_age = 7`)**: exactly 2 trades (SPY 2008-02-06, GLD 2015-03-16), final GBP 8,603. 197 bars cleared the `|t|` gate but only 4 also had a vertex within 7 bars. Luck alone fires that rule on 0.13% of bars (~7 events). This is retained as the diagnostic baseline in `DIAGNOSIS.md`.

Treat the strategy's verdict as **inconclusive at best, with no evidence of an edge**. The next step is a strategy change (exit rule / different statistic), not more parameter search.

---

## 3. Quick start

```
source .venv/bin/activate
pip install yfinance pandas numpy matplotlib pytest
pytest                  # 79 tests, no internet needed, about 5 seconds
python report.py        # downloads data; a few minutes (the grid runs a noise/control check per cell)
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
├── DIAGNOSIS.md               The two-trade diagnosis, Phase 1/2 changes, IS/OOS results, remaining work
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
├── tests/                     79 tests (see section 9), no internet needed
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
   t_threshold = calibrate_threshold(...)             (or calibrate_joint_threshold
                                                       if target_signal_rate is set)
   noise_joint_rate = luck-only fire rate at that threshold/max_age
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
- **`calibrate_joint_threshold(log_steps, window, max_age, target_rate=0.005, n_sims=20000, seed=0) -> float`**: the honest version of the above. It returns the `|t|` cut-off that makes the **full** `decide_target` rule fire on `target_rate` of luck-only bars. Necessary because the vertex gate is strongly dependent on `|t|`: calibrating `|t|` alone (nominally 5%) actually fires the full rule on ~0.08% of bars. Raises `ValueError` if `max_age` is too small for the target.
- **`noise_joint_rate(log_steps, window, max_age, threshold, n_sims=10000, seed=0) -> float`**: share of luck-only bars on which the full rule fires at a given threshold. Use it to compare the real trade count with luck.
- **`window_stats_all(values, window) -> (t, vertex, a)`**: one row per rolling window, in time order. Vectorised `fit_window`; `_fit_windows` is the single batch implementation used by both this and the calibration simulations.
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

Constants (edit these, not magic numbers in the loop): `WINDOW = 30`, `T_QUANTILE = 95.0`, `MAX_AGE_FRACTION = 0.5`, `MAX_AGE = max(1, round(WINDOW * MAX_AGE_FRACTION))` (15), `WARMUP_BARS = 250`, `FX_FEE_RATE = 0.0015`, `INITIAL_BALANCE = 1000`. The recency fraction is a **structural** default (a significant vertex sits near mid-window), not a return-maximising one; it was not selected by ranking grid returns. Expressing the gate as a fraction is deliberate: `evaluation.sensitivity_table` derives each cell's `max_age` from the same fraction, so the default configuration is always one of the grid cells. Edit the fraction, not `MAX_AGE`.

`run_backtest(window_bars=WINDOW, t_quantile=T_QUANTILE, max_age=MAX_AGE, warmup_bars=WARMUP_BARS, target_signal_rate=None, data=None) -> BacktestResult`. Pass `data=(ratio_series, price_history)` to avoid re-downloading when running many backtests. `target_signal_rate=None` keeps the historical marginal-`|t|` calibration; a rate such as `0.005` instead calls `calibrate_joint_threshold` so the **full rule** fires on that share of luck-only bars (and `t_quantile` is ignored).

`BacktestResult` fields: `dates`, `strategy_values`, `baseline_values`, `t_stats` and `vertices` (one entry per traded bar); `prices` (SPY/GLD/GBPUSD on traded bars); `log_ratio` (all bars, including warm-up); `trades` (DataFrame indexed by execution date, column `bought`); `t_threshold`; `noise_joint_rate` (luck-only share of bars on which both gates fire); `trade_count`; `target_signal_rate`; and the settings used. The loop asserts `trade_count == len(trades)`, which catches a queued swap that failed to change holdings.

Algorithm:
1. `log_ratio = np.log(ratio_series)` (a new Series; `ratio_series` is never modified).
2. Calibrate the threshold from `np.diff(log_ratio.iloc[:warmup_bars])`, i.e. **warm-up bars only**.
3. Create fresh portfolios and a fresh strategy. The **baseline buys SPY on the first traded bar** (`ratio_series.index[warmup_bars - 1]`) and is never touched again.
4. For `cutoff` in `range(warmup_bars, len(ratio_series) + 1)`: window = `log_ratio.iloc[cutoff - window_bars : cutoff]` (**rolling**, fixed length); fit; look up `prices` at `window.index[-1]`; `strategy_portfolio.update_conversion(GBPUSD)`; `strategy.update(...)`; record a trade if the held ticker changed; snapshot valuations.

**Alignment invariants:** the first traded bar is index `warmup_bars - 1` and the last is the final bar. `window.index[-1]` is the date used for both the fit and the pricing. `price_history` and `ratio_series` share an index; anything that filters or resamples one must keep this. `warmup_bars` must be at least `window_bars`.

### 6.7 `smooth_graph/evaluation.py`

- Statistics: `holding_timeline`, `max_drawdown`, `summarise(result)` (strategy vs baseline table; returns are measured against `INITIAL_BALANCE`), `hit_rate(result, horizon=15)`. `MIN_TRADES_FOR_STATS = 10` is the threshold below which trade-level statistics are suppressed by the report.
- **`range_metrics(result, start=None, end=None) -> dict`**: returns, annualised return, drawdowns, trades, time invested and `beats_baseline` over a date window. Returns are measured from the last bar before `start`; `beats_baseline` compares growth **within** the window (not inherited levels). `in_sample_metrics` / `out_of_sample_metrics` wrap it at the pre-registered `SPLIT_DATE = 2014-12-31`.
- **`signal_diagnostics(result) -> dict`**: the "why so few trades?" table — significant bars, recent-vertex bars, bars passing both gates, the luck-only rate, the median vertex when significant, and the share of significant bars that also pass the age gate. The gap between the last two numbers is the parameter problem described in `DIAGNOSIS.md`.
- **`shifted_timeline_control(result, n_runs=300)`**: replays the strategy's own holding timeline rotated in time by random amounts (shift 0 excluded). Each run spends exactly as long in cash, SPY and GLD as the real strategy, but at the wrong dates, with identical costs. It asks whether *when* the strategy held each asset added anything.
- **`noise_trade_counts(result, n_sims=200)`**: runs the same rules on histories built by resampling the real daily changes. Compare with `result.trade_count`: if the real count is not clearly above this range, the signal is not firing more than luck would. It preserves fat tails but destroys volatility clustering and autocorrelation. Uses `regression.window_stats_all` (a vectorised `fit_window`) and `_count_trades`.
- `sensitivity_table(data, windows=(20, 30, 45), quantiles=(90, 95, 99), age_fractions=None, noise_sims=100, control_runs=100)`: re-runs the backtest across a grid (`max_age = max(1, round(window * fraction))`). `age_fractions` defaults to `default_age_fractions()` = `(MAX_AGE_FRACTION, 0.35, 0.5, 0.65)` deduplicated, so **the default configuration is always a cell**. Columns include `age_fraction`, `IS_*` and `OOS_*` (split at `SPLIT_DATE`), `noise_trades_median` (same rule on shuffled daily changes) and `beats_control_pct` (share of shifted-timeline runs the cell beats). A high final value with a noise-like trade count is not evidence. **Select on in-sample columns only**, using `best_in_sample_cell`; do not pick the best full-sample row.
- **`is_neighbour_median(table, column="IS_return_pct")`**: median of a column over each cell's 3x3 neighbourhood (quantile x age fraction, same window). **`best_in_sample_cell(table)`** ranks on that median, tie-breaking on the cell's own IS value. This is the pre-registered selection rule; report the chosen cell's OOS value once, never feed it back.
- Charts (each returns a Matplotlib `Figure`): `plot_values`, `plot_log_ratio_with_flips`, `plot_control`.

### 6.8 `report.py`

`main()` loads data once, runs the backtest, prints the summary, **signal diagnostics**, the **in-sample / out-of-sample block**, hit rate, shifted-timeline control, noise check, trade list and (if `RUN_SENSITIVITY`) the sensitivity grid plus the **pre-registered selection check** (IS-robust pick, its OOS outcome, and how many cells beat the OOS baseline). It saves `summary.txt` and the three charts to `report_output/`, then calls `plt.show()`. The hit rate is replaced by an explicit "suppressed" line when there are fewer than `evaluation.MIN_TRADES_FOR_STATS` trades, because a handful of trades say nothing.

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
11. **Do not change the parameters after looking at results.** Choose settings on the pre-registered in-sample slice (`end=SPLIT_DATE`) only; evaluate `out_of_sample_metrics` once. `DIAGNOSIS.md` documents how unstable full-sample selection is and records that 0 of 27 grid cells beat the OOS baseline.
12. **`MAX_AGE` is derived from `MAX_AGE_FRACTION`, and the sensitivity grid uses the same fraction.** Do not hard-code `max_age` in the grid or the default cell will disappear from it.
13. **A `t_threshold` calibrated with `calibrate_threshold` controls only the marginal `|t|` rate.** The actual decision rule fires much less often because of the vertex gate. Use `noise_joint_rate` when describing frequency and `calibrate_joint_threshold` (or `run_backtest(target_signal_rate=...)`) when the frequency itself is the parameter.
14. **`SPLIT_DATE = 2014-12-31` is pre-registered.** It separates in-sample selection from the single out-of-sample evaluation. Do not move it after seeing grid results, and do not choose settings from full-sample or OOS columns. Use `best_in_sample_cell`, which ranks on the 3x3 in-sample neighbourhood median.

---

## 9. Tests

Run `pytest` from the project root (venv activated). Everything runs on **synthetic data**, with no network.

| File | Tests | Covers |
|---|---|---|
| `tests/test_regression.py` | 8 | `fit_window`: recovers known parabolas, vertex sign convention, scale invariance, edge cases |
| `tests/test_calibration.py` | 13 | `calibrate_threshold`: reproducible, monotonic in quantile and window, scale-free, "noise exceeds it about 5% of the time". `calibrate_joint_threshold`: hits the requested **joint** rate, monotone in the target, reproducible, rejects impossible targets. `noise_joint_rate`: the full rule fires far less often than the marginal 5% suggests |
| `tests/test_decision.py` | 9 | `decide_target`: both gates, inclusive boundaries, NaN and infinite inputs |
| `tests/test_strategy.py` | 7 | `QuadraticStrategy`: one-bar delay, never queues what is already held, execute-before-detect ordering |
| `tests/test_backtest.py` | 13 | Rolling log windows, no look-ahead calibration, FX rate current at trade time, trade log, recorded vertices/noise rate, `target_signal_rate` path |
| `tests/test_evaluation.py` | 29 | Statistics, range/IS-OOS metrics, shifted control, noise check (vectorised rules agree with the real strategy trade for trade), signal diagnostics, default grid cell, sensitivity columns, neighbour-median selection, charts |

`tests/helpers.py` provides two datasets:
- `random_walk_data()`: plain random walks, which produce few or no signals.
- `curvy_data()`: a clean sine wave on the log ratio, which **forces trades**.

**Any test of trading behaviour needs `curvy_data()`.** On random walks the strategy barely trades, so a bug in the trading path (a missing FX update, for example) can pass every test unnoticed. When you change behaviour, add a test that fails without your change.

---

## 10. Evaluating results honestly

- **Judge the signal against luck, not just against the baseline.** Use `noise_trade_counts` (is the signal firing more than noise would?) and `shifted_timeline_control` (does the timing add anything over the same time in each asset?).
- **A random-switch-date control is misleading for few-trade strategies.** It sits in cash for a different length of time than the real strategy (about 34% of the history on average, against 10%), which handicaps it. Use the shifted-timeline control instead.
- **Ignore trade-level statistics with few trades.** A hit rate over 2 trades means nothing; wait for dozens.
- **Read the sensitivity grid for stability.** A real effect shows a smooth pattern across neighbouring settings. Results that jump around, or a single good cell, indicate luck. Use `best_in_sample_cell` (3x3 IS neighbourhood median) rather than the best row.
- **Respect the pre-registered split.** Choose settings on `in_sample_metrics` only, then evaluate once with `out_of_sample_metrics`. A full-sample win can be entirely an in-sample phenomenon; that is exactly what the current default shows (+341.7% IS, +224.4% OOS vs +424.1% baseline).
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
7. In 2026-11 the two-trade result was traced to the joint gate: significance selects vertices near the window centre, so `max_age = window/4` admitted only ~2% of significant bars. Phase 1 added the gate diagnostics, joint calibration and noise/control grid columns without changing the default. Phase 2 pre-registered a 2014-12-31 split, added IS/OOS measurement and neighbour-robust selection, and changed `MAX_AGE_FRACTION` to 0.5 structurally (`window/2`), giving 21 trades. The out-of-sample result is below the baseline and 0 of 27 grid cells beat it, so there is still no demonstrated edge. `DIAGNOSIS.md` records the evidence and the remaining phases.

---

## 13. Extending safely

| Change | Where | Notes |
|---|---|---|
| Tune window / quantile / max age | `smooth_graph/backtest.py` constants or `run_backtest` arguments | Follow rule 11 in section 8. Prefer `target_signal_rate` / `calibrate_joint_threshold` when you care about trade frequency; the marginal `t_quantile` does not control it. Read `DIAGNOSIS.md` first. |
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
5. If the bend is statistically significant and its turning point is at most 15 bars old (`window/2`): queue SPY for a cup, GLD for a cap, unless you already hold it.
6. Value the strategy and the buy-and-hold baseline in GBP.
7. Judge the result against luck (noise check, shifted-timeline control), not just against the baseline.

---

## 15. If something looks wrong

- **Very few trades** → read the "Signal diagnostics" block in `report_output/summary.txt`. If almost no significant bars have a recent vertex, the recency gate is the binding constraint, not the threshold; if the joint count is well above the luck rate but still tiny, the threshold is. See `DIAGNOSIS.md`.
- **`ZeroDivisionError` in `usd_to_gbp`** → a portfolio's FX rate was never set (default 0).
- **Valuations off by a constant factor** → GBPUSD used the wrong way round.
- **Balance unexpectedly non-zero for long stretches** → a `buy` returned early because `balance <= commission`, or `withdraw` refused a cost slightly above the balance.
- **Tests pass but a trading bug slips through** → the test data produced no signals; use `curvy_data()`.
- **`KeyError` on `["Close"][ticker]`** → yfinance changed its column layout; inspect the downloaded frame.
- **A download returns an empty frame** → Yahoo rate-limited the request; retry.