# Diagnosis and handoff: why the strategy flips only twice

**Written 2026-11-14.** This file exists so another agent can pick up without the
original conversation. Read it together with `agents.md` (the general codebase
guide). It records what is wrong, what has already been changed, and what is left.

> **Do not run git commands.** The project owner manages version control
> personally. Read and edit files; do not commit, branch, stash or reset.

**Current status:** Phase 1 and Phase 2 are complete. The default recency
fraction is now `0.5` of the window, which gives 21 trades and GBP 14,326, but
the out-of-sample result is **below** the baseline and the trade count is at
the noise rate. Phases 3–4 remain. The details are in sections 5 and 6.

---

## 1. Symptom

`report_output/summary.txt` (original defaults, before Phase 2: window 30,
quantile 95, max_age 7, warm-up 250) reported **2 trades in 21 years**:

```
2008-02-06  SPY
2015-03-16  GLD
```

Final value GBP 8,603 vs the buy-and-hold baseline's 11,366. The obvious
suspicion is that `max_age=7` is an untuned guess. It is, but simply tuning it
on the full sample is **overfitting**, not a fix.

---

## 2. Root cause (measured, not guessed)

The signal is the product of two gates in `regression.decide_target`:

1. `|t_stat| >= t_threshold` (calibrated at the 95th percentile of luck), and
2. `-max_age <= vertex <= 0`.

The gates are nearly mutually exclusive, because conditioning on a large
curvature t-statistic pushes the fitted vertex toward the **centre** of the
window (an OLS parabola's vertex is most precise there).

Measured on the real 2005-2026 data, window 30, threshold 8.56:

| Quantity | Value |
|---|---|
| Traded bars | 5,253 |
| Bars with `|t| >= 8.56` | 197 (3.75%) |
| Bars passing both gates at `max_age=7` | **4** (2.0% of the significant bars) |
| Median vertex when significant | **-14.8 bars** |
| Share of significant bars with vertex in `[-7, 0]` | 2.0% |
| Share of *all* bars with vertex in `[-7, 0]` | 11.2% |

The four passing bars are 2008-02-05/06/07 (cup -> SPY) and 2015-03-13
(cap -> GLD). With the one-bar delay and the "never queue what you already
hold" rule they become the two trades above. **This is the parameterisation
working as written, not a coding bug.**

On an i.i.d. null built from the warm-up steps, this exact rule fires on
**0.13% of bars** (~6.8 events per 21 years). The real data fired 4 times, i.e.
at or below the noise rate. `max_age` is the binding constraint, not the
threshold.

### Why `max_age = window/4` is the wrong default

The significance gate needs data on both sides of the turning point, so a
detectable vertex is usually about half a window old. Empirically, 50.8% of
significant bars have a vertex in `[-15, -7)` and 47.2% are older than 15.
`window/2` (15) is the natural recency default; `window/4` (7) fights the
estimator. A *different statistic* (e.g. the significance of today's slope
`b`, or "the turn has been confirmed by a move of X sigma") would behave
better still, but that is a strategy change (Phase 3).

---

## 3. Why "just tune max_age" is unsafe

Sweep at window 30, quantile 95 (same data):

| max_age | 5 | 7 | 8 | 10 | 12 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|---|---|
| trades | 0 | 2 | 2 | 5 | 13 | 21 | 33 | 35 |
| final GBP | 1,000 | 8,603 | 8,196 | 4,030 | 13,639 | **14,326** | 8,999 | 9,989 |

`max_age=15` gives 14,326 and beats the baseline, so the intuition has a real
basis. But the response is a cliff, not a smooth effect: 8 -> 10 loses 4,166;
12 -> 15 gains 687; 20 loses 5,327. At `max_age=7`, sweeping the quantile gives
13,144 (q50), **14,799 (q60)**, 8,579 (q70), 8,456 (q80), 4,447 (q90), 8,603
(q95), 1,000 (q99). Warm-up length (a nuisance parameter) flips it too:
250 -> 2 trades/8,603; 375 -> 1/11,808; 1,000 -> 1/3,572. A best cell with bad
neighbours is luck, and picking it violates rule 11 in `agents.md`.

At `max_age=15` the noise check has a median of ~26 trades (18-33); the real
data makes 21. More trades makes the harness informative; it does **not**
create evidence of an edge.

---

## 4. Other real problems found

1. **The joint rule was never calibrated.** `calibrate_threshold` controls only
   the marginal `|t|` rate (nominally 5%). The actual decision rule fired on
   0.076% of bars. The parameter `95` is therefore misleading as a tuning knob.
   Phase 1 adds joint-gate calibration and diagnostics.
2. **The default was not in the sensitivity grid.** `MAX_AGE=7` but
   `sensitivity_table` used `round(30*0.25)=8`, so the grid's 30/95 row showed
   8,196 while the summary reported 8,603. Phase 1 makes the default fraction
   explicit (`MAX_AGE_FRACTION`) and guarantees the default cell appears.
3. **No exit / trend filter -> stale positions.** GLD was held for 2,936 bars
   (11.6 years) after 2015-03. SPY returned +339% vs GLD +221% over that span.
   A decomposition replay (not a valid strategy) shows that skipping only the
   2015 trade ends at 11,808 instead of 8,603: **that one signal costs 27% of
   final value**. The 2015 call was right for about a year; the failure is that
   nothing ever closes it. Phase 3.
4. **The "first half" outperformance is mostly an FX/cash artefact.** The
   strategy sits in cash for 529 bars (2005-12 to 2008-02), then buys SPY at
   GBPUSD 1.9608 vs the baseline's entry at 1.7197; GBPUSD ends at 1.3231. That
   entry-rate difference alone is **+14.0%** in GBP, while SPY rose only +9.75%
   locally during the cash spell. The reported 164.8% vs 148.0% is cash +
   currency timing, not signal. The report now flags trade-level stats as
   meaningless with so few trades.
5. **Dead, misleading code in `strategy.SwapStrategy._detect_signal`.** It uses
   the opposite condition (`if abs(gradient) > threshold: return`) and a
   different meaning of `threshold`. Not on the live path; delete in Phase 4.
6. **`trade_count` could disagree with the trade log.** A pending swap incremented
   the counter even if `portfolio.buy` silently no-opped after the sell. Phase 1
   guards the increment and asserts `len(trades) == trade_count` in the
   backtest.
7. **`calibrate_threshold` resamples steps i.i.d.**, destroying volatility
   clustering, and the real exceedance is 3.75% vs the nominal 5%. Consider a
   block bootstrap later (Phase 4).
8. **Reporting overstated weak evidence.** Hit rate over 2 trades, a
   shifted-timeline control with one switch, and a performance-only sensitivity
   table. Phase 1 suppresses the hit rate below 10 trades and adds noise /
   control columns to the grid.
9. **Minor:** `data.py` names GBPUSD `usd_to_gbp`; no download cache;
   `price_history` is not reindexed to the ratio index; `_annualised` divides by
   `INITIAL_BALANCE` rather than the first valuation. Phase 4.

---

## 5. Phase 1 (DONE): make the problem visible (historical)

Goal: make the diagnosis visible, without changing the default strategy's
behaviour. At that point the default was still `max_age=7`, so the strategy
still made 2 trades and ended at 8,603.

- Added `regression.window_stats_all` (moved from `evaluation`) and
  `regression._fit_windows` so single-window and batch fits share one code path.
- Added `regression.calibrate_joint_threshold(log_steps, window, max_age,
  target_rate, ...)`: picks the `|t|` threshold so the **full decision rule**
  fires on `target_rate` of bars under the luck null. This is the parameter
  Phase 2 should use.
- Added `regression.noise_joint_rate(...)`: expected luck-only fire rate at a
  given threshold/max_age.
- `BacktestResult` now records `vertices` (one per traded bar) and
  `noise_joint_rate`.
- `run_backtest(..., target_signal_rate=None)` can calibrate the joint rule
  directly instead of the marginal quantile. Default `None` keeps the old
  behaviour.
- Added `evaluation.signal_diagnostics(result)`, which reports significant
  bars, recent-vertex bars, the joint rate, the luck-implied rate and the
  conditional vertex distribution. `report.py` prints this block.
- `evaluation.sensitivity_table` now defaults to `MAX_AGE_FRACTION` so the
  default cell is in the grid, and adds `noise_trades_median` and
  `beats_control_pct` columns. Runtime grows; lower `noise_sims`/`control_runs`
  if needed.
- `report.py` suppresses the hit rate below `MIN_TRADES_FOR_STATS = 10` and
  labels the shifted control as weak when there are few switches.
- `strategy._execute_pending_swap` increments `trade_count` only if the buy
  actually left a holding; `run_backtest` asserts `trade_count == len(trades)`.

Verify:

```
source .venv/bin/activate
pytest                       # 70 tests at that point (79 after Phase 2), no internet, ~5 s
MPLBACKEND=Agg python report.py   # regenerates report_output/, a few minutes
```

The default run after Phase 1 was unchanged (2 trades, GBP 8,603) and the report
already said why:

```
Signal diagnostics (why this many trades?):
  Bars with |t| >= 8.56: 197 (3.75% of 5253)
  Bars with a vertex within 7 bars: 589 (11.21%)
  Bars passing BOTH gates: 4 (0.076%)
  Luck alone passes both at 0.130% (~6.8 events over this period)
  Median vertex when significant: -14.8 bars (the gate asks for <= 7)
  Share of significant bars that also pass the age gate: 2.0%
Hit rate: suppressed — 2 trades is below the 10 needed for a meaningful read
```

The sensitivity table now contains the default cell `window=30, quantile=95,
max_age=7` and two extra columns (`noise_trades_median`, `beats_control_pct`).
The default row's 2 trades exactly match its noise median of 2.

---

## 6. Phase 2 (DONE): honest parameter selection

Implemented:

- `evaluation.SPLIT_DATE = 2014-12-31` is pre-registered before any grid is run.
  Do not move it after reading results.
- `evaluation.range_metrics(result, start, end)` measures returns, annualised
  return, drawdowns, trades and time invested over any window. Returns are
  measured from the last bar **before** `start`, and `beats_baseline` compares
  growth **within** the window, not inherited final levels.
  `in_sample_metrics` / `out_of_sample_metrics` wrap it for the split.
- `sensitivity_table` now includes `age_fraction`, `IS_*` and `OOS_*` columns.
  Its default age fractions are `MAX_AGE_FRACTION` plus 0.35/0.50/0.65,
  deduplicated, so the default cell is always present exactly once.
- `evaluation.is_neighbour_median(table)` returns the median of an IS column
  over each cell's 3x3 neighbourhood (quantile x age fraction, same window).
  `best_in_sample_cell(table)` selects the best neighbourhood median, breaking
  ties on the cell's own IS value. Selection therefore uses IS data only and
  cannot be captured by an isolated spike.
- `backtest.MAX_AGE_FRACTION = 0.5` (`max_age = 15`). This is a **structural**
  choice, not a return-ranked one: a significant vertex sits near mid-window,
  so `window/2` admits ~53% of significant bars instead of ~2%. The default was
  not chosen by maximising in-sample return; the IS-best cell is reported only
  as a diagnostic.
- `report.py` prints the IS/OOS split and the pre-registered selection check:
  the IS-robust pick, its OOS outcome, and how many grid cells beat the OOS
  baseline.

Default result (window 30, q95, `max_age` 15, threshold 8.56):

| | Strategy | Baseline |
|---|---|---|
| Trades | 21 | 0 |
| Final GBP | 14,326 | 11,366 |
| Annualised | 13.6% | 12.4% |
| Max drawdown | -23.0% | -35.0% |
| In-sample (2005-12-28..2014-12-31) | **+341.7%** | +117.2% |
| Out-of-sample (2015-01-02..2026-11-13) | **+224.4%** | +424.1% |

Honest verdict:

- The default makes 21 trades; the same rules on shuffled daily changes make a
  median of 25 (5th–95th percentile 17–33). The trade count is **at the noise
  rate, not above it**.
- It beats the baseline over the full sample and in-sample, and loses badly out
  of sample (-199.8 percentage points). The full-sample win is an in-sample
  phenomenon.
- Grid: windows 20/30/45 x q90/95/99 x age fractions 0.35/0.50/0.65 (27 cells).
  **0 of 27 settings beat the OOS baseline.** The raw best IS cell
  (window 45 / q90 / max_age 22) returned +310.6% OOS vs +424.1%; the
  neighbour-robust IS pick (window 20 / q90 / max_age 13, IS +193.1%) returned
  +326.4% OOS.
- Conclusion: the harness now produces enough trades to evaluate, but there is
  still **no demonstrated edge**. More parameter search is not the answer;
  Phase 3 is.

Reproduce:

```
source .venv/bin/activate
pytest -q                              # 79 tests, no internet, ~5 s
MPLBACKEND=Agg python report.py        # regenerates report_output/, a few minutes
```

`report_output/summary.txt` now contains an `In-sample / out-of-sample` block
and, after the sensitivity grid, a `Pre-registered selection check` block whose
last line should read `0 of 27 settings beat the baseline out of sample`.

---

## 7. Phase 3 (NOT DONE): strategy design

The biggest economic problem is that a single stale signal can last a decade.
Options, in rough order of preference:

1. **Exit / max holding rule.** Re-evaluate after N bars, or exit when the
   fitted slope turns against the position, or hold SPY as the default when
   curvature is insignificant. This directly attacks the 2015-2026 GLD hold.
2. **Replace the vertex-recency gate** with a test on today's slope
   (`b / se_b`) or a "turn confirmed by a move of X sigma" rule. Expose `b` and
   its standard error on `WindowFit`.
3. **Trend filter**: skip mean-reversion signals that fight a strong long-term
   trend, or size down instead of all-in.

Evaluate any of these with `curvy_data()` tests, the in-sample/out-of-sample
split, `noise_trade_counts`, and `shifted_timeline_control`. Do not judge on
final value alone.

---

## 8. Phase 4 (NOT DONE): cleanup

- Delete or repair `SwapStrategy.update` / `_detect_signal` (dead, inverted,
  different `threshold` semantics).
- Rename `usd_to_gbp` -> `gbp_usd` in `data.py`; add a parquet download cache;
  reindex `price_history` to the ratio index and fail loudly on NaNs.
- Use a block bootstrap in `calibrate_threshold` to preserve volatility
  clustering; report the realised exceedance rate.
- Add a `hit_rate`-style guard everywhere statistics are computed from fewer
  than ~10 observations.
- Consider `_annualised` using the first valuation rather than `INITIAL_BALANCE`.

---

## 9. Key commands and files

```
pytest -q
MPLBACKEND=Agg python report.py
```

- Signal rules: `smooth_graph/regression.py`
- Loop / defaults: `smooth_graph/backtest.py`
- Statistics and controls: `smooth_graph/evaluation.py`
- Report text: `report.py`
- Generated output: `report_output/summary.txt`, `report_output/*.png`
- General guide (invariants, traps): `agents.md`

## 10. Numbers to keep in mind

- **Old default (`max_age = 7`)**: 197 significant bars, 4 pass the age gate,
  2 executed trades (2008, 2015); luck-only joint rate 0.13%/bar (~6.8 events in
  21 years). Keep these as the diagnostic baseline.
- **New default (`max_age = 15`)**: 104 bars pass both gates, 21 trades; the
  same rule on shuffled daily changes makes a median of 25 (17–33), so the real
  count is noise-like. The joint luck rate rises to 2.5% of bars.
- Median vertex when significant is -14.8 bars either way: the recency gate was
  the binding constraint, not the threshold.
- Full sample +1,332.6% vs +1,036.6%; in-sample +341.7% vs +117.2%;
  out-of-sample **+224.4% vs +424.1%**. The OOS loss is the honest read.
- The 27-cell grid has **0 winners** against the +424.1% OOS baseline. Raw best
  IS cell 45/90/22: IS +364.5%, OOS +310.6%. Neighbour-robust IS pick
  20/90/13: IS +193.1%, OOS +326.4%.
- 2015-2026 GLD hold under the old default: 2,936 bars; SPY +339% vs GLD +221%;
  skipping that trade would have ended at 11,808 (decomposition only, not a
  strategy).
- "First half" advantage under the old default: the initial cash spell's GBP
  gain is mostly the GBPUSD entry rate (1.96 vs 1.72), not signal.
