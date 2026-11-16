# Diagnosis and handoff: why the strategy flips only twice

**Written 2026-11-14.** This file exists so another agent can pick up without the
original conversation. Read it together with `agents.md` (the general codebase
guide). It records what is wrong, what has already been changed, and what is left.

> **Do not run git commands.** The project owner manages version control
> personally. Read and edit files; do not commit, branch, stash or reset.

**Current status:** Phases 1, 2 and 3 are complete. The default recency
fraction is `0.5` of the window (`max_age = 15`), and a stale position is now
rotated to SPY when its fitted turn leaves the recency window. The default makes
39 trades and ends at GBP 12,312, but the in-sample result still trails the
baseline, the out-of-sample win is not robust to nearby `max_age` values, and
the trade count is still at the noise rate. Phase 4 remains. The details are in
sections 5–8.

---

## 0. Phase 5 validation: PRE-REGISTERED (written before any validation run)

This section was written **before** running any of the new validation machinery
on real data. The constants live in `smooth_graph/evaluation.py`; they must not
be edited after seeing results. They do not change any strategy rule or default
(`WINDOW`, `T_QUANTILE`, `MAX_AGE_FRACTION`, `FALLBACK`, `WARMUP_BARS`,
`SPLIT_DATE` all stay as they are).

The signal counts as **VALIDATED** only if **all four** checks pass:

| # | Check | Pass rule | Pre-registered constant |
|---|---|---|---|
| 1 | Full-pipeline Monte Carlo | real annualised excess return >= 95th percentile of the synthetic null | `MC_SIMS = 300`, `MC_BLOCK_BARS = 20`, `MC_SEED = 0`, `MC_PERCENTILE = 95.0` |
| 2 | Detour attribution | lower end of the 95% bootstrap CI of mean **net** detour excess is above 0 | `DETOUR_BOOTSTRAP_SIMS = 10000`, `DETOUR_BOOTSTRAP_SEED = 0`, `DETOUR_CI_PCT = 95.0` |
| 3 | Friction | strategy still beats the baseline at twice the current FX fee | `FRICTION_FEE_MULTIPLE = 2.0` |
| 4 | Neighbourhood | default cell beats the baseline in at least 6 of its 9 3x3-neighbourhood cells, in BOTH the full sample and OOS | `NEIGHBOURHOOD_MIN_BEATS = 6`, `NEIGHBOURHOOD_TOTAL_CELLS = 9` |

The one-sided Monte Carlo p-value is pre-registered as
`(1 + #{null >= real}) / (MC_SIMS + 1)`; check 1 is the percentile rule above,
not a post-hoc p-value threshold.

**Clean evidence.** `CLEAN_HOLDOUT_START = 2026-11-14`. Data before it has
already been used for design decisions in Phases 1–3 (including the OOS
window), so only bars on/after it count as clean forward evidence. The current
history ends 2026-11-13, so the forward block must say `no clean data yet`.

**Friction grid.** Check D uses extra friction per swap leg of 0, 0.05, 0.10,
0.20 and 0.30 percentage points (`FRICTION_EXTRAS`). Check 3 separately runs the
1.5x point (total fee `2 * 0.0015 = 0.003`) because twice the current fee is not
one of the five pre-registered display points. The default fee is never changed.

**Assumptions recorded instead of asking:**

1. **A "detour" is a GLD spell.** `detour_attribution` returns one row per
   non-SPY holding spell, so the initial CASH spell is present for
   reconciliation, but the headline mean/median/bootstrap statistics use GLD
   rows only. Cash is not a tactical detour from a SPY position and has no entry
   swap. This is why the summary reports `count`, `gld_count` and `cash_count`.
2. **Detour switching costs.** A CASH -> asset buy costs `f / (1 + f)` of the
   balance; an asset -> asset swap costs `2f / (1 + f)` (sell fee plus the
   reduced buy budget). A GLD spell still open on the final bar is charged only
   the entry swap; an exit swap is charged only when a later bar exists.
3. **Detour reconciliation (corrected after the first results; see 0.1).** The
   first implementation was deliberately approximate: each row's net return was
   multiplied by the strategy's value on the bar before entry. It reported a
   ~GBP 959 gap and blamed compounding. That was wrong. The reconciliation is
   now an exact bar-by-bar log decomposition and must equal
   `log(strategy final / baseline final)` to within 1e-9. The old additive
   `net_excess_gbp` remains only as a `legacy_*` diagnostic.
4. **IS/OOS detour split uses the entry date** against the pre-registered
   `SPLIT_DATE`. A spell that starts in-sample is counted in-sample even if it
   ends after the split.
5. **Tracking-error years.** The daily active return is strategy minus baseline
   daily return. The annualised information ratio is
   `mean(active) / std(active) * sqrt(252)`, and at 2 standard errors the years
   needed are `(2 / IR) ** 2`. If the observed active mean is not positive, the
   years-to-detect is reported as infinite/n/a.
6. **Neighbourhood axis order.** `is_neighbour_median` now sorts quantiles and
   age fractions by value before taking the 3x3 block. The grid lists the
   default age fraction first, so without sorting the default cell would sit on
   an edge and see only 6 cells; the pre-registered check requires a true 9-cell
   neighbourhood. This is an evaluation fix, not a strategy change.
7. **Full-pipeline null.** Synthetic histories resample the **joint** daily log
   returns of SPY, GLD and GBPUSD in the same 20-bar blocks, start from the real
   first price of each series, and have the same length/index as the real
   history. Each run recalibrates its threshold on its own synthetic warm-up.
   Annualised excess is `annualised(strategy) - annualised(synthetic SPY
   buy-and-hold)`, using the same convention as `summarise`.
8. **Friction break-even.** The table's break-even is linear interpolation
   between the supplied extra-friction grid points where the final-value lead
   crosses zero. If it does not cross inside the grid, the report says so rather
   than extrapolating.
9. **No strategy/default changes were made for this phase.** The only backtest
   API change is an optional `fx_fee_rate` argument whose default is the
   existing `FX_FEE_RATE`.

### 0.1 Post-result correction: exact attribution and annual distribution

**Written after the first Phase 5 real-data results.** This is an evaluation
correction only. No strategy rule, default, fee, split date, Monte Carlo seed,
bootstrap seed, CI level, or pre-registered pass threshold was changed. Check 2
still passes only when the lower end of the 95% bootstrap CI of mean net GLD
detour excess is above zero.

**What was wrong with the first reconciliation.** It summed each spell's net
return applied to the strategy value on the bar before entry. That is a local
GBP effect, not a decomposition of the final lead, and it measured a GLD spell
only to its **last held bar**. The position is actually sold at the close of the
**next (swap) bar**, so every GLD detour was one bar short. For this history the
old GLD gross excess summed to +13.41%; including the exit swap bar gives
+20.03%, i.e. the missing bars alone are worth about +6.6 percentage points of
gross return across the 19 spells.

**Exact decomposition used now.** For every traded bar `t > 0`:

```
active[t] = log(strategy[t] / strategy[t-1]) - log(baseline[t] / baseline[t-1])
```

The interval ending at `t` is grouped under the asset held on bar `t-1` (the
asset that earned the return over the interval). A switch at bar `t` is charged
to the spell being **left**, so the GLD exit swap bar belongs to the GLD spell.
The only unattached term is the first-bar entry mismatch
`log(strategy[0] / baseline[0])`, because the baseline has already paid its
entry fee on the first traded bar while the strategy is still in cash. The
reconciliation is:

```
sum(spell active log returns) + first-bar mismatch
    == log(final strategy / final baseline)
```

and is asserted to within `1e-9`. The report prints `Reconciled: yes/no`.

**Before vs after (real data, default Phase 3 settings):**

| Quantity | Before correction | After correction |
|---|---|---|
| Reconciliation method | additive local GBP | exact bar-by-bar log |
| Sum of non-SPY attribution | -13 GBP | n/a (exact log used) |
| Strategy-minus-baseline lead | +946 GBP | +946 GBP |
| Reported gap | +959 GBP ("compounding") | 0 by construction |
| Max log reconciliation error | n/a | 5.2e-15 |
| GLD gross excess, sum of spells | +13.41% | +20.03% |
| GLD mean net excess | +0.11% | +0.45% |
| GLD median net excess | -0.60% | +0.26% |
| GLD share positive | 42.1% | 52.6% |
| 95% bootstrap CI of mean net excess | [-1.52%, +1.92%] | [-1.43%, +2.55%] |
| Check [2] at the SAME threshold (CI low > 0) | FAIL | **FAIL** |

The corrected check still fails: the CI spans zero. Correcting the attribution
changed the numbers, not the verdict.

**Where the old ~GBP 959 gap came from.** It was not return compounding. A
bridge from the old additive sum to the exact lead is:

| Step | GBP |
|---|---:|
| Old additive sum | -13.2 |
| Include the exit swap bar (corrected local attribution) | +140.3 |
| Grow local effects to the final portfolio value | +784.6 |
| Add the first-bar entry mismatch | +18.4 |
| Exact log/cost residual (log vs simple arithmetic) | +16.0 |
| Exact strategy-minus-baseline lead | **+946.2** |

The log-vs-simple residual is only about GBP 16, so compounding of returns is
not the explanation. The gap is (a) the omitted exit bar and (b) the fact that a
GBP effect earned early also grows with the portfolio. The exact decomposition
removes both ambiguities.

**Exact contributions by holding (default run):**

| Holding | Spells | Bars | Active log % | Gross log % | Cost log % | GBP effect to end |
|---|---:|---:|---:|---:|---:|---:|
| CASH | 1 | 35 | -1.232 | -1.082 | -0.150 | -153 |
| GLD | 19 | 121 | +14.778 | +20.478 | -5.700 | +1,691 |
| SPY | 20 | 5,097 | -5.700 | 0.000 | -5.700 | -722 |
| First-bar mismatch | - | - | +0.150 | - | - | +18 |
| **Total** | | | **+7.996** | | | **+946** |

The SPY active log return is exactly the switch-cost component; the gross
SPY-vs-SPY component is 0.000% (no unexplained SPY return). The GLD spells
supply the gross lead, but total switch costs and the initial cash spell give
most of it back.

**Leave-one-out (neutralise one GLD spell's active log return):**

| GLD spell | Active log % | Final GBP | Lead vs baseline GBP |
|---|---:|---:|---:|
| 2015-08-17 -> 2015-08-24 | +13.38 | 10,770 | -595 |
| 2006-07-11 -> 2006-07-14 | +5.95 | 11,601 | +235 |
| 2008-05-23 -> 2008-06-03 | -5.51 | 13,010 | +1,644 |

The lead is concentrated: removing the 2015 GLD detour alone puts the strategy
below the baseline. This is the same fragility described in sections 7 and 10,
not evidence of a stable edge.

**Annual return distribution (new evaluation chart).** `report.py` now saves
`4_annual_returns.png` and, when the Monte Carlo runs, `5_mc_null.png`. Trailing
one-year returns use `value[t] / value[t-252] - 1` on every traded bar with a
full year behind it. The windows overlap heavily, so the report prints:
"Windows overlap: about 5,000 daily observations are roughly 21 independent
years. Do not read the smoothness as precision." The current full-sample
statistics (5,001 valid windows) are:

| Series | Mean | Median | SD | Skew | Excess kurtosis | 5th pct | Worst | Best | Share negative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Strategy | +13.96% | +14.16% | 13.24 | -0.42 | 0.80 | -11.54% | -34.52% | +60.76% | 11.26% |
| Baseline | +13.52% | +13.22% | 12.63 | -0.17 | 0.27 | -8.86% | -30.60% | +58.68% | 11.94% |
| Active | +0.44 pp | 0.00 pp | 5.78 | +1.90 | 5.45 | -5.92 pp | -10.33 pp | +28.57 pp | 43.25% |

Active has excess kurtosis > 1, so the report notes that the normal curve
understates the left tail. The active series also has many exact zeros (bars
where the strategy and baseline held the same SPY position), which is why the
share ahead in the chart is about 39% while 43.25% of windows are negative.

**Calendar years (partial first/last years excluded from the statistics):**
full-year means are strategy +13.4%, baseline +13.0%, active +0.4 pp. The
partial years are 2005 (starts 2005-12-28) and 2026 (ends 2026-11-13).

**Assumptions recorded instead of asking for the annual chart.**

1. A year is 252 traded bars, the same `BARS_PER_YEAR` used elsewhere.
2. The calendar-year table uses the last traded bar of the previous year as the
   base for a full year; the first and last calendar years are flagged partial
   when trading starts after 1 January or ends before 31 December.
3. The fitted normal curve is `pdf(x) * N * bin_width`, i.e. scaled to histogram
   counts. If the sample SD is zero, the curve is skipped with a message.
4. Active return is the arithmetic difference of the two trailing one-year
   returns, in percentage points, not a ratio.

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
   nothing ever closes it. Phase 3 adds the exit rule; the longest GLD hold is
   now 14 bars.
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
pytest                       # 70 tests at that point (79 after Phase 2, 91 after Phase 3), no internet, ~10 s
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
- Conclusion: the harness produced enough trades to evaluate, but there was
  still **no demonstrated edge**. More parameter search was not the answer;
  Phase 3 (the exit rule) was.

Reproduce:

```
source .venv/bin/activate
pytest -q                              # 79 tests at that point (91 now), no internet, ~10 s
MPLBACKEND=Agg python report.py        # regenerates report_output/, a few minutes
```

`report_output/summary.txt` now contains an `In-sample / out-of-sample` block
and, after the sensitivity grid, a `Pre-registered selection check` block. These
Phase 2 numbers are historical: Phase 3 changes the exit behaviour, so the
current default is 39 trades (section 7). The Phase 2 lesson still stands: the
21-trade default returned +341.7% IS and +224.4% OOS against a +424.1% OOS
baseline, and 0 of 27 grid cells beat that baseline.

---

## 7. Phase 3 (DONE): the exit rule

**The problem.** A signal only says "a turn just happened". Nothing in the
pre-Phase-3 rules said when the position becomes stale. Under the old
`max_age = 7` default, a 2015 GLD signal was held for 2,936 bars, essentially to
the end of the data. Under Phase 2's `max_age = 15`, the longest GLD hold was
still 1,156 bars (2016-11-03..2021-06-09), through a period when SPY
outperformed. The strategy was not wrong about the turn; it simply never let go.

**The rule.** When a position is held and no fresh entry signal fires, keep it
while the fitted turning point still supports it:

```
-max_age <= vertex <= 0
```

If the vertex has aged out (`vertex < -max_age`), moved into the future
(`vertex > 0`), or is undefined (NaN, i.e. `a == 0`), the position has lost its
support and is rotated into `FALLBACK`, which defaults to **SPY**. `None`
disables the rule and reproduces the old behaviour. Do not confuse this with the
entry gate: the exit deliberately does **not** re-check the t-statistic. An
insignificant but recent turn still supports the position; only a stale or
undefined turn triggers the exit.

**Why SPY, not cash.** SPY is the buy-and-hold benchmark and the natural neutral
holding for a long-only SPY/GLD allocator. Rotating to cash would itself be a
market-timing bet, and the experiments confirmed it: every cash fallback was
dramatically worse (`vertex_stale -> CASH` ended at GBP 1,081 vs GBP 12,312 for
SPY). Cash remains available before the first trade, as before.

**Why this rule, not a max holding period.** A maximum holding period (say 252
bars) is an arbitrary new parameter that ignores what the fit says. The
vertex-recency rule reuses the parameter the entry decision already has
(`max_age`) and asks the same question on the way out: *is the turn that
justified this position still recent?* It adds no new tuned number. Alternatives
tried and rejected on structural grounds: a slope-significance exit (adds a new
threshold), a curvature-flip exit (fires on every insignificant wiggle), and a
hard max-hold (arbitrary).

**Default result** (window 30, q95, `max_age` 15, fallback SPY):

| | Strategy | Baseline |
|---|---|---|
| Trades | 39 | 0 |
| Final GBP | 12,312 | 11,366 |
| Annualised | 12.8% | 12.4% |
| Max drawdown | -38.6% | -35.0% |
| In-sample (2005-12-28..2014-12-31) | +106.3% | +117.2% |
| Out-of-sample (2015-01-02..2026-11-13) | **+496.8%** | +424.1% |
| Longest GLD hold | **14 bars** | n/a |

Compare with Phase 2 (21 trades, GBP 14,326): the exit rule lowers the full-sample
return but fixes the staleness. The longest GLD hold falls from 1,156 bars to 14.

**Honest verdict: still no demonstrated edge.**

- **In-sample it loses** (+106.3% vs +117.2%). The full-sample win is not an
  in-sample win.
- **Out-of-sample it wins** (+496.8% vs +424.1%), but the margin is fragile.
  Nearby `max_age` values reverse it: 10 gives +409.7% (below baseline), 15 gives
  +496.8%, 20 gives +378.9% (below). A genuine effect should not change sign
  between neighbouring recency settings. Report the OOS win, but do not present
  it as robust.
- **The trade count is still luck-like**: 39 real trades vs a shuffled-daily-
  changes median of 43 (5th–95th percentile 32–55). The exit rule fires at the
  rate noise would.
- **The shifted-timeline control is encouraging but not decisive**: the strategy
  beat 89% of 300 runs that spend the same time in cash/SPY/GLD at the wrong
  dates (median GBP 9,993). Hit rate over 39 trades was 61.5%, now just above
  the 10-trade reporting floor.
- **A 39-trade, single-pair backtest cannot establish an edge.** The honest
  summary is: the exit rule fixes a real defect and is structurally motivated,
  but the evidence for outperformance remains weak and parameter-sensitive. The
  default cell also happens to be the highest-OOS cell in the 27-cell grid; that
  is a coincidence of this sample, not a selection criterion, and it is exactly
  why the neighbouring-parameter check matters.

**Reproduce:**

```
source .venv/bin/activate
pytest -q                              # 117 tests, no internet, ~11 s
MPLBACKEND=Agg python report.py        # regenerates report_output/, a few minutes
```

The report prints the exit policy, the longest holding and the longest
non-fallback *invested* holding, plus the usual diagnostics, the exact
attribution and the annual-return distribution. The Phase 3 grid is re-run
inside the normal sensitivity table; do not select on its OOS column.

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

## 8.5 Phase 5 (DONE: validation harness only)

The four gates in section 0 are implemented in `smooth_graph/evaluation.py` and
printed by `report.py`:

- `block_bootstrap_joint_prices` / `full_pipeline_monte_carlo`
- `detour_attribution` (with exact log reconciliation and bootstrap CI)
- `exact_attribution` (bar-by-bar decomposition; section 0.1)
- `tracking_error_stats`
- `friction_sensitivity` (uses the new evaluation-only
  `run_backtest(fx_fee_rate=...)` override; the default is unchanged)
- `forward_metrics` from `CLEAN_HOLDOUT_START`
- `neighbourhood_check` / `validation_verdict`
- `trailing_annual_returns`, `annual_return_statistics`, `calendar_year_returns`,
  `plot_annual_returns` and `plot_mc_null` (annual distribution charts)

`is_neighbour_median` now sorts its axes by value so the default cell really
gets a 9-cell 3x3 neighbourhood instead of a 6-cell edge block. This is an
evaluation fix, not a strategy change.

The harness is tested entirely on synthetic data (`tests/test_validation.py`
and `tests/test_annual_returns.py`; `pytest` is green at 117 tests). The
`Validation verdict` block in `report_output/summary.txt` is authoritative:
`VALIDATED` only if all four checks pass, otherwise `NOT VALIDATED`. On the
current real-data run all four checks fail, so the verdict remains
**NOT VALIDATED**. Check 2 uses the corrected exact attribution from section
0.1 with the original pre-registered threshold; its corrected CI is
[-1.43%, +2.55%], still spanning zero.

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
- **Phase 2 default (`max_age = 15`, no exit rule)**: 104 bars pass both gates,
  21 trades; the same rule on shuffled daily changes makes a median of 25
  (17–33), so the real count is noise-like. The joint luck rate is 2.5% of bars.
- **Phase 3 default (`max_age = 15`, fallback SPY)**: 39 trades; final
  GBP 12,312 vs GBP 11,366; annualised 12.8% vs 12.4%; max drawdown -38.6% vs
  -35.0%; in-sample +106.3% vs +117.2%; out-of-sample +496.8% vs +424.1%.
  Longest GLD hold 14 bars (down from 1,156). Noise check: real 39 vs median 43
  [32–55]. Shifted-timeline control: beat 89% of 300 runs (median 9,993–10,014
  depending on the run). Hit rate 61.5% over 39 trades.
- The OOS win is **not robust** to `max_age`: 10 gives +409.7% (below the
  +424.1% baseline), 15 gives +496.8%, 20 gives +378.9% (below). Do not present
  it as a stable edge.
- Median vertex when significant is -14.8 bars either way: the recency gate was
  the binding constraint, not the threshold.
- Full sample under Phase 3 +1,131.2% vs +1,036.6%. The Phase 3 grid has
  **7 of 27 cells** beating the OOS baseline (the default is one of them); the
  neighbour-robust IS pick (45/90/max_age 22) returned +487.7% OOS but only
  +72.8% IS against a +117.2% IS baseline. Selection on IS would have chosen a
  configuration that lost to the baseline in-sample, which is the honest
  warning: the grid is unstable.
- Under the old default the 2015–2026 GLD hold was 2,936 bars; under Phase 2 it
  was 1,156 bars (2016-11-03..2021-06-09), and Phase 3 cuts the longest GLD hold
  to 14 bars.
- "First half" advantage under the old default: the initial cash spell's GBP
  gain is mostly the GBPUSD entry rate (1.96 vs 1.72), not signal.
