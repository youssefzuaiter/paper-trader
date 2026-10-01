# Long-term core — design (phase 3)

**Status:** for review · **Answers:** `docs/long-term-core-brief.md` · **Written:** 2026-09-29 ·
**Read:** the brief; `docs/backtester-design.md` §0, 3, 7, 8, 10, 12–13; `backtest/strategies.py`,
`costs.py`, `fees.json`, `registry.py`, `metrics.py`, `data.py`, `calendar.py`, `pipeline.py`
(`Plan`, `Series`, `hold_series`), `runs.py` (`cmd_run`), `cli.py`, `fetch.py`; for §8, `tier0.py`
and the headers of `risk_router/`. **Not read:** the system-design PDF (§3 cash buffer, §7 policy
and Allocator). The brief's summary of it is used; §9 lists what to reconcile with it.

> **Not financial advice.** This design builds a tool that shows trade-offs. It ranks nothing and
> recommends no mix. Tickers are instruments to test.

## 0. Decisions

| Brief §7 | Decision | § |
|---|---|---|
| 1. Many assets, rules, M4 | One portfolio engine (`backtest/portfolio.py`) replaces the body of `run_hold`. Rules and weights live in a pure, stdlib-only `core_alloc.py` that paper mode imports too. Decide at the close of *t* from a point-in-time view. Sells fill at the open of *t+1*, then buys, capped by cash. M4 uses trailing 63-session volatility. | 3 |
| 2. Crypto alignment | BTC/USD sampled on the NYSE grid. Its "close" is the last 5-minute bar that ends at or before the session's `close_at` (16:00, or 13:00 on half-days). Its "open" is the bar starting at `open_at`. The weekend's move goes into Monday's return, as it does for ETFs. | 2 |
| 3. Registration and report | `register-core` freezes the whole grid, windows, stress dates, cost levels, sizes, tax rates, raise-cash scenarios and **reading rules** before any run. `run-core` refuses anything else. The report is ordered by the brief's seven questions. | 4, 5 |
| 4. Raise cash | A deterministic greedy plan: free cash, then the buffer, then water-filling on the post-withdrawal over-weights, then pro-rata. Lot choice is tax-aware and the plan is grossed up for costs. It is property-tested and backtested against a pro-rata baseline. | 6 |
| 5. Paper mode | A separate paper account and a separate router app. An owner-edited, read-only policy file. Hard ceilings live in a new `tier0_core.py`, and `tier0.py` is untouched. The router recomputes every plan and rejects a mismatch. Kill switch, breaker, receipts and paper lock are imported, not copied. | 8 |
| 6. Leakage and consistency | Twelve checks (C1–C12), each with a broken twin that must fail. These include the M4 future canary, a point-in-time oracle and an exact reproduction of phase 1's S0/S1. | 7 |

Four facts from the code shape the rest:

1. **Phase 1's return metrics are additive.** `metrics.max_drawdown` and the report's "Total" and
   "Annualised" sum daily returns. That is harmless over 1.7 years and wrong over 10.7. The core
   needs compound (wealth-path) metrics. The phase 1 functions stay as they are, for reproduction.
2. **`fees.json` has no rows before 2023.** `FeeRow.applies` is false before 2023-02-27 (SEC) and
   2024-01-01 (TAF, CAT), so 2016–2022 sales would pay **zero** fees without any error. Fixing
   that edits `fees.json`. But `registry.reproduce` checks code-shipped inputs against the *current*
   tree, not the checked-out commit, so any edit would break every phase 1 reproduction. The
   registry is fixed first (§1).
3. **`run_hold` sizes from the previous close and can take cash negative.** It computes
   `qty = value(prev close) × target / open` for every symbol, so an overnight gap-down leaves a
   small overdraft. This must be kept exactly for the S0/S1 reproduction. New runs use a cash-safe
   profile instead (§3.4).
4. **The news router would dismantle a core held in the same account.** It has a $50 gross cap, 5
   positions and 10 buys a day, and it sells everything at close − 10 minutes. The core therefore
   needs its own paper account and its own app, not a flag (§8).

## 1. Module changes

| Module | Change |
|---|---|
| `backtest/registry.py` | `reproduce` verifies code-shipped inputs (labelled relative to `ROOT`) inside the temporary worktree, and data inputs against the data root. **This comes first**, with a test that editing `fees.json` after an experiment does not break that experiment's reproduction. |
| `backtest/fees.json` | Pre-2023 SEC Section 31 and FINRA TAF rows, filled from the SEC fee-rate advisories and FINRA's schedule at build start (from memory: not trusted). Crypto rows with a new optional `asset_class` field (default `equity`), taken from Alpaca's crypto fee schedule. Existing rows are unchanged. `_check_coverage` runs per (fee, asset class). |
| `backtest/costs.py` | Spread groups become a symbol → group table: `tight` and `wide` keep phase 1's numbers and members, `etf` and `etf_wide` are new, and so is `crypto`. `CostLevel` gains fields with defaults at the end, so `LEVELS` values for the 8 stocks do not change. |
| `backtest/fetch.py` (+ `ResearchData.crypto_bars`) | `CorePaths` under `.cache/backtest/core/`: calendar 2016-01-01 → 2026-12-31; daily `all` and `raw` bars for every instrument in the brief's §3 table; BTC/USD 5-minute bars from 2021-01-01; optionally an ETF quote sample (O10). **Phase 1 files are never touched**, so their manifests stay valid. |
| `backtest/daily.py` (new) | `DailyMarket` (calendar + daily bars + BTC samples), `DailyView` (decision side, bound to one instant), fill accessors (never passed to decision code). |
| `core_alloc.py` (new, repo root, stdlib only) | Weight functions (fixed; inverse-volatility), rules (calendar, band, combined), `plan_rebalance`, `plan_raise_cash`, lot and tax arithmetic. Pure functions of `Decimal` and floats, like `tier0.py`, so the router image can import them without numpy. |
| `backtest/portfolio.py` (new) | The engine: `simulate(market, spec, sessions, level, fees, profile)` → one record per day. |
| `backtest/strategies.py` | `run_hold` keeps its signature and `DailyDay` but becomes a call to `portfolio.simulate` with the `PHASE1` profile. There is no second engine (C7 checks this by AST). |
| `backtest/metrics.py` | Compound metrics (§5.1). A cached stationary-bootstrap index matrix, drawn in exactly the order `block_bootstrap` draws today, so phase 1 intervals are unchanged and every core series in a window shares one set of resamples. |
| `backtest/report.py` | `core_report` (§5.2). |
| `backtest/leakage_core.py` (new) | C1–C12 (§7), reusing `leakage.Check`. |
| `backtest/schema.sql`, `store.py` | New `core_day` table (run, date, value, cash, weights JSON, traded, spread, slippage, fees, tax, deferred flag). Orders go into `trade_event` as `filled`, with reason `rebalance`, `raise_cash` or `tax`. Portable DDL only. |
| `backtest/cli.py` | `ingest-core`, `register-core`, `run-core`, `leakage-core`, `raise-cash`. Each registers an experiment first, and `reproduce` covers all of them through `replay-experiment`. |

`engine.py`, `sim_broker.py`, `walkforward.py` and everything in phase 1's news path are untouched.

## 2. Data and the crypto cutoff (Q2)

**Instruments.** One primary per class (O2): US stocks **VTI**, non-US **VXUS**, bonds **BND**, gold
**IAU**, real estate **VNQ**, cash **BIL**, crypto **BTC/USD**. SPY, IEF, TLT, GLD and SGOV are
fetched and used only in registered robustness rows (§4). Bars are SIP. Alpaca's `all`
adjustment is used for performance and `raw` for share counts, cent rounding and fees.

**Point in time.** A daily bar for session *d* is readable from `close_at(d)` + 15 minutes, which
is phase 1's P2 rule applied to the calendar's close. Adjusted *levels* are not point in time,
because later dividends rescale earlier prices. Adjusted *ratios* are, provided the adjustment is
proportional (to verify, §9). So decision code only ever uses returns and weights. C5 enforces
this.

**Crypto on the NYSE grid.**

- BTC close(*d*) = close of the last 5-minute bar ending at or before `close_at(d)`. On half-days
  that is 13:00, not 16:00, so all assets share one cutoff per session. The cutoff is converted
  from America/New_York to UTC, so daylight saving time is handled.
- BTC open(*d*) = open of the 5-minute bar starting at `open_at(d)`.
- If the chosen bar is more than 60 minutes older than its instant, the session is flagged and
  the run fails. It never forward-fills silently.
- Weekend and holiday moves fall into the next session's return, exactly as an ETF's weekend
  news falls into Monday's gap. Volatility and correlations are therefore computed on one
  sampling grid.
- Alpaca's daily crypto bar is not used. Its day boundary is not the NYSE close, and C6's broken
  twin uses it and must fail.
- Five-minute bars keep 09:30, 13:00 and 16:00 on bar boundaries at about 600k rows. One-minute
  bars would work too, at five times the size.

**Windows** (frozen at registration). **Full:** from the first session after M4's 63-session
warm-up (about 2016-04-05) to the last complete session fetched. **Crypto:** the same rule from
BTC's first session (about 2021-04-06). Every mix in a window starts on the same day, so M4 is
never compared over a different period.

## 3. The daily simulation (Q1)

### 3.1 State and loop

State in `Decimal`: cash; per asset, quantity in adjusted units plus tax lots (raw cost basis,
date); the frozen target weights of the last rebalance; any pending plan. For each session *k*:

1. **Open (fill side).** If a plan is pending:
   - (a) *Router-faithful breaker.* If the portfolio's value at the open is ≤ −2.5% against the
     previous close and the plan contains any buy, the whole plan is deferred and re-decided at
     this close. This is live behaviour (§8.3) and is counted in the report.
   - (b) Sells fill at *o*·(1 − h·m − σ).
   - (c) Buys fill at *o*·(1 + h·m + σ). Each buy's notional comes from the plan, scaled
     pro-rata down to the cash actually available after the sells.
   - Orders below `min_order_usd` (default $1, Alpaca's fractional minimum, to verify) are
     skipped, and the drift stays.
2. **Close.** Mark at adjusted closes. Record value, weights, traded notional, costs by
   component and realised gains.
3. **Decision (as of `close_at` + 15 min).** Pay any tax due (§3.5) and any registered cash need
   (§6). Then `rule.due(view, state)`. If it is due, `weights(view)` gives new targets and
   `plan_rebalance` gives sells (in quantity) and buys (in notional), both computed from close
   *k*. The plan fills at open *k+1*.

The initial build is the plan decided at the close of the warm-up's last session, sized on
`capital`.

### 3.2 Rules

| Rule | Checked at | Rebalances when |
|---|---|---|
| none | never (after the initial build) | never |
| monthly / quarterly / yearly | close of the period's last session (the calendar is published in advance, so this is not leakage) | always |
| band ±5 / ±10 | every close | any \|wᵢ − targetᵢ\| > band (absolute points, D6) |
| monthly-check ±5 *(= phase 1 S1)* | close of each month's last session | any weight outside ±5 |
| quarterly-check ±5 | close of each quarter's last session | any weight outside ±5 |

These are eight rules. The two combined ones answer §2.4's "combinations", and one of them is S1.
A rebalance always goes fully back to target, as S1 does. A ±5 band on BTC's 5% sleeve fires at 0%
or 10%, and the report states this.

### 3.3 Weights

- **Fixed** (M1–M3 and their crypto versions): constants from the registration.
- **M4, inverse volatility.** σᵢ is the sample standard deviation of the last 63 session log
  returns, closes (*t*−63 … *t*). The weights are wᵢ ∝ 1/σᵢ, recomputed **only when a rebalance
  is due**. Between rebalances the band check measures drift against the frozen weights of the
  last rebalance. M8 fixes BTC at 5% and risk-balances the other 95%; BTC is not inverse-vol
  weighted.
- **Determinism.** Weights are quantised to 10⁻⁶ in `Decimal`, and the rounding residual goes to
  the largest weight, so they sum to exactly 1. Paper mode calls the same function and gets the
  same bits.

### 3.4 Sizing profiles and prices

| Profile | Used by | Sizing | Prices for fees and rounding | Fee rounding | Breaker |
|---|---|---|---|---|---|
| `PHASE1` | S0/S1 reproduction only | `qty = value(prev close) × w / open` for every asset; cash may go negative | adjusted | per order (as `run_hold`) | off |
| `CORE` | every registered core run | sells by quantity, buys by notional, both from close *t*; buys capped by cash | fills and deltas in adjusted units; **raw** shares = adjusted units × (adj/raw) for TAF/CAT, and the raw price for $0.0001 rounding | the level's (D12: daily at optimistic/central, per order at pessimistic) | on |

Market orders only, so D1 (limit buys) does not bind. D3 becomes two registered capital sizes: $100,000,
as in phase 1, and the owner's size (O5). The fee floor and the $1 minimum matter at small sizes.
Dividends are reinvested in the paying asset through the adjusted bars. That is the total return the
brief asks for. Paper mode receives cash instead, which the next rebalance sweeps in (§8.4).

### 3.5 Taxes

Every sale realises gains against lots. The method is average cost by default (O7). Gains are netted
per calendar year and losses carried forward. Tax = rate × max(net gain, 0). It is decided at the
year's last close and paid by a raise-cash plan (§6) at the next open, and that plan can realise gains
of its own in the new year. The rate defaults to 0. The registered illustrative rates, 15% and 30%, are
**labelled illustrative everywhere**. The report adds an after-tax liquidation value at the end, so
"none" is not flattered by its unrealised gains.

## 4. Candidate set and the registration step (Q3)

**Proposed change to the brief's set: none to the mixes.** Two combined rules are added (§3.2).
Robustness rows are registered but are not candidates:

| Family | Window | Mixes × rules × levels × sizes | Runs |
|---|---|---|---|
| Core | full | M1–M4 × 8 × 3 × 2 | 192 |
| Crypto comparison | crypto | M1–M8 × 8 × 3 × 2 | 384 |
| Tax (illustrative 15%, 30%) | full | M1–M4 × 8 × central × owner's size × 2 rates | 64 |
| Instruments | full | M2 with IEF, then TLT, for BND; M3 with SPY for VTI, GLD for IAU; rules none and quarterly; central; owner's size | 8 |
| Buffer and raise cash | full | M1–M4 with a 0% and 5% BIL buffer; rules none and quarterly; central; scenarios in §6 | 16 + scenarios |

M1 = 100% VTI. M2 = 60 VTI / 40 BND. M3 = 20% each of VTI, VXUS, BND, IAU and VNQ. M4 = inverse
volatility over the same five. M5–M8 = M1–M4 × 0.95 + 5% BTC/USD.

**`python -m backtest register-core`** writes one experiment (command `core-registration`) before any
core number exists. It refuses to run if any `run-core` experiment is already in the store, unless the
new registration names the one it supersedes, and a superseded registration still counts. Its
parameters are:

- **Universe:** tickers, classes and the mix definitions above; M4's lookback and estimator.
- **Grid:** rules with exact semantics, the three cost levels with the new spread groups, the crypto
  fee, the fee-table version, the two sizes, `min_order_usd`, the sizing profile and the breaker
  flag.
- **Windows:** the exact first and last sessions of both windows, the two halves of each (split at
  the midpoint session) and the **stress windows**. These are S&P 500 closing peak to trough, to be
  confirmed on SPY closes and then frozen: 2018-09-20 → 2018-12-24; 2020-02-19 → 2020-03-23;
  2022-01-03 → 2022-10-12; 2025-02-19 → 2025-04-08.
- **Tax:** rates, lot method, netting.
- **Raise cash:** the scenarios (§6.3).
- **Statistics:** bootstrap resamples (10,000), mean block (5 days), seed, and the risk-free proxy
  (BIL total return).
- **Reading rules** (below), the **owner's tolerance** (O5) and the fixed text of the history
  warning.

`run-core` takes a registration id, reruns it and records the registration's parameters verbatim.
Any difference means a new registration. **Reading rules**, the core's counterpart of phase 1's pass
rule:

- **R1.** Two configurations differ on a metric only if all of these hold: the paired 95% interval
  excludes zero at the central level; the sign agrees at all three levels; and it agrees in both
  halves. Otherwise the report prints *indistinguishable on this history*.
- **R2.** Rebalancing *helps on return* for a mix under a rule if R1 holds for its mean daily excess
  over "none". It *helps on risk* if R1 holds, with a negative sign, for the paired volatility
  difference. Maximum drawdown is shown, with no interval.
- **R3.** Crypto *changes the picture* for Mₖ if R1 holds for M₍ₖ₊₄₎ − Mₖ on mean return or on
  volatility. The drawdown difference is always shown.
- **R4.** A mix is marked *within your stated tolerance* if, at the pessimistic level, its maximum
  drawdown and longest underwater spell stay within the owner's registered numbers. The history
  warning applies.
- **No winner field.** The report never says which mix to hold.

**Counting.** The report prints "grid of G runs, experiment N of M on this window", with M from
`registry.configuration`. The deflated Sharpe uses the number of *selectable* configurations in
the window (mix × rule: 32 full, 64 crypto), not cost levels or sizes, which the owner cannot
choose (O6).

**Lock-box.** There is none, because there is no fitting and every row is in-sample. The defences
are pre-registration, a small fixed grid, R1 and the deflated Sharpe. The **paper period (§8) is
the out-of-sample test.**

## 5. Report (the brief's §2)

### 5.1 Metrics (daily series of wealth Wₜ = Vₜ/V₀, net of all costs)

| Metric | Definition |
|---|---|
| CAGR | W_T^(365.25 / calendar days) − 1 |
| Volatility | std(daily returns) × √252 |
| Sharpe | mean(r − r_BIL) / std(r − r_BIL) × √252. Phase 1's raw Sharpe is kept for the S0/S1 table. |
| Sortino | mean(r − r_BIL) / downside deviation (below 0) × √252 |
| Calmar | CAGR / \|max drawdown\| |
| Max drawdown | min over t of (Wₜ / max₍ₛ≤ₜ₎ Wₛ − 1), **compound**, with peak, trough and recovery dates |
| Longest underwater | peak to recovery, in sessions and calendar days; "not recovered (≥ n)" if open at the end |
| Worst calendar year | compound return per year; partial first and last years labelled |
| Turnover | one-way traded notional / mean value, per year; rebalances per year; breaker deferrals |
| Costs | spread, slippage and fees per year in bps of mean value, per level; tax paid (illustrative) |

Intervals use the stationary block bootstrap: mean block 5 days, 10,000 resamples, recorded seed,
**the same resample indices for every series in a window**, so paired differences are coherent.
They cover CAGR, volatility, Sharpe, Sortino, Calmar, and the paired differences in mean and in
volatility. They are computed at the central level. The other levels are shown as point estimates,
and R1 reads their signs. Maximum drawdown gets no interval; §2.5's stress windows are its evidence
(O14).

### 5.2 Layout

- **0. Header:** provenance, registration id, grid and experiment count, **the history warning**,
  and the not-advice line. The history warning is fixed text: *about 10.5 years of one long bull
  market with four sharp shocks; no 2008, no 2000–2002; rates near zero for half of it; differences
  smaller than the intervals are not evidence; nothing here predicts the next ten years.*
- **1. How bad (§2.1):** mix × {none, quarterly} at central: maximum drawdown (dates), longest
  underwater, worst year, and R4's mark.
- **2. What it earns (§2.2):** the same rows, with CAGR, volatility, Sharpe, Sortino and Calmar
  [95%].
- **3. Does rebalancing help (§2.3, §2.4):** per mix, all eight rules. Rows show Δ mean vs none
  [95%], Δ volatility [95%], maximum drawdown, turnover, rebalances a year, costs by component at
  three levels, and R2's verdict.
- **4. Stress (§2.5):** mix × window. Rows show return over the window, drawdown inside it, and days
  to regain the pre-window peak, plus each instrument's own return in the window. The 2022 rows
  show bonds falling with stocks.
- **5. Crypto (§2.6):** crypto window only, Mₖ beside M₍ₖ₊₄₎ for every rule. Rows show the paired
  differences and R3's verdict, with BTC's 5-minute cutoff restated.
- **6. Raise cash (§2.7):** §6.3's scenarios.
- **7. Robustness:** halves, cost levels, both sizes, instrument substitutions, taxes at 15% and 30%
  (illustrative).
- **8. Checks:** the ids of the last passing `leakage-core` run and of the phase 1 reproduction.
- **9. Not modelled:** dividend withholding, currency (O7), intraday execution, and the
  borrow-free assumption.

## 6. The "raise cash" plan (Q4)

### 6.1 Rule (`core_alloc.plan_raise_cash`)

**Input.** Amount *X* net of costs and tax, need date *D*, the state at the decision close, target
weights, the level's cost rates and the tax spec. The last sell session *s* is the latest with
settle(*s*) ≤ *D*: T+1 from 2024-05-28, T+2 before, and immediate for crypto. If no such session is
after the decision, the plan refuses with the reason `too_late`. It also refuses with
`exceeds_portfolio` if *X* > the liquidation value net of costs.

The plan draws on these sources in order, gross-up included:

1. Free cash above the reserve.
2. The buffer sleeve (BIL), down to zero.
3. **Water-filling on post-withdrawal over-weights.** With V′ = V − X, each asset's excess is
   eᵢ = valueᵢ − targetᵢ·V′. Sell from the largest excess until it equals the next largest, then
   both, and so on. Every dollar sold here *reduces* drift.
4. When no excess is left, sell pro-rata to target weights, which keeps the mix on target.

**Within an asset**, lots are chosen by the tax method. With a rate above zero and a method that
allows it, the highest cost basis goes first. **Between assets tied within 0.5 points of excess**,
the lower cost per dollar (h + σ + fees) goes first.

**Gross-up.** Proceeds needed = X + spread, slippage, fees and tax on the sells. Tax depends on
the lots, so the plan is solved incrementally per lot. The result is net ≥ X and exceeds it by
less than one cent plus rounding.

**Output.** The orders; cost by component; realised gain and tax; drift after (maximum
\|w − target\| and L1); and whether the next rule check would trigger.

### 6.2 Tests (unit and property-based, seeded)

- Net proceeds ≥ X, and less than X + $0.01 + one rounding step per order.
- Nothing is sold below zero.
- The buffer is used before any mix asset.
- No under-weight asset is sold while an over-weight one remains.
- With tax 0 and equal costs, the plan equals pure water-filling.
- HIFO picks the highest-basis lot.
- `too_late` and `exceeds_portfolio` are refused.
- Crypto settles the same day.
- The same inputs give the same plan in `core_alloc` whether called from the engine or directly.

### 6.3 Backtested scenarios (registered)

- *X* ∈ {2%, 10%, 25%} of value.
- *D* at each stress window's trough (the worst time to need cash) and on 20 seeded random dates.
- Mixes M1–M4, buffer 0% and 5%, rules none and quarterly, central level.
- Every scenario runs twice: **this plan** and a **pro-rata baseline** (sell every asset in
  proportion).

Reported: cost, tax, drift after, the drawdown at the time of sale, and the portfolio value one
year later. The last measures what "selling bonds, not stocks, at the bottom" was worth on this
history.

## 7. Leakage and consistency tests (Q6)

Each check has a **broken twin that must fail**. `leakage-core` runs them all on real data as a
registered experiment, and `run-core` refuses to start without a passing one at the same commit.

| # | Check | Broken twin |
|---|---|---|
| C1 | **Future canary for M4.** At 500 seeded decision instants, replace every price after *t* with 10⁶× itself; M4's weights must be bit-identical. Also: changing close *t* changes them, and changing close *t*−63 does not (the window is exactly right). | The window includes *t*+1. |
| C2 | **Point-in-time oracle.** `DailyView` at 10,000 random instants equals a brute-force scan; any read past *t* raises `LookAheadError`. | A view that serves session *t*'s bar before `close_at` + 15 min. |
| C3 | **Execution lag.** Every fill's session is strictly after its decision session and uses that session's open. | Fill at the decision close. |
| C4 | **Band trigger.** Recomputing every band decision from the stored end-of-day positions gives the same triggers. | The trigger reads *t*+1's open. |
| C5 | **Adjustment invariance.** Multiplying all adjusted prices before a random date by *k* leaves every decision identical and every return equal to within 10⁻¹². | Volatility on price differences instead of returns. |
| C6 | **Crypto cutoff.** No BTC sample uses a bar ending after its session's `close_at`, including on half-days and across daylight saving time. | Alpaca's daily crypto bar. |
| C7 | **Phase 1 through the new engine.** `run_hold` via the `PHASE1` profile reproduces, **exactly**, a golden file of every `DailyDay` (all levels, 2025-01-02 → 2026-09-18) generated once from commit `f700c0d` (its hash committed), and the S0/S1 rows of run `x-20260927-112403-badb5a8d` (clean, f700c0d). Plus an AST check that `run_hold` only calls `portfolio.simulate`. | `PHASE1` with cash-safe sizing, or with daily fee rounding. |
| C8 | **Accounting.** Every day, value = cash + Σ positions. Under `CORE`, cash ≥ 0, weights sum to 1 and fees ≥ 0. | Buys not capped by cash. |
| C9 | **Independent reference.** At the frictionless level, a vectorised numpy constant-mix and buy-and-hold computation (separate code) matches the engine to 10⁻¹⁰. | The reference shifted by one day. |
| C10 | **Registration lock.** `run-core` refuses without a registration, with changed parameters, or without a passing `leakage-core`. | The guard removed. |
| C11 | **Data completeness.** Every instrument has a bar for every session in its window, with no gaps filled. | Forward-fill. |
| C12 | **One implementation.** The paper Allocator and the router import `core_alloc`'s functions (identity check), and `tier0.py`'s constants equal phase 1's (the existing L10 source check). | A copied function. |

A `reproduce` of a `run-core` experiment must give identical metrics and report hash.

## 8. Paper-trading mode (Q5): designed, not built

### 8.1 Separation

The core trades a **second Alpaca paper account** (O11), through a **separate app**
(`risk_router/core_app.py`) and a separate deployment and keys. At startup it refuses to run unless
the account number equals the policy's `account`. The news app refuses that account number too.
The mode is fixed by which app runs, never by a request field or an environment flag. News-trading
limits in `tier0.py` stay exactly as they are.

### 8.2 The policy file (`policy/core.toml`, read with stdlib `tomllib`)

It is owner-edited and changed only by a reviewed commit. It is mounted read-only (ConfigMap,
`readOnly: true`), and no agent has a write path to it. Its sha256 goes on every receipt.

| Field | Meaning |
|---|---|
| `account` | the core paper account number |
| `registration`, `evidence_run` | the registration and `run-core` experiment ids the choice rests on |
| `mix` | instrument → weight, or `inverse_vol` with lookback and fixed sleeves |
| `rule` | one of §3.2's rules |
| `buffer` | BIL sleeve target and minimum |
| `limits` | per-order ceiling, turnover per rebalance, `min_order_usd`, cash reserve |
| `effective_from` | first session the policy applies |

At load, the router validates the policy against `tier0_core.py`. A policy beyond any ceiling
disables core trading (fail closed) and sends a receipt saying why.

### 8.3 Router limits for this mode (`tier0_core.py`, hardcoded `Final`s, reviewed like `tier0.py`)

| Limit | Proposed (paper) | Replaces |
|---|---|---|
| `ALLOWED_SYMBOLS` | the registered instruments only | any symbol |
| `MAX_CRYPTO_WEIGHT_PCT` | 10 | — |
| `MAX_TURNOVER_PER_REBALANCE_PCT` | 25 (one-way, of equity); the initial build needs a control-signed `fund` command | $50 gross cap |
| `MAX_ORDER_NOTIONAL_USD` | 100,000 | $10 per order |
| `MAX_REBALANCE_PLANS_PER_MONTH` | 2 (plus deferrals of the same plan) | 10 buys a day |
| `MAX_ORDERS_PER_DAY` | 2 × the number of instruments | — |
| Order types | ETFs: market, day, submitted 09:25–09:29 ET (to fill at the open, as modelled); BTC: market IOC at 09:30 ET; no margin, no shorts | marketable limit + stop |
| Exits | **none**: no stop-loss, take-profit or session exit on core positions | sell everything at close − 10 min |
| Plan check | the router recomputes the plan with `core_alloc` from the policy, Alpaca's positions and the day's bars, and rejects any difference (`plan_mismatch`) | `prob_up` gate |

**Unchanged and imported, not copied:**

- **Kill switch:** the same `StateStore` latch. PFW's halt stops both apps.
- **Daily loss breaker:** `tier0.CIRCUIT_BREAKER_DAILY_PNL_PCT`, the same equity vs `last_equity`
  test. OPEN (any buy) is blocked, so a rebalance containing a buy is deferred **whole** to the next
  session, exactly as the backtest models (§3.1). REDUCE-only raise-cash plans still pass.
- **Signed receipts:** outbox → PFW, HMAC.
- **Paper lock:** `AsyncAlpaca`'s host check.

### 8.4 Daily flow

1. **16:20 ET.** The Allocator runs `core_alloc` on the published bars. If the rule is due, it signs
   a plan with its own secret, and the router verifies it and receipts it (`plan`).
2. **09:25 ET, next session.** Sells go in. Buys follow after the sells fill, capped by cash, then
   BTC.
3. Receipts go out (`filled`, with pre and post weights and the policy hash).
4. **After the close.** Reconcile with Alpaca's positions. Dividends received as cash wait for the
   next rebalance.
5. **Shadow line.** The engine re-simulates the same day on the same data. Each order's paper fill
   versus the modelled fill (implementation shortfall) is stored. After a few months this
   **measures** *h* and σ, which checks the cost levels with evidence.

**Raise cash** is an owner request on the control route (amount, need-by). The plan is receipted and
executes only after a control-signed confirmation.

## 9. To verify when the build starts

1. Alpaca's `all` adjustment is proportional for cash dividends (C5 relies on this), and BIL's
   monthly distributions appear in it.
2. The Section 31 and TAF rates for 2016–2022 (sources, as `fees.json` already does). Alpaca's
   crypto fee tiers, and whether they changed since 2021.
3. The fractional minimum ($1?), and whether fractional and notional orders placed before 09:30
   fill at the opening print. Paper crypto order rules (TIF, minimums).
4. BTC/USD 5-minute bars from 2021-01-01 on the free endpoint, and how many 5-minute intervals
   have no trade.
5. The exact first sessions after warm-up, and the stress dates against SPY closes.
6. The system-design PDF's §3 (buffer, raise-cash order) and §7 (policy fields, the Allocator's
   role). Reconcile any difference before registration.
7. Runtime of the whole grid on one core at `nice 19` with `OMP_NUM_THREADS=1`. Expected: minutes
   for the simulation. The bootstrap uses the cached index matrix in chunks.

## 10. Open decisions for you

| # | Decision | Default here | Recommendation |
|---|---|---|---|
| O1 | Longer history (FRED Treasury yields, Ken French library, Shiller data, Stooq index series) | Not used | Keep out of v1. If you want 2000–2002 and 2008, pick one source after reading its terms, as a separate registered study; it cannot use the same ETF fills. |
| O2 | Instrument per class | VTI, VXUS, BND, IAU, VNQ, BIL | As default. GLD instead of IAU if you prefer liquidity to the lower expense ratio. |
| O3 | M4 lookback and estimator | 63 sessions, std of log returns | As default. One registered lookback; a 252 variant adds 24 selectable configurations (M4 and M8 × 8 rules across the two windows). |
| O4 | Rule set | 8 rules incl. two combined | As default (S1's rule has to be in it anyway). |
| O5 | Your size and tolerance | $100,000 plus your size; tolerance unset | Register your intended size and your limits (maximum drawdown, years underwater) **before** the first run. |
| O6 | Deflated Sharpe count | Selectable configurations (mix × rule per window) | As default. The report prints all runs too. |
| O7 | Tax model | Rate 0; 15% and 30% illustrative; average cost; annual netting | Confirm the lot method your country allows. If you are not a US tax resident, US-source dividends are usually withheld at source (rate set by treaty): add a withholding parameter (default 0) and ask a local adviser. Currency effects (a USD portfolio measured in your home currency) are out of scope. |
| O8 | Buffer | 0% and 5% BIL, as a scenario family | Set your real buffer (months of expenses) before registration. |
| O9 | Crypto | BTC/USD only, fixed 5% | As default. Check that Alpaca offers crypto to your live account before relying on it. |
| O10 | ETF spreads | Phase 1's two spread groups reused (placeholder) | Fetch a small free quote sample of the ETFs and BTC (as D9) and set the levels from it before registration. |
| O11 | Core account | Separate paper account | Yes: sharing one with the news router breaks both. |
| O12 | Breaker for the core | −2.5% latch; the whole rebalance is deferred | As default (the brief keeps the breaker). The report counts the deferrals, so you see what it costs. |
| O13 | Policy format and change process | TOML, reviewed commit, restart | As default. |
| O14 | Interval for drawdown and Calmar | 5-day blocks; no interval for maximum drawdown | As default. A 21-day block is a registered sensitivity if you want one. |
| O15 | Paper execution time | At the open (matches the backtest) | As default. A 10:00 execution is cheaper but needs intraday bars and a new registration. |
| O16 | Registry fix before any core work | Yes | Yes: without it, extending `fees.json` breaks every phase 1 reproduction. |
