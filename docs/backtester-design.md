# Backtester — design (phase 1)

**Status:** for review · **Answers:** `docs/backtester-brief.md` · **Written:** 2026-09-26 ·
**Read:** the brief and its §2 files only; facts that live in other files are listed in §11, to
verify when the build starts.

## 0. Decisions

| Brief §7 | Decision | § |
|---|---|---|
| 1. Replay style | Hybrid: vectorised up to the `prediction` table, then a minute-by-minute replay through the unmodified live router. Only the replay produces P&L. | 1 |
| 2. Fill model | Raw 1-min SIP bars ± per-symbol half-spreads; limit buys fill only through the limit, at the limit; market orders at the next open ± costs; three levels; fees rounded up to the cent. | 3 |
| 3. Router offline | Broker and clock are already injectable; `SimAlpaca` speaks Alpaca's JSON; one refactor makes the `prob_up` gate a constructor argument. | 4 |
| 4. Walk-forward | Monthly refits, expanding window, 5-session embargo, first scored month 2025-01; lock-box from 2026-09-21. | 6 |
| 5. Store | DuckDB + Parquet, portable DDL; PostgreSQL + TimescaleDB for the live writers. | 7 |
| 6. Leakage tests | Twelve, each with a broken twin that must fail. | 9 |
| 7. Registry | A row before any computation; commit, data hash, parameters, seeds; exact `reproduce`. | 8 |

Three facts from the code shape the rest:

1. **The live router cannot trade the current model.** `ROUTER_MIN_PROB_UP` is 0.60; in
   `report.md` no tradeable article reached even 0.55. A live-faithful backtest makes no trades (§4).
2. **The live router cannot execute S2.** Overnight signals are rejected as `market_closed`, then as
   `stale_signal`. S2 needs a pre-open scheduler, simulated here as strategy code (D4).
3. **At $10 an order, fees may dominate.** If each fee is rounded up to the cent per order (to
   verify), a $10 sale pays at least 10 bps, the size of `report.md`'s edges (§3, D3).

## 1. Architecture and modules

Predictions do not depend on the portfolio; trades do. No-pyramiding, five positions, $50 gross
(working buys included), ten buys a day and the four-hour cooldown make each trade depend on the
earlier ones, which a vectorised P&L gets wrong. So a **research layer** (NumPy, scikit-learn:
`train_return_model.py` generalised) builds events → samples → walk-forward folds → `prediction`
and `outcome` rows, and an **execution layer** replays each session minute by minute: strategies
emit `TradeSignal`s, the production `RiskRouter` decides, `SimAlpaca` fills (§4). Vectorised gate
tables are *screening*, never results. Budget: 21 GBM fits ≈ 1 min; a replay takes seconds because
idle minutes are skipped; the whole suite runs in under 15 min.

New package `backtest/`, tests in `tests/backtest/`. Reused as they are: `swarm/features.py`,
`tier0.py`, `risk_router/policy.py`, `risk_router/gatekeeper.py` (after §4), `swarm/alpaca_data.py`,
and `RegularBars`, `DailyHistory`, `build_dataset`, `split` from `swarm/train_return_model.py`
(legacy mode).

| Module | Interface (signatures only) |
|---|---|
| `calendar.py` | `Calendar.session(d)` → `Session(open_at, close_at, half_day)`; `session_at(ts)`; `next_session(ts)` |
| `data.py` | `MarketData.as_of(t)` → `PitView` with `daily(sym)`, `quote_bar(sym)`; `MarketData.fill_bars(sym, not_before)`; `factor(sym, session)` |
| `events.py` | news → `event`, `event_symbol`; `cluster(events, window, jaccard)` → `story_id` |
| `dataset.py` | `build(events, market, cfg)` → `Samples`: features as of `made_at`, labels; `cfg.mode` legacy or v2 |
| `walkforward.py` | `Schedule(...)`; `folds(samples)`; `run(samples, schedule)` → `list[FoldResult]` (model, thresholds, move tables, predictions) |
| `outcomes.py` | `resolve(predictions, market)` → `list[Outcome]` |
| `costs.py` | `CostLevel`; `FeeTable.sale_fees(at, qty, price)` → `Decimal`; `FillModel.try_fill(order, bar)` → `Fill` or `None` |
| `sim_broker.py` | `SimAlpaca`: async `clock`, `positions`, `orders(status, after)`, `latest_quote`, `submit_order`, `account`; `advance(t)`; `mark(mode)` |
| `engine.py` | `Engine(market, calendar, strategy, level, params).run(window)` → `RunResult` |
| `strategies.py` | `signals_at(t, view)` → `list[TradeSignal]`; `direct_orders_at(t, view)` for S0/S1 |
| `store.py`, `schema.sql` | `Store.put_events / put_predictions / put_outcomes / append_trade_events / register / finish` |
| `registry.py` | `begin(hypothesis, params)` → `Experiment`; `manifest(paths)` → hash; `reproduce(id)` |
| `metrics.py`, `report.py` | `block_bootstrap(daily, stat, mean_block, n, seed)`; one report per strategy × level |
| `cli.py` | `python -m backtest` with `ingest`, `walkforward`, `run`, `reproduce`, `lockbox` |

## 2. Data and point-in-time rules

Cached and reused: the 46,370 articles; adjusted daily bars (features as live, 1-day and 1-month
outcomes, S0/S1); adjusted 30-min bars (reproduction only). New, free, about 10 minutes of paced
requests: **raw 1-min SIP bars** (quotes, fills, stops, intraday outcomes, v2 labels), **raw daily
bars** (adjustment factor = adjusted close / raw close), the **calendar** (`/v2/calendar`) and news
**`updated_at`** (revision diagnostic, §9). All as Parquet under `.cache/backtest/`, listed in a
manifest (§8).

Feature and strategy code only sees a `PitView` bound to one instant, which raises
`LookAheadError` on any access beyond it. Fill code reads `fill_bars`, which never feeds a decision.

| Rule | Enforced in |
|---|---|
| P1 An event is usable from `known_at` (`created_at` in the backfill, receipt time live); a prediction on it is made at `known_at` + `signal_latency` (default 60 s; 0 in legacy mode). | `PitView` |
| P2 A daily bar is usable from 16:15 New York on its session (`completed_sessions`, unchanged). | `PitView.daily` |
| P3 Quotes and marks use a 1-min bar from its end (live quotes are real-time IEX); an intraday bar as a model input (none today) only from its end + 16 min. | `PitView` |
| P4 An order fills only on a bar starting at or after decision + `latency_bars` minutes (≥ 1, default 1). | `SimAlpaca`, assertion |
| P5 A model scoring at *t* was trained only on labels resolved before its fold start − embargo, and fold start ≤ *t*. | `walkforward`, assertion |
| P6 Thresholds, calibration and move tables come from the fold's own training window. | `walkforward` |
| P7 Nothing from the lock-box is readable outside a registered lock-box run. | `MarketData`, `Store` |

The legacy label breaks P4: `forward_return` uses `bisect_left`, so an article stamped 10:00:00
enters at the open of the bar starting 10:00:00. Legacy mode is exempt from P4 and serves only the
reproduction.

**Calendar.** Sessions and half-days come from the calendar, never from weekday rules.
`SimAlpaca.clock()` serves `is_open`, `next_open` and `next_close` from it, so the router's own
`session_close` logic handles a 13:00 close as live (no entries from 12:30, exit at 12:50).
`features.is_regular_hours` stays unchanged: it is part of the model's input contract.

**Corporate actions.** Trading uses **raw** prices (quotes, limits, quantities, fills, fees):
`plan_buy` rounds to cents and divides by price and dollar ATR, and fees are per share, so on
adjusted prices NVDA before its June 2024 10-for-1 split would be sized at ten times its real share
count. The `atr` passed to `plan_buy` is computed as live does, then converted to raw dollars with
the last session's factor. Features use **adjusted** daily bars, as training and serving do (ratios
do not depend on the adjustment date). Intraday returns are identical in both spaces, so the 30-min
cache stays valid for labels; S2/S3 positions never cross a session (the engine flags any that
do). S0/S1 and the 1-day and 1-month outcomes use adjusted closes (dividends reinvested free): a
slightly flattered benchmark, so every comparison against it is conservative.

## 3. Fill and cost model

*d* = decision time; *h* = half-spread; *m* = opening multiplier for quotes and fills in a
session's first 5 minutes (1 otherwise); σ = slippage. All in `Decimal`: buys round up and sells
down to $0.0001; cash debits round up and credits down to the cent.

1. **Quote.** mid = close of the last regular 1-min bar that ended at or before *d*;
   bid = mid·(1 − h·m) rounded down to the cent, ask = mid·(1 + h·m) rounded up; timestamp = that
   bar's end. The router's own 60 s `MAX_QUOTE_AGE` then applies.
2. **Limit buy** (`plan_buy`'s *L*, day order). Eligible bars start at or after *d* + latency, in
   the same session. It fills on the first eligible bar whose low is strictly below *L*, **at
   *L***: the brief's "worse of the limit and the open" read literally (never above *L*, no price
   improvement; D1). Unfilled at the close: expired. Fills are whole; `order_class: oto` is
   rejected loudly (impossible at $10 an order).
3. **Market order** (router exits; S0/S1): at the open *o* of the first eligible bar, sells at
   *o*·(1 − h·m − σ), buys at *o*·(1 + h·m + σ).
4. **Exit detection.** The live exit pass runs every 15 s. Here the production `run_exit_pass`
   runs twice a minute, with positions marked first at the previous bar's low, then at its close:
   stops fire on any touch, take-profits only on a close through the target, and the session exit
   fires at close − 10 min as live.

| | optimistic | central | pessimistic |
|---|---:|---:|---:|
| *h*, AAPL MSFT NVDA AMZN GOOGL (bps) | 0.5 | 1.5 | 4 |
| *h*, META TSLA AMD (bps) | 1 | 2.5 | 6 |
| *m*, first 5 minutes | 1 | 2 | 3 |
| σ, market orders (bps) | 0 | 2 | 5 |
| S2 round trip before fees, first group (bps) | ≈ 26 | ≈ 32 | ≈ 46 |
| … with D1's alternative entry | ≈ 1 | ≈ 9 | ≈ 46 |

These names quote one or two cents wide most of the day (0.2–1 bp half-spread) and several times
wider at the open, where S2 trades. Under the literal entry rule the 25 bps limit buffer is most of
every round trip. A free sample of historical quotes would measure *h* instead of assuming it (D9).

**Fees.** No commission; sales pay regulatory fees. They live in an effective-dated table (fee,
side, basis, rate, cap, rounding) filled at build start from the SEC, FINRA and Alpaca fee pages and
hashed into the manifest. At $10, rounding is what matters: if each fee is rounded up to the cent
per order (verify), a $10 sale pays at least $0.01 of FINRA TAF (10 bps), plus $0.01 of SEC fee
while the Section 31 rate is above zero (from memory: $27.80 per million until 13 May 2025, then
zero; verify). Reports split P&L into gross, spread and slippage, and fees, and add fees at your
real-money size (D3).

## 4. Running the live router offline

`RiskRouter(alpaca, guard, on_submitted=None, now=...)` already takes its broker and clock as
arguments. It reads time only through `self._now()` and the market only through `clock()`,
`positions()`, `orders()`, `latest_quote()` and `submit_order()`; `exit_reason`, `session_close`,
`PortfolioSnapshot.from_alpaca`, `check_can_open` and `tier0.plan_buy` are pure. The injected clock
the brief asked for therefore exists, and **`SimAlpaca`** only has to implement those calls (plus
whatever the guard reads, §11) in Alpaca's JSON shapes: decimals as strings, ISO timestamps,
`qty_available` net of working sells, `market_value` at the current mark, `submitted_at` and
`filled_at` on orders. Even the router's parsing then runs as in production. A contract test checks
the payloads against recorded Alpaca paper responses. Exit `client_order_id`s are `uuid4` in the
live code, so reproduction compares every field but that one.

**Refactors** (behaviour-preserving, each tested; no monkeypatching anywhere, L10):

1. **Gate threshold as a constructor argument:** `RiskRouter(..., min_prob_up=tier0.ROUTER_MIN_PROB_UP)`,
   read by `_entry_gate`. Production wiring passes nothing, and a test asserts it. Every risk
   limit stays an un-injectable `Final` that the simulation never varies (D2).
2. **Estimator extraction:** the GBM, its sigmoid calibration and the move table become top-level
   functions in `swarm/train_return_model.py` (`fit_calibrated`, `move_table`), used by
   `train_and_evaluate` and the walk-forward alike. `report.md` must still reproduce exactly.
3. **Guard, if needed:** if `ExecutionGuard` reads Alpaca or the wall clock itself, it gets the
   same injection.

**Replay loop.** Decisions happen on minute boundaries; a signal created inside a minute is decided
at the next one. At each boundary *t*, with the router's clock at *t*:

1. `advance(t)` resolves the bar that just ended for every order live before it began (fills
   stamped *t*) and marks positions.
2. Exits: `run_exit_pass` at the low mark, then at the close mark.
3. Entries: signals with `created_at` in (*t* − 1 min, *t*] go to `handle_signal`, highest
   `prob_up` first, then by symbol. Their orders can fill from the bar starting at *t* +
   `latency_bars`.

A boundary is skipped when nothing is pending, or before close − 10 min when the live
`exit_reason` finds nothing at either mark (L12). Every `Decision`, rejections included, becomes a
`trade_event`, so reports can say why signals did not trade. The −2.5% breaker cannot bind at $50 of
exposure on a $100k account; it is wired in anyway.

## 5. Strategies

S2, S3 and B1–B3 trade through the production router; baselines send gate-passing signals, so
only limits, sizing and fills bind. S0 and S1 cannot (eight positions breach the $50 gross cap):
they send market orders after the close, filled at the next adjusted daily open (rule 3). Exits are
always the router's.

| | Rule |
|---|---|
| S0 | Equal weight in the 8 symbols at the first open of the comparison window, held. |
| S1 | S0; on each month's first session, if any weight is outside 12.5% ± 5 percentage points, rebalance all to 12.5% (D6). |
| S2 | A symbol's night: events known between the previous close and this open. At open + 1 min a pre-open scheduler re-scores them with the fold model valid then; score = mean `prob_up` (D7); one signal per symbol, best first, with the fold's S2 move table and threshold and a raw-dollar `atr`. |
| S3 | Events known in a session before close − 30 min; one signal per event at `made_at`; the fold's S3 threshold and move table. |
| B1 | Every symbol, every session, at open + 1 min in seeded random order: the unconditional intraday drift under the same limits. |
| B2 | S2 without the probability gate. |
| B3 | Placebo: S2's daily entry count on random symbols; 100 seeds give S2's null distribution. |

**Move tables per population:** serving's table comes from in-session articles; S2 needs one from
overnight events, or the router's non-positive-edge rule gets the wrong numbers. **Collapsing:**
events for one symbol whose normalised headlines (lower-cased, punctuation and digits stripped)
have token-set Jaccard ≥ 0.7 within 6 hours share a `story_id`; collapsed runs keep each story's
first event, and S2/S3 are reported both ways. **Capital base:** S2, S3 and the baselines hold at
most $50, so daily P&L is divided by $50 (and shown per dollar deployed); S0/S1 are per dollar.

## 6. Walk-forward and lock-box

| Setting | Value |
|---|---|
| Refit | first session of each month (~1,000 samples a month) |
| Window | expanding (rolling 12 months as a variant); first scored month 2025-01, after ≥ 12 months of training; 21 folds to 2026-09 |
| Embargo | 5 sessions between the last resolved training label and the fold start (labels resolve within a session; `PURGE` is 4 days today) |
| Calibration | last 25% of each window, after the same purge (the 60 : 20 of `split`) |
| Per fold, from its window only | model, sigmoid calibration, S2 and S3 move tables and thresholds (P5, P6) |
| Scoring | a prediction whose `made_at` falls in the fold's month uses that fold |
| Hyper-parameters | `report.md`'s GBM settings, `random_state` 7, fixed (D11) |

**Threshold rule** (per fold and strategy, fixed in advance): on the fold's calibration segment,
the lowest of 0.40, 0.42, …, 0.60 with at least 30 qualifying symbol-days and a positive mean
return net of the central round-trip cost; if none qualifies, the fold does not trade. The whole
grid is also reported, marked not selectable.

**Labels.** *Legacy* (reproduction only): `RegularBars.forward_return` on the 30-min cache,
features at publication. *v2* (walk-forward): entry = open of the first 1-min bar the live path
could fill on (in session: `made_at` + latency; overnight: the scheduler at open + 1 min plus
latency, i.e. the 09:32 bar); exit = the session's last regular close; up if > 0.25%; features as
of `made_at`. Reports show the idealised 09:30 entry next to v2: a move made in the first two
minutes cannot be traded.

**Lock-box.** Development data is everything up to the last cached article (2026-09-17 20:00 New
York), the old test segment included. The 2026-09-18 session is an embargo: 09-17's overnight
labels resolve in it. The lock-box holds events known after 2026-09-18 16:00 New York and sessions
from Monday 2026-09-21; cached bars from 09-21 on (label padding) are quarantined by P7. Before it
is opened, the final model is fit on all development data by the fold procedure and its version,
thresholds and pass criteria (D5) are registered; shadow-mode predictions go to the store from now
on. Opening is registered and counted. At most ~160 overnight symbol-nights accrue a month; 1,000
of them (6–7 months) give an AUC interval of about ±0.04, enough to tell 0.55 from 0.50 but not
0.52. P&L on five $10 positions a day needs longer, so prediction metrics decide first (D8).

## 7. Store

**DuckDB + Parquet now; PostgreSQL + TimescaleDB once the live agents write.** DuckDB is embedded
(one file, no server), free, columnar, reads Parquet natively and speaks SQL close to PostgreSQL's.
Its single-writer limit is irrelevant for research and is exactly why live writers need PostgreSQL.
Portable DDL: BIGINT, INTEGER, DOUBLE, DECIMAL(p,s), VARCHAR, BOOLEAN, DATE, TIMESTAMPTZ (UTC) and
JSON only; no LIST, STRUCT, MAP or sequences. Keys are source ids or content hashes, so writes are
idempotent and never collide. Bars stay in Parquet behind views (hypertables, live).

| Table | Key | Columns |
|---|---|---|
| `event` | `feed:source_id` | feed, publisher, `published_at`, `updated_at`, `known_at` and its basis (published / received), headline, headline_hash, n_symbols, story_id, dedup_version, event_type and subject_verified (later), mode |
| `event_symbol` | event, symbol | |
| `model` | model_version | feature names, windows, label definition, params, seed, commit, data hash, artifact sha256, thresholds, move tables, experiment |
| `prediction` | hash(event, symbol, model, made_at) | `made_at`, mode (backtest / shadow / live), experiment, inputs (JSON, name → value), prob_up, expected_move_pct, horizon, reference_at, reference_price |
| `outcome` | prediction, horizon | horizon ∈ {5m, 30m, 2h, close, 1d, 1mo}, resolved_at, ret, max favourable / adverse, price space, status (ok / truncated / no_data) |
| `trade_event` | run, trade, seq | at, state, reason_code, side, qty DECIMAL(20,9), price DECIMAL(18,4), fees DECIMAL(18,2), order ids, prediction, detail (the router's `Decision`) |
| `experiment` | experiment_id | hypothesis, parent, status, window, lock-box flag, commit, dirty flag, data hash and manifest, params, seeds, environment, metrics, conclusion |
| `run` | run_id | experiment, strategy, cost level, variant, metrics |

Trade states: `signal` → `rejected(code)`, or `signal` → `submitted` → `filled` or `expired` →
`exit_submitted(stop_loss / take_profit / session_close)` → `exit_filled`. The `trade` view folds a
lifecycle into one row (entry, exit, gross, costs by component, net); `event_memory` joins each
event to its predictions and outcomes. Outcomes run from the v2 entry price for every prediction,
traded or not; horizons past the close are truncated and flagged. Nothing that builds features or
chooses parameters may read them (L9).

**Live writers** (later, through `PostgresStore` on the same DDL): the news ingester writes
`event` (`known_at` = receipt time); the Inference agent writes `prediction`; the router's
`on_submitted` hook and exit decisions write `trade_event` through the recorder the simulation
already uses; a job after each close resolves outcomes. DuckDB attaches PostgreSQL (free `postgres`
extension), so one set of queries covers backtest, shadow and live rows, told apart by `mode`.

## 8. Experiment registry and exact reproduction

Every command that produces a number first writes an `experiment` row with a required one-line
hypothesis, then adds metrics and a conclusion, or marks it failed. Rows are never deleted, only
superseded, so each report prints "configuration N of M tried on this window". A run records the
git commit (a dirty tree is refused unless `--allow-dirty`, and such runs cannot be cited); a
manifest of every input file (path, size, sha256), its combined hash and the sample-matrix digest;
the fully resolved parameters (cost levels, fee-table version, latencies, schedule, thresholds,
dedup, strategy settings); all seeds, a fixed `OMP_NUM_THREADS` and library versions; and its
outputs (metrics, report hash, model versions). `python -m backtest reproduce <id>` checks the
commit out into a temporary worktree, verifies the manifest, re-runs with the recorded parameters
and seeds, and requires identical metrics and report hash.

## 9. Leakage and consistency tests

Each test has a broken twin that violates its rule on purpose and must fail; the twins stay in the
suite, so this is re-checked on every run.

| Test | Passes when | Broken twin (must fail) |
|---|---|---|
| L1 Future canary (P1–P3) | A canary equal to the sample's own label, known 30 days after its session, joined as-of like any input: AUC moves < 0.01, importance < 0.005 | Joined on session date: AUC > 0.9 |
| L2 Label shuffling (P6) | Labels permuted in symbol-day blocks, 10 seeds (slow), walk-forward + S2: AUC interval contains 0.5; S2 inside the B3 placebo band | Thresholds chosen on each fold's scored month: S2 beats the band |
| L3 No bar after the decision (P2–P4) | `PitView` and `SimAlpaca` assertions never fire in full S2 and S3 runs; 10,000 random decision times match a brute-force oracle | `latency_bars` = 0 with the legacy `bisect_left` entry |
| L4 Reproduce `report.md` | Legacy mode, same scikit-learn version: 33,312 samples (19,790 / 6,588 / 6,663); every AUC, Brier, log loss, quantile, gate and calibration row as printed | `PURGE` = 0, or `bisect_right` entry |
| L4b Engine agrees with labels | Each legacy sample replayed through `SimAlpaca` as a frictionless, unconstrained trade returns its `fwd` to 1e-12 | Fill on the bar containing the publication time |
| L5 Training cut-off (P5) | For every prediction: latest resolved training label + embargo ≤ fold start ≤ `made_at` | Embargo of −1 session |
| L6 Story straddling | No `story_id` in both a fold's training window and its scored month | Embargo 0 |
| L7 Lock-box (P7) | Reading lock-box data outside a registered run raises | Guard disabled |
| L8 Feature parity | Stored `inputs` equal an independent `daily_features(completed_sessions(bars, made_at))`, 1,000 predictions | Cut-off one session late |
| L9 Outcome isolation | `dataset`, `walkforward`, `strategies` import nothing from `outcomes` | Such an import added |
| L10 One engine | Router, policy, `plan_buy`, `exit_reason` are the production objects (identity, code hash); production wiring uses the tier-0 gate | `MAX_NOTIONAL_USD` monkeypatched |
| L11 Calendar | No fills on holidays; on half-days no entry from 12:30, exit at 12:50 | Weekday rule |
| L12 Minute skipping | A fixture month with a stop and a target gives identical logs with and without skipping | Exit passes skipped unconditionally |

Diagnostic until `updated_at` arrives: AUC by the gap between `updated_at` and `published_at`. An
edge concentrated in later-revised articles would mean the headline we hold may postdate `known_at`.

## 10. Statistics and reports

Per strategy × cost level (× collapsed or not, for S2/S3): daily returns on the capital base; total
and annualised return; excess over S0 and, for S2/S3, over B1 and B2, plus S2's percentile in B3;
max drawdown and its duration; Sharpe with a 95% stationary block-bootstrap interval over days (mean
block 5 days, 10,000 resamples, recorded seed), and the same for mean excess return and AUC;
turnover; costs by component; **independent bets: trading days with a position, distinct
symbol-days and trades, always all three**; rejections by code; "configuration N of M" and the
deflated Sharpe ratio for that N. Robustness: half-years, cost levels, threshold grid, latency of
1, 2 and 5 bars, monthly vs quarterly refits, collapsed vs not. Every report opens with the brief's
selection-bias warning (8 surviving large caps picked in 2026, so absolute returns are inflated;
compare with S0).

**Proposed pass rule (D5):** at the central level, S2's mean daily excess over B1 has a 95%
interval above zero across the walk-forward folds and is positive in each half-year, and the
pre-registered lock-box criteria confirm it. A result that holds only at the optimistic level fails.

## 11. To verify when the build starts

1. `risk_router/alpaca_async.py`, `guards.py`, `schemas.py`: the signatures and return types
   `SimAlpaca` must match, what `ExecutionGuard` reads, and `TradeSignal` validation.
2. `swarm/return_model.py` and the agents: how live derives `predicted_move_pct` and the dollar
   `atr`. The engine imports these, never copies them.
3. The cache: if the 30-min bars include extended hours, `RegularBars.build` (weekday bars
   09:30–15:30) labels half-days with an after-hours close; count the samples affected and report
   them with the reproduction. And why does `report.md`'s training span start 2023-11-25 when the
   fetch starts 2024-01-02? If the news API filters on `updated_at`, some headlines we hold are
   revisions.
4. Alpaca's fee schedule: SEC and FINRA TAF rates by date, CAT fees, rounding per order.

## 12. Open decisions for you

| # | Decision | Default here | Recommendation |
|---|---|---|---|
| D1 | Entry fill price | Pays the limit at every level (the brief, literally): ≈ 26 / 32 / 46 bps a round trip before fees | Keep it for pessimistic; use min(*L*, *o*·(1 + h·m + σ)) at optimistic and central (≈ 1 / 9 bps) so the levels bracket reality |
| D2 | Router's `prob_up` gate as a constructor argument | Yes, defaulting to the tier-0 constant | Yes: otherwise no threshold below 0.60 can be tested |
| D3 | Size that decides go / no-go | Tier-0 $10, where fees may cost 10–20 bps a round trip | Report both; decide on your real-money size |
| D4 | S2 pre-open scheduler | Simulated as strategy code | Build it live only if S2 passes, calling `handle_signal` unchanged |
| D5 | Pass criteria | §10 | Register them before the first S2 run |
| D6 | S1 bands | ±5 percentage points, absolute | Absolute; relative ±5% trades almost every month |
| D7 | S2 score per symbol-night | Mean `prob_up` | Mean; max and latest as registered variants |
| D8 | Lock-box timetable | Open at ≥ 1,000 overnight symbol-nights, again at 12 months | As default |
| D9 | New free fetches | 1-min raw bars, raw daily bars, calendar, news `updated_at` | Yes, plus a small quote sample to measure spreads |
| D10 | Store | DuckDB + Parquet (adds `duckdb` to `requirements.txt`) | As default |
| D11 | Hyper-parameters | `report.md`'s GBM settings, fixed | Fixed for v1; any tuning is a registered experiment on development folds |

## 13. Owner decisions (2026-09-26)

Decided by the owner after review. These override the defaults above wherever they differ.

| # | Decision |
|---|---|
| D1 | **Bracket reality.** Pessimistic level: limit buys fill at the limit *L* (the brief's literal rule). Optimistic and central levels: fill at min(*L*, *o*·(1 + h·m + σ)). |
| D2 | **Yes, backtest only.** The router's `prob_up` gate becomes a constructor argument defaulting to `tier0.ROUTER_MIN_PROB_UP` (0.60). The live app never passes it, and a test asserts that the live router's threshold equals the constant. Every other risk limit stays hardcoded. |
| D3 | **Report both, decide later.** Every report shows results at $10 (the Tier-0 cap) and at a second, configurable order size. The owner fixes that real-money size, and registers it with the D5 pass criteria, **before the first S2 result is read**. |
| D4–D11 | **Accepted as recommended** in section 12. |

**Verified fact for section 3 and item 4 of section 11** (Alpaca's regulatory-fees support page,
dated February 2026, read 2026-09-25): the SEC fee is "rounded up to the nearest penny" and the
FINRA TAF is "applied on a per-trade basis, rounded up to the nearest penny, and capped", both on
sales only. A CAT fee also applies, on buys and sells. Rates and caps still need filling from the
SEC and FINRA pages at build start. So a $10 sale pays at least $0.01 of TAF (10 bps), plus $0.01
of SEC fee whenever the Section 31 rate is above zero.
