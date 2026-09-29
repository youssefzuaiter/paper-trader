# Backtester — design brief (phase 1)

**Status:** brief for a design session. **Deliverable of that session:** `docs/backtester-design.md`.
**Owner:** Youssef Zuaiter · **Written:** 2026-09-25 · **Parent documents:**
`~/Documents/ai-portfolio-agent-system-design.pdf` (sections 6, 10, 15) and `deploy/k8s/README.md`.

This system is meant to graduate from paper trading to real money one day. The backtester is
the tool that decides whether any strategy deserves that. A backtester that flatters results is
worse than none, so every rule below leans toward being pessimistic.

---

## 1. What the backtester must answer

1. **Does a strategy make money after realistic costs, on data it was never tuned on?**
2. **Does it beat simple baselines** on the same stocks over the same period?
3. **How robust is the result:** across time periods, cost assumptions and thresholds, and with
   duplicate news collapsed?
4. **For every prediction the system makes (live or simulated): what actually happened next?**
   (event memory and prediction accountability)
5. **What was tried, and what was concluded?** (experiment registry, so a result can never be
   quietly cherry-picked from many attempts)

## 2. What already exists (read these, nothing else unless needed)

| File | Why it matters |
|---|---|
| `swarm/train_return_model.py` | Dataset build, labels (`RegularBars.forward_return`), point-in-time daily features (`DailyHistory`), the purged 60/20/20 split, evaluation. The backtester generalises this. |
| `swarm/features.py` | `completed_sessions`, `daily_features`, `feature_vector`, `is_tradeable`: the no-look-ahead rules, shared with the live agents. |
| `swarm/alpaca_data.py` | Paced free-plan client for news and bars (SIP with the 15-minute delay). |
| `tier0.py` | `plan_buy`: the exact pricing and sizing the live router uses. |
| `risk_router/policy.py`, `risk_router/gatekeeper.py` | Portfolio limits (`check_can_open`), entry gate, exits (stop, target, session close). |
| `models/return_model/report.md` | What the current model does and does not know. |

**Cached data** in `.cache/training/` (gitignored, reusable): 46,370 news articles for 8 symbols
(AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, AMD), 2024-01-02 → 2026-09-18; daily and 30-minute
SIP bars (split- and dividend-adjusted); FinBERT scores for every headline.

## 3. What we already learned (do not re-litigate; design to confirm or refute)

- Next-session horizon: no signal (test AUC 0.49).
- Same-session close: weak signal overall (test AUC 0.524; 0.532 one article per symbol-day).
- **The signal is in articles published outside market hours, entered at the next open**
  (AUC 0.549 calibration, 0.520 test). **In-session articles: noise** (0.508).
- FinBERT sentiment alone is below chance; momentum and RSI carry what the model knows.
- No probability threshold was profitable in *both* the calibration and test periods.
- **The 2026-03-10 → 2026-09-17 test segment has been looked at repeatedly** (the horizon
  diagnostic). It is no longer an untouched test set. A clean confirmation needs data from after
  2026-09-18: a new lock-box and/or shadow-mode results.

## 4. Non-negotiable rules

**No look-ahead.**
- A fact enters the simulation only at the moment it became known: news at its `created_at`,
  daily bars only once `completed_sessions` says so, filings (later) at their filing time.
- An order can only fill on a bar that starts *after* the decision. Decision latency is at least
  one bar; the design should make it a parameter.
- Models are retrained walk-forward: a model scoring day D was trained only on data whose
  **labels** had resolved before D, with a purge/embargo gap.
- Thresholds, horizons and hyper-parameters are chosen on validation data, never on the lock-box.

**One engine.** Sizing and limits must call the *same code* the live router runs
(`tier0.plan_buy`, `risk_router.policy.check_can_open`, the exit rules), not a re-implementation.
If the live code needs refactoring to be callable offline (for example, the exit logic reading an
injected clock instead of Alpaca), do that refactor. Behaviour must not fork.

**Pessimistic fills.**
- Limit buys fill only if the bar trades *through* the limit, at the worse of the limit and the
  open; market orders fill at the next bar's open plus a spread and slippage allowance.
  *(Amended by design decision D1: this rule applies at the pessimistic cost level only; see
  `docs/backtester-design.md` section 13.)*
- Historical bid-ask spreads are not in the cache. Model them as per-symbol basis points, and
  **report every result under at least three cost levels** (optimistic, central, pessimistic). A
  strategy that only works at the optimistic level does not pass.
- Alpaca charges no commission on US stocks, but regulatory fees apply to sales; model them as a
  small per-sale cost.
- Use the exchange calendar (Alpaca's `/v2/calendar`, free): holidays and 13:00 half-days.
- Adjusted prices are fine for returns. The design must state how corporate actions affect fills
  and position quantities (raw vs adjusted), and choose explicitly.

**Honest statistics.**
- Report the number of independent bets (trading days, symbol-days), not just trades.
- Confidence intervals by block bootstrap over days, not by treating trades as independent.
- **Selection bias warning:** the 8 symbols were picked in 2026 and are large companies that
  survived and grew. Absolute returns are therefore inflated. Compare every strategy against
  buy-and-hold of the *same* 8 symbols, and say so in every report.

## 5. The shared store: events, predictions, outcomes, experiments

One point-in-time store feeds the backtester, event memory, prediction accountability and the
experiment registry (live agents write to it later too). Minimum entities:

| Entity | Key fields |
|---|---|
| `event` | id, source, symbol(s), `known_at`, headline, event type (later), subject-verified flag (later) |
| `prediction` | event id, model version, `made_at`, inputs (feature vector), prob_up, expected move, horizon |
| `outcome` | prediction id, realised return at 5 min, 30 min, 2 h, same-day close, 1 day, 1 month |
| `trade` | strategy run id, state transitions with timestamps and reasons (the trade state machine) |
| `experiment` | id, hypothesis, data window, model and code version, parameters, metrics, conclusion |

The design document (section 6) says PostgreSQL + TimescaleDB. For a local research v1,
DuckDB + Parquet may be simpler and is also free. **Decide, with a reason, and keep the schema
portable.**

## 6. Strategies for the first experiments

| # | Strategy | Why |
|---|---|---|
| S0 | Buy-and-hold, equal weight, the 8 symbols | The bar every strategy must clear |
| S1 | Monthly rebalancing to equal weight with ±5% bands | The long-term core from the design document |
| S2 | **Overnight news → buy at the next open, sell at that session's close**, gated by walk-forward `prob_up`, router limits applied | The only signal the evidence supports |
| S3 | In-session news, same-day exit (the router's current rule) | Expected to fail; confirms the backtester agrees with the evaluation |

For S2 and S3: collapse duplicate stories (same symbol, near-identical headline within a few
hours) and report both with and without collapsing.

## 7. Questions the design must answer

1. Event-driven replay or vectorised? (Event-driven is closer to live; vectorised is faster. A
   hybrid is fine if it is justified.)
2. The fill model: its exact rules, and the spread and slippage numbers for each cost level.
3. How the router's live code is made callable offline without changing its behaviour.
4. The walk-forward schedule: window lengths, retraining frequency, embargo, and where the new
   lock-box begins.
5. Store choice and schema; how the live agents will write the same records later.
6. **The leakage test plan.** At minimum:
   - a "future canary" feature that must never help;
   - label shuffling, which must produce no edge;
   - an assertion that no bar used for a decision starts after the decision time;
   - a reproduction of `report.md`'s numbers through the new engine.
7. What gets written to the experiment registry for each run, and how a run is reproduced
   exactly (data hash, code commit, parameters, random seeds).

## 8. Constraints

- Free and self-hosted only (Python 3.13, the existing `.venv`, NumPy, scikit-learn; DuckDB if
  chosen). No paid data.
- Must run on this laptop: the full 8-symbol, ~33k-sample walk-forward should finish in minutes,
  not hours.
- Decimal arithmetic for money and quantities, as everywhere else in the repo; floats for
  features and statistics.
- Tests for every leakage rule. A backtester without tests is not trusted.

## 9. Done when (for the build that follows the design)

- `report.md`'s headline numbers are reproduced through the new engine, which is the proof that
  both are consistent.
- S0–S3 run end to end with a report per strategy: returns vs S0, max drawdown, Sharpe with
  bootstrap interval, turnover, costs paid, number of independent bets, all at three cost levels.
- Every leakage test passes, and each one has been shown to fail when its rule is deliberately
  broken.
- Every run writes a reproducible experiment record.
- Event memory holds each prediction joined to its outcomes at every horizon.

---

## How to run the design session (keep it cheap)

Start a **new** session in `~/paper-trader`, select **Claude Fable 5.1** at **high** effort, and
paste:

> Read `docs/backtester-brief.md`, then only the files in its section 2 table. Write
> `docs/backtester-design.md`: answer every question in section 7, specify the module layout
> and interfaces, the store schema, the fill and cost model, the walk-forward schedule and the
> leakage test plan, and list open decisions for me. Do not write implementation code. Keep it
> under about 8 pages.

Then end that session and build in a new **Opus 5.5** session from `docs/backtester-design.md`.
