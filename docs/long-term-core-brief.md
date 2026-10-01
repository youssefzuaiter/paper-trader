# Long-term core — design brief (phase 3)

**Status:** brief for a design session. **Deliverable of that session:** `docs/long-term-core-design.md`.
**Owner:** Youssef Zuaiter · **Written:** 2026-09-29 · **Builds on:** `docs/backtester-design.md`
(the backtester, merged in PR #2) and `~/Documents/ai-portfolio-agent-system-design.pdf` §7 (the
Allocator and the investment policy).

This system is meant to manage real money one day, for the long term. Phase 1 showed that
predicting prices from news does not work here. Phase 3 builds what the long-term goal actually
needs: **a diversified mix of assets, rebalanced by written rules, judged honestly by the
backtester.** Nothing in this phase predicts prices.

> **Not financial advice.** Which mix to hold, and how much money, are the owner's decisions. The
> backtester's job is to show the trade-offs of each candidate honestly. The candidates below are
> standard reference portfolios chosen to span the range, not recommendations.

---

## 1. Lessons carried over from phase 1 (design to respect them)

- **Eight tech stocks are one bet.** Buy-and-hold of the 8 (S0) and monthly rebalancing among them
  (S1) both fell **−34%** at the worst point. Rebalancing between similar assets does not reduce
  risk. Diversification has to come from assets that behave differently.
- **1.7 years is too short.** Both S0's and S1's Sharpe intervals included zero. Phase 3 uses the
  longest free history available.
- **Selection bias.** Returns of hand-picked 2026 winners are inflated. Phase 3 uses broad funds,
  not picked stocks.
- **The machinery works.** Registry, pre-registered criteria, reproduction, cost levels and fee
  table (all from phase 1) are reused as they are.

## 2. What the backtester must answer

For each candidate mix and each rebalancing rule, over 2016-01-04 → the present:

1. **How bad does it get?** Maximum drawdown, the longest time underwater (peak to recovery), and
   the worst calendar year.
2. **What does it earn, for that risk?** Annualised return, volatility, Sharpe, Sortino, Calmar, with
   day-block bootstrap intervals as in phase 1.
3. **Does rebalancing help** compared with buying the same mix once and never touching it? At what
   cost (turnover, fees, spread)?
4. **How often should it rebalance?** Calendar (monthly, quarterly, yearly) against drift bands
   (±5 or ±10 percentage points), and combinations of the two.
5. **Stress behaviour** in the four real shocks the data covers: the 2018 Q4 sell-off, the 2020
   COVID crash, **2022 (stocks and bonds fell together)**, and the 2025 drop.
6. **Does a small crypto slice change the picture?** Evaluated on 2021-01-01 onward only (Bitcoin's
   data starts there), always shown next to the same mix without crypto over the same period.
7. **The cash buffer and the "raise cash" plan** (design document §3): if the owner needs X by
   date D, which positions are trimmed, what does that cost, and how far does the mix drift?

## 3. Data (verified 2026-09-29, free with the paper keys)

| Role | Instruments to fetch | Daily history from |
|---|---|---|
| US stocks | VTI, SPY | 2016-01-04 |
| Non-US stocks | VXUS | 2016-01-04 |
| Bonds | BND, IEF, TLT | 2016-01-04 |
| Gold | GLD, IAU | 2016-01-04 |
| Real estate | VNQ | 2016-01-04 |
| Cash | BIL (SGOV only from 2020-05-28) | 2016-01-04 |
| Crypto | BTC/USD (Alpaca crypto) | 2021-01-01 |

Tickers are instruments to *test*, not picks. Use total returns (dividend-adjusted bars) for
performance, and state how fills use raw prices (the phase 1 D1/fees decisions still apply).

**Honest limit:** about 10.7 years is one long bull market with a few sharp crashes. It contains
no 2008 and no 2000–2002. The design must say this in every report, and must not treat small
differences between mixes as meaningful. Longer free histories exist elsewhere (for example FRED
series or index data), but each has its own terms of use: **list them as an open decision; don't
use them without the owner's go-ahead.**

## 4. Candidate mixes (to be registered before any result is seen)

Register the full set **once, before the first run**, so no mix can be picked after seeing results.
Every configuration counts toward "configuration N of M" and the deflated Sharpe, as in phase 1.

| # | Reference mix | Why it is in the set |
|---|---|---|
| M1 | 100% US stocks | The simplest benchmark; the phase 1 lesson in fund form |
| M2 | 60% stocks / 40% bonds | The classic balanced portfolio |
| M3 | Equal weight across stocks, non-US stocks, bonds, gold, REITs | Naive diversification across classes |
| M4 | Risk-balanced across the same classes (weights ∝ 1 / trailing volatility, recomputed at each rebalance using only past data) | Diversifying risk rather than dollars |
| M5–M8 | M1–M4 with 5% crypto taken proportionally from the rest (2021+ only) | Question 6 |

Each mix × each rebalancing rule (none, monthly, quarterly, yearly, ±5 pt band, ±10 pt band) × three
cost levels. The design may propose changes to this set **before registration**, with reasons.

## 5. Non-negotiable rules (carried from phase 1)

- **No look-ahead.** A rebalancing decision uses prices up to the close of day *t* and executes at
  the open of *t+1*. Trailing volatility for M4 uses only past days.
- **One engine.** Reuse the backtester's daily simulation (S0/S1), fee table, cost levels, store and
  registry. Do not fork them.
- **Pessimistic by default.** The three cost levels as in phase 1; fractional ETF orders; fees per
  the Alpaca schedule already in `fees.json`.
- **Taxes are country-dependent.** Add a "tax drag on realised gains" parameter, default 0, and show
  sensitivity at a couple of illustrative rates, labelled as illustrative.
- **Reproducible.** Every run registered, committed, and reproducible exactly, as now.
- **CPU budget.** One process at the lowest priority by default (the owner's machine, the owner's fans).

## 6. From backtest to paper trading (design it, don't build it yet)

Once the owner chooses a mix, the Allocator (design document §7.2) runs it on paper through the Risk
Router. **The router's current Tier-0 limits were written for $10 news trades** ($10 per order, $50
gross, 5 positions, 10 buys a day, sell everything 10 minutes before the close). A long-term core
needs different limits: whole-portfolio weights, monthly trades, positions held for years. The design
must propose:

- an **owner-edited policy file** (design document §7.1) for the mix, bands and limits, read-only to
  the agent;
- **how the router's hardcoded limits change** for this mode. This is a deliberate, reviewed change,
  never a side effect, and the news-trading mode keeps its own limits;
- **what stays exactly as it is:** the kill switch, the daily loss breaker, signed receipts, the
  paper-only lock.

## 7. Questions the design must answer

1. How the daily simulation is extended to many assets, calendar and band rules, and M4's volatility
   weights.
2. How crypto's 24/7 prices are aligned with market-hours ETFs (a fixed daily cutoff time).
3. The exact registration step (mixes, rules, cost levels, criteria) and the report layout.
4. How the "raise cash" plan is chosen (buffer first, then most over-weight, cost- and tax-aware) and
   how it is tested.
5. What the paper-trading mode (§6) needs from the router and the policy file.
6. The leakage and consistency tests: at least a "future price" canary for M4's volatility, a check
   that decisions at *t* never use prices after *t*, and a reproduction of phase 1's S0 and S1
   numbers for the 8 stocks through the extended simulation.

## 8. Done when (for the build that follows)

- Every registered mix × rule × cost level runs from committed code and reproduces exactly.
- The report answers §2's seven questions, with intervals, stress windows and the history warning.
- Phase 1's S0/S1 numbers are reproduced through the extended simulation.
- Every new leakage test passes, and fails when its rule is deliberately broken.
- The design for §6 (policy file, router mode) is written and reviewed, but not built.

---

## How to run the design session

Start a **new** session in `~/paper-trader` on branch `feat/long-term-core`. Choose **Opus 5.5**
(Pro limits, free) or **Fable 5.1** (credits, about $5–15), at **high** effort, and paste:

> Read `docs/long-term-core-brief.md`, then `docs/backtester-design.md` (sections 0, 3, 7, 8, 10,
> 13) and only the backtester modules you need for the daily simulation, registry and costs.
> Write `docs/long-term-core-design.md`: answer every question in section 7, specify the module
> changes, the registration step, the report, the leakage tests and the paper-trading mode, and
> list open decisions for me. Do not write implementation code. Keep it under about 8 pages.

Then build in another new **Opus 5.5** session from the design.
