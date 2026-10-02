# Long-term core: build notes

**Written:** 2026-10-01 · **For:** whoever reviews `feat/long-term-core` against
`docs/long-term-core-design.md`. Every place the build departs from the design, and why.

## Departures from the design

| Design | Built | Why |
|---|---|---|
| §1: core fetches in `backtest/fetch.py` | `backtest/core_fetch.py` | Keeps phase 1's fetcher untouched; same helpers, its own cache folder `.cache/backtest/core/`. |
| §1: `raise-cash` CLI command | The registered raise-cash scenarios run inside `run-core` | The command was meant for registered scenarios; an on-demand owner query belongs to paper mode (§8.4), which is designed, not built. |
| §3.5: raise-cash gross-up includes tax | Gross-up covers costs only; tax is paid at the year's end | Tax is netted yearly (§3.5 itself), so a sale owes nothing on the day. The year-end tax plan is its own raise-cash plan. |
| §5.1: CAGR = W^(365.25/calendar days) − 1 | CAGR annualised by 252 sessions a year | One estimator for the point and its bootstrap interval (resampled series have no calendar). Over these windows the two differ by under 0.1% of the CAGR. |
| §7 C1: "changing close t−63 does not" change M4's weights | Changing *t* or *t*−63 changes them, *t*−64 does not | The design's own window is closes *t*−63 … *t* (63 returns), so *t*−63 must matter. The check tests both edges. |
| §7 C5: rescale prices before a random date, every return equal | Each instrument rescaled by its own factor before a seeded date; every decision, weight and return *before* that date identical | Rescaling changes the real return across the date, so "every return" cannot hold. A single factor for all instruments would also let a price-difference volatility pass (normalised weights cancel it), so the twin could not fail. |
| §7 C6 twin: Alpaca's daily crypto bar | Two twins: the bar *after* the close, and a fixed 21:00 UTC cutoff | Both break the exact rule (half-days and daylight saving included) without fetching another file. |
| §7 C9: frictionless engine to 1e-10 | Same, after the frictionless level skips cent truncation of buys | Cent truncation alone is 1e-7 on $100,000. Measured worst difference: 7.4e-13. |
| §3.1: buys capped by cash | Plus a $1 cash reserve and a per-order fee allowance | Daily fee rounding up to a cent per fee type would otherwise overdraw a fully invested account by cents. |

## Facts found while building (not in the design)

- **Crypto fees before 2023-03-13 are not published by Alpaca.** The 2023 tier-1 taker fee (0.25%) is
  applied from 2021, stated in `fees.json`.
- **FINRA paused the TAF from 2026-10-01 to 2026-12-31** (filed 2026-09-15). No backtest window reaches it;
  revisit before core paper trading, once it is known whether Alpaca passes the pause through.
- **Alpaca has no crypto quotes before 2023**, so BTC/USD's spread comes from 22 sessions 2023–2026.
- **BTC/USD 5-minute bars** start 2021-01-01 with some quiet intervals, but no NYSE open or close sample
  is more than 5 minutes old.
- **The dividend adjustment is proportional** (the adjusted/raw ratio steps only on ex-dates), which C5's
  premise needs.
- **The 2022 stress window** is 2022-01-03 → 2022-10-12 on SPY closes, as the design said.

## Order of operations for a citable result

1. Commit everything (a dirty tree cannot be cited).
2. `python -m backtest leakage-core`: C1–C12 must all pass at that commit.
3. `python -m backtest register-core`: freezes the grid, reading rules and the owner's tolerance.
4. `python -m backtest run-core <registration id>`: refuses without 2 and 3.
5. `python -m backtest reproduce <run-core id>`: identical metrics and report hash.

No core performance number was printed or read before step 3; dry runs printed structure only.

## Results and what they taught (2026-10-01)

Registration `x-20261001-112412-3f5633a7`, checks `x-20261001-112250-98f74c76` (all 26 rows pass), run
`x-20261001-112421-9bfa07ac` at commit b8bdeb6, reproduced exactly as `x-20261001-112713-3ee8a03c`
(report hash dfb49052…). Report: `.cache/backtest/reports/run-core.md`.

**A flaw in the registered reading rule R1, found in the results:** R1 has no minimum effect size. For M1
(100% VTI) the calendar rules only ever sweep idle cash: one $4.49 trade in ten years, worth about $14. The
two return series are nearly identical, so the paired interval is microscopically narrow, excludes zero,
and R1 prints "helps on return; adds risk". The registered rule stands for this run (it cannot be changed
after the results). A future registration should add a materiality threshold (for example, a difference
must exceed one basis point a year) and apply it only to evidence gathered after it, such as the paper
period.

**On the edge:** M1's maximum drawdown is -34.98% at the pessimistic level against the owner's -35%:
"within your tolerance" by 0.02 points.

## Second run, after review fixes (2026-10-01)

A review of the first run found three low-severity issues, fixed in commit 5493a47:

1. Buys reserved the spread and slippage on top of a fill price that already contains them, so that
   fraction sat idle as cash (M1 'none' kept $5.50 of $10,000). Buys now reserve only fees plus an
   explicit cent-rounding allowance; idle cash is the $1 reserve plus cents. This was the root cause of
   M1's "helps on return" artifact: the one $4.49 trade was this idle cash being swept.
2. Turnover counted the initial purchase; it no longer does.
3. The raise-cash table counted sub-cent noise as "higher"; it now needs more than a cent.

C11's twin was also strengthened to forward-fill a bar through the served market.

Run `x-20261001-121225-9c4ebec5` (run-core **2 of 2** on this window, same registration
`x-20261001-112412-3f5633a7`, checks `x-20261001-121158-87ca6a16` at 5493a47), reproduced exactly as
`x-20261001-121518-6c66ad74` (report hash 4100e37d…). Against the first run: every figure moves by
rounding only (CAGRs within 0.01 point); the only verdict that changes is M1's, now "indistinguishable on
this history" under every rule. All other R2 and R3 verdicts, and sections 1 and 2, are unchanged. Both
runs stay in the registry; this one is the one to cite. R1's missing materiality threshold remains a
flaw of the registration, now without a visible symptom.

## The core in shekels (2026-10-02)

The owner spends primarily in shekels and gave the go-ahead for a new data source: the Bank of Israel's
representative USD/ILS rate (RER_USD_ILS, quoted with the source attributed). FRED has no shekel series
(the Federal Reserve's H.10 list does not include it). `fx-core` (commit 5e3b94c, checks
`x-20261002-144521-1527b27d`) is experiment `x-20261002-144550-d272c7ed`, reproduced exactly as
`x-20261002-144604-0e366514` (report hash 910b67bd…). Report: `.cache/backtest/reports/fx-core.md`.

The dollar fell 19.4% against the shekel over the window (₪3.802 → ₪3.063), so shekel growth is about two
points a year lower than dollar growth for every mix. The dollar rose in the 2020 and 2022 crashes, so
falls in shekels were smaller than in dollars, while day-to-day swings were larger. Every row stays within
the owner's −35% / 4-year tolerance in shekels. 131 of 2,637 sessions had no Bank of Israel rate that day
and used the last one published before it.
