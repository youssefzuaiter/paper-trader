# Long-term core: paper mode

**Status:** built and tested; the pre-flight passes against the real paper account (`PA3EKDCB37WB`); not deployed;
no order placed · **Updated:** 2026-10-03 · **Designs:** `docs/long-term-core-design.md` §8, reconciled with the
system design PDF §3 and §7 · **Operating it:** `docs/core-runbook.md`

> Not financial advice. The policy is the owner's own choice: M3 with a 5% BIL cash buffer, rebalanced
> quarterly, chosen 2026-10-02 from registered run `x-20261001-121225-9c4ebec5`.

## What runs

| Part | File | Can trade? | Does |
|---|---|---|---|
| Policy | `policy/core.toml` | no | The owner's mix, rule, buffer, limits and first tradable session (`effective_from`); read-only to every agent; its sha256 is in every journal entry |
| Hard limits | `tier0_core.py` | no | Ceilings the policy must stay inside; per-plan checks |
| Shared decision | `core_paper.py` | no | The snapshot, the decision, order payloads; calls `core_alloc`, the backtest's own functions |
| Allocator | `swarm/core_allocator.py` | **no** | After each close: reads, decides, signs, proposes. A day with no session is a no-op |
| Core router | `risk_router/core_gatekeeper.py`, `core_app.py` | **yes, the only part** | Recomputes every plan, enforces limits and approvals, executes at the open, journals, and closes anything it cannot finish |
| Alerts | `risk_router/core_alerts.py` | no | Pushes the events that need a person to a webhook (ntfy, Slack, Discord). Optional; never blocks the router |
| Console | `risk_router/core_ctl.py` | no (asks the router) | The owner's signed `status` / `approve` / `raise-cash` / `halt` |
| Pre-flight | `risk_router/core_preflight.py` | no | Read-only readiness check, and a dry run of the Allocator |
| Paper report | `backtest/paper_report.py` | no | Turns the journal into the out-of-sample evidence the design promised, under rules fixed before any order |

The news router (`risk_router/app.py`, `tier0.py`) is unchanged except for one refusal: it will not
start on the core's account.

## One day

1. **16:20 New York.** The Allocator asks the router for the facts only it knows (and the router first closes any
   plan that can no longer finish, so those facts are true), reads the account and the day's closes, and decides with
   `core_alloc.rebalance_due`, the same rule the backtest used. For this policy that means the last session of each
   quarter, or the re-decision of a plan that did not complete. A plan that would execute before `effective_from` is
   not made.
2. The router **rebuilds the same snapshot from its own reads**, recomputes the plan, and rejects any
   difference (`plan_mismatch`). Then the limits: per-order ceiling, 25% turnover per rebalance, two plans a
   month, a weekly cap on trades and value, the registered instruments only. The same proposal sent twice (a retry)
   changes nothing, and a proposal cannot replace a plan that is already trading.
3. **Approval.** The initial build waits for the owner's signed `fund` command, and is refused if the
   account holds more than the policy's $10,000 funded cap. A plan trading more than $3,000, and every
   raise-cash plan, waits for the owner's signed approval (`core_ctl approve`). Anything else is approved automatically.
4. **09:20–09:27 next session.** Sells go in as limit orders at the previous close − 3%: Alpaca fills
   orders received before 09:28 at the official opening price, as the backtest assumed. Once they fill, buys go in
   as marketable notional limits (ask + 0.5%), scaled to the cash the account actually has. A plan with no sells
   (the initial build) sends its buys pre-open at the close + 3%. Each order has a deterministic client id, so a
   submission that fails part-way resumes on the next 15-second tick without duplicating anything, and an order
   Alpaca refuses outright is recorded without freezing the rest of the plan.
5. **Every plan is closed, one way or another.** `done` when every order filled; `done` with unfilled orders,
   `abandoned` (the process died or an order never settled: after 15:30 New York or the next day), `expired` (not
   approved in time, or the 09:20–09:27 window passed), `deferred` (the breaker), `halted` (the kill switch). Anything
   short of a full fill sets `forced`, and the next close re-decides what is left from the account's real positions.
   An unfinished *first* build stays "the build" (it needs the owner's fund command again, and is exempt from the
   turnover cap, so it can actually be completed).
6. Every step goes to `core-journal.jsonl`: append-only, each entry carrying the previous entry's hash. The file
   itself is the source of truth for the chain; the state file keeps an anchor. At start-up a broken chain, or a
   journal cut short, **disables trading** and says why on `/health`.

**Guards, imported unchanged from the news router.** The kill switch blocks everything. The −2.5% daily-loss
breaker blocks any plan that buys; a sells-only raise-cash plan still passes. There are no stop-losses, no
take-profits and no session exits: core positions are held for years.

**One difference from the backtest, stated plainly.** The backtest defers the *whole* plan if the portfolio is down
2.5% at the open. Live, the 09:20–09:27 check cannot see the open's gap (the account's equity still equals the last
close), so it passes; the breaker bites later, when the buys go in after the sells have filled. A block there abandons
the rest of the plan and the close re-decides it. The effect is that on such a day the sells may execute and the buys
wait a day, where the backtest would have done neither. It matters on a rebalance day that also gaps down 2.5%: rare,
and it only delays the buys.

## Operating it

See `docs/core-runbook.md` for setup, the first run, the routine, and what to do about each alert. Secrets are
`CORE_ALPACA_KEY_ID`, `CORE_ALPACA_SECRET_KEY`, `CORE_PLAN_SECRET`, `WEBHOOK_SECRET` (and optionally
`CORE_ALERT_WEBHOOK_URL`); `CORE_PLAN_SECRET` and `WEBHOOK_SECRET` must differ, and the pre-flight checks it.

## Verified while building (2026-10-02)

- Alpaca: $1 fractional minimum, 9-decimal quantities; fractional orders must be `day` orders; `opg` (on
  the open) is not available for fractional shares; "any market orders received before 9:28 will be
  filled at the Nasdaq Official Opening Price", and limit orders are protected on the opening print; limit
  orders accept notional amounts; crypto orders take `gtc` or `ioc` only (not needed by this policy).
- Moving the due rule into `core_alloc.rebalance_due` left the backtest engine bit-identical (224 runs
  compared day by day).
- C12 passes over 65 files: the Allocator, the router and the engine all call `core_alloc`'s functions,
  and none defines its own copy (the twin, a copied function, is caught).
- **Replay (2026-10-03, `tests/test_core_replay.py`).** The real `CoreRouter` was driven session by session through
  501 real sessions (2022-01-03 to 2023-12-29, which contains the year stocks and bonds fell together) and compared
  with `portfolio.simulate`, the registered engine, at the frictionless level on the same prices and calendar. Its
  wealth path stays within **6e-6** of the engine's on every day (0.06 bp of the portfolio), and it traded on exactly
  the same 8 days (the initial build and seven quarter-ends, found from a real calendar with holidays). The test can
  fail: a router rebalancing monthly instead of quarterly is 38 bp off, a 1-point tilt in two weights is 9 bp off,
  while the harmless difference of ignoring the $1 reserve is 0.2 bp and correctly passes (tolerance 1 bp).
  This shows the paper pipeline runs the evidence's arithmetic over time; it does not show how Alpaca fills real orders.

## Hardened (2026-10-03)

A review of the first version found real defects. Each was reproduced against the real router first (a failing
test), then fixed:

| Found | Effect | Fix |
|---|---|---|
| A plan caught in `selling`/`buying` when the process died was never closed (`tick()` only expired `approved`/`awaiting_approval` ones) | After sells filled, the cash sat idle until the next quarter-end (about 6% of the account in the reproduction) | `reconcile()` closes every plan that cannot finish; `forced` re-decides it at the close |
| The breaker, checked again when the buys go in, raised out of `tick()` on every call | Buys never placed, an exception every 15 seconds | A block abandons the plan cleanly and journals `buy_blocked` |
| A 4xx on one order raised out of `tick()` for the whole 7-minute window | One refused order froze the whole plan | Recorded as `rejected`; the rest proceeds |
| A resumed submission re-sent orders it had already sent | Duplicate sells in the reproduction | Orders are skipped by client id; buys are sized once |
| The same proposal sent again reset an approved plan to awaiting approval; a new proposal could replace a plan mid-trade | An approval silently undone; live orders orphaned | Idempotent on the plan id; `plan_in_flight` refuses a replacement |
| `effective_from` was parsed and never read | A documented control that did nothing | Enforced in the shared decision, so both sides agree |
| Unfilled orders at the deadline left the plan "done" with no re-decision | Drift waited for the next quarter | An incomplete plan sets `forced` |
| An unfinished first build could not be completed (the remainder broke the 25% turnover cap) | A deadlock needing hand-placed orders | The remainder is still "the build" |
| The journal head lived in the state file | Losing it forked the chain; a journal cut short still verified | Head read from the file; the state file is an anchor; checked at start-up |
| The tick and the HTTP routes interleave at every `await` | Two callers could both abandon one stale plan (double history, double alert), or a proposal could supersede a plan whose sells were half submitted | One lock around everything that changes the plan; "in flight" means *orders exist*, not just a status |
| `raise_cash` read the day's unpublished bar and could make a plan whose window had already passed | A plan born expired | Uses the last published close; asked during the session it says `ask_after_the_close` |
| Nothing stopped a second router on one state directory | A leftover process plus a restarted container places every order twice | An exclusive lock on `.core.lock`; the second router disables itself |
| No way for a human to make the signed approval | The approval gate was unusable | `core_ctl` |
| Four tests pinned the placeholder account number | The branch was red | Tests no longer depend on the owner's account or start date; the committed policy is itself tested against the hard limits |

## Open items

- **Receipts to PFW are off** (`receipts_to_pfw = false`). PFW's ledger has no notion of a second account,
  and the core's fills would mix with the news account's. The journal is the record until PFW is ready.
- **No order has been placed.** Everything above is verified against fakes and, for the read-only paths, the real
  paper API. Fills, the opening-print behaviour and alert delivery are first seen on the first funded session.
- **Hosting is undecided.** The image builds and was smoke-tested locally on 2026-10-03: it starts as a non-root
  user, fails closed on a key Alpaca rejects (with `/health` saying why), writes its state volume and passes its
  own health check. It has not been run on a host.
  The single-instance lock covers one host; two hosts on separate disks would each read their own state.
- **The live breaker is checked at the buys, not at the open** (see "One difference from the backtest" above).
- **"The Allocator cannot trade" is a property of its code, not of its credentials.** It reads the account with the
  same Alpaca key the router trades with (Alpaca's paper keys cannot be scoped read-only). What stops a compromised
  Allocator is that the router recomputes every plan and needs the owner's separately-keyed approval for anything
  large, not that the Allocator lacks a key.
- **The drawdown ladder is off** (system design §7.4). It would block the rebalancing buys the evidence
  assumed in 2020 and 2022; turning it on is the owner's call and makes paper results diverge from the backtest.
- **The TAF pause** (FINRA, 2026-10-01 to 12-31) is not in `fees.json`: changing it would alter the
  registered fee-table version. It does not reach the backtest window, and paper accounts charge no fees.
- **The paper report measures the plumbing, not the market.** Alpaca's paper fills come from its simulator. What
  the report does not do is replay the policy through the backtest engine over the paper dates to compare the
  account's path with the modelled one; that needs fresh bars fetched outside the registered cache (extending it
  would break the reproducibility of the registered run) and is the natural next step once there is data.
- A raise-cash larger than the 25% turnover cap is refused; a withdrawal that big is a manual decision.
