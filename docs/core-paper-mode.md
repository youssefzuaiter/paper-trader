# Long-term core: paper mode

**Status:** built and tested, not deployed, no order placed · **Written:** 2026-10-02 · **Designs:**
`docs/long-term-core-design.md` §8, reconciled with the system design PDF §3 and §7.

> Not financial advice. The policy is the owner's own choice: M3 with a 5% BIL cash buffer, rebalanced
> quarterly, chosen 2026-10-02 from registered run `x-20261001-121225-9c4ebec5`.

## What runs

| Part | File | Can trade? | Does |
|---|---|---|---|
| Policy | `policy/core.toml` | no | The owner's mix, rule, buffer and limits; read-only to every agent; its sha256 is in every journal entry |
| Hard limits | `tier0_core.py` | no | Ceilings the policy must stay inside; per-plan checks |
| Shared decision | `core_paper.py` | no | The snapshot, the decision, order payloads; calls `core_alloc`, the backtest's own functions |
| Allocator | `swarm/core_allocator.py` | **no** | After each close: reads, decides, signs, proposes |
| Core router | `risk_router/core_gatekeeper.py`, `core_app.py` | **yes, the only part** | Recomputes every plan, enforces limits and approvals, executes at the open, journals |

The news router (`risk_router/app.py`, `tier0.py`) is unchanged except for one refusal: it will not
start on the core's account.

## One day

1. **16:20 New York.** The Allocator reads the core account and the day's closes, asks the router whether
   the last plan was deferred, and decides with `core_alloc.rebalance_due`, the same rule the backtest used.
   For this policy, that means the last session of each quarter.
2. The router **rebuilds the same snapshot from its own reads**, recomputes the plan, and rejects any
   difference (`plan_mismatch`). Then the limits: per-order ceiling, 25% turnover per rebalance, two plans a
   month, a weekly cap on trades and value, the registered instruments only.
3. **Approval.** The initial build waits for the owner's signed `fund` command, and is refused if the
   account holds more than the policy's $10,000 funded cap. A plan trading more than $3,000, and every
   raise-cash plan, waits for the owner's signed approval. Anything else is approved automatically.
4. **09:20–09:27 next session.** Sells go in as limit orders at the previous close − 3%: Alpaca fills
   orders received before 09:28 at the official opening price, as the backtest assumed. Once they fill,
   buys go in as marketable notional limits (ask + 0.5%), scaled to the cash the account actually has. A
   plan with no sells (the initial build) sends its buys pre-open at the close + 3%.
5. Every step goes to `core-journal.jsonl`: append-only, each entry carrying the previous entry's hash.

**Guards, imported unchanged from the news router.** The kill switch blocks everything. PFW's halt
reaches both apps through the same control secret, but each app must be called. The −2.5% daily-loss
breaker defers any plan that buys, **whole**, to be re-decided at the next close, exactly as the backtest
modelled. A sells-only raise-cash plan still passes. There are no stop-losses, no take-profits and no
session exits: core positions are held for years.

## What you will need to do (deployment, task 6)

1. **Create a second Alpaca paper account with a $10,000 starting balance.** At a $100,000 balance the
   router refuses the build (`over_funded`).
2. Put its account number (`PA…`) in `policy/core.toml` (`account = "PA…"`) through a reviewed commit.
   Until then the core app starts disabled and says why on `/health`.
3. Secrets, set by you, never in chat or git: `CORE_ALPACA_KEY_ID`, `CORE_ALPACA_SECRET_KEY` (the new
   account's keys), `CORE_PLAN_SECRET` (32+ random characters, shared by the Allocator and the core router),
   and the existing `WEBHOOK_SECRET` for your signed control calls.
4. Run the core app (`uvicorn risk_router.core_app:create_core_app --factory`) and schedule the Allocator
   after 16:20 New York on each session (`python -m swarm.core_allocator`).
5. The first evening, the Allocator proposes the initial build; you approve it with the signed `fund`
   command; it executes at the next open.

## Verified while building (2026-10-02)

- Alpaca: $1 fractional minimum, 9-decimal quantities; fractional orders must be `day` orders; `opg` (on
  the open) is not available for fractional shares; "any market orders received before 9:28 will be
  filled at the Nasdaq Official Opening Price", and limit orders are protected on the opening print; limit
  orders accept notional amounts; crypto orders take `gtc` or `ioc` only (not needed by this policy).
- Moving the due rule into `core_alloc.rebalance_due` left the backtest engine bit-identical (224 runs
  compared day by day).
- C12 passes over 65 files: the Allocator, the router and the engine all call `core_alloc`'s functions,
  and none defines its own copy (the twin, a copied function, is caught).
- Tests (30) cover: the policy's limits and its equality with the registered configuration; the shared
  decision; plan mismatch and forged inputs; the fund command and the funded cap; the turnover cap; a
  whole execution day (sells at the open, buys with the cash the account has); breaker deferral; the kill
  switch; expiry and re-decision; raise cash (buffer first, passes the breaker); wrong accounts both ways;
  the journal's hash chain; the app failing closed with halt still working; the Allocator's signed
  proposal admitted end to end.

## Open items

- **Receipts to PFW are off** (`receipts_to_pfw = false`). PFW's ledger has no notion of a second account,
  and the core's fills would mix with the news account's. The journal is the record until PFW is ready.
- **The drawdown ladder is off** (system design §7.4). It would block the rebalancing buys the evidence
  assumed in 2020 and 2022; turning it on is the owner's call and makes paper results diverge from the
  backtest.
- **The TAF pause** (FINRA, 2026-10-01 to 12-31) is not in `fees.json`: changing it would alter the
  registered fee-table version. It does not reach the backtest window, and paper accounts charge no fees.
- **The shadow line's daily comparison** (`core_paper.shortfall_bps`) is in place; the monthly report
  over the journal is task 7.
- A raise-cash larger than the 25% turnover cap is refused; a withdrawal that big is a manual decision.
