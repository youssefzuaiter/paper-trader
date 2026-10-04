# Long-term core: runbook

**For:** the owner · **Written:** 2026-10-03 · `docs/core-paper-mode.md` says what the parts are and why; this
says what you *do* with them.

> Paper account, no real money, not financial advice. Every command below is run from the repo root, reads
> `.env`, and is named `./.venv/bin/python -m …`.

## What runs, and when

| Part | When | You |
|---|---|---|
| **Core router** `risk_router.core_app` | Always on. It may only send orders 09:20–09:27 New York on a session day, so it must be up then, and its state directory must survive restarts. | leave it running |
| **Allocator** `swarm.core_allocator` | Once per session, after 16:20 New York. A day with no session exits 0, so a daily schedule is fine. | schedule it |
| **Console** `risk_router.core_ctl` | When something needs you: `status`, `approve`, `raise-cash`, `halt`, `test-alert`. | answer it |
| **Preflight** `risk_router.core_preflight` | Before the first evening, and after any change to keys, policy or host. Read-only. | run it |
| **Report** `backtest.paper_report` | Monthly. Read-only. | read it |
| **Website mirror** `risk_router.core_sync` | Inside the router, if `CORE_PFW_SYNC_URL` is set. Read-only. | watch `/trading/core` |

Raising cash (`core_ctl raise-cash`) must be asked **after 16:20 New York**: the plan is decided on a published close
and executes at the next open, so asked during the session it would aim at an open that has already passed, and the
router says so (`ask_after_the_close`). Starting a second router on the same state directory is refused (it would
place every order twice); the second one disables itself and says why.

New York 09:20–09:27 is 16:20–16:27 in UTC+3 while New York is on daylight time, and an hour later when it is not.

## One-time setup

1. **A second Alpaca *paper* account** with a **$10,000** starting balance (the default is $100,000 and the router
   refuses to build on more than the policy's funded cap: `over_funded`). Its `PA…` number is already in
   `policy/core.toml`.
2. **Secrets in `.env`** (names in `.env.example`; values never in git or in a chat):
   `CORE_ALPACA_KEY_ID`, `CORE_ALPACA_SECRET_KEY`, `CORE_PLAN_SECRET`, `WEBHOOK_SECRET`, `CORE_ROUTER_URL`.
   Generate each secret with `python3 -c "import secrets; print(secrets.token_hex(32))"`.
   **`CORE_PLAN_SECRET` and `WEBHOOK_SECRET` must be different.** The first signs the Allocator's proposals, the
   second signs your approvals and halts; if they were equal, a compromised Allocator could approve its own plans.
   The preflight fails if they match.
3. **Alerts.** Nothing else will tell you a plan is waiting for your approval. Easiest is [ntfy](https://ntfy.sh): install
   the phone app, subscribe to a long random topic name, then in `.env`:
   `CORE_ALERT_WEBHOOK_URL=https://ntfy.sh/<your-topic>` and `CORE_ALERT_FORMAT=ntfy`. (Slack and Discord incoming
   webhooks work with the default `json` format.) Check it: `core_ctl test-alert`.
4. **Optional but quick: `python -m pytest tests/test_core_replay.py -s`** (2 seconds, needs the research caches on this
   machine). It replays the real router through two years of real history and prints how closely it tracks the
   registered engine (about 0.06 bp). Run it again after any change to the router.
5. **`core_preflight` must say `READY`.** It checks the policy against the hard limits, the secrets (by name, never
   by value), the account, the funding, the exchange calendar, that SIP daily bars are readable for every
   instrument, and the journal; and it prints the plan the Allocator would propose.
6. **Start the router:** `uvicorn risk_router.core_app:create_core_app --factory --port 8080`, or the image from
   `Dockerfile.core` with a volume at `/data`. `GET /health` should say `"status": "ok"`.
7. **Schedule the Allocator** after 16:20 New York. A UTC schedule that is right all year is 21:30 (that is 17:30
   in summer and 16:30 in winter in New York; the Allocator refuses to run before 16:20 there):
   `30 21 * * 1-5  cd /path/to/paper-trader && ./.venv/bin/python -m swarm.core_allocator >> core-state/allocator.log 2>&1`

## Watching it on the website (optional)

PFW (the website) can show the core: its status, account, holdings against target, the open plan, plan history and
the journal. It is **read-only**: it cannot approve, halt or trade, because the website holds no credential for the
router. Approving a plan is still `core_ctl approve`, from the router's host.

1. In PFW's deployment, `PAPER_TRADING_USER_EMAIL` must name the account that should see the core (it already does,
   for the trading agent), and the migration `core_agent_mirror` must be applied.
2. In this repository's `.env`: `CORE_PFW_SYNC_URL=https://<your PFW domain>/api/webhooks/core`. It is signed with
   `WEBHOOK_SECRET`, the same value PFW already holds for the trade receipts. `core_preflight` checks the setting
   (`website mirror`) and never blocks on it.
3. Start the router. Within a minute the website shows the journal; within five, the router's status; within the
   hour, the account. `GET /health` shows the mirror under `pfw_sync` (`last_success_at`, `pending_entries`,
   `last_error`).

How it works, in one paragraph: the router sends from a cursor over its own journal and moves it only when PFW says it
holds the entries, so there is no second queue to lose, and a website that is down just means "send again later". The
cursor lives in `core-sync-state.json` beside the journal; losing it is harmless. A copy of the journal on the website
re-checks its own hash chain on every page load.

## The first run

1. Run the Allocator for the last published session: `./.venv/bin/python -m swarm.core_allocator --session 2026-10-02`.
   (Preflight already showed you its answer: six buys, about $9,999 in total.)
2. `core_ctl status` shows the **initial build** waiting for you. Read the orders, then
   `core_ctl approve --fund`. The `--fund` flag is the signed *fund command*: you are saying the account really holds
   the cash and you want it spent.
3. Before **09:27 New York** on the first session (`effective_from` in the policy, 2026-10-05), the router submits the
   six buys as collared limit orders, which Alpaca fills at the opening print. Miss the window and nothing is lost:
   the plan expires and is proposed again that evening.
4. `core_ctl status` afterwards shows no plan open. `backtest.paper_report` shows the build and its fills.

## Routine

| When | What |
|---|---|
| Each evening | Nothing. The Allocator runs by itself; you hear from ntfy only if something needs you. |
| Quarter-end | A rebalance is proposed. Under $3,000 traded it runs by itself; over that it waits for `core_ctl approve`. |
| Weekly | `core_ctl status`, or open `/health`: an empty `attention` list is the all-clear. |
| Monthly | `./.venv/bin/python -m backtest.paper_report --out reports/2026-11.md`. Read *Incidents*; the reading rules are printed at its top. |
| Daily | Back up `core-state/` (three small files: the journal, the plan state, the kill switch). It is the record the report is built from, and a lost disk loses it. |

## When something goes wrong

| What you are told | What it means | What to do |
|---|---|---|
| **A plan needs your approval** | The initial build, a plan over $3,000, or a raise-cash plan. | `core_ctl approve` (it shows the orders first). Before 09:27 New York on the execution day, or it expires and is re-decided that evening. |
| **A rebalance was abandoned part-way** | The router died, or an order never settled, between the sells and the buys. It cancelled what was left. | Nothing: the next close re-decides it from the real positions. If it repeats, find out why the host was down. |
| **A plan missed its session** | Not approved in time, or the router was down 09:20–09:27. | Nothing: re-decided at that evening's close, one day late. |
| **The breaker deferred a plan / the buys were blocked after the open** | The account was down ≥ 2.5% on the day. | Nothing. Sells may have executed; no exposure was added; the close re-decides. (The breaker cannot see a gap before the open, so this can only show up after the sells fill. The backtest defers the whole plan; paper cannot.) |
| **Alpaca refused an order** | A 4xx: insufficient quantity, a halted symbol, a rule. The rest of the plan carried on. | Read the `error` in the journal. One-off: ignore. Repeated: investigate. |
| **Sells/buys did not fill in time** | The limit collar (3%) was not reached on a big gap, or Alpaca was slow. | Nothing: the remainder is re-decided at the close. |
| **The router disagrees with the Allocator** (`plan_mismatch`) | Two programs computed different plans from what should be identical inputs. | **Stop and look.** Approve nothing. Usually the account or the bars changed between the two reads: run the Allocator once more. If it persists, the two hosts run different code: compare versions. |
| **A plan broke a limit** | The hard limits refused it (e.g. turnover over 25%). | A rebalance that large means a position changed by hand or something is badly wrong. There is no override, on purpose. Find out why before touching the limits. |
| **The execution loop keeps failing** | Three ticks in a row (45 s) failed; repeats hourly. | Check Alpaca's status and the host's log. Orders are at-most-once by client id, so retries are safe. |
| **The website mirror has not updated for 30 minutes** | `GET /health` lists it under `attention`; `pfw_sync.last_error` says why. `HTTP 403`: `WEBHOOK_SECRET` differs between here and PFW. `HTTP 503 paper_trading_user_unresolved`: PFW's `PAPER_TRADING_USER_EMAIL` matches no user. `HTTP 503 core_mirror_unavailable`: PFW's `core_agent_mirror` migration has not been applied. `HTTP 404`: PFW has not deployed the route yet. `chain_conflict`: PFW holds a different history for an entry (the journal was restored from an older copy, or PFW's data was changed). | Trading is unaffected, and nothing is lost: the journal is the outbox. Fix the cause and it catches up by itself. For `chain_conflict`, compare `core-state/core-journal.jsonl` with the website's journal before changing anything. |
| **The journal failed its integrity check** | An entry was altered, removed, or the files are from different times. **Trading is disabled until this is resolved.** | **Do not delete anything.** Copy `core-state/` aside, compare with your backup, and see whether the damage is the last entry (a restore fixes it) or the middle (investigate). |

**The kill switch:** `core_ctl halt` stops all core trading and cancels open orders. It survives restarts. To resume,
delete `core-router-state.json` in the state directory on the router's host and restart it.

## Where to run it

The router has two hard needs: **up at 09:20–09:27 New York on session days**, and **a state directory that survives
a restart**. Everything else is flexible. These are the honest trade-offs; none has been run on a real host (the image itself builds and boots locally).

- **Your own machine.** Free, and it works if the machine is awake in the window. If it sleeps, the plan expires and
  is re-decided that evening: a day's delay, not a loss. Fine for the first weeks, while you watch.
- **A small always-on host** (a VPS of a few dollars a month, or a container host with a volume) running
  `Dockerfile.core` with `/data` on a persistent volume. The right shape for months.
- `/health` is unauthenticated on purpose (so a monitor can read it). It shows plan ids and statuses, the policy hash
  and what needs attention, never a secret. If the router is reachable from the internet, put your host's access
  control in front of it; the signed routes are protected either way.
- **Render's free tier does not fit**: it sleeps when idle (misses the window) and has no persistent disk.

## Changing the policy

`policy/core.toml` changes only by a reviewed commit, then a restart (it is baked into the image). A different policy
hash starts a new period: the report refuses to mix two, so run it with `--since <the day the new policy took
effect>`. The hard ceilings in `tier0_core.py` change the same way and are deliberately not configurable.
