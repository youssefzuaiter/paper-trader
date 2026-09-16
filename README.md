# Tier-0 Paper Trading Agent

Sandbox-only equities agent. Sentiment inference gated by a deterministic
execution layer, with HMAC-signed trade receipts pushed to the PFW Next.js
dashboard.

**Nothing here can touch real money.** `broker.py` hardcodes `paper=True` as a
module-level `Final`, and re-asserts after construction that the resolved client
points at `https://paper-api.alpaca.markets` with `sandbox=True`. There is no
code path that builds a live `TradingClient`.

Everything in the stack is free: Alpaca paper accounts, `alpaca-py`, and the
FinBERT checkpoint (`ProsusAI/finbert`, MIT) all cost nothing, and market data is
simulated so no data subscription is required.

---

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
cp .env.example .env      # then fill in ALPACA_* from the paper dashboard
./.venv/bin/uvicorn main:app --reload --port 8000
```

Paper keys: <https://app.alpaca.markets/paper/dashboard/overview> → *API Keys*.
`WEBHOOK_SECRET` is pre-generated in `.env`; copy that same value into PFW.

---

## Module map

| File | Phase | Responsibility |
|------|-------|----------------|
| `config.py`    | 1 | `BrokerSettings` + `WebhookSettings`, validated independently. |
| `broker.py`    | 1 | Alpaca `TradingClient`, sandbox enforced structurally. |
| `inference.py` | 2 | Simulated quotes/news + FinBERT-shaped async inference. |
| `execution.py` | 3 | Tier-0 gates, sizing, order construction, submission. |
| `webhook.py`   | 4 | Canonical JSON receipt, HMAC-SHA256, `httpx` delivery. |
| `outbox.py`    | — | Durable JSONL queue of receipts PFW never acknowledged; re-signed and replayed with backoff. |
| `reconcile.py` | — | Re-derives settlement receipts from Alpaca's own order history (startup + hourly + on demand). |
| `main.py`      | 1 | FastAPI app, lifespan, HTTP surface. |

---

## Settings are split by trust boundary

`config.py` exposes two Pydantic models that validate **independently**:

| Model | Holds | Needed for |
|-------|-------|-----------|
| `BrokerSettings` | `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`, pinned `base_url` | *Placing* an order |
| `WebhookSettings` | `WEBHOOK_SECRET`, `WEBHOOK_URL`, `USD_ILS_RATE` | *Delivering* a receipt |

This is not tidiness. A single all-or-nothing settings object meant a
missing `ALPACA_API_KEY_ID` raised while merely *building a receipt* —
blocking the one path that has no business touching the broker. Signing a
receipt for a fill that already happened must not depend on credentials for
placing a new one.

So the service boots and serves the full receipt path with no Alpaca
credentials at all. `GET /health` reports `broker_configured` and
`webhook_configured` separately; `/signals/execute` and `/account` return
`503` naming the exact missing variables, and nothing else is affected.

`BrokerSettings.base_url` is present but **never read from the
environment** — it is pinned to the paper host by a validator that rejects
the live one. Phase 1 made sandbox execution structural; routing it through
`.env` would hand it back to whoever can edit that file.

---

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/health` | Liveness + active Tier-0 limits + `outbox_pending`. No credentials needed. |
| `GET`  | `/account` | Paper account snapshot. |
| `GET`  | `/telemetry` | The in-memory event feed (last 50 wake/evaluate/reject/execute/settle/sleep events). |
| `POST` | `/signals/evaluate` | **Dry run.** Ingest → infer → gate. Submits nothing. |
| `POST` | `/signals/execute` | Full pipeline, including order + receipt. |
| `POST` | `/control/reconcile` | Re-send settlements for every fill Alpaca reports in `lookback_hours` (default 24, max 720). HMAC-verified inbound like `/control/halt`. Idempotent on the PFW side. |
| `POST` | `/analyze/transaction` | Cash-flow anomaly check (Phase 3, ad hoc) — HMAC-verified inbound, same trust boundary as `/control/halt`. See [Cash-flow anomaly detection](#cash-flow-anomaly-detection). |
| `GET`  | `/docs` | OpenAPI UI. |

```bash
curl -X POST localhost:8000/signals/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"ticker":"NVDA","headline":"NVDA beats estimates and raises full-year guidance"}'
```

Omit `headline` to pull one from the simulated feed. A Tier-0 rejection is a
normal `200` with `approved: false` — the engine declining to trade is an
expected outcome, not an error.

Both `/signals/*` endpoints run the same `scheduler.run_signal_cycle` pipeline
as the autonomous loop, so a manual call shows up in `/telemetry` (a dry run is
labelled `(dry run)`) and posts a scenario-metrics event to PFW like any
autonomous cycle — previously neither happened, which left PFW's Agent
Activity page empty whenever `AUTONOMOUS_MODE` was off.

---

## Tier-0 rules

Hardcoded `Final` constants in `execution.py`. They are deliberately **not**
environment-configurable and not reachable from any request body — a limit a
caller can widen is not a limit.

| Constant | Value | Effect |
|----------|-------|--------|
| `MIN_PREDICTED_GAIN_PCT` | `10` | Reject any signal predicting `< +10%`. |
| `MAX_NOTIONAL_USD` | `10` | Hard cap on per-order exposure. |
| `STOP_LOSS_PCT` | `5` | Stop distance below the entry limit. |
| `LIMIT_BUFFER_PCT` | `0.25` | Marketable-limit buffer above the ask. |

Sizing is `Decimal` throughout with explicit rounding modes — entry rounds **up**,
stop rounds **down**, quantity rounds **down** — so rounding can only ever tighten
risk. A post-condition re-checks the notional against the cap and suppresses the
order if it fails. The single `float` cast happens at the Alpaca API boundary.

### Constraint: fractional orders cannot carry a broker-side stop

Alpaca's server rejects the `bracket` / `OCO` / `OTO` order classes on fractional
quantities. With a $10 cap, almost every liquid symbol sizes fractionally, so
both branches are live. `_choose_stop_strategy` takes the strongest stop the
broker will actually accept:

| Sizing | `stop_loss_kind` | Behaviour |
|--------|------------------|-----------|
| Whole shares (≥ 1, integral) | `native_oto` | Real resting stop child order at Alpaca. |
| Fractional | `engine_tracked` | Broker cannot hold it; stop price is computed, logged at `WARNING`, and carried on the receipt. |

The stop is **never silently dropped** — but on the fractional path it is not yet
enforced. Closing that gap needs a position monitor that polls marks and submits
the exit; that does not exist yet and is the highest-value next piece of work.

---

## Webhook contract

`POST {WEBHOOK_URL}` → `http://localhost:3000/api/webhooks/trades`

| Header | Value |
|--------|-------|
| `X-Signature-Timestamp` | Unix seconds |
| `X-Signature-256` | `sha256=<hex digest>` |
| `X-Idempotency-Key` | Mirrors `idempotency_key` in the body |

Signed material is `f"{timestamp}." + body` (Stripe's scheme) — binding the
timestamp into the MAC is what makes a captured receipt un-replayable.

Two invariants worth preserving if you edit `webhook.py`:

1. **Sign the bytes you send.** The body is serialised once (sorted keys, no
   whitespace) and passed to `httpx` as `content=`, never `json=`. Handing httpx
   a dict lets it re-serialise, and the digest would stop describing the
   transmitted bytes.
2. **Money crosses the wire as integers.** Minor units (cents / agorot) as JSON
   integers; share quantities and FX rates as decimal *strings*. PFW stores these
   as `BigInt` and `Decimal(30, 18)`; a JSON float round-trip corrupts both.

Field names mirror PFW's `Trade` model so the route can persist without a
translation layer:

```json
{
  "schema_version": 1,
  "idempotency_key": "<client_order_id — 32 hex chars>",
  "broker": "alpaca", "environment": "paper",
  "order_id": "...", "client_order_id": "...", "order_status": "accepted",
  "symbol": "NVDA", "side": "buy", "quantity": "0.082236842",
  "currency": "USD",
  "native_price_amount": 12160, "native_total_amount": 1000,
  "exchange_rate_at_entry": "3.700000",
  "price_agorot": 44992, "total_agorot": 3700,
  "limit_price": "121.60", "stop_price": "115.52",
  "stop_loss_kind": "engine_tracked",
  "executed_at": "2026-09-05T11:24:27Z",
  "signal": { "model_name": "...", "predicted_move_pct": 11.9592,
              "confidence": 0.874027, "headline": "..." }
}
```

Receiving side — read the **raw** body before any JSON parsing, or the digest
will not match:

```ts
import { createHmac, timingSafeEqual } from "node:crypto";

const raw = await req.text();                       // raw bytes, not req.json()
const ts  = req.headers.get("x-signature-timestamp") ?? "";
const sig = (req.headers.get("x-signature-256") ?? "").replace("sha256=", "");

if (Math.abs(Date.now() / 1000 - Number(ts)) > 300) return new Response(null, { status: 401 });

const expected = createHmac("sha256", process.env.WEBHOOK_SECRET!)
  .update(`${ts}.${raw}`)
  .digest("hex");

const a = Buffer.from(expected), b = Buffer.from(sig);
if (a.length !== b.length || !timingSafeEqual(a, b)) return new Response(null, { status: 401 });
```

Delivery never raises: the trade has already executed, so a webhook failure is
reported as structured status rather than unwinding the order.

### Durable delivery (`outbox.py`)

Three quick inline retries (~1.5s) cover a blip. Anything longer used to mean
the receipt was simply lost — for an order already sitting at the broker. Now a
retryable failure (transport error, 5xx, 429) is appended to
`outbox/pending.jsonl` and a background loop replays it every 30s with capped
exponential backoff (30s → 5min) until PFW returns 2xx. Each replay **re-signs
with a fresh timestamp** — PFW rejects a signature older than 300s, so the
original headers can't be reused — while the body bytes stay identical, and
`idempotency_key` is what lets PFW dedupe a replay whose 2xx was lost in
flight. A permanent 4xx goes to `outbox/dead-letter.jsonl` instead; entries
older than 7 days are dead-lettered too. `GET /health` reports
`outbox_pending`, which PFW's dashboard badge surfaces as "N undelivered
receipts". The `outbox/` directory is runtime state, gitignored.

### Reconciliation (`reconcile.py`)

The outbox only helps once this process *knows* a delivery failed. Two
real cases slip past it: a settlement PFW acknowledged with a 200 but had
actually dropped (a race in PFW's own route, since fixed), and an order
whose receipts were never generated because PFW was down at submit time.
Alpaca is the source of truth for fills, so 15s after startup and then
hourly, `reconcile_recent_fills` lists every CLOSED order in the last 24h,
rebuilds each FILLED one's settlement receipt with the same
`build_settlement_receipt(order)` the live WebSocket path uses, and
re-sends it. PFW handles every outcome idempotently — already settled →
`200`, stranded pending → settled `201`, never seen → created `201` — so
re-sending is always safe. A `201` is logged as `RECOVERED` and shows on
PFW's Agent Activity page as a `settle (reconciled)` event. Trigger a
pass by hand (e.g. after a long PFW outage) with `POST /control/reconcile`
and `{"lookback_hours": 168}`.

### The receipt's ILS figures are informational

`exchange_rate_at_entry`, `price_agorot` and `total_agorot` are computed from
this service's fixed `USD_ILS_RATE` (default 3.7). PFW does **not** book them:
it re-prices `native_price_amount` at its own Frankfurter-synced rate at
receipt time (its law is "convert once, at execution, at the real rate") and
logs a warning if the two rates differ by more than 2%. Keep sending them —
the schema still requires them — but don't expect them to match what PFW
stores.

---

## Cash-flow anomaly detection

`models/autoencoder.py` is a lightweight PyTorch autoencoder (11 input
features: log-normalized amount, cyclical hour-of-day, one-hot category)
that flags a single transaction as anomalous by how badly it reconstructs
— the standard autoencoder-anomaly-detection paradigm, trained only on
synthetic "normal" traffic (`models/train_autoencoder.py`; there is no
real transaction history here to train against — this service has no
database connection to PFW's ledger at all). `POST /analyze/transaction`
runs a transaction through it and compares the reconstruction MSE against
a Z-score threshold derived at training time from a held-out normal
validation set's own error distribution (default `z > 2.5`).

Unlike `/signals/*`, this is genuinely trained, not a placeholder (see
below) — an untrained model's threshold would be statistically
meaningless, and `is_anomaly: true` is a direct, user-facing flag against
someone's own financial data, not an internal trading signal sitting
behind separate hard-coded safety caps. Retrain with
`python -m models.train_autoencoder` if the category vocabulary
(`CATEGORY_VOCAB` in `models/autoencoder.py`) ever changes — training and
inference share that constant so the two can't silently drift apart.

Same inbound HMAC trust boundary as `/control/halt`: the caller (PFW's
own server) signs the request with the shared `WEBHOOK_SECRET`, verified
here via `webhook.verify()`. No new secret needed. This endpoint has no
way to look up anything about the caller's account — it only ever sees
the transaction fields in the request body.

---

## What is a placeholder

`inference.py` ships a fixed-weight `torch.nn.Module` over a SHA-256
pseudo-embedding, plus a small lexical prior so the stub behaves intelligibly
during development. It has the real model's contract — 3-class logits in
ProsusAI/finbert's label order (`0=positive, 1=negative, 2=neutral`), softmax,
blocking forward dispatched through `asyncio.to_thread` so a CPU-bound
transformer never stalls the event loop.

Output is **deterministic in `(ticker, headline)`**. A Tier-0 engine whose
upstream signal is random is untestable: the same headline must always produce
the same order.

To swap in the real model: `pip install transformers`, replace the body of
`_load_model`, and tokenise in `_embed` instead of hashing. Nothing downstream of
`predict_move` changes.

Also simulated: quotes (`fetch_quote`) and the news feed
(`fetch_latest_headline`), both deterministic with time-based drift.
