"""Risk & Routing agent — the swarm's gatekeeper.

The only service in the target architecture that may place an order.
Inference Agents send it signals; it applies the kill switch and the daily
circuit breaker, blocks duplicate accumulation and portfolio-limit
breaches, prices and sizes through ``tier0``, submits to Alpaca without
blocking the event loop, and exits positions at their stop-loss or
take-profit.

Module map::

    alpaca_async.py  non-blocking, paper-only Alpaca REST client (httpx)
    state.py         kill-switch / breaker latch persisted to disk
    guards.py        ExecutionGuard: halt + circuit breaker, per order intent
    policy.py        portfolio limits over a snapshot of Alpaca's own state
    gatekeeper.py    RiskRouter: signal → guard → policy → size → submit; exits
    app.py           FastAPI surface; every execution route is guarded

Run with ``uvicorn risk_router.app:app --port 8080``.
"""
