"""Backtester (phase 1): the tool that decides whether a strategy deserves real money.

Design: ``docs/backtester-design.md``. A research layer builds events,
samples, walk-forward folds and predictions; an execution layer replays
each session minute by minute through the production ``RiskRouter``
against a simulated Alpaca. Only the replay produces P&L.

Module map::

    lockbox.py      where development data ends and the lock-box begins (P7)
    schema.sql      the portable point-in-time schema
    store.py        DuckDB store: idempotent writes, P7 on every read
    parquet.py      Parquet in and out through DuckDB
    fetch.py        the free fetches: raw 1-min and daily bars, calendar, news, quote sample
    calendar.py     exchange sessions and half-days; Alpaca's /v2/clock
    data.py         MarketData and the point-in-time PitView (P2-P4)
    events.py       news as events; duplicate stories
    dataset.py      samples: legacy (report.md) and v2 (the live path's timing)
    walkforward.py  monthly folds, embargo, thresholds and move tables (P5, P6)
    outcomes.py     every prediction at 5m, 30m, 2h, close, 1d, 1mo
    costs.py        fills, spreads, slippage, fees (fees.json)
    sim_broker.py   SimAlpaca, the server behind the production client
    engine.py       the minute-by-minute replay through the router
    strategies.py   S0-S3 and the baselines B1-B3
    leakage.py      checks L1-L12 and their broken twins
    registry.py     experiments, manifests, exact reproduction
    metrics.py      block-bootstrap intervals over days, drawdown, deflated Sharpe
    report.py       reports
    cli.py          python -m backtest <command>
"""
