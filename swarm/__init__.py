"""The swarm's upstream agents: Ingestion → Quantitative → Inference.

Each agent is a small FastAPI app (``/health`` for Kubernetes probes) whose
lifespan runs one worker loop. They talk over Redis Streams consumer groups
and hand trade signals to the Risk & Routing agent (``risk_router/``), the
only component that can place orders.

Module map::

    common.py              watchlist, stream names, Redis connection
    alpaca_data.py         async Alpaca market-data client: news + bars, rate-limited
    features.py            pure feature maths shared by training and serving
    bus.py                 Redis Streams publish / consumer-group loop / dead letters
    ingestion.py           Ingestion Agent: live news WebSocket + gap backfill
    quant.py               Quantitative Agent: volatility + technical features
    return_model.py        the trained scikit-learn up-move model, loaded for serving
    inference_agent.py     Inference Agent: FinBERT + return model → signed signals
    train_return_model.py  offline training on real news and realised returns
"""
