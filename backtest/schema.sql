-- The point-in-time store: events, predictions, outcomes, trades, experiments.
--
-- Portable DDL (design §7): runs unchanged on DuckDB now and PostgreSQL
-- later. Types are limited to BIGINT, INTEGER, DOUBLE, DECIMAL(p,s),
-- VARCHAR, BOOLEAN, DATE, TIMESTAMPTZ (always UTC) and JSON; no LIST,
-- STRUCT, MAP or sequences. Keys are source ids or content hashes, so a
-- repeated write is a no-op and two different rows can never share a key.
-- Money is DECIMAL; features and statistics are DOUBLE.

CREATE TABLE IF NOT EXISTS event (
    event_id          VARCHAR PRIMARY KEY,   -- feed:source_id
    feed              VARCHAR NOT NULL,      -- alpaca
    source_id         VARCHAR NOT NULL,
    publisher         VARCHAR,               -- benzinga
    published_at      TIMESTAMPTZ NOT NULL,  -- the feed's created_at
    updated_at        TIMESTAMPTZ,           -- the feed's updated_at, when fetched
    known_at          TIMESTAMPTZ NOT NULL,
    known_at_basis    VARCHAR NOT NULL,      -- published (backfill) | received (live)
    headline          VARCHAR NOT NULL,
    headline_hash     VARCHAR NOT NULL,
    n_symbols         INTEGER NOT NULL,      -- every ticker the article names
    story_id          VARCHAR,
    dedup_version     VARCHAR,
    event_type        VARCHAR,               -- later
    subject_verified  BOOLEAN,               -- later
    mode              VARCHAR NOT NULL       -- backfill | live
);

CREATE TABLE IF NOT EXISTS event_symbol (
    event_id  VARCHAR NOT NULL,
    symbol    VARCHAR NOT NULL,
    PRIMARY KEY (event_id, symbol)
);

CREATE TABLE IF NOT EXISTS model (
    model_version    VARCHAR PRIMARY KEY,
    experiment_id    VARCHAR,
    fold_month       DATE,
    fold_start       TIMESTAMPTZ,
    embargo_cutoff   TIMESTAMPTZ,
    feature_names    JSON NOT NULL,
    windows          JSON NOT NULL,          -- train / calibration spans and counts
    label            JSON NOT NULL,
    params           JSON NOT NULL,
    seed             INTEGER NOT NULL,
    git_commit       VARCHAR NOT NULL,
    data_hash        VARCHAR NOT NULL,
    artifact_sha256  VARCHAR NOT NULL,
    thresholds       JSON NOT NULL,
    move_tables      JSON NOT NULL
);

CREATE TABLE IF NOT EXISTS prediction (
    prediction_id      VARCHAR PRIMARY KEY,  -- hash(event, symbol, model, made_at)
    event_id           VARCHAR NOT NULL,
    symbol             VARCHAR NOT NULL,
    model_version      VARCHAR NOT NULL,
    made_at            TIMESTAMPTZ NOT NULL,
    mode               VARCHAR NOT NULL,     -- backtest | shadow | live
    experiment_id      VARCHAR,
    inputs             JSON NOT NULL,        -- feature name -> value
    prob_up            DOUBLE NOT NULL,
    expected_move_pct  DOUBLE,
    horizon            VARCHAR NOT NULL,
    reference_at       TIMESTAMPTZ,          -- v2 entry: open of the first fillable bar
    reference_price    DECIMAL(18,4)         -- raw dollars
);

CREATE TABLE IF NOT EXISTS outcome (
    prediction_id   VARCHAR NOT NULL,
    horizon         VARCHAR NOT NULL,        -- 5m | 30m | 2h | close | 1d | 1mo
    resolved_at     TIMESTAMPTZ,
    ret             DOUBLE,
    max_favourable  DOUBLE,
    max_adverse     DOUBLE,
    price_space     VARCHAR NOT NULL,        -- raw | adjusted
    status          VARCHAR NOT NULL,        -- ok | truncated | no_data
    PRIMARY KEY (prediction_id, horizon)
);

CREATE TABLE IF NOT EXISTS trade_event (
    run_id           VARCHAR NOT NULL,
    trade_id         VARCHAR NOT NULL,
    seq              INTEGER NOT NULL,
    occurred_at      TIMESTAMPTZ NOT NULL,
    state            VARCHAR NOT NULL,       -- signal | rejected | submitted | filled | expired
                                             -- | exit_submitted | exit_filled | exit_expired
    reason_code      VARCHAR,
    symbol           VARCHAR NOT NULL,
    side             VARCHAR,
    qty              DECIMAL(20,9),
    price            DECIMAL(18,4),          -- fill price, costs included
    ref_price        DECIMAL(18,4),          -- the fill bar's open: the frictionless price
    cash             DECIMAL(18,2),          -- signed; debits rounded up, credits down
    fees             DECIMAL(18,6),          -- per order, or exact with a daily rounding row
    spread_cost      DECIMAL(18,6),
    slippage_cost    DECIMAL(18,6),
    order_id         VARCHAR,
    client_order_id  VARCHAR,
    prediction_id    VARCHAR,
    detail           JSON,                   -- the router's Decision, verbatim
    PRIMARY KEY (run_id, trade_id, seq)
);

CREATE TABLE IF NOT EXISTS experiment (
    experiment_id  VARCHAR PRIMARY KEY,
    created_at     TIMESTAMPTZ NOT NULL,
    hypothesis     VARCHAR NOT NULL,
    command        VARCHAR NOT NULL,
    parent_id      VARCHAR,
    status         VARCHAR NOT NULL,         -- running | done | failed
    superseded_by  VARCHAR,
    window_start   DATE,
    window_end     DATE,
    lockbox        BOOLEAN NOT NULL,
    git_commit     VARCHAR NOT NULL,
    git_dirty      BOOLEAN NOT NULL,
    data_hash      VARCHAR NOT NULL,
    manifest       JSON NOT NULL,
    params         JSON NOT NULL,
    seeds          JSON NOT NULL,
    environment    JSON NOT NULL,
    metrics        JSON,
    report_hash    VARCHAR,
    conclusion     VARCHAR,
    finished_at    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS run (
    run_id         VARCHAR PRIMARY KEY,
    experiment_id  VARCHAR NOT NULL,
    strategy       VARCHAR NOT NULL,
    cost_level     VARCHAR NOT NULL,
    variant        VARCHAR NOT NULL,
    params         JSON NOT NULL,
    metrics        JSON
);

-- One row per trade: its lifecycle folded, P&L split into gross (at the
-- fill bars' opens), spread, slippage, other (limit price and cent
-- rounding), fees and net. Daily fee-rounding rows (trade_id 'fees') are
-- not trades; runs add them to the total.
CREATE OR REPLACE VIEW trade AS
SELECT
    run_id,
    trade_id,
    max(symbol)                                                      AS symbol,
    min(CASE WHEN state = 'signal' THEN occurred_at END)                      AS signal_at,
    max(CASE WHEN state = 'rejected' THEN reason_code END)           AS rejected_code,
    max(CASE WHEN state = 'filled' THEN occurred_at END)                      AS entry_at,
    max(CASE WHEN state = 'filled' THEN price END)                   AS entry_price,
    max(CASE WHEN state = 'filled' THEN qty END)                     AS qty,
    max(CASE WHEN state = 'exit_submitted' THEN reason_code END)     AS exit_reason,
    max(CASE WHEN state = 'exit_filled' THEN occurred_at END)                 AS exit_at,
    max(CASE WHEN state = 'exit_filled' THEN price END)              AS exit_price,
    sum(CASE WHEN state = 'filled' THEN -qty * ref_price
             WHEN state = 'exit_filled' THEN qty * ref_price END)    AS gross_pnl,
    sum(coalesce(spread_cost, 0))                                    AS spread_cost,
    sum(coalesce(slippage_cost, 0))                                  AS slippage_cost,
    sum(coalesce(fees, 0))                                           AS fees,
    sum(coalesce(cash, 0))                                           AS cash_pnl,
    sum(coalesce(cash, 0)) - sum(coalesce(fees, 0))                  AS net_pnl
FROM trade_event
WHERE trade_id <> 'fees'
GROUP BY run_id, trade_id;

-- Event memory: every prediction joined to what happened next, per horizon.
CREATE OR REPLACE VIEW event_memory AS
SELECT
    e.event_id, e.published_at, e.known_at, e.headline, e.story_id,
    p.prediction_id, p.symbol, p.model_version, p.made_at, p.mode, p.experiment_id,
    p.prob_up, p.expected_move_pct, p.reference_at, p.reference_price,
    max(CASE WHEN o.horizon = '5m' THEN o.ret END)     AS ret_5m,
    max(CASE WHEN o.horizon = '30m' THEN o.ret END)    AS ret_30m,
    max(CASE WHEN o.horizon = '2h' THEN o.ret END)     AS ret_2h,
    max(CASE WHEN o.horizon = 'close' THEN o.ret END)  AS ret_close,
    max(CASE WHEN o.horizon = '1d' THEN o.ret END)     AS ret_1d,
    max(CASE WHEN o.horizon = '1mo' THEN o.ret END)    AS ret_1mo,
    count(o.horizon)                                   AS n_outcomes
FROM event e
JOIN prediction p ON p.event_id = e.event_id
LEFT JOIN outcome o ON o.prediction_id = p.prediction_id
GROUP BY e.event_id, e.published_at, e.known_at, e.headline, e.story_id,
         p.prediction_id, p.symbol, p.model_version, p.made_at, p.mode, p.experiment_id,
         p.prob_up, p.expected_move_pct, p.reference_at, p.reference_price;
