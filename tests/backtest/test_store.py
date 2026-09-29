"""The point-in-time store: portable DDL, idempotent writes, P7 on every read."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from backtest.lockbox import DEV_EVENTS_END, issue_key
from backtest.store import KEYS, SCHEMA_PATH, Store, StoreConflict, canonical_json

T0 = datetime(2025, 3, 3, 14, 30, tzinfo=UTC)


def event(n: int, known_at: datetime = T0, **overrides: Any) -> dict[str, Any]:
    row = {
        "event_id": f"alpaca:{n}", "feed": "alpaca", "source_id": str(n), "publisher": "benzinga",
        "published_at": known_at, "updated_at": None, "known_at": known_at, "known_at_basis": "published",
        "headline": f"headline {n}", "headline_hash": f"h{n}", "n_symbols": 1, "mode": "backfill",
    }
    row.update(overrides)
    return row


def prediction(n: int, event_id: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "prediction_id": f"p{n}", "event_id": event_id, "symbol": "AAPL", "model_version": "m1",
        "made_at": T0 + timedelta(minutes=1), "mode": "backtest", "experiment_id": "x1",
        "inputs": {"rsi14": 55.0, "mom5": 1.2}, "prob_up": 0.41, "expected_move_pct": 0.05,
        "horizon": "close", "reference_at": T0 + timedelta(minutes=2), "reference_price": Decimal("187.1500"),
    }
    row.update(overrides)
    return row


def experiment_row(eid: str = "x1", **overrides: Any) -> dict[str, Any]:
    row = {
        "experiment_id": eid, "created_at": T0, "hypothesis": "h", "command": "legacy", "status": "running",
        "lockbox": False, "git_commit": "abc", "git_dirty": False, "data_hash": "d", "manifest": {"files": []},
        "params": {}, "seeds": {"gbm": 7}, "environment": {"python": "3.13"},
    }
    row.update(overrides)
    return row


@pytest.fixture
def store() -> Store:
    s = Store()
    yield s
    s.close()


# --- schema -------------------------------------------------------------------------

def test_declared_keys_match_the_ddl(store: Store) -> None:
    rows = store._rows("SELECT table_name, constraint_column_names FROM duckdb_constraints() "
                       "WHERE constraint_type = 'PRIMARY KEY'", [])
    assert {r["table_name"]: tuple(r["constraint_column_names"]) for r in rows} == KEYS


def test_ddl_uses_only_portable_types() -> None:
    """The DDL must run on PostgreSQL unchanged: no LIST, STRUCT, MAP or sequences."""
    ddl = SCHEMA_PATH.read_text(encoding="utf-8")
    tables = re.findall(r"CREATE TABLE IF NOT EXISTS \w+ \((.*?)\n\);", ddl, flags=re.DOTALL)
    assert len(tables) == len(KEYS)
    allowed = re.compile(r"^(BIGINT|INTEGER|DOUBLE|DECIMAL\(\d+,\d+\)|VARCHAR|BOOLEAN|DATE|TIMESTAMPTZ|JSON)(\s|$)")
    for body in tables:
        for line in body.splitlines():
            line = line.split("--")[0].strip().rstrip(",")
            if not line or line.startswith("PRIMARY KEY"):
                continue
            _name, sql_type = line.split(None, 1)
            assert allowed.match(sql_type), line
    code = "\n".join(line.split("--")[0] for line in ddl.splitlines()).upper()
    for forbidden in ("LIST", "STRUCT", "MAP(", "SEQUENCE", "[]"):
        assert forbidden not in code


# --- writes -------------------------------------------------------------------------

def test_writes_are_idempotent_and_collisions_are_refused(store: Store) -> None:
    assert store.put_events([event(1), event(2)], [{"event_id": "alpaca:1", "symbol": "AAPL"}]) == 2
    assert store.put_events([event(1)], [{"event_id": "alpaca:1", "symbol": "AAPL"}]) == 0  # same content

    with pytest.raises(StoreConflict):
        store.put_events([event(1, headline="a revised headline")], [])
    with pytest.raises(StoreConflict):  # two versions of one key in a single batch
        store.put_events([event(3), event(3, headline="other")], [])
    assert len(store.events()) == 2


def test_values_round_trip_exactly(store: Store) -> None:
    store.put_events([event(1)], [])
    store.put_predictions([prediction(1, "alpaca:1")])
    (p,) = store.predictions()
    assert p["reference_price"] == Decimal("187.1500")
    assert p["made_at"] == T0 + timedelta(minutes=1) and p["made_at"].utcoffset() == timedelta(0)
    assert p["inputs"] == canonical_json({"mom5": 1.2, "rsi14": 55.0})
    # the same prediction with its JSON keys in another order is the same row
    assert store.put_predictions([prediction(1, "alpaca:1", inputs={"mom5": 1.2, "rsi14": 55.0})]) == 0


@pytest.mark.parametrize(("field", "value", "error"), [
    ("known_at", datetime(2025, 3, 3, 14, 30), ValueError),  # noqa: DTZ001 — naive on purpose: which zone?
    ("prob_up", float("nan"), ValueError),
])
def test_bad_values_are_refused(store: Store, field: str, value: Any, error: type[Exception]) -> None:
    row = event(1) if field == "known_at" else prediction(1, "alpaca:1")
    row[field] = value
    with pytest.raises(error):
        (store.put_events([row], []) if field == "known_at" else store.put_predictions([row]))


def test_money_is_never_a_float(store: Store) -> None:
    store.put_events([event(1)], [])
    with pytest.raises(TypeError):
        store.put_predictions([prediction(1, "alpaca:1", reference_price=187.15)])


def test_updated_at_is_filled_once(store: Store) -> None:
    store.put_events([event(1)], [])
    later = T0 + timedelta(days=2)
    store.set_updated_at({"alpaca:1": later})
    store.set_updated_at({"alpaca:1": later})  # same value: fine
    with pytest.raises(StoreConflict):
        store.set_updated_at({"alpaca:1": later + timedelta(hours=1)})
    assert store.events()[0]["updated_at"] == later


# --- P7 ------------------------------------------------------------------------------

def test_lockbox_rows_are_invisible_without_a_key(tmp_path) -> None:
    path = tmp_path / "store.duckdb"
    with Store(path) as s:
        s.put_events([event(1, DEV_EVENTS_END - timedelta(seconds=1)), event(2, DEV_EVENTS_END)],
                     [{"event_id": "alpaca:1", "symbol": "AAPL"}, {"event_id": "alpaca:2", "symbol": "AAPL"}])
        s.put_predictions([prediction(1, "alpaca:1"), prediction(2, "alpaca:2")])
        s.put_outcomes([{"prediction_id": pid, "horizon": "close", "ret": 0.01, "price_space": "raw",
                         "status": "ok"} for pid in ("p1", "p2")])
        assert [e["event_id"] for e in s.events()] == ["alpaca:1"]
        assert [p["prediction_id"] for p in s.predictions()] == ["p1"]
        assert [o["prediction_id"] for o in s.outcomes()] == ["p1"]
        assert [m["prediction_id"] for m in s.event_memory()] == ["p1"]
        assert len(s.event_symbols()) == 1
    with Store(path, lockbox=issue_key("x-lockbox")) as s:
        assert len(s.events()) == 2 and len(s.predictions()) == 2


# --- experiments -------------------------------------------------------------------------

def test_experiments_are_registered_once_finished_once_and_never_deleted(store: Store) -> None:
    store.register(experiment_row())
    with pytest.raises(StoreConflict):
        store.register(experiment_row())
    store.finish("x1", status="done", metrics={"auc": 0.52}, report_hash="r", conclusion="c",
                 finished_at=T0 + timedelta(hours=1))
    with pytest.raises(StoreConflict):
        store.finish("x1", status="failed", metrics=None, report_hash=None, conclusion=None,
                     finished_at=T0 + timedelta(hours=2))
    store.register(experiment_row("x2", created_at=T0 + timedelta(days=1)))
    store.supersede("x1", "x2")
    rows = store.experiments()
    assert [(r["experiment_id"], r["status"], r["superseded_by"]) for r in rows] == [
        ("x1", "done", "x2"), ("x2", "running", None)]
    assert not any(name.startswith("delete") for name in dir(Store))


# --- views -----------------------------------------------------------------------------

def test_trade_view_splits_pnl_into_its_components(store: Store) -> None:
    def te(seq: int, state: str, **kw: Any) -> dict[str, Any]:
        return {"run_id": "r1", "trade_id": "t1", "seq": seq, "occurred_at": T0 + timedelta(minutes=seq), "state": state,
                "symbol": "AAPL", **kw}

    store.append_trade_events([
        te(0, "signal"),
        te(1, "submitted", side="buy", qty=Decimal("0.05"), price=Decimal("200.50")),
        te(2, "filled", side="buy", qty=Decimal("0.05"), price=Decimal("200.0600"), ref_price=Decimal("200.00"),
           cash=Decimal("-10.01"), spread_cost=Decimal("0.0015"), slippage_cost=Decimal("0.0020")),
        te(3, "exit_submitted", reason_code="session_close", side="sell", qty=Decimal("0.05")),
        te(4, "exit_filled", side="sell", qty=Decimal("0.05"), price=Decimal("201.9400"), ref_price=Decimal("202.00"),
           cash=Decimal("10.09"), fees=Decimal("0.02"), spread_cost=Decimal("0.0015"),
           slippage_cost=Decimal("0.0015")),
    ])
    (t,) = store.trades("r1")
    assert t["gross_pnl"] == Decimal("0.10")            # 0.05 x (202 - 200)
    assert t["cash_pnl"] == Decimal("0.08")             # -10.01 + 10.09
    assert t["net_pnl"] == Decimal("0.06")              # after 0.02 of fees
    assert t["exit_reason"] == "session_close"
    assert (t["entry_price"], t["exit_price"]) == (Decimal("200.0600"), Decimal("201.9400"))
    with pytest.raises(StoreConflict):  # the lifecycle is append-only
        store.append_trade_events([te(4, "exit_filled", price=Decimal("1"))])


def test_models_and_runs(store: Store) -> None:
    store.put_models([{
        "model_version": "wf-2025-01-abc", "experiment_id": "x1", "fold_month": date(2025, 1, 1),
        "fold_start": T0, "embargo_cutoff": T0, "feature_names": ["a"], "windows": {}, "label": {},
        "params": {"max_iter": 300}, "seed": 7, "git_commit": "abc", "data_hash": "d", "artifact_sha256": "s",
        "thresholds": {"S2": 0.46}, "move_tables": {},
    }])
    store.put_run({"run_id": "r1", "experiment_id": "x1", "strategy": "S2", "cost_level": "central",
                   "variant": "all", "params": {}})
    store.set_run_metrics("r1", {"sharpe": 0.1})
    assert store.models()[0]["fold_month"] == date(2025, 1, 1)
    assert store.runs("x1")[0]["metrics"] == canonical_json({"sharpe": 0.1})


def _model(experiment_id: str, **overrides: object) -> dict:
    return {
        "model_version": "wf-2025-01-abc", "experiment_id": experiment_id, "fold_month": date(2025, 1, 1),
        "fold_start": T0, "embargo_cutoff": T0, "feature_names": ["a"], "windows": {}, "label": {},
        "params": {"max_iter": 300}, "seed": 7, "git_commit": "abc", "data_hash": "d", "artifact_sha256": "s",
        "thresholds": {"S2": 0.46}, "move_tables": {}, **overrides,
    }


def test_the_same_fold_can_belong_to_several_experiments(store: Store) -> None:
    """A clean re-run or a reproduction fits the same content-addressed model_version under a new
    experiment (and commit). That is its own row, not a clash."""
    store.put_models([_model("x-dirty")])
    store.put_models([_model("x-clean", git_commit="def", artifact_sha256="s2")])
    assert [m["experiment_id"] for m in store.models(experiment_id="x-clean")] == ["x-clean"]
    assert len(store.models()) == 2
    with pytest.raises(StoreConflict):  # within one experiment a key still means one content
        store.put_models([_model("x-clean", thresholds={"S2": 0.5})])


def test_a_store_keyed_the_old_way_is_migrated_with_its_rows(tmp_path: Path) -> None:
    path = tmp_path / "old.duckdb"
    with Store(path) as s:  # rebuild the model table as it was before 2026-09-27
        s.put_models([_model("x1")])
        con = s._con
        con.execute("CREATE TABLE model_old AS SELECT * FROM model")
        con.execute("DROP TABLE model")
        ddl = SCHEMA_PATH.read_text(encoding="utf-8")
        old = re.search(r"CREATE TABLE IF NOT EXISTS model \(.*?\n\);", ddl, flags=re.DOTALL).group(0)
        old = old.replace("model_version    VARCHAR NOT NULL,", "model_version    VARCHAR PRIMARY KEY,")
        old = old.replace(",\n    PRIMARY KEY (model_version, experiment_id)", "")
        con.execute(old)
        con.execute("INSERT INTO model SELECT * FROM model_old")
        con.execute("DROP TABLE model_old")
        key = con.execute("SELECT constraint_column_names FROM duckdb_constraints() "
                          "WHERE table_name = 'model' AND constraint_type = 'PRIMARY KEY'").fetchone()[0]
        assert tuple(key) == ("model_version",)
    with Store(path) as s:
        key = s._con.execute("SELECT constraint_column_names FROM duckdb_constraints() "
                             "WHERE table_name = 'model' AND constraint_type = 'PRIMARY KEY'").fetchone()[0]
        assert tuple(key) == ("model_version", "experiment_id")
        assert [m["experiment_id"] for m in s.models()] == ["x1"]
        s.put_models([_model("x2")])  # the collision that blocked clean re-runs
        assert len(s.models()) == 2
