"""The point-in-time store (design §7): DuckDB on the portable ``schema.sql``.

DuckDB now, PostgreSQL + TimescaleDB once the live agents write: DuckDB is
one embedded file, free, columnar and reads Parquet natively; its single
writer is irrelevant to research and is exactly why live writers will need
PostgreSQL on the same DDL.

Writes are idempotent by key. Writing a row whose key exists with the same
content is a no-op; the same key with *different* content raises
``StoreConflict`` instead of silently keeping either version. Rows are
staged column by column (``unnest`` of one list per column): DuckDB's
``executemany`` inserts row by row, 200x slower at this size.

Reads of events, predictions and outcomes stop at the lock-box (rule P7)
unless the store was opened with a ``LockBoxKey``. There is deliberately
no public raw-SQL method: every read goes through one that applies P7.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, Self

import duckdb

from backtest.lockbox import DEV_EVENTS_END, LockBoxKey

SCHEMA_PATH: Final[Path] = Path(__file__).with_name("schema.sql")

#: Primary keys, as declared in schema.sql (a test holds the two together).
KEYS: Final[dict[str, tuple[str, ...]]] = {
    "event": ("event_id",),
    "event_symbol": ("event_id", "symbol"),
    "model": ("model_version", "experiment_id"),
    "prediction": ("prediction_id",),
    "outcome": ("prediction_id", "horizon"),
    "trade_event": ("run_id", "trade_id", "seq"),
    "experiment": ("experiment_id",),
    "run": ("run_id",),
    "core_day": ("run_id", "day"),
}


class StoreConflict(RuntimeError):
    """A write reused an existing key for different content."""


def canonical_json(value: Any) -> str:
    """One text per value, so stored JSON compares equal iff the values do.
    NaN and infinities are refused: they are not JSON."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return _require_aware(value).astimezone(UTC).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"naive datetime {value!r}: the store holds UTC instants only")
    return value


class Store:
    def __init__(self, path: Path | str = ":memory:", *, lockbox: LockBoxKey | None = None) -> None:
        self.path = str(path)
        self._lockbox = lockbox
        self._con = duckdb.connect(self.path)
        self._con.execute("SET TimeZone = 'UTC'")
        self._con.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        self._migrate_model_key()
        self._types = {
            table: dict(self._con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position", [table]).fetchall())
            for table in KEYS
        }

    def _migrate_model_key(self) -> None:
        """Stores created before 2026-09-27 keyed ``model`` on ``model_version`` alone, so a second
        walk-forward (a clean re-run, or ``reproduce``) collided with the first. Rebuild the table
        with the (model_version, experiment_id) key, keeping every row. A no-op once migrated."""
        key = self._con.execute(
            "SELECT constraint_column_names FROM duckdb_constraints() "
            "WHERE table_name = 'model' AND constraint_type = 'PRIMARY KEY'").fetchone()
        if key is None or tuple(key[0]) == KEYS["model"]:
            return
        ddl = re.search(r"CREATE TABLE IF NOT EXISTS model \(.*?\n\);",
                        SCHEMA_PATH.read_text(encoding="utf-8"), flags=re.DOTALL)
        if ddl is None:
            raise RuntimeError("schema.sql has no model table")
        self._con.execute("BEGIN TRANSACTION")
        try:
            self._con.execute("ALTER TABLE model RENAME TO model_pre_migration")
            self._con.execute(ddl.group(0))
            self._con.execute("INSERT INTO model SELECT * FROM model_pre_migration")
            self._con.execute("DROP TABLE model_pre_migration")
            self._con.execute("COMMIT")
        except Exception:
            self._con.execute("ROLLBACK")
            raise

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def lockbox_open(self) -> bool:
        return self._lockbox is not None

    # --- writes --------------------------------------------------------------

    def put_events(self, events: Sequence[Mapping[str, Any]], symbols: Sequence[Mapping[str, Any]]) -> int:
        n = self._insert("event", events)
        self._insert("event_symbol", symbols)
        return n

    def set_story_ids(self, story_ids: Mapping[str, str], dedup_version: str) -> None:
        """Re-clustering is an explicit, versioned update, never a silent overwrite."""
        if not story_ids:
            return
        self._con.execute("CREATE OR REPLACE TEMP TABLE _story (event_id VARCHAR, story_id VARCHAR)")
        self._con.execute("INSERT INTO _story SELECT unnest(?), unnest(?)",
                          [list(story_ids), list(story_ids.values())])
        self._con.execute(
            "UPDATE event SET story_id = s.story_id, dedup_version = ? FROM _story s "
            "WHERE event.event_id = s.event_id", [dedup_version])

    def set_updated_at(self, updated_at: Mapping[str, datetime]) -> None:
        """Fill ``updated_at`` from a later fetch. Only NULLs are filled; a
        different value already stored is a conflict."""
        if not updated_at:
            return
        self._con.execute("CREATE OR REPLACE TEMP TABLE _upd (event_id VARCHAR, updated_at TIMESTAMPTZ)")
        self._con.execute("INSERT INTO _upd SELECT unnest(?), unnest(?)",
                          [list(updated_at), [_require_aware(v) for v in updated_at.values()]])
        clash = self._con.execute(
            "SELECT e.event_id FROM event e JOIN _upd u USING (event_id) "
            "WHERE e.updated_at IS NOT NULL AND e.updated_at <> u.updated_at LIMIT 5").fetchall()
        if clash:
            raise StoreConflict(f"updated_at already stored with other values for {[c[0] for c in clash]}")
        self._con.execute("UPDATE event SET updated_at = u.updated_at FROM _upd u "
                          "WHERE event.event_id = u.event_id AND event.updated_at IS NULL")

    def put_models(self, rows: Sequence[Mapping[str, Any]]) -> int:
        return self._insert("model", rows)

    def put_predictions(self, rows: Sequence[Mapping[str, Any]]) -> int:
        return self._insert("prediction", rows)

    def put_outcomes(self, rows: Sequence[Mapping[str, Any]]) -> int:
        return self._insert("outcome", rows)

    def append_trade_events(self, rows: Sequence[Mapping[str, Any]]) -> int:
        return self._insert("trade_event", rows)

    def put_run(self, row: Mapping[str, Any]) -> None:
        self._insert("run", [row])

    def put_runs(self, rows: Sequence[Mapping[str, Any]]) -> int:
        return sum(self._insert("run", batch) for batch in chunks(rows, 50_000))

    def put_core_days(self, rows: Sequence[Mapping[str, Any]]) -> int:
        return sum(self._insert("core_day", batch) for batch in chunks(rows, 50_000))

    def core_days(self, run_id: str) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM core_day WHERE run_id = ? ORDER BY day", [run_id])

    def set_run_metrics(self, run_id: str, metrics: Mapping[str, Any]) -> None:
        self._con.execute("UPDATE run SET metrics = ? WHERE run_id = ? AND metrics IS NULL",
                          [canonical_json(metrics), run_id])

    def register(self, row: Mapping[str, Any]) -> None:
        """A new experiment row. Never overwritten, never deleted."""
        if self._con.execute("SELECT 1 FROM experiment WHERE experiment_id = ?",
                             [row["experiment_id"]]).fetchone():
            raise StoreConflict(f"experiment {row['experiment_id']} is already registered")
        self._insert("experiment", [row])

    def finish(self, experiment_id: str, *, status: str, metrics: Mapping[str, Any] | None,
               report_hash: str | None, conclusion: str | None, finished_at: datetime) -> None:
        """Close a running experiment exactly once."""
        if status not in {"done", "failed"}:
            raise ValueError(f"status must be done or failed, not {status!r}")
        current = self._con.execute("SELECT status FROM experiment WHERE experiment_id = ?",
                                    [experiment_id]).fetchone()
        if current is None:
            raise KeyError(experiment_id)
        if current[0] != "running":
            raise StoreConflict(f"experiment {experiment_id} is already {current[0]}")
        self._con.execute(
            "UPDATE experiment SET status = ?, metrics = ?, report_hash = ?, conclusion = ?, finished_at = ? "
            "WHERE experiment_id = ?",
            [status, None if metrics is None else canonical_json(metrics), report_hash, conclusion,
             _require_aware(finished_at), experiment_id])

    def supersede(self, old_id: str, new_id: str) -> None:
        for eid in (old_id, new_id):
            if self._con.execute("SELECT 1 FROM experiment WHERE experiment_id = ?", [eid]).fetchone() is None:
                raise KeyError(eid)
        self._con.execute("UPDATE experiment SET superseded_by = ? WHERE experiment_id = ? AND superseded_by IS NULL",
                          [new_id, old_id])

    # --- reads -----------------------------------------------------------------

    def _visible(self, alias: str = "e") -> tuple[str, list[Any]]:
        """The P7 clause for queries joined to ``event`` as ``alias``."""
        if self._lockbox is not None:
            return "TRUE", []
        return f"{alias}.known_at < ?", [DEV_EVENTS_END]

    def events(self) -> list[dict[str, Any]]:
        clause, params = self._visible()
        return self._rows(f"SELECT * FROM event e WHERE {clause} ORDER BY e.known_at, e.event_id", params)

    def event_symbols(self) -> list[dict[str, Any]]:
        clause, params = self._visible()
        return self._rows(f"SELECT s.* FROM event_symbol s JOIN event e USING (event_id) WHERE {clause} "
                          "ORDER BY s.event_id, s.symbol", params)

    def predictions(self, *, experiment_id: str | None = None,
                    model_version: str | None = None) -> list[dict[str, Any]]:
        clause, params = self._visible()
        sql = f"SELECT p.* FROM prediction p JOIN event e USING (event_id) WHERE {clause}"
        if experiment_id is not None:
            sql, params = sql + " AND p.experiment_id = ?", [*params, experiment_id]
        if model_version is not None:
            sql, params = sql + " AND p.model_version = ?", [*params, model_version]
        return self._rows(sql + " ORDER BY p.made_at, p.prediction_id", params)

    def outcomes(self, *, experiment_id: str | None = None) -> list[dict[str, Any]]:
        clause, params = self._visible()
        sql = (f"SELECT o.* FROM outcome o JOIN prediction p USING (prediction_id) "
               f"JOIN event e USING (event_id) WHERE {clause}")
        if experiment_id is not None:
            sql, params = sql + " AND p.experiment_id = ?", [*params, experiment_id]
        return self._rows(sql + " ORDER BY o.prediction_id, o.horizon", params)

    def event_memory(self, *, experiment_id: str | None = None) -> list[dict[str, Any]]:
        clause, params = self._visible("m")
        sql = f"SELECT * FROM event_memory m WHERE {clause}"
        if experiment_id is not None:
            sql, params = sql + " AND m.experiment_id = ?", [*params, experiment_id]
        return self._rows(sql + " ORDER BY m.made_at, m.prediction_id", params)

    def models(self, *, experiment_id: str | None = None) -> list[dict[str, Any]]:
        if experiment_id is None:
            return self._rows("SELECT * FROM model ORDER BY fold_month, model_version", [])
        return self._rows("SELECT * FROM model WHERE experiment_id = ? ORDER BY fold_month, model_version",
                          [experiment_id])

    def trade_events(self, run_id: str) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM trade_event WHERE run_id = ? ORDER BY occurred_at, trade_id, seq", [run_id])

    def trades(self, run_id: str) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM trade WHERE run_id = ? ORDER BY signal_at, trade_id", [run_id])

    def runs(self, experiment_id: str) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM run WHERE experiment_id = ? ORDER BY run_id", [experiment_id])

    def experiment(self, experiment_id: str) -> dict[str, Any]:
        rows = self._rows("SELECT * FROM experiment WHERE experiment_id = ?", [experiment_id])
        if not rows:
            raise KeyError(experiment_id)
        return rows[0]

    def experiments(self, *, command: str | None = None) -> list[dict[str, Any]]:
        if command is None:
            return self._rows("SELECT * FROM experiment ORDER BY created_at, experiment_id", [])
        return self._rows("SELECT * FROM experiment WHERE command = ? ORDER BY created_at, experiment_id",
                          [command])

    def _rows(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        cursor = self._con.execute(sql, params)
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

    # --- the one insert path -------------------------------------------------------

    def _insert(self, table: str, rows: Sequence[Mapping[str, Any]]) -> int:
        """Stage ``rows``, refuse key collisions with different content,
        insert what is new. Returns the number of new rows."""
        if not rows:
            return 0
        types = self._types[table]
        unknown = set().union(*(r.keys() for r in rows)) - set(types)
        if unknown:
            raise ValueError(f"{table}: unknown columns {sorted(unknown)}")
        columns = list(types)
        values = [[_to_db(r.get(c), types[c]) for r in rows] for c in columns]
        stage = f"_stage_{table}"
        self._con.execute(f"CREATE OR REPLACE TEMP TABLE {stage} AS SELECT * FROM {table} LIMIT 0")
        select = ", ".join(f"CAST(unnest(?) AS {types[c]}) AS {c}" for c in columns)
        self._con.execute(f"INSERT INTO {stage} SELECT * FROM (SELECT {select})", values)

        key = KEYS[table]
        on_key = " AND ".join(f"s.{k} = t.{k}" for k in key)
        differs = " OR ".join(f"s.{c} IS DISTINCT FROM t.{c}" for c in columns if c not in key)
        clashes = self._con.execute(
            f"SELECT {', '.join('s.' + k for k in key)} FROM {stage} s JOIN {table} t ON {on_key} "
            f"WHERE {differs or 'FALSE'} LIMIT 5").fetchall()
        within = self._con.execute(
            f"SELECT {', '.join(key)} FROM (SELECT DISTINCT * FROM {stage}) GROUP BY {', '.join(key)} "
            "HAVING count(*) > 1 LIMIT 5").fetchall()
        if clashes or within:
            raise StoreConflict(f"{table}: key reused with different content: {clashes or within}")
        before = self._count(table)
        self._con.execute(f"INSERT INTO {table} SELECT DISTINCT * FROM {stage} ON CONFLICT DO NOTHING")
        self._con.execute(f"DROP TABLE {stage}")
        return self._count(table) - before

    def _count(self, table: str) -> int:
        return int(self._con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


def _to_db(value: Any, sql_type: str) -> Any:
    if value is None:
        return None
    if sql_type == "JSON":
        return canonical_json(json.loads(value) if isinstance(value, str) else value)
    if sql_type.startswith("DECIMAL"):
        if isinstance(value, float):
            raise TypeError(f"money must be Decimal or str, not float ({value!r})")
        return str(value)
    if sql_type.startswith("TIMESTAMP"):
        if not isinstance(value, datetime):
            raise TypeError(f"expected an aware datetime, got {value!r}")
        return _require_aware(value)
    if sql_type == "DOUBLE":
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite DOUBLE {value!r}")
    return value


def chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]
