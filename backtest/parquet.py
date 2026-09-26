"""Parquet in and out through DuckDB, so no pyarrow or pandas is needed.

Writes are atomic (a temporary file, then ``os.replace``): an interrupted
fetch leaves the previous file or none, never half of one.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    return con


def write(path: Path, columns: Mapping[str, Sequence[Any]], types: Mapping[str, str], *,
          order_by: str | None = None) -> int:
    """Write equal-length ``columns`` (one Python list per column) as ``types``."""
    names = list(types)
    if set(columns) != set(names):
        raise ValueError(f"columns {sorted(columns)} != declared {sorted(names)}")
    lengths = {len(columns[n]) for n in names}
    if len(lengths) != 1:
        raise ValueError(f"columns differ in length: {lengths}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    con = _connect()
    try:
        select = ", ".join(f'CAST(unnest(?) AS {types[n]}) AS "{n}"' for n in names)
        con.execute(f"CREATE TABLE t ({', '.join(f'\"{n}\" {types[n]}' for n in names)})")
        if lengths.pop():
            con.execute(f"INSERT INTO t SELECT * FROM (SELECT {select})", [list(columns[n]) for n in names])
        order = f" ORDER BY {order_by}" if order_by else ""
        con.execute(f"COPY (SELECT * FROM t{order}) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        count = int(con.execute("SELECT count(*) FROM t").fetchone()[0])
    finally:
        con.close()
    os.replace(tmp, path)
    return count


def read_rows(path: Path, sql_where: str = "", params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    con = _connect()
    try:
        cursor = con.execute(f"SELECT * FROM read_parquet('{path}') {sql_where}", list(params))
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
    finally:
        con.close()


def read_numpy(path: Path, select: str, sql_tail: str = "", params: Sequence[Any] = ()) -> dict[str, np.ndarray]:
    """Columns as NumPy arrays, e.g. ``select="epoch(t)::BIGINT AS t, o, h"``."""
    con = _connect()
    try:
        return con.execute(f"SELECT {select} FROM read_parquet('{path}') {sql_tail}", list(params)).fetchnumpy()
    finally:
        con.close()
