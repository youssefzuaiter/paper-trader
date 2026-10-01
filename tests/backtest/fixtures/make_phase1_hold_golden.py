"""Write phase1_hold_golden.json: every DailyDay of phase 1's S0 and S1 at the three cost levels.

Run once, from a worktree of commit f700c0d (the clean phase 1 run x-20260927-112403-badb5a8d),
against this repository's caches:

    git worktree add --detach /tmp/p1 f700c0d
    PYTHONPATH=/tmp/p1 python /tmp/p1/../<this file> --data-root ~/paper-trader --out <fixture>

Check C7 requires the long-term core's engine, in its PHASE1 profile, to reproduce this file exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from decimal import Decimal
from pathlib import Path

from backtest.costs import LEVELS, FeeTable
from backtest.data import MarketData
from backtest.fetch import SYMBOLS, Paths
from backtest.pipeline import Plan
from backtest.strategies import run_hold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.data_root)
    import backtest
    code = Path(backtest.__file__).resolve().parent.parent
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code, check=True, capture_output=True,
                            text=True).stdout.strip()
    market = MarketData.load(Paths(root / ".cache" / "backtest"), symbols=SYMBOLS,
                             training_cache=root / ".cache" / "training")
    plan = Plan()
    sessions = market.calendar.sessions_between(plan.first_session, plan.last_session)
    fees = FeeTable.load()
    series = {}
    for name, band in (("S0", None), ("S1", plan.s1_band)):
        for level in ("optimistic", "central", "pessimistic"):
            days = run_hold(market, SYMBOLS, sessions, LEVELS[level], fees, capital=Decimal(100000), band=band)
            series[f"{name}:{level}"] = [[d.date.isoformat(), str(d.value), str(d.traded), str(d.costs)] for d in days]
    body = json.dumps(series, sort_keys=True, separators=(",", ":"))
    out = {"generated_from_commit": commit, "symbols": list(SYMBOLS), "first_session": plan.first_session.isoformat(),
           "last_session": plan.last_session.isoformat(), "s1_band": str(plan.s1_band), "capital": "100000",
           "fee_table_version": fees.version, "series_sha256": hashlib.sha256(body.encode()).hexdigest(),
           "series": series}
    Path(args.out).write_text(json.dumps(out, indent=0, sort_keys=True) + "\n", encoding="utf-8")
    print(commit, out["series_sha256"], {k: len(v) for k, v in series.items()})


if __name__ == "__main__":
    main()
