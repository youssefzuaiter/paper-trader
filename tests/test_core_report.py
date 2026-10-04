"""The paper-period report. Its input is the journal the real router writes, so the tests drive the real router
through a build and a rebalance and read back what it wrote; the parser cannot drift from the writer unseen."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from test_core_paper import SESSIONS, Clock, FakeBroker, built, policy, policy_file
from test_core_resilience import START, FlakyBroker, at_the_sells, build_portfolio, rally_and_submit_sells

import core_paper
from backtest import paper_report as pr
from risk_router.core_gatekeeper import CoreStore, Journal

D = Decimal
COMMITTED_SHA = policy().sha256


@pytest.fixture(autouse=True)
def restore_the_calendar() -> Iterator[None]:
    backup = list(SESSIONS)
    yield
    SESSIONS[:] = backup


def entries_of(tmp_path: Path) -> list[dict[str, Any]]:
    return pr.parse_entries((tmp_path / "journal.jsonl").read_text())


def opens_for(fills: list[pr.Fill], bps: str = "3") -> dict[tuple[str, date], Decimal]:
    """Session opens placed so that every fill paid exactly ``bps`` basis points against its open."""
    k = D(bps) / 10000
    return {(f.symbol, f.session): f.price / (1 + (k if f.side == "buy" else -k)) for f in fills}


async def a_build_and_a_rebalance(tmp_path: Path) -> tuple[Any, FlakyBroker]:
    broker, clock = FlakyBroker(), Clock(START)
    router = await built(tmp_path, broker, clock)
    await rally_and_submit_sells(router, broker, clock)
    broker.fill_all("sell")
    clock.set("2027-01-04", 9, 40)
    await router.tick()
    broker.fill_all("buy")
    clock.set("2027-01-04", 9, 50)
    await router.tick()
    assert router.store.data.plan is None
    return router, broker


def status_of(router: Any) -> dict[str, Any]:
    return router.journal.status(repair=False)


# --- reading what the router wrote -----------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_report_reads_a_journal_the_real_router_wrote(tmp_path: Path) -> None:
    router, _ = await a_build_and_a_rebalance(tmp_path)
    entries = entries_of(tmp_path)
    fills = pr.extract_fills(entries)
    report = pr.build_report(entries, policy(), journal_status=status_of(router), opens=opens_for(fills),
                             generated=datetime(2027, 1, 5, tzinfo=UTC))
    assert report.invalid is None and report.shas_seen == [COMMITTED_SHA]
    assert [(p.kind, p.outcome) for p in report.plans] == [("initial", "done"), ("rebalance", "done")]
    assert all(p.filled == p.orders > 0 for p in report.plans)
    assert report.incidents == []
    ex = report.execution
    assert ex is not None and ex.n == len(fills) > 6 and ex.unpriced == 0
    assert ex.mean_bps == pytest.approx(3.0, abs=1e-6)
    assert ex.by_side["buy"][1] == pytest.approx(3.0, abs=1e-6) and ex.by_side["sell"][1] == pytest.approx(3.0, abs=1e-6)
    assert "too few" in ex.placement
    text = pr.render(report)
    assert "Reading rules" in text and "No incident" in text and "too few to say" in text
    assert "Alpaca's paper fills come from its simulator" in text


@pytest.mark.asyncio
async def test_an_abandoned_plan_is_an_incident_and_its_fills_still_count(tmp_path: Path) -> None:
    router, broker, clock = await at_the_sells(tmp_path)
    broker.fill_all("sell")
    clock.set("2027-01-05", 9, 25)
    await router.tick()
    entries = entries_of(tmp_path)
    report = pr.build_report(entries, policy(), journal_status=status_of(router))
    outcomes = [(p.kind, p.outcome) for p in report.plans]
    assert ("rebalance", "abandoned") in outcomes
    assert [i.event for i in report.incidents] == ["plan_abandoned"]
    assert any(f.side == "sell" for f in pr.extract_fills(entries)), "an abandoned plan's sells did trade"
    text = pr.render(report)
    assert "Incidents to look at" in text and "plan_abandoned" in text


@pytest.mark.asyncio
async def test_a_plan_that_waited_for_approval_shows_how_long(tmp_path: Path) -> None:
    router, _ = await a_build_and_a_rebalance(tmp_path)
    report = pr.build_report(entries_of(tmp_path), policy(), journal_status=status_of(router))
    initial = report.plans[0]
    assert initial.approval_wait is not None and initial.approval_wait.total_seconds() >= 0


@pytest.mark.asyncio
async def test_since_keeps_the_fills_of_a_plan_decided_inside_the_window_only(tmp_path: Path) -> None:
    router, _ = await a_build_and_a_rebalance(tmp_path)
    entries = entries_of(tmp_path)
    everything = pr.windowed_fills(entries, None)
    later = pr.windowed_fills(entries, "2026-12-01")
    first = pr.windowed_fills(entries, "2026-12-31")
    assert 0 < len(later) < len(everything) and {f.session for f in later} == {date(2027, 1, 4)}
    assert {f.plan_id for f in first} == {f.plan_id for f in later}
    report = pr.build_report(entries, policy(), journal_status=status_of(router), since="2026-12-01")
    assert [p.kind for p in report.plans] == ["rebalance"]


# --- the integrity rules (P1) -------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_failed_integrity_check_prints_no_evidence(tmp_path: Path) -> None:
    router, _ = await a_build_and_a_rebalance(tmp_path)
    path = tmp_path / "journal.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:3] + lines[4:]) + "\n")                  # one entry removed
    status = Journal(path, CoreStore(tmp_path / "core-plan-state.json"), "x").status(repair=False)
    report = pr.build_report(pr.parse_entries(path.read_text()), policy(), journal_status=status)
    text = pr.render(report)
    assert report.invalid and "INVALID" in text and "Nothing below this line is evidence" in text
    assert "## 2. Behaviour" not in text and "## 3." not in text


def test_a_second_policy_hash_in_the_period_invalidates_it() -> None:
    entries = [{"at": "2026-10-05T10:00:00+00:00", "event": "started", "policy_sha256": "a" * 64, "prev": ""},
               {"at": "2026-11-05T10:00:00+00:00", "event": "started", "policy_sha256": "b" * 64, "prev": "x"}]
    report = pr.build_report(entries, policy(), journal_status={"ok": True, "reason": None})
    assert report.invalid and "2 policy hash(es)" in report.invalid


def test_after_a_policy_change_each_period_can_be_reported_on_its_own() -> None:
    entries = [{"at": "2026-10-05T10:00:00+00:00", "event": "started", "policy_sha256": "a" * 64, "prev": ""},
               {"at": "2026-11-05T10:00:00+00:00", "event": "started", "policy_sha256": COMMITTED_SHA, "prev": "x"}]
    status = {"ok": True, "reason": None}
    assert pr.build_report(entries, policy(), journal_status=status).invalid, "both periods together: refused"
    assert pr.build_report(entries, policy(), journal_status=status, since="2026-11-01").invalid is None


def test_a_journal_from_a_policy_other_than_the_committed_one_is_invalid() -> None:
    entries = [{"at": "2026-10-05T10:00:00+00:00", "event": "started", "policy_sha256": "a" * 64, "prev": ""}]
    report = pr.build_report(entries, policy(), journal_status={"ok": True, "reason": None})
    assert report.invalid and "committed policy" in report.invalid


def test_an_empty_journal_is_a_valid_empty_report() -> None:
    report = pr.build_report([], policy(), journal_status={"ok": True, "reason": None})
    text = pr.render(report)
    assert report.invalid is None and "No plan has been proposed yet" in text and "No order has filled yet" in text


# --- execution quality (P3, P4) ---------------------------------------------------------------------------------------------

def fills_of(n: int, side: str = "buy", symbol: str = "VTI") -> list[pr.Fill]:
    """``n`` fills of one share at $100, each on its own day (so each has its own session open)."""
    return [pr.Fill(f"p{i}", date(2026, 10, 5) + timedelta(days=i), symbol, side, D("1"), D("100")) for i in range(n)]


def analyse(fills: list[pr.Fill], per_fill_bps: list[float]) -> pr.ExecutionStats:
    opens = {}
    for f, bps in zip(fills, per_fill_bps, strict=True):
        k = D(str(bps)) / 10000
        opens[(f.symbol, f.session)] = f.price / (1 + (k if f.side == "buy" else -k))
    stats = pr.analyse_execution(fills, opens)
    assert stats is not None
    return stats


def test_below_thirty_fills_the_mean_is_stated_but_never_placed() -> None:
    stats = analyse(fills_of(29), [4.0, 5.0] * 14 + [4.5])
    assert stats.n == 29 and "too few" in stats.placement and "29 fills" in stats.placement


def test_from_thirty_fills_the_mean_is_placed_among_the_registered_levels() -> None:
    # VTI one-way: optimistic 1.7, central 2.4+2 = 4.4, pessimistic 5.9+5 = 10.9 bp
    where = {}
    # (label, mean, noise): the "optimistic" case needs noise wide enough that its interval reaches the level
    for label, centre, noise in (("better", 0.2, 0.05), ("optimistic", 1.6, 1.0), ("optcentral", 3.0, 0.05),
                                 ("centpess", 7.5, 0.05), ("worse", 14.0, 0.05)):
        values = [centre + (noise if i % 2 else -noise) for i in range(40)]
        where[label] = analyse(fills_of(40), values).placement
    assert "better than the optimistic level" in where["better"]
    assert "at or near the optimistic level" in where["optimistic"]
    assert "between the optimistic and central levels" in where["optcentral"]
    assert "between the central and pessimistic levels" in where["centpess"]
    assert "worse than the pessimistic level: investigate" in where["worse"]


def test_the_modelled_cost_is_the_registered_half_spread_plus_slippage() -> None:
    stats = analyse(fills_of(2), [3.0, 3.0])
    assert stats.modelled_bps["optimistic"] == pytest.approx(1.7) and stats.modelled_bps["central"] == pytest.approx(4.4)
    assert stats.modelled_bps["pessimistic"] == pytest.approx(10.9)


def test_a_fill_far_from_the_open_is_listed_as_a_fault() -> None:
    stats = analyse(fills_of(3), [2.0, 3.0, 61.0])
    assert [round(v) for _, v in stats.outliers] == [61]
    assert "more than 25 bp from the open" in pr.render(_report_with(stats))


def test_a_fill_with_no_session_open_is_counted_not_guessed() -> None:
    fills = fills_of(3)
    stats = pr.analyse_execution(fills, {(fills[0].symbol, fills[0].session): D("100")})
    assert stats is not None and stats.n == 1 and stats.unpriced == 2


def test_sells_are_measured_in_the_direction_that_costs() -> None:
    sell = [pr.Fill("p", date(2026, 10, 5), "VTI", "sell", D("1"), D("99.97"))]
    stats = pr.analyse_execution(sell, {("VTI", date(2026, 10, 5)): D("100")})
    assert stats is not None and stats.by_side["sell"][1] == pytest.approx(3.0)


def _report_with(stats: pr.ExecutionStats) -> pr.Report:
    report = pr.Report(datetime(2026, 11, 1, tzinfo=UTC), COMMITTED_SHA, {"ok": True, "reason": None})
    report.execution = stats
    return report


# --- where the portfolio stands ------------------------------------------------------------------------------------------------

def test_standing_shows_each_holdings_drift_and_the_cash_outside_the_mix() -> None:
    pol = policy()
    qty = {s: D(0) for s in pol.symbols} | {"VTI": D("25"), "VXUS": D("19"), "BND": D("19"), "IAU": D("19"),
                                              "VNQ": D("19"), "BIL": D("5")}
    snap = core_paper.Snapshot(date(2026, 12, 31), date(2027, 1, 4), qty, D("100"), dict.fromkeys(pol.symbols, D("100")))
    rows, cash = pr.standing(pol, snap)
    by = {r.symbol: r for r in rows}
    assert by["VTI"].gap > 0 and by["VXUS"].gap < 0
    assert sum(r.now for r in rows) + cash == pytest.approx(D(1))
    report = pr.build_report([], pol, journal_status={"ok": True, "reason": None}, snap=snap)
    assert "Largest drift" in pr.render(report) and "As of the 2026-12-31 close" in pr.render(report)


# --- reading opens from Alpaca ---------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_opens_are_read_from_the_daily_bar_one_request_per_session() -> None:
    class Bars(FakeBroker):
        calls: list[tuple[str, str]] = []

        async def daily_bars(self, symbols: list[str], start: str, end: str) -> dict[str, list[dict[str, Any]]]:
            self.calls.append((start, end))
            return {s: [{"t": f"{start}T04:00:00Z", "o": "101.25", "c": "102"}] for s in symbols}

    broker = Bars()
    fills = [pr.Fill("a", date(2026, 10, 5), "VTI", "buy", D(1), D("101.3")),
             pr.Fill("a", date(2026, 10, 5), "BND", "buy", D(1), D("70")),
             pr.Fill("b", date(2027, 1, 4), "VTI", "sell", D(1), D("101.2"))]
    opens = await pr.fetch_opens(broker, fills)
    assert opens[("VTI", date(2026, 10, 5))] == D("101.25") and len(opens) == 3
    assert broker.calls == [("2026-10-05", "2026-10-05"), ("2027-01-04", "2027-01-04")]


# --- the command line --------------------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_command_reads_the_state_directory_and_writes_a_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                       capsys: pytest.CaptureFixture[str]) -> None:
    from risk_router.core_app import create_core_app
    monkeypatch.setenv("CORE_PLAN_SECRET", "p" * 40)
    path = policy_file(tmp_path)
    monkeypatch.setenv("CORE_POLICY_PATH", str(path))
    broker, clock = FakeBroker(), Clock(START)
    app = create_core_app(alpaca=broker, background=False, state_dir=tmp_path, policy_path=path, now=clock)
    async with app.router.lifespan_context(app):
        router = app.state.router
        await build_portfolio(router, broker, clock)
    out = tmp_path / "report.md"
    code = await pr.main_async(argparse.Namespace(state_dir=str(tmp_path), since=None, offline=True, out=str(out)))
    assert code == 0 and "written to" in capsys.readouterr().out
    text = out.read_text()
    assert "Long-term core: paper period report" in text and "| initial |" in text and "Not read (offline" in text

    code = await pr.main_async(argparse.Namespace(state_dir=str(tmp_path / "nowhere"), since=None, offline=True, out=None))
    assert code == 1 and "no journal" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_the_command_refuses_to_present_a_tampered_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                 capsys: pytest.CaptureFixture[str]) -> None:
    from risk_router.core_app import create_core_app
    monkeypatch.setenv("CORE_PLAN_SECRET", "p" * 40)
    path = policy_file(tmp_path)
    monkeypatch.setenv("CORE_POLICY_PATH", str(path))
    broker, clock = FakeBroker(), Clock(START)
    app = create_core_app(alpaca=broker, background=False, state_dir=tmp_path, policy_path=path, now=clock)
    async with app.router.lifespan_context(app):
        await build_portfolio(app.state.router, broker, clock)
    lines = (tmp_path / "core-journal.jsonl").read_text().splitlines()
    (tmp_path / "core-journal.jsonl").write_text("\n".join(lines[:2] + lines[3:]) + "\n")
    code = await pr.main_async(argparse.Namespace(state_dir=str(tmp_path), since=None, offline=True, out=None))
    assert code == 1 and "INVALID" in capsys.readouterr().out


def test_the_reading_rules_state_what_paper_fills_can_and_cannot_prove() -> None:
    rules = pr.READING_RULES
    assert "written on 2026-10-03, before any paper order existed" in rules
    assert "simulator, not the market" in rules and "30 or more" in rules and "Materiality" in rules
