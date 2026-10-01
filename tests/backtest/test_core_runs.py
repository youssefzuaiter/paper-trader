"""register-core's guard: once a run-core result exists, a new registration must name the one it supersedes."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from backtest import core_runs
from backtest.core_grid import registration_params
from backtest.store import Store


def _row(store: Store, experiment_id: str, command: str, params: dict) -> None:
    store.register({"experiment_id": experiment_id, "created_at": datetime.now(UTC), "hypothesis": "h",
                    "command": command, "parent_id": None, "status": "running", "window_start": None,
                    "window_end": None, "lockbox": False, "git_commit": "0" * 40, "git_dirty": False,
                    "data_hash": "-", "manifest": {}, "params": params, "seeds": {}, "environment": {}})
    store.finish(experiment_id, status="done", metrics={}, report_hash=None, conclusion="c",
                 finished_at=datetime.now(UTC))


def test_a_new_registration_after_results_must_name_what_it_supersedes(tmp_path: Path) -> None:
    def noop_report(*_args, **_kw) -> Path:
        return tmp_path / "r.md"

    params = {"root": str(tmp_path), "registration": registration_params(), "supersedes": None}
    with Store() as store:
        _row(store, "x-reg", core_runs.REGISTRATION, {"registration": registration_params()})
        _row(store, "x-run", core_runs.RUN, {"registration": "x-reg"})
        with pytest.raises(PermissionError, match="supersedes"):
            core_runs.cmd_register_core(params, store, allow_dirty=True, parent_id=None, write_report=noop_report)
        with pytest.raises(ValueError, match="not a completed core registration"):
            core_runs.cmd_register_core({**params, "supersedes": "x-run"}, store, allow_dirty=True, parent_id=None,
                                        write_report=noop_report)


def test_run_core_refuses_a_registration_whose_grid_changed() -> None:
    with Store() as store:
        changed = {**registration_params(), "sizes_usd": ["100000", "5000"]}
        _row(store, "x-reg", core_runs.REGISTRATION, {"registration": changed})
        with pytest.raises(PermissionError, match="differs from the registration"):
            core_runs.registered(store, "x-reg")
        _row(store, "x-ok", core_runs.REGISTRATION, {"registration": registration_params()})
        assert core_runs.registered(store, "x-ok")["sizes_usd"] == ["100000", "10000"]
        with pytest.raises(PermissionError, match="no passing leakage-core"):
            core_runs.passing_leakage(store, "0" * 40)


def test_registration_writes_its_report_with_decimal_parameters(tmp_path: Path) -> None:
    """The grid's cost levels carry Decimals; the registration's report must serialise them."""
    written = {}

    def capture(_root, _exp, _name, body, _store) -> Path:
        written["body"] = body
        return tmp_path / "r.md"

    params = {"root": str(tmp_path), "registration": registration_params(), "supersedes": None}
    with Store() as store:
        out = core_runs.cmd_register_core(params, store, allow_dirty=True, parent_id=None, write_report=capture)
        assert store.experiment(out["experiment"])["status"] == "done"
    assert '"half_spread_bps"' in written["body"] and "History warning" in written["body"]
