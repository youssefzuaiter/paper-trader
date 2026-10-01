"""The experiment registry: a row before the work, provenance, exact reproduction."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from backtest import registry
from backtest.cli import main, registered_criteria
from backtest.lockbox import LockBoxKey
from backtest.store import Store

ROOT = Path(__file__).resolve().parents[2]
REAL = (ROOT / ".cache" / "backtest" / "calendar.parquet").exists() and (ROOT / ".cache" / "training").exists()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


def fresh_repo(path: Path) -> Path:
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "test")
    (path / "a.txt").write_text("a")
    git(path, "add", "a.txt")
    git(path, "commit", "-qm", "init")
    return path


def begin(store: Store, repo: Path, **kw):
    params = {"hypothesis": "h", "command": "test", "params": {"x": 1}, "seeds": {"s": 7},
              "inputs": [repo / "a.txt"], "data_root": repo, "repo": repo}
    params.update(kw)
    return registry.begin(store, **params)


def test_a_dirty_tree_is_refused_unless_explicitly_allowed(tmp_path: Path) -> None:
    repo = fresh_repo(tmp_path / "r")
    (repo / "b.txt").write_text("uncommitted")
    with Store() as store:
        with pytest.raises(registry.DirtyTree):
            begin(store, repo)
        exp = begin(store, repo, allow_dirty=True)
        row = store.experiment(exp.experiment_id)
        assert row["git_dirty"] is True and row["status"] == "running"
        assert row["git_commit"] == git(repo, "rev-parse", "HEAD").strip()


def test_the_row_exists_before_the_work_and_closes_once(tmp_path: Path) -> None:
    repo = fresh_repo(tmp_path / "r")
    with Store() as store:
        with pytest.raises(RuntimeError), begin(store, repo) as exp:
            assert store.experiment(exp.experiment_id)["status"] == "running"
            raise RuntimeError("boom")
        failed = store.experiment(exp.experiment_id)
        assert failed["status"] == "failed" and "boom" in failed["conclusion"]

        with begin(store, repo) as exp:
            report_hash = exp.finish({"auc": 0.5}, report_body="body", conclusion="ok")
        done = store.experiment(exp.experiment_id)
        assert done["status"] == "done" and done["report_hash"] == report_hash == hashlib.sha256(b"body").hexdigest()
        env = json.loads(done["environment"])
        assert {"python", "numpy", "scikit-learn", "duckdb", "OMP_NUM_THREADS"} <= set(env)
        assert json.loads(done["seeds"]) == {"s": 7}


def test_a_one_line_hypothesis_is_required(tmp_path: Path) -> None:
    repo = fresh_repo(tmp_path / "r")
    with Store() as store:
        for bad in ("", "  ", "two\nlines"):
            with pytest.raises(ValueError):
                begin(store, repo, hypothesis=bad)


def test_the_manifest_catches_a_changed_input(tmp_path: Path) -> None:
    repo = fresh_repo(tmp_path / "r")
    recorded = registry.manifest([repo / "a.txt"], repo)
    registry.verify(recorded, repo)
    (repo / "a.txt").write_text("b")
    with pytest.raises(registry.ManifestMismatch):
        registry.verify(recorded, repo)
    assert registry.manifest([repo / "a.txt"], repo)["hash"] != recorded["hash"]


def test_configuration_n_of_m_counts_the_same_command_on_the_same_window(tmp_path: Path) -> None:
    repo = fresh_repo(tmp_path / "r")
    window = (date(2025, 1, 1), date(2026, 9, 1))
    with Store() as store:
        ids = []
        for _ in range(3):
            with begin(store, repo, window=window) as exp:
                exp.finish({}, report_body=None, conclusion="c")
                ids.append(exp.experiment_id)
        with begin(store, repo, window=(date(2024, 1, 1), date(2024, 12, 1))) as other:
            other.finish({}, report_body=None, conclusion="c")
        # Counted as of each run, so a report's "N of M" never changes after it is written.
        assert [registry.configuration(store, i) for i in ids] == [(1, 1), (2, 2), (3, 3)]
        # A reproduction is not a configuration: it reports its parent's count and adds none.
        with begin(store, repo, window=window, parent_id=ids[1]) as rerun:
            rerun.finish({}, report_body=None, conclusion="c")
        assert registry.configuration(store, rerun.experiment_id) == (2, 2)
        with begin(store, repo, window=window) as later:
            later.finish({}, report_body=None, conclusion="c")
        assert registry.configuration(store, later.experiment_id) == (4, 4)
        assert registry.configuration(store, ids[2]) == (3, 3)


def test_a_strategy_run_cites_the_walkforward_its_params_name(tmp_path: Path) -> None:
    """A reproduction re-runs the original's params, so it must cite the same walk-forward, not
    whichever is newest (the walk-forward's own reproduction, for instance)."""
    from backtest.runs import cited_walkforward

    repo = fresh_repo(tmp_path / "r")
    with Store() as store:
        with begin(store, repo, command="walkforward") as first:
            first.finish({}, report_body=None, conclusion="c")
        with begin(store, repo, command="walkforward", parent_id=first.experiment_id) as reproduction:
            reproduction.finish({}, report_body=None, conclusion="c")
        assert cited_walkforward(store)["experiment_id"] == first.experiment_id  # reproductions are skipped
        assert cited_walkforward(store, first.experiment_id)["experiment_id"] == first.experiment_id


def test_the_lockbox_opens_only_inside_a_registered_lockbox_experiment(tmp_path: Path) -> None:
    repo = fresh_repo(tmp_path / "r")
    with Store() as store:
        plain = begin(store, repo)
        with pytest.raises(PermissionError):
            registry.open_lockbox(store, plain)
        lockbox = begin(store, repo, lockbox=True)
        assert isinstance(registry.open_lockbox(store, lockbox), LockBoxKey)
        lockbox.finish({}, report_body=None, conclusion="c")
        with pytest.raises(PermissionError):
            registry.open_lockbox(store, lockbox)
        plain.fail("unused")


def test_criteria_must_be_registered_before_strategy_results(tmp_path: Path) -> None:
    store_path = tmp_path / "s.duckdb"
    with Store(store_path) as store, pytest.raises(PermissionError):
        registered_criteria(store)
    main(["register-criteria", "--order-notional", "1000", "--lockbox-criteria", "AUC interval above 0.5",
          "--allow-dirty", "--store", str(store_path)])
    with Store(store_path) as store:
        criteria = registered_criteria(store)
    assert criteria["order_notional_usd"] == "1000" and "optimistic level fails" in criteria["pass_rule"]


# --- exact reproduction, end to end ---------------------------------------------------------------------

def _copy_code(dest: Path) -> Path:
    """This repository's code (tracked and new files, nothing ignored) as a fresh one-commit repo."""
    files = git(ROOT, "ls-files").splitlines() + git(ROOT, "ls-files", "--others", "--exclude-standard").splitlines()
    for name in files:
        if (ROOT / name).is_file():
            (dest / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest / name)
    git(dest, "init", "-q")
    git(dest, "config", "user.email", "test@example.com")
    git(dest, "config", "user.name", "test")
    git(dest, "add", "-A")
    git(dest, "commit", "-qm", "snapshot")
    return dest


@pytest.mark.skipif(not REAL, reason="needs the fetched caches under .cache/")
def test_reproduce_reruns_the_commit_and_matches_bit_for_bit(tmp_path: Path) -> None:
    repo = _copy_code(tmp_path / "repo")
    store_path = tmp_path / "store.duckdb"
    env = {**os.environ, "PYTHONPATH": str(repo)}
    common = ["--data-root", str(ROOT), "--store", str(store_path)]
    subprocess.run([sys.executable, "-m", "backtest", "legacy", *common], cwd=repo, env=env, check=True,
                   capture_output=True, text=True)
    with Store(store_path) as store:
        (original,) = store.experiments(command="legacy")
    assert original["status"] == "done" and not original["git_dirty"]
    assert (tmp_path / "reports" / f"{original['experiment_id']}-legacy-reproduction.md").exists()
    assert not (ROOT / ".cache" / "backtest" / "reports" / f"{original['experiment_id']}-legacy-reproduction.md").exists()

    out = subprocess.run([sys.executable, "-m", "backtest", "reproduce", original["experiment_id"], *common],
                         cwd=repo, env=env, check=True, capture_output=True, text=True).stdout
    result = json.loads(out)
    assert result["metrics_identical"] and result["report_hash_identical"]
    with Store(store_path) as store:
        rerun = store.experiment(result["rerun"])
    assert rerun["parent_id"] == original["experiment_id"] and rerun["git_commit"] == original["git_commit"]
    assert "worktree" not in git(repo, "worktree", "list").split("\n", 1)[-1]  # cleaned up


def test_an_experiment_run_on_a_dirty_tree_cannot_be_reproduced(tmp_path: Path) -> None:
    repo = fresh_repo(tmp_path / "r")
    (repo / "b.txt").write_text("uncommitted")
    store_path = tmp_path / "s.duckdb"
    with Store(store_path) as store:
        exp = begin(store, repo, allow_dirty=True)
        exp.finish({}, report_body="b", conclusion="c")
    with pytest.raises(registry.DirtyTree):
        registry.reproduce(store_path, exp.experiment_id, repo=repo)


def test_a_code_file_keeps_its_path_when_data_lives_elsewhere(tmp_path: Path) -> None:
    """``reproduce`` runs worktree code against the original data root. ``fees.json`` ships with the
    code, so it sits outside that root; it must still be recorded as ``backtest/fees.json``."""
    from backtest.registry import ROOT, manifest

    data = tmp_path / "data.txt"
    data.write_text("x", encoding="utf-8")
    fees = ROOT / "backtest" / "fees.json"
    recorded = manifest([data, fees], tmp_path)
    assert [f["path"] for f in recorded["files"]] == sorted(["data.txt", "backtest/fees.json"])
    with pytest.raises(ValueError):
        manifest([Path("/etc/hosts")], tmp_path)


def test_a_later_commit_editing_a_code_input_does_not_break_an_older_reproduction(tmp_path: Path) -> None:
    """Design O16: extending ``fees.json`` in a later commit must leave earlier experiments
    reproducible. Code-shipped inputs are checked in the recorded commit's worktree; data inputs
    are still checked against the data root, so a changed data file is still caught."""
    repo = fresh_repo(tmp_path / "r")
    (repo / ".gitignore").write_text("data/\n")
    (repo / "fees.json").write_text('{"v": 1}')
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "fees v1")
    (repo / "data").mkdir()
    (repo / "data" / "bars.txt").write_text("bars")
    recorded = registry.manifest([repo / "fees.json", repo / "data" / "bars.txt"], repo)
    commit = git(repo, "rev-parse", "HEAD").strip()

    (repo / "fees.json").write_text('{"v": 2}')  # a later commit adds fee rows
    git(repo, "commit", "-qam", "fees v2")
    with pytest.raises(registry.ManifestMismatch):  # the old check: against the current tree
        registry.verify(recorded, repo)

    worktree = tmp_path / "tree"
    git(repo, "worktree", "add", "--detach", str(worktree), commit)
    tracked = frozenset(git(worktree, "ls-files").splitlines())
    assert "fees.json" in tracked and "data/bars.txt" not in tracked
    registry.verify(recorded, repo, code_root=worktree, code_files=tracked)

    (repo / "data" / "bars.txt").write_text("changed bars")
    with pytest.raises(registry.ManifestMismatch, match="data/bars.txt"):
        registry.verify(recorded, repo, code_root=worktree, code_files=tracked)
