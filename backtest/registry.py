"""The experiment registry (design §8): nothing produces a number without a row first.

``begin`` writes an ``experiment`` row *before* any computation: a required
one-line hypothesis, the git commit (a dirty tree is refused unless
``allow_dirty``, and such runs cannot be cited or reproduced), a manifest of
every input file (path, size, sha256) and its combined hash, the fully
resolved parameters, every seed, and the environment. The run then finishes
with metrics, a report hash and a conclusion, or is marked failed. Rows are
never deleted, only superseded, so a report can say "configuration N of M
tried on this window".

A report is a provenance header (ids, times, configuration counts) above a
body; the **body** is hashed, so a faithful reproduction has the same hash.

``reproduce`` checks the recorded commit out into a temporary worktree,
verifies the manifest, re-runs the command with the recorded parameters and
seeds, and requires identical metrics and report hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Self

from backtest.lockbox import LockBoxKey, issue_key
from backtest.store import Store

ROOT: Final[Path] = Path(__file__).resolve().parent.parent


class DirtyTree(RuntimeError):
    """Uncommitted changes: the run could not be reproduced from its commit."""


class ManifestMismatch(RuntimeError):
    """An input file changed since the experiment recorded it."""


class NotReproduced(RuntimeError):
    pass


# --- provenance ------------------------------------------------------------------------------

def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def git_state(repo: Path = ROOT) -> tuple[str, bool]:
    """``(commit, dirty)``: dirty means tracked changes or untracked files outside ignored paths."""
    commit = git(repo, "rev-parse", "HEAD")
    return commit, bool(git(repo, "status", "--porcelain"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _input_label(path: Path, root: Path) -> str:
    """``path`` relative to the data ``root``, or else to the code tree (``ROOT``).

    A reproduction runs committed code from a temporary worktree against the original data root,
    so an input that ships with the code (``backtest/fees.json``) lives outside the data root.
    Labelling it against the code tree gives it the same path it had in the original run, where
    code and data shared one repository.
    """
    for base in (root.resolve(), ROOT):
        if path.is_relative_to(base):
            return str(path.relative_to(base))
    raise ValueError(f"{path} is in neither the data root {root} nor the code tree {ROOT}")


def manifest(paths: Sequence[Path], root: Path) -> dict[str, Any]:
    """Every input file relative to ``root`` (size, sha256) and one hash over all of them."""
    files = []
    for path in sorted({p.resolve() for p in paths}):
        files.append({"path": _input_label(path, root), "size": path.stat().st_size,
                      "sha256": sha256_file(path)})
    combined = hashlib.sha256("\n".join(f"{f['path']}:{f['size']}:{f['sha256']}" for f in files).encode())
    return {"root": str(root.resolve()), "files": files, "hash": combined.hexdigest()}


def verify(recorded: Mapping[str, Any], root: Path) -> None:
    for f in recorded["files"]:
        path = root / f["path"]
        if not path.exists():
            raise ManifestMismatch(f"{f['path']} is missing")
        if path.stat().st_size != f["size"] or sha256_file(path) != f["sha256"]:
            raise ManifestMismatch(f"{f['path']} changed since the experiment recorded it")


def code_hash(repo: Path = ROOT) -> str:
    """One hash of every tracked and untracked (not ignored) file as it is on disk:
    it identifies a dirty tree's exact code, and equals a clean re-run's after commit."""
    names = sorted(set(git(repo, "ls-files").splitlines())
                   | set(git(repo, "ls-files", "--others", "--exclude-standard").splitlines()))
    h = hashlib.sha256()
    for name in names:
        path = repo / name
        if path.is_file():
            h.update(f"{name}\0{sha256_file(path)}\n".encode())
    return h.hexdigest()


def environment() -> dict[str, Any]:
    import duckdb
    import joblib
    import numpy
    import scipy
    import sklearn

    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": numpy.__version__,
            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "duckdb": duckdb.__version__,
            "joblib": joblib.__version__, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}


# --- experiments --------------------------------------------------------------------------------

@dataclass
class Experiment:
    store: Store
    experiment_id: str
    command: str
    params: dict[str, Any]
    finished: bool = False

    def finish(self, metrics: Mapping[str, Any], *, report_body: str | None, conclusion: str) -> str | None:
        report_hash = hashlib.sha256(report_body.encode()).hexdigest() if report_body is not None else None
        self.store.finish(self.experiment_id, status="done", metrics=metrics, report_hash=report_hash,
                          conclusion=conclusion, finished_at=datetime.now(UTC))
        self.finished = True
        return report_hash

    def fail(self, reason: str) -> None:
        self.store.finish(self.experiment_id, status="failed", metrics=None, report_hash=None,
                          conclusion=reason[:2000], finished_at=datetime.now(UTC))
        self.finished = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, kind: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        if exc is not None and not self.finished:
            self.fail(f"{kind.__name__}: {exc}")
        elif not self.finished:
            self.fail("finished without metrics")


def begin(store: Store, *, hypothesis: str, command: str, params: Mapping[str, Any], seeds: Mapping[str, Any],
          inputs: Sequence[Path], data_root: Path, window: tuple[date, date] | None = None, lockbox: bool = False,
          parent_id: str | None = None, allow_dirty: bool = False, repo: Path = ROOT) -> Experiment:
    """Register an experiment before computing anything."""
    hypothesis = hypothesis.strip()
    if not hypothesis or "\n" in hypothesis:
        raise ValueError("an experiment needs a one-line hypothesis")
    commit, dirty = git_state(repo)
    if dirty and not allow_dirty:
        raise DirtyTree("uncommitted changes: commit first, or pass --allow-dirty (such a run cannot be cited)")
    files = manifest(inputs, data_root)
    experiment_id = f"x-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    store.register({
        "experiment_id": experiment_id, "created_at": datetime.now(UTC), "hypothesis": hypothesis,
        "command": command, "parent_id": parent_id, "status": "running", "window_start": window[0] if window else None,
        "window_end": window[1] if window else None, "lockbox": lockbox, "git_commit": commit, "git_dirty": dirty,
        "data_hash": files["hash"], "manifest": files, "params": dict(params), "seeds": dict(seeds),
        "environment": {**environment(), "code_hash": code_hash(repo)},
    })
    return Experiment(store, experiment_id, command, dict(params))


def open_lockbox(store: Store, experiment: Experiment) -> LockBoxKey:
    """The only way to read lock-box data: a registered lock-box experiment, counted."""
    row = store.experiment(experiment.experiment_id)
    if not row["lockbox"] or row["status"] != "running":
        raise PermissionError("the lock-box opens only inside a running experiment registered with lockbox=True")
    return issue_key(experiment.experiment_id)


def configuration(store: Store, experiment_id: str) -> tuple[int, int]:
    """``(N, M)``: configuration N of the M experiments run with the same command on the same window,
    counted as of the original run. Reproductions are not configurations: one reports its parent's
    count, and none is counted, so neither a reproduction nor a later run changes an earlier report."""
    row = store.experiment(experiment_id)
    anchor = store.experiment(row["parent_id"]) if row["parent_id"] else row
    same = [r for r in store.experiments(command=anchor["command"])
            if (r["window_start"], r["window_end"]) == (anchor["window_start"], anchor["window_end"])
            and r["parent_id"] is None and r["created_at"] <= anchor["created_at"]]
    return [r["experiment_id"] for r in same].index(anchor["experiment_id"]) + 1, len(same)


# --- reproduction ----------------------------------------------------------------------------------

def reproduce(store_path: Path, experiment_id: str, *, repo: Path = ROOT, python: str = sys.executable,
              timeout: int = 3600) -> dict[str, Any]:
    """Re-run ``experiment_id`` from its commit in a temporary worktree; require
    identical metrics and report hash. Returns both experiments' ids and hashes."""
    with Store(store_path) as store:
        original = store.experiment(experiment_id)
    if original["git_dirty"]:
        raise DirtyTree(f"{experiment_id} ran on a dirty tree: there is no commit to reproduce it from")
    if original["status"] != "done":
        raise NotReproduced(f"{experiment_id} is {original['status']}, not done")
    recorded = json.loads(original["manifest"])
    verify(recorded, Path(recorded["root"]))
    with tempfile.TemporaryDirectory(prefix="backtest-reproduce-") as tmp:
        worktree = Path(tmp) / "tree"
        git(repo, "worktree", "add", "--detach", str(worktree), original["git_commit"])
        try:
            env = {**os.environ, "PYTHONPATH": str(worktree),
                   "OMP_NUM_THREADS": json.loads(original["environment"]).get("OMP_NUM_THREADS") or "8"}
            args = [python, "-m", "backtest", "replay-experiment", experiment_id, "--store", str(store_path.resolve())]
            done = subprocess.run(args, cwd=worktree, env=env, capture_output=True, text=True, timeout=timeout,
                                  check=False)
            if done.returncode != 0:
                raise NotReproduced(f"the re-run failed:\n{done.stdout[-2000:]}\n{done.stderr[-4000:]}")
        finally:
            git(repo, "worktree", "remove", "--force", str(worktree))
    with Store(store_path) as store:
        children = [r for r in store.experiments(command=original["command"]) if r["parent_id"] == experiment_id]
        rerun = children[-1]
    same_metrics = rerun["metrics"] == original["metrics"]
    same_report = rerun["report_hash"] == original["report_hash"]
    result = {"original": experiment_id, "rerun": rerun["experiment_id"], "metrics_identical": same_metrics,
              "report_hash_identical": same_report, "report_hash": rerun["report_hash"]}
    if not (same_metrics and same_report):
        raise NotReproduced(json.dumps(result))
    return result
