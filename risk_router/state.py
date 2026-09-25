"""The router's two pieces of state that must survive a pod restart.

Everything else the router decides on — positions, working orders, how
many buys went out today, when a symbol was last sold — is read from
Alpaca, the source of truth, on every decision. Two facts are not
Alpaca's to know:

* **The kill switch.** In the monolith an emergency halt lived in memory
  and "a deliberate restart" cleared it. Under Kubernetes a restart is not
  deliberate — a liveness failure, a node drain or an OOM kill restarts
  the pod on its own — so a memory-only halt would silently lift itself.
  It is persisted here, and clearing it is an operator action (delete the
  file, then restart the pod).
* **The breaker latch.** Once the daily circuit breaker trips it stays
  tripped for the rest of that New York trading day, even if equity
  recovers intraday or the pod restarts.

A small JSON file on the StatefulSet's volume, rewritten atomically.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

logger = logging.getLogger("risk_router.state")


@dataclass
class _Snapshot:
    halted_at: str | None = None
    halt_reason: str | None = None
    breaker_tripped_on: str | None = None  # ISO date, America/New_York
    breaker_detail: str | None = None


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> _Snapshot:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return _Snapshot()
        except (OSError, ValueError) as exc:
            # Unreadable state must not read as "not halted". Fail closed:
            # treat a corrupt file as a halt the operator has to clear.
            logger.critical("Router state %s is unreadable (%s) — starting HALTED", self.path, exc)
            return _Snapshot(halted_at="unknown", halt_reason=f"state file unreadable: {exc}")
        return _Snapshot(**{k: raw.get(k) for k in _Snapshot.__dataclass_fields__})

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".state-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(asdict(self._data), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)

    # --- kill switch -------------------------------------------------------

    @property
    def halted(self) -> bool:
        return self._data.halted_at is not None

    @property
    def halt_reason(self) -> str | None:
        return self._data.halt_reason

    def set_halted(self, at_iso: str, reason: str) -> None:
        with self._lock:
            self._data.halted_at = at_iso
            self._data.halt_reason = reason
            self._save()

    # --- breaker latch -----------------------------------------------------

    def breaker_tripped_on(self) -> date | None:
        value = self._data.breaker_tripped_on
        return date.fromisoformat(value) if value else None

    @property
    def breaker_detail(self) -> str | None:
        return self._data.breaker_detail

    def trip_breaker(self, trading_day: date, detail: str) -> None:
        with self._lock:
            self._data.breaker_tripped_on = trading_day.isoformat()
            self._data.breaker_detail = detail
            self._save()
