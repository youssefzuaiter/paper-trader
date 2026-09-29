"""Where development data ends and the lock-box begins (design §6, rule P7).

Development data is every event known before 2026-09-18 00:00 UTC (the
end of the cached news fetch, 20:00 New York on 09-17) and every session
up to and including 2026-09-18, the embargo session in which 09-17's
overnight labels resolve. The lock-box holds events known from 2026-09-18
16:00 New York and sessions from Monday 2026-09-21.

Nothing on the lock-box side is readable without a ``LockBoxKey``, and the
only way to get one is ``registry.open_lockbox``, which records the opening
against a registered experiment. ``MarketData`` and ``Store`` both check.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Final

from swarm.common import NEW_YORK

#: Development events are known strictly before this instant.
DEV_EVENTS_END: Final[datetime] = datetime(2026, 9, 18, 0, 0, tzinfo=UTC)
#: The session between development and lock-box: dev labels resolve in it.
EMBARGO_SESSION: Final[date] = date(2026, 9, 18)
#: Lock-box events are known at or after this instant.
LOCKBOX_EVENTS_FROM: Final[datetime] = datetime(2026, 9, 18, 16, 0, tzinfo=NEW_YORK)
#: Lock-box sessions start on this date.
LOCKBOX_SESSIONS_FROM: Final[date] = date(2026, 9, 21)

_ISSUER = object()


class LockBoxError(PermissionError):
    """Lock-box data was requested outside a registered lock-box run."""


class LockBoxKey:
    """Proof that a lock-box experiment was registered before the data was read.

    Constructed only by ``registry.open_lockbox``; the private issuer
    argument makes an accidental ``LockBoxKey(...)`` elsewhere fail loudly.
    """

    __slots__ = ("experiment_id",)

    def __init__(self, experiment_id: str, *, issuer: object) -> None:
        if issuer is not _ISSUER:
            raise LockBoxError("a LockBoxKey is issued only by registry.open_lockbox")
        self.experiment_id = experiment_id


def issue_key(experiment_id: str) -> LockBoxKey:
    """For ``registry.open_lockbox`` only."""
    return LockBoxKey(experiment_id, issuer=_ISSUER)


def event_readable(known_at: datetime, key: LockBoxKey | None) -> bool:
    return key is not None or known_at < DEV_EVENTS_END


def session_readable(session: date, key: LockBoxKey | None) -> bool:
    return key is not None or session < LOCKBOX_SESSIONS_FROM
