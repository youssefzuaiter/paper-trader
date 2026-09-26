"""News articles as point-in-time events, and duplicate stories (design §2 P1, §5, §7).

An article is usable from ``known_at``: its ``created_at`` in the backfill,
the receipt time when the live ingester writes it (later). The headline we
hold may be a revision (the news API filters on ``updated_at``, §11), so
``updated_at`` is kept for the revision diagnostic.

**Stories.** Events sharing a symbol whose normalised headlines (lower-case,
punctuation and digits stripped) have token-set Jaccard >= 0.7 within six
hours are one story. Linkage is single (a chain of near-duplicates is one
story); the story id is its earliest event. Collapsed runs keep each
story's first event.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, Final

FEED: Final[str] = "alpaca"
STORY_WINDOW: Final[timedelta] = timedelta(hours=6)
STORY_JACCARD: Final[float] = 0.7
DEDUP_VERSION: Final[str] = "jaccard0.70-6h-single-v1"
_NOT_LETTERS: Final[re.Pattern[str]] = re.compile(r"[^a-z\s]+")


@dataclass(frozen=True, slots=True)
class Event:
    event_id: str
    source_id: str
    publisher: str
    published_at: datetime
    updated_at: datetime | None
    known_at: datetime
    headline: str
    symbols: tuple[str, ...]  # every ticker the article names
    n_symbols: int
    story_id: str | None = None


def headline_hash(headline: str) -> str:
    return hashlib.sha256(headline.strip().encode()).hexdigest()


def normalise(headline: str) -> frozenset[str]:
    return frozenset(_NOT_LETTERS.sub(" ", headline.lower()).split())


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def from_articles(fetched: Sequence[Mapping[str, Any]], cached: Sequence[Mapping[str, Any]]) -> tuple[
        list[Event], dict[str, int]]:
    """Development events: every fetched article, plus cached ones the
    re-fetch no longer returns. A cached article keeps its cached headline
    (the text report.md and the FinBERT cache used); the counts say how
    often the re-fetch disagreed."""
    counts = {"fetched": len(fetched), "cached": len(cached), "cached_only": 0, "fetched_only": 0,
              "headline_changed": 0}
    by_id = {str(a["id"]): a for a in fetched}
    events: dict[str, Event] = {}
    for a in cached:
        sid = str(a["id"])
        live = by_id.get(sid)
        if live is None:
            counts["cached_only"] += 1
        elif (live["headline"] or "").strip() != (a.get("headline") or "").strip():
            counts["headline_changed"] += 1
        created = datetime.fromisoformat(a["created_at"])
        events[sid] = Event(
            event_id=f"{FEED}:{sid}", source_id=sid, publisher=a.get("source") or "", published_at=created,
            updated_at=live["updated_at"] if live is not None else None, known_at=created,
            headline=(a.get("headline") or "").strip(), symbols=tuple(a.get("symbols") or ()),
            n_symbols=len(a.get("symbols") or ()))
    for sid, a in by_id.items():
        if sid in events:
            continue
        counts["fetched_only"] += 1
        events[sid] = Event(
            event_id=f"{FEED}:{sid}", source_id=sid, publisher=a.get("source") or "", published_at=a["created_at"],
            updated_at=a["updated_at"], known_at=a["created_at"], headline=(a["headline"] or "").strip(),
            symbols=tuple(s for s in (a["symbols"] or "").split(",") if s),
            n_symbols=len([s for s in (a["symbols"] or "").split(",") if s]))
    ordered = sorted(events.values(), key=lambda e: (e.known_at, e.event_id))
    return ordered, counts


def cluster(events: Sequence[Event], watch: frozenset[str], *, window: timedelta = STORY_WINDOW,
            threshold: float = STORY_JACCARD) -> dict[str, str]:
    """``event_id -> story_id`` for every event naming a watched symbol."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    order = {e.event_id: i for i, e in enumerate(sorted(events, key=lambda e: (e.known_at, e.event_id)))}

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:  # the earlier event stays the root, so it names the story
            if order[ra] < order[rb]:
                parent[rb] = ra
            else:
                parent[ra] = rb

    recent: dict[str, deque[tuple[datetime, str, frozenset[str]]]] = defaultdict(deque)
    for e in sorted(events, key=lambda e: (e.known_at, e.event_id)):
        mine = [s for s in e.symbols if s in watch]
        if not mine:
            continue
        parent[e.event_id] = e.event_id
        tokens = normalise(e.headline)
        for symbol in mine:
            queue = recent[symbol]
            while queue and queue[0][0] < e.known_at - window:
                queue.popleft()
            for _, other, other_tokens in queue:
                if jaccard(tokens, other_tokens) >= threshold:
                    union(e.event_id, other)
            queue.append((e.known_at, e.event_id, tokens))
    return {eid: "story:" + find(eid).split(":", 1)[1] for eid in parent}


def with_stories(events: Sequence[Event], stories: Mapping[str, str]) -> list[Event]:
    return [replace(e, story_id=stories.get(e.event_id)) for e in events]


def store_rows(events: Iterable[Event], *, mode: str = "backfill") -> tuple[list[dict[str, Any]],
                                                                          list[dict[str, Any]]]:
    rows, symbols = [], []
    for e in events:
        rows.append({
            "event_id": e.event_id, "feed": FEED, "source_id": e.source_id, "publisher": e.publisher,
            "published_at": e.published_at, "updated_at": e.updated_at, "known_at": e.known_at,
            "known_at_basis": "published" if mode == "backfill" else "received", "headline": e.headline,
            "headline_hash": headline_hash(e.headline), "n_symbols": e.n_symbols, "story_id": e.story_id,
            "dedup_version": DEDUP_VERSION if e.story_id else None, "mode": mode,
        })
        symbols.extend({"event_id": e.event_id, "symbol": s} for s in dict.fromkeys(e.symbols))
    return rows, symbols


def from_store(rows: Sequence[Mapping[str, Any]], symbol_rows: Sequence[Mapping[str, Any]]) -> list[Event]:
    by_event: dict[str, list[str]] = defaultdict(list)
    for r in symbol_rows:
        by_event[r["event_id"]].append(r["symbol"])
    return [Event(event_id=r["event_id"], source_id=r["source_id"], publisher=r["publisher"] or "",
                  published_at=r["published_at"], updated_at=r["updated_at"], known_at=r["known_at"],
                  headline=r["headline"], symbols=tuple(by_event[r["event_id"]]), n_symbols=r["n_symbols"],
                  story_id=r["story_id"])
            for r in rows]
