"""Strategies (design §5). S2, S3 and the baselines B1-B3 trade through the
production router; S0 and S1 cannot (eight positions breach the $50 gross
cap) and run as a daily simulation of market orders at the next adjusted open.

Each router strategy builds a session's signals from what was known at the
moment it emits them: stored inputs (features as of ``made_at``), the
walk-forward fold valid at that moment, and a ``PitView`` for the dollar
ATR. Signals carry what the live path would carry:

* ``predicted_move_pct``: the fold's move table through the production
  ``ReturnModel.expected_move``;
* ``atr``: the production ``quant.atr_usd`` on adjusted bars, converted to
  raw dollars with the last session's factor (orders are sized in the
  dollars that trade, design §2);
* ``created_at``: S3 the article's (as the Inference Agent sets it), S2 the
  scheduler's own time (its signals are created at the open).

Baselines send gate-passing signals (probability 1 - rank, a positive
nominal move), so only the router's limits, sizing and the fills bind.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from typing import Final

import numpy as np

from backtest import costs
from backtest.calendar import Calendar, Session
from backtest.costs import CostLevel, FeeTable
from backtest.data import MarketData
from backtest.dataset import Samples
from backtest.engine import PendingSignal
from backtest.walkforward import FoldResult, _first_per_story, fold_for, serving_models
from risk_router.schemas import TradeSignal
from swarm.common import NEW_YORK
from swarm.features import MIN_DAILY_BARS, daily_features
from swarm.quant import atr_usd

SCHEDULER_OFFSET: Final[timedelta] = timedelta(minutes=1)
NO_GATE: Final[Decimal] = Decimal(0)
CLOSED: Final[Decimal] = Decimal(1)       # a fold that does not trade: nothing can pass
NOMINAL_MOVE_PCT: Final[float] = 1.0      # baselines: any positive edge passes the router's rule


def raw_atr(market: MarketData, symbol: str, at: datetime) -> Decimal | None:
    """The ATR a signal would carry at ``at``, in raw dollars."""
    history = market.as_of(at).daily(symbol)
    if len(history) < MIN_DAILY_BARS:
        return None
    adjusted = atr_usd(daily_features(history), history[-1])
    factor = market.factor(symbol, history[-1].t.astimezone(NEW_YORK).date())
    value = (Decimal(str(adjusted)) / Decimal(repr(factor))).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
    return value if value > 0 else None


def _scheduler_time(session: Session) -> datetime:
    return session.open_at + SCHEDULER_OFFSET


def _signal(signal_id: str, symbol: str, source: str, model: str, prob: float, move: float, atr: Decimal | None,
            created_at: datetime, headline: str | None = None) -> TradeSignal:
    return TradeSignal(signal_id=signal_id, symbol=symbol, source=source, model_name=model[:128],
                       prob_up=min(max(prob, 0.0), 1.0), predicted_move_pct=move,
                       confidence=min(max(prob, 0.0), 1.0), headline=headline, atr=atr, created_at=created_at)


# --- S2: overnight news, next open to that close ----------------------------------------------------

@dataclass
class S2:
    """At open + 1 minute a pre-open scheduler re-scores each symbol's night
    (events known between the previous close and this open) with the fold
    valid then; score = mean ``prob_up`` (D7); one signal per symbol at or
    above the fold's S2 threshold, best first."""

    samples: Samples
    folds: Sequence[FoldResult]
    collapsed: bool = False
    name: str = "S2"

    def __post_init__(self) -> None:
        self._key = "S2_collapsed" if self.collapsed else "S2"
        night = np.flatnonzero(self.samples.night)
        if self.collapsed:
            night = _first_per_story(self.samples, night)
        self._by_session: dict[int, np.ndarray] = {}
        for session in np.unique(self.samples.session[night]):
            self._by_session[int(session)] = night[self.samples.session[night] == session]

    def _fold(self, session: Session) -> FoldResult | None:
        return fold_for(list(self.folds), _scheduler_time(session))

    def min_prob_up(self, session: Session) -> Decimal:
        fold = self._fold(session)
        threshold = fold.thresholds.get(self._key) if fold else None
        return CLOSED if threshold is None else Decimal(str(threshold))

    def scores(self, session: Session, calendar: Calendar) -> list[tuple[str, float, list[int]]]:
        """``(symbol, score, sample indices)`` for the session's symbol-nights, best first."""
        fold = self._fold(session)
        idx = self._by_session.get(calendar.index[session.date])
        if fold is None or idx is None:
            return []
        prob = fold.model.predict_proba(self.samples.X[idx])[:, 1]
        out = []
        for symbol in sorted(set(self.samples.symbol[idx])):
            mine = self.samples.symbol[idx] == symbol
            out.append((str(symbol), float(prob[mine].mean()), [int(i) for i in idx[mine]]))
        return sorted(out, key=lambda s: (-s[1], s[0]))

    def signals_for(self, session: Session, market: MarketData) -> list[PendingSignal]:
        fold = self._fold(session)
        threshold = fold.thresholds.get(self._key) if fold else None
        if threshold is None:
            return []
        at = _scheduler_time(session)
        table = serving_models(fold)[self._key]
        pending = []
        for symbol, score, members in self.scores(session, market.calendar):
            if score < threshold:
                continue
            signal = _signal(f"{self.name}-{session.date:%Y%m%d}-{symbol}", symbol, f"backtest/{self.name}",
                             fold.model_version, score, table.expected_move(score), raw_atr(market, symbol, at), at)
            pending.append(PendingSignal(at, signal, None, tuple(str(self.samples.event_id[i]) for i in members)))
        return pending


# --- S3: in-session news, same-day exit ------------------------------------------------------------------

@dataclass
class S3:
    """One signal per event known in session before close - 30 min, at its
    ``made_at``, when the fold's S3 threshold and a positive expected move
    pass — the Inference Agent's own filter before it calls the router."""

    samples: Samples
    folds: Sequence[FoldResult]
    prediction_ids: Mapping[tuple[str, str, str], str]  # (event_id, symbol, model_version) -> prediction_id
    collapsed: bool = False
    name: str = "S3"

    def __post_init__(self) -> None:
        self._key = "S3_collapsed" if self.collapsed else "S3"
        tradeable = np.flatnonzero(self.samples.tradeable)
        if self.collapsed:
            tradeable = _first_per_story(self.samples, tradeable)
        self._by_session: dict[int, np.ndarray] = {}
        for session in np.unique(self.samples.session[tradeable]):
            self._by_session[int(session)] = tradeable[self.samples.session[tradeable] == session]
        self._prob: dict[int, float] = {}
        for fold in self.folds:
            for k, i in enumerate(fold.scored):
                self._prob[int(i)] = float(fold.prob_scored[k])

    def min_prob_up(self, session: Session) -> Decimal:
        fold = fold_for(list(self.folds), session.open_at)
        threshold = fold.thresholds.get(self._key) if fold else None
        return CLOSED if threshold is None else Decimal(str(threshold))

    def signals_for(self, session: Session, market: MarketData) -> list[PendingSignal]:
        idx = self._by_session.get(market.calendar.index[session.date])
        if idx is None:
            return []
        pending = []
        for i in idx:
            made_at = datetime.fromtimestamp(float(self.samples.made_at[i]), NEW_YORK)
            fold = fold_for(list(self.folds), made_at)
            threshold = fold.thresholds.get(self._key) if fold else None
            prob = self._prob.get(int(i))
            if threshold is None or prob is None or prob < threshold:
                continue
            move = serving_models(fold)[self._key].expected_move(prob)
            if move <= 0:
                continue
            event_id, symbol = str(self.samples.event_id[i]), str(self.samples.symbol[i])
            news_id = event_id.split(":", 1)[1]
            signal = _signal(f"{news_id}-{symbol}-{fold.model_version}"[:64], symbol, f"backtest/{self.name}",
                             fold.model_version, prob, move, raw_atr(market, symbol, made_at),
                             datetime.fromtimestamp(float(self.samples.published_at[i]), NEW_YORK))
            pending.append(PendingSignal(made_at, signal, self.prediction_ids.get((event_id, symbol,
                                                                                   fold.model_version)),
                                         (event_id,)))
        return pending


# --- baselines -------------------------------------------------------------------------------------------

def _gate_passing(symbols: Sequence[str], session: Session, market: MarketData, name: str) -> list[PendingSignal]:
    """In the given order: probability 1 - rank/1000 keeps it through the router's ordering."""
    at = _scheduler_time(session)
    return [PendingSignal(at, _signal(f"{name}-{session.date:%Y%m%d}-{symbol}", symbol, f"backtest/{name}",
                                      f"baseline/{name}", 1.0 - rank / 1000, NOMINAL_MOVE_PCT,
                                      raw_atr(market, symbol, at), at))
            for rank, symbol in enumerate(symbols)]


@dataclass
class B1:
    """Every symbol, every session, at open + 1 minute in seeded random order."""

    symbols: Sequence[str]
    seed: int
    name: str = "B1"

    def min_prob_up(self, session: Session) -> Decimal:
        return NO_GATE

    def signals_for(self, session: Session, market: MarketData) -> list[PendingSignal]:
        order = list(self.symbols)
        random.Random(f"{self.seed}:{session.date}").shuffle(order)
        return _gate_passing(order, session, market, self.name)


@dataclass
class B2:
    """S2 without the probability gate: every symbol with a night, best score first."""

    s2: S2
    name: str = "B2"

    def min_prob_up(self, session: Session) -> Decimal:
        return NO_GATE

    def signals_for(self, session: Session, market: MarketData) -> list[PendingSignal]:
        return _gate_passing([s for s, _, _ in self.s2.scores(session, market.calendar)], session, market,
                             self.name)


@dataclass
class B3:
    """Placebo: as many entries as S2 made that day, on random symbols (one seed of the null)."""

    symbols: Sequence[str]
    entries_by_date: Mapping[date, int]
    seed: int
    name: str = "B3"

    def min_prob_up(self, session: Session) -> Decimal:
        return NO_GATE

    def signals_for(self, session: Session, market: MarketData) -> list[PendingSignal]:
        k = self.entries_by_date.get(session.date, 0)
        if k == 0:
            return []
        rng = random.Random(f"{self.seed}:{session.date}")
        return _gate_passing(rng.sample(list(self.symbols), k), session, market, f"{self.name}s{self.seed}")


# --- S0 and S1: the long-term core, daily --------------------------------------------------------------------

@dataclass
class DailyDay:
    date: date
    value: Decimal       # mark-to-market at the adjusted close, after costs
    traded: Decimal      # notional bought and sold that day
    costs: Decimal       # spread, slippage and fees that day


def run_hold(market: MarketData, symbols: Sequence[str], sessions: Sequence[Session], level: CostLevel,
             fees: FeeTable, *, capital: Decimal = Decimal("100000"), band: Decimal | None = None) -> list[DailyDay]:
    """S0 (``band=None``): equal weight at the first open, held. S1: on each
    month's first session, if any weight at the previous close is outside
    12.5% +/- ``band``, every symbol is rebalanced to 12.5%. Orders are
    decided after a close and filled at the next open, market, ± h·m + σ
    (the open is in the opening window, so m applies), adjusted prices."""
    closes = {s: {b.t.astimezone(NEW_YORK).date(): b for b in market.daily[s]} for s in symbols}
    cash, qty = capital, {s: Decimal(0) for s in symbols}
    target = Decimal(1) / len(symbols)
    out: list[DailyDay] = []
    for k, session in enumerate(sessions):
        traded = cost = Decimal(0)
        rebalance = k == 0 or (band is not None and session.date.month != sessions[k - 1].date.month
                               and _outside_band(qty, closes, sessions[k - 1].date, target, band))
        if rebalance:
            value = cash + sum((qty[s] * Decimal(repr(closes[s][sessions[k - 1].date].c)) for s in symbols),
                               Decimal(0)) if k else capital
            for s in symbols:
                bar = closes[s][session.date]
                open_ = Decimal(repr(bar.o))
                want = (value * target / open_).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)
                delta = want - qty[s]
                if delta == 0:
                    continue
                side = "buy" if delta > 0 else "sell"
                fill = costs.market_fill(side, open_, s, level, level.open_multiplier)
                amount = abs(delta)
                # $100k per rebalance: fee rounding is immaterial here; per order is the conservative reading
                charged = sum(fees.order_fees(session.date, side, amount, fill.price, "per_order").values(), Decimal(0))
                if side == "buy":
                    cash -= costs.cash_debit(amount, fill.price, level) + charged
                else:
                    cash += costs.cash_credit(amount, fill.price, level) - charged
                qty[s] = want
                traded += amount * open_
                cost += amount * abs(fill.price - open_) + charged
        value = cash + sum((qty[s] * Decimal(repr(closes[s][session.date].c)) for s in symbols), Decimal(0))
        out.append(DailyDay(session.date, value, traded, cost))
    return out


def _outside_band(qty: Mapping[str, Decimal], closes: Mapping[str, Mapping[date, object]], day: date,
                  target: Decimal, band: Decimal) -> bool:
    values = {s: qty[s] * Decimal(repr(closes[s][day].c)) for s in qty}
    total = sum(values.values(), Decimal(0))
    return total > 0 and any(abs(v / total - target) > band for v in values.values())
