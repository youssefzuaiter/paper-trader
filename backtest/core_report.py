"""The long-term core's report (core design §5): numbers from runs, verdicts from the registered rules.

Every run becomes a daily return series net of all costs, with the owner's
withdrawals taken out (a withdrawal is not a loss): rₜ = (Vₜ + flowₜ) / Vₜ₋₁ − 1.
Intervals are the stationary block bootstrap over days with **one index
matrix per window**, shared by every series in it, so a paired difference is
resampled on the same days on both sides. They are computed at the central
level; the other levels are point estimates, which R1 reads for their sign.

The report is ordered by the brief's seven questions. It never names a mix to
hold: there is no "best" column and no winner field.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

import numpy as np

from backtest import core_grid as G
from backtest import metrics
from backtest.core_grid import Config
from backtest.daily import DailyMarket
from backtest.portfolio import Day

TRADING_DAYS = metrics.TRADING_DAYS
#: "Higher" in the raise-cash table means by more than a cent: sub-cent rounding is not a difference.
HIGHER = 0.01


@dataclass
class Run:
    """One run, compacted to arrays as soon as it is simulated (thousands of runs × 2,637 days of
    ``Day`` objects would not fit in memory). ``sale`` keeps the few days the raise-cash section reads."""
    cfg: Config
    first: int                      # session index of the first day
    dates: list[date]
    value: np.ndarray               # end-of-day value (float of the Decimal)
    flow: np.ndarray
    traded: np.ndarray
    exec_cost: np.ndarray
    spread: np.ndarray
    slippage: np.ndarray
    fees: np.ndarray
    rebalanced: np.ndarray
    deferred: np.ndarray
    r: np.ndarray                   # daily returns, computed in Decimal then floated
    end_value: Decimal
    tax_paid: Decimal
    withdrawn: Decimal
    unrealised: Decimal | None
    tax_base_pending: Decimal | None
    sale: dict[int, dict[str, Any]]  # day index -> {"weights": {...}, "cost": float}
    summaries: dict[bool, dict[str, Any]] = field(default_factory=dict)  # per run: never keyed by id()

    @classmethod
    def from_days(cls, cfg: Config, days: Sequence[Day], first: int) -> Run:
        prev = [cfg.size] + [d.value for d in days[:-1]]
        sale: dict[int, dict[str, Any]] = {}
        if cfg.need is not None:
            k = next(i for i, d in enumerate(days) if d.date == cfg.need[0])
            for i in (k, k + 1):
                sale[i] = {"weights": {s: float(w) for s, w in days[i].weights.items()},
                           "cost": float(days[i].exec_cost + days[i].fees)}

        def arr(name: str) -> np.ndarray:
            return np.array([float(getattr(d, name)) for d in days])

        return cls(cfg, first, [d.date for d in days], arr("value"), arr("flow"), arr("traded"), arr("exec_cost"),
                   arr("spread"), arr("slippage"), arr("fees"), np.array([d.rebalanced for d in days]),
                   np.array([d.deferred for d in days]),
                   np.array([float((d.value + d.flow) / p - 1) for d, p in zip(days, prev, strict=True)]),
                   days[-1].value, sum((d.tax for d in days), Decimal(0)), sum((d.flow for d in days), Decimal(0)),
                   days[-1].unrealised, days[-1].tax_base_pending, sale)

    def returns(self) -> np.ndarray:
        return self.r


def riskfree(market: DailyMarket, first: int, last: int) -> np.ndarray:
    """BIL's total return per session (adjusted close to close): the risk-free proxy."""
    return np.array([float(market.close(G.CASH, k) / market.close(G.CASH, k - 1) - 1) for k in range(first, last + 1)])


# --- statistics ---------------------------------------------------------------------------------------------

STATS: dict[str, Callable[..., Any]] = {
    "cagr": lambda r, rf: metrics.cagr(r),
    "volatility": lambda r, rf: metrics.volatility(r),
    "sharpe": metrics.sharpe_excess,
    "sortino": metrics.sortino,
    "calmar": lambda r, rf: metrics.calmar(r),
}


def _finite(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


class Window:
    """One window's shared resamples, risk-free series and halves."""

    def __init__(self, name: str, market: DailyMarket) -> None:
        start, end = G.WINDOWS[name]
        self.name = name
        self.first, self.last = market.index(start), market.index(end)
        self.n = self.last - self.first + 1
        self.rf = riskfree(market, self.first, self.last)
        self.indices = metrics.index_matrix(self.n, mean_block=G.MEAN_BLOCK, resamples=G.RESAMPLES,
                                            seed=G.BOOTSTRAP_SEED)
        self.mid = self.n // 2  # halves: [0, mid) and [mid, n)

    def interval(self, name: str, r: np.ndarray) -> metrics.Interval:
        stat = STATS[name]
        draws = metrics.resampled(lambda a, b: stat(a, b), self.indices, r, self.rf)
        return metrics.interval_from(float(stat(r, self.rf)), draws, resamples=G.RESAMPLES, seed=G.BOOTSTRAP_SEED)

    def paired(self, kind: str, a: np.ndarray, b: np.ndarray) -> metrics.Interval:
        """``mean``: mean daily difference; ``volatility``: annualised volatility difference."""
        if kind == "mean":
            stat = lambda x, y: np.mean(x - y, axis=-1)
        else:
            stat = lambda x, y: metrics.volatility(x) - metrics.volatility(y)
        draws = metrics.resampled(stat, self.indices, a, b)
        return metrics.interval_from(float(stat(a, b)), draws, resamples=G.RESAMPLES, seed=G.BOOTSTRAP_SEED)

    def halves(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return r[: self.mid], r[self.mid:]


def point(kind: str, a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a - b)) if kind == "mean" else float(metrics.volatility(a) - metrics.volatility(b))


def r1(window: Window, kind: str, central: tuple[np.ndarray, np.ndarray],
       levels: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> tuple[str, metrics.Interval]:
    """R1: differs only if the central interval excludes zero, the sign agrees at all three levels,
    and it agrees in both halves; otherwise indistinguishable on this history."""
    iv = window.paired(kind, *central)
    sign = np.sign(iv.estimate)
    excludes = iv.low > 0 or iv.high < 0
    levels_agree = all(np.sign(point(kind, a, b)) == sign for a, b in levels.values())
    halves_agree = all(np.sign(point(kind, ha, hb)) == sign
                       for ha, hb in zip(window.halves(central[0]), window.halves(central[1]), strict=True))
    if excludes and levels_agree and halves_agree and sign != 0:
        return ("higher" if sign > 0 else "lower"), iv
    return "indistinguishable", iv


# --- one run's figures -----------------------------------------------------------------------------------

def summary(run: Run, window: Window | None = None) -> dict[str, Any]:
    key = window is not None
    if key not in run.summaries:
        run.summaries[key] = _summary(run, window)
    return run.summaries[key]


def _summary(run: Run, window: Window | None) -> dict[str, Any]:
    r = run.returns()
    dd = metrics.drawdown_detail(r)
    dates = run.dates
    years = [d.year for d in dates]
    by_year = metrics.year_returns(list(r), years)
    first_year, last_year = years[0], years[-1]
    mean_value = float(np.mean(run.value))
    n_years = len(r) / TRADING_DAYS

    def when(i: int | None) -> str | None:
        if i is None:
            return None
        return (dates[i - 1] if i > 0 else dates[0]).isoformat()

    out: dict[str, Any] = {
        "end_value": str(run.end_value),
        "cagr": float(metrics.cagr(r)), "volatility": float(metrics.volatility(r)),
        "max_drawdown": dd.depth, "drawdown_peak": when(dd.peak), "drawdown_trough": when(dd.trough),
        "drawdown_recovered": when(dd.recovery),
        "longest_underwater_sessions": dd.longest_under, "longest_underwater_open": dd.open_at_end,
        "longest_underwater_from": when(dd.longest_under_from), "longest_underwater_to": when(dd.longest_under_to),
        "worst_year": min((y for y in by_year if first_year < y < last_year), key=lambda y: by_year[y],
                          default=min(by_year, key=lambda y: by_year[y])),
        # One-way turnover after the initial build: buying the mix on day one is not turnover.
        "turnover_per_year": float(np.sum(run.traded[1:])) / 2 / mean_value / n_years,
        "rebalances_per_year": float(np.sum(run.rebalanced[1:])) / n_years,
        "deferrals": int(np.sum(run.deferred)),
        "exec_cost_bps_per_year": float(np.sum(run.exec_cost)) / mean_value / n_years * 1e4,
        "spread_bps_per_year": float(np.sum(run.spread)) / mean_value / n_years * 1e4,
        "slippage_bps_per_year": float(np.sum(run.slippage)) / mean_value / n_years * 1e4,
        "fees_bps_per_year": float(np.sum(run.fees)) / mean_value / n_years * 1e4,
        "tax_paid": str(run.tax_paid),
        "withdrawn": str(run.withdrawn),
    }
    out["worst_year_return"] = by_year[out["worst_year"]]
    if window is not None:
        out["sharpe"] = _finite(float(metrics.sharpe_excess(r, window.rf)))
        out["sortino"] = _finite(float(metrics.sortino(r, window.rf)))
        out["calmar"] = _finite(float(metrics.calmar(r)))
    if run.cfg.tax_rate and run.unrealised is not None:
        base = run.unrealised + (run.tax_base_pending or Decimal(0))
        out["after_tax_liquidation_value"] = str(run.end_value - run.cfg.tax_rate * max(base, Decimal(0)))
    return out


def within_tolerance(s: Mapping[str, Any]) -> bool:
    years = s["longest_underwater_sessions"] / TRADING_DAYS
    return s["max_drawdown"] >= float(G.TOLERANCE["max_drawdown"]) and years <= float(G.TOLERANCE["underwater_years"])


# --- formatting --------------------------------------------------------------------------------------------

def pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f}%"


def iv_pct(i: metrics.Interval, digits: int = 1) -> str:
    return f"{pct(i.estimate, digits)} [{pct(i.low, digits)}, {pct(i.high, digits)}]"


def iv_num(i: metrics.Interval) -> str:
    return f"{i.estimate:.2f} [{i.low:.2f}, {i.high:.2f}]"


def bp(x: float) -> str:
    return f"{x * 1e4:+.2f} bp"


def iv_bp(i: metrics.Interval) -> str:
    return f"{bp(i.estimate)} [{bp(i.low)}, {bp(i.high)}]"


def table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out + [""]


# --- the report -------------------------------------------------------------------------------------------

def build(runs: Mapping[str, Run], market: DailyMarket, *, registration_id: str, leakage_id: str,
          configuration: tuple[int, int], grid_runs: int, scenario_runs: int) -> tuple[str, dict[str, Any]]:
    """The report body (hashed) and the metrics recorded on the experiment."""
    windows = {name: Window(name, market) for name in G.WINDOWS}
    owner = G.OWNER_SIZE

    def get(family: str, window: str, mix: str, rule: str, level: str, size: Decimal = owner, **kw: Any) -> Run:
        return runs[Config(family, window, mix, rule, level, size, **kw).run_key]

    lines: list[str] = []
    m: dict[str, Any] = {"registration": registration_id, "leakage_core": leakage_id}

    # 0. header
    lines += ["# Long-term core: how the reference mixes behaved", "",
              f"> **History warning.** {G.HISTORY_WARNING}", "", f"> {G.NOT_ADVICE}", "",
              f"- Registration: `{registration_id}`; checks: `{leakage_id}` (C1-C12, each with a broken twin).",
              (f"- Grid of {grid_runs} runs and {scenario_runs} raise-cash scenarios; this is run-core experiment "
              f"{configuration[0]} of {configuration[1]} on this window."),
              (f"- Windows: full {G.WINDOWS['full'][0]} to {G.WINDOWS['full'][1]} ({windows['full'].n} sessions); "
              f"crypto {G.WINDOWS['crypto'][0]} to {G.WINDOWS['crypto'][1]} ({windows['crypto'].n} sessions)."),
              (f"- Tables are at the owner's size (${owner:,}) and the central cost level unless they say otherwise. "
              f"Intervals: 95%, stationary block bootstrap over days (mean block {G.MEAN_BLOCK}, {G.RESAMPLES:,} "
              f"resamples, seed {G.BOOTSTRAP_SEED}), one set of resamples per window."),
              (f"- Your tolerance (registered): maximum drawdown {pct(float(G.TOLERANCE['max_drawdown']), 0)}, "
              f"longest underwater {G.TOLERANCE['underwater_years']} years, read at the pessimistic level (R4)."), ""]

    full = windows["full"]
    rows_bad, rows_earn = [], []
    trial_sharpes = []
    for mix in G.BASE_MIXES:
        for rule in G.RULES:
            run = get("core", "full", mix, rule, "central")
            trial_sharpes.append(float(np.mean(run.returns() - full.rf) / np.std(run.returns() - full.rf, ddof=1)))
    for mix in G.BASE_MIXES:
        for rule in ("none", "quarterly"):
            run = get("core", "full", mix, rule, "central")
            s = summary(run, full)
            pess = summary(get("core", "full", mix, rule, "pessimistic"), full)
            mark = "yes" if within_tolerance(pess) else "no"
            under_years = s["longest_underwater_sessions"] / TRADING_DAYS
            rows_bad.append([mix, rule, pct(s["max_drawdown"]), f"{s['drawdown_peak']} → {s['drawdown_trough']}",
                             s["drawdown_recovered"] or "not recovered",
                             f"{under_years:.1f} y" + (" (still under)" if s["longest_underwater_open"] else ""),
                             f"{s['worst_year']} {pct(s['worst_year_return'])}", mark])
            r = run.returns()
            ivs = {name: full.interval(name, r) for name in STATS}
            excess = r - full.rf
            dsr = metrics.deflated_sharpe(float(np.mean(excess) / np.std(excess, ddof=1)), trial_sharpes, len(r),
                                          float(_skew(excess)), float(_kurtosis(excess)))
            rows_earn.append([mix, rule, iv_pct(ivs["cagr"]), iv_pct(ivs["volatility"]), iv_num(ivs["sharpe"]),
                              iv_num(ivs["sortino"]), iv_num(ivs["calmar"]), f"{dsr:.2f}"])
            m[f"full:{mix}:{rule}"] = {**{k: v for k, v in s.items() if not isinstance(v, str) or k == "end_value"},
                                       **{f"{k}_interval": [ivs[k].low, ivs[k].high] for k in STATS},
                                       "within_tolerance": mark == "yes", "deflated_sharpe": dsr}
    lines += ["## 1. How bad does it get?", "",
              "Compound drawdown on the wealth path, net of every cost. R4's mark uses the pessimistic level.", ""]
    lines += table(["Mix", "Rule", "Max drawdown", "Peak → trough", "Back at the peak", "Longest underwater",
                    "Worst calendar year", "Within your tolerance (R4)"], rows_bad)
    lines += ["## 2. What does it earn, for that risk?", "",
              ("Sharpe and Sortino are over BIL's total return. DSR: the deflated Sharpe ratio's probability, "
              f"against all {G.selectable('full')} selectable mix × rule configurations in this window (O6)."), ""]
    lines += table(["Mix", "Rule", "CAGR", "Volatility", "Sharpe", "Sortino", "Calmar", "DSR"], rows_earn)

    # 3. rebalancing
    lines += ["## 3. Does rebalancing help, and how often?", "",
              ("Each rule against buying the same mix once and never touching it ('none'). Δ mean is the mean daily "
              "return difference; R2 reads R1 on it (return) and on the volatility difference (risk). Turnover "
              "excludes the initial purchase."), ""]
    for mix in G.BASE_MIXES:
        none = {lv: get("core", "full", mix, "none", lv).returns() for lv in G.LEVELS}
        rows = []
        for rule in G.RULES:
            if rule == "none":
                continue
            this = {lv: get("core", "full", mix, rule, lv).returns() for lv in G.LEVELS}
            ret_verdict, ret_iv = r1(full, "mean", (this["central"], none["central"]),
                                     {lv: (this[lv], none[lv]) for lv in G.LEVELS})
            risk_verdict, risk_iv = r1(full, "volatility", (this["central"], none["central"]),
                                       {lv: (this[lv], none[lv]) for lv in G.LEVELS})
            s = summary(get("core", "full", mix, rule, "central"))
            costs = [summary(get("core", "full", mix, rule, lv))["exec_cost_bps_per_year"]
                     + summary(get("core", "full", mix, rule, lv))["fees_bps_per_year"] for lv in G.LEVELS]
            helps = []
            if ret_verdict == "higher":
                helps.append("helps on return")
            if ret_verdict == "lower":
                helps.append("hurts return")
            if risk_verdict == "lower":
                helps.append("helps on risk")
            if risk_verdict == "higher":
                helps.append("adds risk")
            verdict = "; ".join(helps) or "indistinguishable on this history"
            rows.append([rule, iv_bp(ret_iv), iv_pct(risk_iv, 2), pct(s["max_drawdown"]),
                         f"{s['turnover_per_year']:.2f}", f"{s['rebalances_per_year']:.1f}",
                         " / ".join(f"{c:.1f}" for c in costs), verdict])
            m[f"rebalance:{mix}:{rule}"] = {"mean_diff": [ret_iv.estimate, ret_iv.low, ret_iv.high],
                                            "vol_diff": [risk_iv.estimate, risk_iv.low, risk_iv.high],
                                            "verdict": verdict}
        lines += [f"### {mix} ({G.registration_params()['mixes'].get(mix, '')})", ""]
        lines += table(["Rule", "Δ mean vs none", "Δ volatility vs none", "Max drawdown", "Turnover / yr",
                        "Rebalances / yr", "Costs bps/yr (opt / central / pess)", "R2"], rows)

    # 4. stress
    lines += ["## 4. Stress: the four real shocks", "",
              ("Return from the S&P 500's closing peak to its trough, the deepest fall inside that span, and how "
              "many sessions after the trough the portfolio regained its value at the peak."), ""]
    rows = []
    for label, (peak, trough) in G.STRESS.items():
        for mix in G.BASE_MIXES:
            for rule in ("none", "quarterly"):
                run = get("core", "full", mix, rule, "central")
                st = stress(run, peak, trough)
                rows.append([label, mix, rule, pct(st["return"]), pct(st["drawdown"]), st["regained"]])
                m[f"stress:{label}:{mix}:{rule}"] = st
    lines += table(["Shock", "Mix", "Rule", "Peak → trough", "Deepest fall inside", "Sessions to regain"], rows)
    inst = [G.CASH, *G.FIVE]
    rows = [[label, *(pct(float(market.close(s, market.index(t)) / market.close(s, market.index(p)) - 1)) for s in inst)]
            for label, (p, t) in G.STRESS.items()]
    lines += ["Each instrument on its own (total return, peak to trough):", ""]
    lines += table(["Shock", *inst], rows)

    # 5. crypto
    crypto = windows["crypto"]
    lines += ["## 5. Does a 5% crypto slice change the picture?", "",
              (f"Crypto window only ({G.WINDOWS['crypto'][0]} on). BTC/USD is priced at each NYSE session's close "
              "(the last 5-minute bar ending by 16:00, or 13:00 on half-days); weekend moves count toward Monday. "
              "R3 reads R1 on M(k+4) − Mk."), ""]
    rows = []
    for base, with_btc in zip(G.BASE_MIXES, G.CRYPTO_MIXES, strict=True):
        for rule in G.RULES:
            a = {lv: get("crypto", "crypto", with_btc, rule, lv).returns() for lv in G.LEVELS}
            b = {lv: get("crypto", "crypto", base, rule, lv).returns() for lv in G.LEVELS}
            mean_v, mean_iv = r1(crypto, "mean", (a["central"], b["central"]), {lv: (a[lv], b[lv]) for lv in G.LEVELS})
            vol_v, vol_iv = r1(crypto, "volatility", (a["central"], b["central"]),
                               {lv: (a[lv], b[lv]) for lv in G.LEVELS})
            dd_a = summary(get("crypto", "crypto", with_btc, rule, "central"))["max_drawdown"]
            dd_b = summary(get("crypto", "crypto", base, rule, "central"))["max_drawdown"]
            changes = "yes" if (mean_v != "indistinguishable" or vol_v != "indistinguishable") else "no"
            rows.append([f"{with_btc} vs {base}", rule, iv_bp(mean_iv), iv_pct(vol_iv, 2),
                         f"{pct(dd_a)} vs {pct(dd_b)}", f"{changes} (return: {mean_v}; volatility: {vol_v})"])
            m[f"crypto:{with_btc}:{rule}"] = {"mean_diff": [mean_iv.estimate, mean_iv.low, mean_iv.high],
                                              "vol_diff": [vol_iv.estimate, vol_iv.low, vol_iv.high],
                                              "changes_the_picture": changes == "yes"}
    lines += table(["Pair", "Rule", "Δ mean", "Δ volatility", "Max drawdown (with vs without)", "R3"], rows)

    # 6. raise cash
    lines += raise_cash_section(runs, market, m)

    # 7. robustness
    lines += ["## 7. Robustness", "", "### Halves of the full window (CAGR, central)", ""]
    rows = []
    for mix in G.BASE_MIXES:
        for rule in ("none", "quarterly"):
            r = get("core", "full", mix, rule, "central").returns()
            h1, h2 = full.halves(r)
            rows.append([mix, rule, pct(float(metrics.cagr(h1))), pct(float(metrics.cagr(h2)))])
    lines += table(["Mix", "Rule", "First half", "Second half"], rows)
    lines += ["### Cost levels and sizes (CAGR)", ""]
    rows = []
    for mix in G.BASE_MIXES:
        for rule in ("none", "quarterly"):
            cells = [pct(float(metrics.cagr(get("core", "full", mix, rule, lv, size).returns())), 2)
                     for size in G.SIZES for lv in G.LEVELS]
            rows.append([mix, rule, *cells])
    lines += table(["Mix", "Rule", *(f"${s:,} {lv}" for s in G.SIZES for lv in G.LEVELS)], rows)
    lines += ["### Instrument substitutions (central)", ""]
    rows = []
    for mix, sub in G.SUBSTITUTIONS:
        for rule in G.SCENARIO_RULES:
            alt = summary(get("instruments", "full", mix, rule, "central", substitute=tuple(sub.items())))
            base = summary(get("core", "full", mix, rule, "central"))
            rows.append([mix, ", ".join(f"{a} → {b}" for a, b in sub.items()), rule,
                         f"{pct(alt['cagr'])} vs {pct(base['cagr'])}", f"{pct(alt['volatility'])} vs {pct(base['volatility'])}",
                         f"{pct(alt['max_drawdown'])} vs {pct(base['max_drawdown'])}"])
    lines += table(["Mix", "Substitution", "Rule", "CAGR (alt vs base)", "Volatility", "Max drawdown"], rows)
    lines += ["### Taxes (illustrative rates, labelled illustrative)", "",
              ("Average-cost lots, gains netted per calendar year, losses carried forward, tax paid at the next "
              "open by selling. Dividends are reinvested in the adjusted prices and taxed with the gains at sale, "
              "not each year as income: illustrative only. After-tax liquidation value taxes the unrealised gains "
              "at the end too, so 'none' is not flattered."), ""]
    rows = []
    for mix in G.BASE_MIXES:
        for rule in ("none", "quarterly", "monthly", "band5"):
            base_end = get("core", "full", mix, rule, "central").end_value
            cells = [f"${base_end:,.0f}"]
            for t in G.TAX_RATES:
                s = summary(get("tax", "full", mix, rule, "central", tax_rate=t))
                cells.append(f"${Decimal(s['after_tax_liquidation_value']):,.0f} (paid ${Decimal(s['tax_paid']):,.0f})")
            rows.append([mix, rule, *cells])
    lines += table(["Mix", "Rule", "No tax: end value", *(f"{t * 100:.0f}% (illustrative): after-tax liquidation"
                                                           for t in G.TAX_RATES)], rows)
    lines += ["### A cash buffer (5% BIL), central", ""]
    rows = []
    for mix in G.BASE_MIXES:
        for rule in G.SCENARIO_RULES:
            with_b = summary(get("buffer", "full", mix, rule, "central", buffer=Decimal("0.05")))
            without = summary(get("buffer", "full", mix, rule, "central", buffer=Decimal(0)))
            rows.append([mix, rule, f"{pct(with_b['cagr'])} vs {pct(without['cagr'])}",
                         f"{pct(with_b['volatility'])} vs {pct(without['volatility'])}",
                         f"{pct(with_b['max_drawdown'])} vs {pct(without['max_drawdown'])}"])
    lines += table(["Mix", "Rule", "CAGR (5% vs 0%)", "Volatility", "Max drawdown"], rows)

    # 8, 9
    lines += ["## 8. Checks", "", (f"- Leakage and consistency checks: `{leakage_id}` (C1-C12 pass, every broken twin "
              "fails); C7 inside it reproduces phase 1's S0 and S1 exactly through this engine."), "",
              "## 9. Not modelled", "",
              "- Dividend withholding for a non-US resident, and income tax on dividends each year (O7).",
              "- Currency: results are in US dollars, not your home currency.",
              ("- Intraday execution: orders fill at the open, at the measured opening spreads and the levels' "
              "slippage; market impact at these sizes is assumed nil."),
              "- Borrowing, margin and shorting: none are used.",
              "- History before 2016 (no 2008, no 2000-2002): see the warning at the top (O1).", ""]
    m["grid_runs"] = grid_runs
    m["scenario_runs"] = scenario_runs
    return "\n".join(lines), m


def _skew(x: np.ndarray) -> float:
    d = x - x.mean()
    return float(np.mean(d ** 3) / np.mean(d ** 2) ** 1.5)


def _kurtosis(x: np.ndarray) -> float:
    d = x - x.mean()
    return float(np.mean(d ** 4) / np.mean(d ** 2) ** 2)


def stress(run: Run, peak: date, trough: date) -> dict[str, Any]:
    dates = run.dates
    values = list(run.value + run.flow)
    i, j = dates.index(peak), dates.index(trough)
    span = values[i:j + 1]
    running = np.maximum.accumulate(span)
    after = next((k - j for k in range(j, len(values)) if values[k] >= values[i]), None)
    return {"return": values[j] / values[i] - 1, "drawdown": float(np.min(np.array(span) / running - 1)),
            "regained": after if after is not None else "not by the end"}


def raise_cash_section(runs: Mapping[str, Run], market: DailyMarket, m: dict[str, Any]) -> list[str]:
    """§6.3: this plan against selling everything pro rata, on the same needs."""
    pairs: dict[tuple[str, Decimal, str, Decimal, date], dict[str, Run]] = {}
    for run in runs.values():
        c = run.cfg
        if c.family == "raise_cash" and c.need is not None:
            pairs.setdefault((c.mix, c.buffer, c.rule, c.need[1], c.need[0]), {})[c.raise_method] = run
    troughs = {t for _, t in G.STRESS.values()}
    agg: dict[tuple[str, Decimal, str, Decimal], list[dict[str, float]]] = {}
    for (mix, buffer, rule, amount, need_day), both in sorted(pairs.items()):
        row = {}
        for method, run in both.items():
            k = run.dates.index(need_day)
            weights_after = run.sale[k + 1]["weights"]
            spec = run.cfg.spec()
            view = market.view(run.first + k) if spec.needs_view else None
            targets = {s: float(w) for s, w in spec.weights(view).items()}
            drift = max(abs(weights_after.get(s, 0.0) - targets.get(s, 0.0)) for s in set(targets) | set(weights_after))
            row[f"{method}_cost"] = run.sale[k + 1]["cost"]
            row[f"{method}_drift"] = drift
            row[f"{method}_value_1y"] = float(run.value[min(k + 1 + TRADING_DAYS, len(run.value) - 1)])
            row["drawdown_at_sale"] = float(run.value[k] / np.max(run.value[: k + 1]) - 1)
        row["trough"] = float(need_day in troughs)
        agg.setdefault((mix, buffer, rule, amount), []).append(row)
    lines = ["## 6. Raising cash", "",
             ("If you need X by a date, this plan sells free cash first, then the buffer, then whatever is most "
             "over-weight after the withdrawal, and only then pro rata. The baseline sells every holding pro rata. "
             "Same needs, same dates (the four stress troughs and 20 seeded dates), central level, $10,000. "
             "Value one year later compares what the two plans left behind; 'higher' means by more than a cent."), ""]
    rows = []
    for (mix, buffer, rule, amount), items in sorted(agg.items()):
        n = len(items)
        diff = [i["plan_value_1y"] - i["pro_rata_value_1y"] for i in items]
        trough_diff = [i["plan_value_1y"] - i["pro_rata_value_1y"] for i in items if i["trough"]]
        rows.append([mix, f"{buffer * 100:.0f}%", rule, f"${amount:,.0f}",
                     f"${np.mean([i['plan_cost'] for i in items]):.2f} vs ${np.mean([i['pro_rata_cost'] for i in items]):.2f}",
                     (f"{pct(float(np.mean([i['plan_drift'] for i in items])))} vs "
                     f"{pct(float(np.mean([i['pro_rata_drift'] for i in items])))}"),
                     f"${np.mean(diff):+,.2f} ({sum(d > HIGHER for d in diff)} of {n} higher)",
                     f"${np.mean(trough_diff):+,.2f}" if trough_diff else "—"])
        m[f"raise_cash:{mix}:{buffer}:{rule}:{amount}"] = {"mean_value_1y_diff": float(np.mean(diff)),
                                                          "plan_higher": int(sum(d > HIGHER for d in diff)), "n": n}
    lines += table(["Mix", "Buffer", "Rule", "Need", "Sale cost (plan vs pro rata)", "Drift after",
                    "Value 1 year later, plan − pro rata", "At the 4 stress troughs"], rows)
    return lines
