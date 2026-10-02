"""The long-term core in shekels: a separately registered translation study (owner's go-ahead 2026-10-02).

The core's results are in US dollars; the owner spends primarily in shekels. This study re-runs a fixed
set of the registered configurations through the same engine and translates each day's dollar value
into shekels. Nothing here trades or decides: the exchange rate never feeds a decision, so it is a
translation, not a strategy, and point-in-time timing does not arise.

Settings, fixed in code before any shekel figure was computed:

* **Rate:** the Bank of Israel's representative USD/ILS rate (series RER_USD_ILS, the Bank's open
  SDMX API; Bank of Israel publications may be quoted with the source attributed). For each NYSE
  session date, that date's rate; on a session the Bank did not publish (an Israeli holiday or a
  Friday), the last rate published before it. Those sessions are counted in the report.
* **Start:** the $10,000 is converted at the rate for the session whose close decided the initial
  purchase (the session before the window), so day one's return includes that day's currency move.
* **Rows:** M1-M4 untouched and quarterly; M3 under all eight rules; M3 with a 5% BIL buffer
  (untouched and quarterly). Central costs, the owner's $10,000; R4's tolerance read at pessimistic.
* **Measures:** the core's compound metrics on the shekel path, beside the dollar ones; intervals for
  growth and volatility from the same stationary bootstrap (mean block 5, 10,000 resamples, seed 0);
  the four stress windows; the exchange rate's own path.

    python -m backtest fx-core
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import numpy as np

from backtest import core_grid as G
from backtest import core_report, metrics, parquet, portfolio, registry
from backtest.core_fetch import CorePaths
from backtest.costs import CORE_LEVELS, FeeTable
from backtest.daily import DailyMarket
from backtest.store import Store

logger = logging.getLogger("backtest.core_fx")

COMMAND: Final[str] = "fx-core"
SERIES: Final[str] = "RER_USD_ILS"
FX_START: Final[date] = date(2015, 12, 1)
FX_END: Final[date] = date(2026, 9, 30)
FX_URL: Final[str] = ("https://edge.boi.org.il/FusionEdgeServer/sdmx/v2/data/dataflow/BOI.STATISTICS/EXR/1.0/"
                      f"{SERIES}?format=csv&startperiod={FX_START.isoformat()}&endperiod={FX_END.isoformat()}")
SOURCE: Final[str] = "Bank of Israel, representative exchange rate USD/ILS (series RER_USD_ILS), boi.org.il"
FX_TYPES: Final[dict[str, str]] = {"day": "DATE", "ils_per_usd": "DOUBLE"}

#: (mix, rule, buffer) rows of the study, all at the owner's size.
ROWS: Final[tuple[tuple[str, str, Decimal], ...]] = (
    *((m, r, Decimal(0)) for m in G.BASE_MIXES for r in ("none", "quarterly")),
    *(("M3", r, Decimal(0)) for r in G.RULES if r not in ("none", "quarterly")),
    *(("M3", r, Decimal("0.05")) for r in ("none", "quarterly")),
)


def fx_path(root: Path) -> Path:
    return CorePaths.under(root / ".cache" / "backtest").root / "usd_ils_boi.parquet"


def fetch(path: Path) -> int:
    """Download the series once (an existing file is never overwritten: it is hashed into manifests)."""
    import csv
    import io

    import httpx

    response = httpx.get(FX_URL, timeout=60.0)
    response.raise_for_status()
    rows = [r for r in csv.DictReader(io.StringIO(response.text)) if r.get("SERIES_CODE") == SERIES]
    if not rows:
        raise RuntimeError(f"no {SERIES} rows from the Bank of Israel")
    cols = {"day": [date.fromisoformat(r["TIME_PERIOD"]) for r in rows],
            "ils_per_usd": [float(r["OBS_VALUE"]) for r in rows]}
    return parquet.write(path, cols, FX_TYPES, order_by="day")


def load_rates(path: Path) -> dict[date, float]:
    return {r["day"]: r["ils_per_usd"] for r in parquet.read_rows(path, "ORDER BY day")}


def rates_on(dates: list[date], published: dict[date, float]) -> tuple[list[float], list[date]]:
    """The rate for each date: that date's, else the last one published before it. Returns the rates
    and the dates that used an earlier publication. A date before the first publication is an error."""
    days = sorted(published)
    out, carried = [], []
    j = -1
    for d in dates:
        while j + 1 < len(days) and days[j + 1] <= d:
            j += 1
        if j < 0:
            raise ValueError(f"no USD/ILS rate published on or before {d}")
        out.append(published[days[j]])
        if days[j] != d:
            carried.append(d)
    return out, carried


def shekel_returns(usd_values: list[Decimal], capital: Decimal, rate_start: float, rates: list[float]) -> np.ndarray:
    """Daily returns of the shekel value: V_ILS(t) = V_USD(t) × rate(t), from capital × the start rate."""
    ils = [float(capital) * rate_start] + [float(v) * r for v, r in zip(usd_values, rates, strict=True)]
    return np.array([ils[i + 1] / ils[i] - 1 for i in range(len(rates))])


def params() -> dict[str, Any]:
    return {"source": SOURCE, "series": SERIES, "url": FX_URL, "fx_period": [FX_START.isoformat(), FX_END.isoformat()],
            "rule": "each NYSE session's own rate; else the last rate published before it",
            "start": "capital converted at the rate of the session that decided the initial purchase",
            "rows": [[m, r, str(b)] for m, r, b in ROWS], "size_usd": str(G.OWNER_SIZE), "level": "central",
            "tolerance_level": "pessimistic", "tolerance": G.TOLERANCE, "window": [d.isoformat() for d in
                                                                                  G.WINDOWS["full"]],
            "statistics": {"resamples": G.RESAMPLES, "mean_block_days": G.MEAN_BLOCK, "seed": G.BOOTSTRAP_SEED},
            "stress_windows": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in G.STRESS.items()}}


def cmd_fx_core(run_params: dict[str, Any], store: Store, *, allow_dirty: bool, parent_id: str | None,
                write_report: Any) -> dict[str, Any]:
    from backtest import core_runs

    root = Path(run_params["root"])
    path = fx_path(root)
    if not path.exists():
        logger.info("fetching %s", SERIES)
        fetch(path)
    commit, dirty = registry.git_state()
    leakage_id = "dirty run: not checked" if allow_dirty and dirty else core_runs.passing_leakage(store, commit)
    with registry.begin(store, hypothesis="How the registered reference mixes look in shekels, the owner's spending "
                        "currency (a translation, not a strategy)", command=COMMAND, params=run_params,
                        seeds={"bootstrap": G.BOOTSTRAP_SEED}, inputs=[*core_runs.core_inputs(root), path],
                        data_root=root, window=G.WINDOWS["full"], allow_dirty=allow_dirty,
                        parent_id=parent_id) as exp:
        market = core_runs.load_market(root)
        body, results = build(market, load_rates(path), leakage_id)
        exp.finish(results, report_body=body, conclusion=f"{len(ROWS)} rows in shekels")
        report = write_report(root, exp, COMMAND, body, store)
    return {"experiment": exp.experiment_id, "report": str(report)}


def _simulate(market: DailyMarket, mix: str, rule: str, buffer: Decimal, level: str) -> list[portfolio.Day]:
    family = "buffer" if buffer else "core"
    cfg = G.Config(family, "full", mix, rule, level, G.OWNER_SIZE, buffer=buffer)
    first, last = market.index(G.WINDOWS["full"][0]), market.index(G.WINDOWS["full"][1])
    return portfolio.simulate(market, cfg.spec(), first, last, CORE_LEVELS[level], FeeTable.load())


def _usd_returns(days: list[portfolio.Day]) -> np.ndarray:
    prev = [G.OWNER_SIZE] + [d.value for d in days[:-1]]
    return np.array([float((d.value + d.flow) / p - 1) for d, p in zip(days, prev, strict=True)])


def _figures(r: np.ndarray, dates: list[date]) -> dict[str, Any]:
    dd = metrics.drawdown_detail(r)
    by_year = metrics.year_returns(list(r), [d.year for d in dates])
    years = sorted(by_year)
    worst = min((y for y in years[1:-1]), key=lambda y: by_year[y])

    def when(i: int | None) -> str | None:
        return None if i is None else (dates[i - 1] if i > 0 else dates[0]).isoformat()

    return {"cagr": float(metrics.cagr(r)), "volatility": float(metrics.volatility(r)), "max_drawdown": dd.depth,
            "peak": when(dd.peak), "trough": when(dd.trough), "recovered": when(dd.recovery),
            "longest_under_years": dd.longest_under / metrics.TRADING_DAYS, "under_open": dd.open_at_end,
            "worst_year": worst, "worst_year_return": by_year[worst], "growth": float(np.prod(1 + r))}


def build(market: DailyMarket, published: dict[date, float], leakage_id: str) -> tuple[str, dict[str, Any]]:
    first, last = market.index(G.WINDOWS["full"][0]), market.index(G.WINDOWS["full"][1])
    dates = market.dates[first:last + 1]
    rates, carried = rates_on(dates, published)
    (rate_start,), _ = rates_on([market.dates[first - 1]], published)
    indices = metrics.index_matrix(len(dates), mean_block=G.MEAN_BLOCK, resamples=G.RESAMPLES, seed=G.BOOTSTRAP_SEED)

    def interval(stat: Any, r: np.ndarray) -> metrics.Interval:
        return metrics.interval_from(float(stat(r)), metrics.resampled(stat, indices, r), resamples=G.RESAMPLES,
                                     seed=G.BOOTSTRAP_SEED)

    pct, iv = core_report.pct, core_report.iv_pct
    results: dict[str, Any] = {"carried_sessions": len(carried), "rate_start": rate_start, "rate_end": rates[-1]}
    rows_main, rows_stress = [], []
    for mix, rule, buffer in ROWS:
        days = _simulate(market, mix, rule, buffer, "central")
        usd = _usd_returns(days)
        ils = shekel_returns([d.value + d.flow for d in days], G.OWNER_SIZE, rate_start, rates)
        fu, fi = _figures(usd, dates), _figures(ils, dates)
        pess = _figures(shekel_returns([d.value + d.flow for d in _simulate(market, mix, rule, buffer, "pessimistic")],
                                       G.OWNER_SIZE, rate_start, rates), dates)
        within = (pess["max_drawdown"] >= float(G.TOLERANCE["max_drawdown"])
                  and pess["longest_under_years"] <= float(G.TOLERANCE["underwater_years"]))
        label = f"{mix}{' + 5% BIL' if buffer else ''}"
        start_ils = float(G.OWNER_SIZE) * rate_start
        rows_main.append([label, rule, f"₪{start_ils:,.0f} → ₪{start_ils * fi['growth']:,.0f}",
                          iv(interval(metrics.cagr, ils)), f"{pct(fu['cagr'])}",
                          iv(interval(metrics.volatility, ils)), f"{pct(fu['volatility'])}",
                          f"{pct(fi['max_drawdown'])} ({fi['peak']} → {fi['trough']})", pct(fu["max_drawdown"]),
                          f"{fi['longest_under_years']:.1f} y" + (" (still under)" if fi["under_open"] else ""),
                          f"{fi['worst_year']} {pct(fi['worst_year_return'])}", "yes" if within else "no"])
        results[f"{label}:{rule}"] = {"ils": fi, "usd": fu, "within_tolerance_ils": within}
        if buffer == 0 and rule in ("none", "quarterly"):
            for name, (peak, trough) in G.STRESS.items():
                i, j = dates.index(peak), dates.index(trough)
                usd_v = [float(d.value + d.flow) for d in days]
                ils_v = [v * r for v, r in zip(usd_v, rates, strict=True)]
                rows_stress.append([name, mix, rule, pct(ils_v[j] / ils_v[i] - 1), pct(usd_v[j] / usd_v[i] - 1)])
    i0, i1 = dates.index(G.STRESS["2020 COVID crash"][0]), dates.index(G.STRESS["2020 COVID crash"][1])
    lines = ["# The long-term core in shekels", "",
             f"> **History warning.** {G.HISTORY_WARNING}", "", f"> {G.NOT_ADVICE}", "",
             f"- Source: {SOURCE}. Quoted with the source attributed, as the Bank permits.",
             (f"- A translation, not a strategy: the same runs as the core (checks `{leakage_id}`), each day's dollar "
             "value times that session's USD/ILS rate. Central costs, $10,000; your tolerance read at pessimistic."),
             (f"- {len(carried)} of {len(dates)} sessions had no Bank of Israel rate that day (an Israeli holiday or a "
             "Friday) and used the last published rate."),
             (f"- The dollar cost ₪{rate_start:.3f} when the money was invested and ₪{rates[-1]:.3f} at the end "
             f"({pct(rates[-1] / rate_start - 1)} for the dollar against the shekel). In the 2020 crash it went from "
             f"₪{rates[i0]:.3f} to ₪{rates[i1]:.3f}."), "",
             "## Each mix in shekels, beside dollars", "",
             "Intervals: 95%, stationary block bootstrap over days (mean block 5, 10,000 resamples, seed 0).", ""]
    lines += core_report.table(["Mix", "Rule", "₪ start → end", "CAGR in ₪", "CAGR in $", "Volatility in ₪",
                                "in $", "Max drawdown in ₪ (peak → trough)", "in $", "Longest underwater (₪)",
                                "Worst year (₪)", "Within −35% / 4 y in ₪"], rows_main)
    lines += ["## The four shocks in shekels", "", "Peak to trough of the S&P 500, as in the core report.", ""]
    lines += core_report.table(["Shock", "Mix", "Rule", "In ₪", "In $"], rows_stress)
    lines += ["## What this does not show", "",
              ("- Israeli tax on gains and on dividends, and US dividend withholding: depend on your residency; ask a "
              "local adviser."),
              "- Currency conversion costs when moving shekels to dollars and back (bank or broker spreads).",
              "- Whether the shekel's path over these ten years says anything about the next ten: it does not.", ""]
    return "\n".join(lines), results
