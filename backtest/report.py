"""Reports (design §10). Every report opens with the brief's selection-bias
warning; a provenance header (experiment, time, configuration N of M) sits
above a body that is a pure function of code, data, parameters and seeds,
and only the body is hashed into the registry.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any, Final

import numpy as np

from backtest import metrics
from backtest.engine import RunResult

SELECTION_BIAS: Final[str] = (
    "> **Selection bias.** The 8 symbols (AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, AMD) were picked in 2026 "
    "and are large companies that survived and grew, so absolute returns are inflated. Every strategy is "
    "compared with buy-and-hold of the same 8 symbols (S0).")


def document(header: Mapping[str, Any], body: str) -> str:
    lines = [f"<!-- {k}: {v} -->" for k, v in header.items()]
    return "\n".join(lines) + "\n\n" + body


# --- the report.md reproduction ---------------------------------------------------------------------

def _flatten(value: Any, path: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        return [item for k in value for item in _flatten(value[k], f"{path}/{k}" if path else str(k))]
    if isinstance(value, list):
        return [item for k, v in enumerate(value) for item in _flatten(v, f"{path}[{k}]")]
    return [(path, value)]


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def reproduction_report(ours: Mapping[str, Any], meta: Mapping[str, Any], checks: Sequence[Any],
                        findings: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Every number of ``report.md`` next to its recomputation, compared exactly."""
    theirs = {"metrics": meta["metrics"], "move_table": meta["move_table"], "n_samples": meta["n_samples"],
              "digest": meta["version"].rsplit("-", 1)[1]}
    left, right = dict(_flatten(ours)), dict(_flatten(theirs))
    paths = sorted(set(left) | set(right))
    rows = [(p, right.get(p), left.get(p), left.get(p) == right.get(p) and p in left and p in right) for p in paths]
    matched = sum(ok for *_, ok in rows)
    m, t = meta["metrics"], meta["metrics"]["test"]
    lines = [
        f"# Reproduction of `models/return_model/report.md` ({meta['version']})",
        "",
        SELECTION_BIAS,
        "",
        (f"**{matched} of {len(rows)} numbers reproduced exactly** (bit for bit, compared after the same JSON "
        f"round trip `meta.json` went through), in legacy mode: the training script's own `build_dataset`, "
        "`split` and the shared estimator `fit_calibrated`, on the cached articles and bars, with nothing from "
        "the lock-box sessions in memory."),
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
        *(f"| {c.name} | {'pass' if c.passed else '**FAIL**'} | {_check_detail(c)} |" for c in checks),
        "",
        "## Headline numbers",
        "",
        "| | report.md | reproduced |",
        "|---|---:|---:|",
        f"| samples | {meta['n_samples']} | {ours['n_samples']} |",
        f"| dataset digest | {theirs['digest']} | {ours['digest']} |",
        *(f"| {seg} samples / up rate | {v['n']} / {v['up_rate']:.4f} | "
          f"{ours['metrics']['segments'][seg]['n']} / {ours['metrics']['segments'][seg]['up_rate']:.4f} |"
          for seg, v in m["segments"].items()),
        *(f"| {k} | {t[k]:.4f} | {ours['metrics']['test'][k]:.4f} |"
          for k in ("auc_model", "auc_sentiment_only", "auc_model_one_article_per_symbol_day", "auc_model_tradeable",
                    "auc_model_tradeable_one_per_symbol_day", "brier_model", "brier_base_rate", "log_loss_model",
                    "log_loss_base_rate")),
        "",
        "## What the legacy label gets wrong (§11, reproduced as is, fixed in v2)",
        "",
        *(f"- {line}" for line in findings.get("notes", [])),
        "",
        "## Every number",
        "",
        "| Path | report.md | reproduced | exact |",
        "|---|---:|---:|:---:|",
        *(f"| `{p}` | {_fmt(a)} | {_fmt(b)} | {'✓' if ok else '✗'} |" for p, a, b, ok in rows),
        "",
    ]
    return "\n".join(lines), {"numbers": len(rows), "exact": matched}


def _check_detail(check: Any) -> str:
    d = check.detail
    if check.name == "L4":
        return f"{d['n_differences']} differences (purge {d['purge_s'] / 86400:g} days, {d['entry']} entry)"
    if check.name == "L4b":
        return (f"{d['samples']} samples, largest abs(return − fwd) {d['max_abs_error']:.1e}, {d['unfilled']} unfilled"
                + (" (twin)" if d.get("twin") else ""))
    return ", ".join(f"{k} {v}" for k, v in d.items() if not isinstance(v, (list, dict)))[:200]


# --- strategies ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Summary:
    strategy: str
    level: str
    days: int
    total_return: float
    annualised: float
    sharpe: metrics.Interval
    mean_excess: dict[str, metrics.Interval]
    max_drawdown: float
    drawdown_days: int
    turnover: float
    costs: dict[str, float]
    bets: dict[str, int]
    rejections: dict[str, int]
    percentile_in_placebo: float | None


def independent_bets(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Always all three: days with a position, distinct symbol-days, trades."""
    fills = [r for r in rows if r["state"] == "filled"]
    days = {r["occurred_at"].date() for r in fills}
    return {"days_with_a_position": len(days), "symbol_days": len({(r["symbol"], r["occurred_at"].date())
                                                                    for r in fills}), "trades": len(fills)}


def cost_components(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    out: Counter[str] = Counter()
    for r in rows:
        for key in ("spread_cost", "slippage_cost", "fees"):
            if r.get(key) is not None:
                out[key] += float(r[key])
    return dict(out)


def summarise(result: RunResult, capital_base: Decimal, benchmarks: Mapping[str, Mapping[date, float]], *,
              placebo_totals: Sequence[float] | None = None, seed: int = 0, resamples: int = metrics.RESAMPLES
              ) -> Summary:
    daily = np.array(result.daily_returns(capital_base))
    dates = [d.date for d in result.days]
    excess = {}
    for name, series in benchmarks.items():
        diff = np.array([r - series[d] for r, d in zip(daily, dates, strict=True) if d in series])
        excess[name] = metrics.block_bootstrap(diff, np.mean, seed=seed, n=resamples)
    depth, duration = metrics.max_drawdown(daily)
    traded = sum(float(r["qty"] * r["price"]) for r in result.rows
                 if r["state"] in {"filled", "exit_filled"} and r.get("qty") is not None)
    total = float(daily.sum())
    percentile = None
    if placebo_totals:
        percentile = float(np.mean(np.asarray(placebo_totals) <= total) * 100)
    return Summary(
        strategy=result.strategy, level=result.level, days=len(daily), total_return=total,
        annualised=float(daily.mean() * metrics.TRADING_DAYS) if len(daily) else 0.0,
        sharpe=metrics.block_bootstrap(daily, metrics.sharpe, seed=seed, n=resamples), mean_excess=excess,
        max_drawdown=depth, drawdown_days=duration,
        turnover=traded / float(capital_base) / max(len(daily), 1), costs=cost_components(result.rows),
        bets=independent_bets(result.rows), rejections=dict(result.rejections), percentile_in_placebo=percentile)


def strategy_report(summaries: Sequence[Summary], *, configuration: tuple[int, int], trial_sharpes: Sequence[float],
                    notes: Sequence[str] = ()) -> str:
    """One strategy at every cost level (and variant), in the design's order."""
    name = summaries[0].strategy
    n, m = configuration
    lines = [f"# {name}", "", SELECTION_BIAS, "",
             (f"Configuration {n} of {m} tried on this window. Intervals: stationary block bootstrap over days "
             f"(mean block {metrics.MEAN_BLOCK} days)."), "",
             ("| Level | Days | Total | Annualised | Sharpe [95%] | Deflated Sharpe | Max drawdown (days) | "
             "Turnover/day |"),
             "|---|---:|---:|---:|---|---:|---|---:|"]
    for s in summaries:
        daily_sharpe = s.sharpe.estimate / np.sqrt(metrics.TRADING_DAYS)
        dsr = metrics.deflated_sharpe(daily_sharpe, [x / np.sqrt(metrics.TRADING_DAYS) for x in trial_sharpes],
                                      s.days, 0.0, 3.0)
        lines.append(f"| {s.level} | {s.days} | {s.total_return:+.2%} | {s.annualised:+.2%} | "
                     f"{s.sharpe.estimate:.2f} [{s.sharpe.low:.2f}, {s.sharpe.high:.2f}] | {dsr:.2f} | "
                     f"{s.max_drawdown:.2%} ({s.drawdown_days}) | {s.turnover:.2f} |")
    lines += ["", "## Excess over the baselines (mean daily, 95% interval)", "",
              "| Level | " + " | ".join(summaries[0].mean_excess) + " | Percentile in B3 |",
              "|---|" + "---|" * (len(summaries[0].mean_excess) + 1)]
    for s in summaries:
        cells = [f"{i.estimate:+.4%} [{i.low:+.4%}, {i.high:+.4%}]" for i in s.mean_excess.values()]
        pct = "—" if s.percentile_in_placebo is None else f"{s.percentile_in_placebo:.0f}"
        lines.append(f"| {s.level} | " + " | ".join(cells) + f" | {pct} |")
    lines += ["", "## Independent bets, costs and rejections", "",
              "| Level | Days with a position | Symbol-days | Trades | Spread | Slippage | Fees | Rejections |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for s in summaries:
        rejections = ", ".join(f"{k} {v}" for k, v in sorted(s.rejections.items())) or "—"
        lines.append(f"| {s.level} | {s.bets['days_with_a_position']} | {s.bets['symbol_days']} | {s.bets['trades']} "
                     f"| ${s.costs.get('spread_cost', 0):.2f} | ${s.costs.get('slippage_cost', 0):.2f} "
                     f"| ${s.costs.get('fees', 0):.2f} | {rejections} |")
    if notes:
        lines += ["", *(f"- {note}" for note in notes)]
    return "\n".join(lines) + "\n"
