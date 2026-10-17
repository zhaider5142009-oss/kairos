"""Rich terminal dossiers — human-grade presentation of KAIROS output.

Every renderer takes the plain dicts produced by the engine / backtest /
execution layers and turns them into an institutional-grade terminal report.
"""

from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

BRAND = "KAIROS"

DIR_STYLE = {1: ("BULLISH", "bold green"), 0: ("NEUTRAL", "bold yellow"),
             -1: ("BEARISH", "bold red")}


def _dirval(v) -> int:
    s = str(v).strip().lower()
    if s in ("1", "+1", "bullish", "buy", "up", "strong buy"):
        return 1
    if s in ("-1", "bearish", "sell", "down", "strong sell"):
        return -1
    return 0


def _dir(v) -> Text:
    label, style = DIR_STYLE.get(_dirval(v), DIR_STYLE[0])
    return Text(label, style=style)


def _pct(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v * 100:+.{digits}f}%"


def _num(v: float | None, digits: int = 2) -> str:
    return "—" if v is None else f"{v:,.{digits}f}"


def _verdict_style(v: str) -> str:
    return {"PROFITABLE": "bold green", "MARGINAL": "yellow",
            "UNPROFITABLE": "red"}.get(v, "white")


# ---------------------------------------------------------------------------
# Prediction dossier
# ---------------------------------------------------------------------------
def render_prediction(res: dict[str, Any], console: Console | None = None,
                      council: bool = True) -> None:
    console = console or Console()
    if not res.get("ok"):
        console.print(Panel(f"[red]{res.get('error', 'prediction failed')}",
                            title=f"{BRAND} — {res.get('symbol', '?')}", border_style="red"))
        return

    console.print(Rule(style="bold blue"))
    header = (f"[bold]{BRAND} DOSSIER[/] — [cyan]{res['symbol']}[/] · {res['name']}\n"
              f"[dim]{res['asset_class']} · {res['market']} · {res['sector']} · "
              f"{res['currency']} · as of {res['as_of']} · {res['history_bars']} bars · "
              f"{res['computation_ms']}ms[/]")
    console.print(Panel(header, border_style="blue"))

    sig = res["signal"]
    dir_txt = _dir(sig["direction"])
    conf_bar = "█" * int(sig["confidence"] * 10) + "░" * (10 - int(sig["confidence"] * 10))

    # ---- verdict banner
    verdict = Table.grid(padding=(0, 2))
    verdict.add_column(justify="center")
    verdict.add_row(f"{dir_txt}   score [bold]{sig['score']:+.2f}[/]   "
                    f"conf [bold]{sig['confidence']:.2f}[/] {conf_bar}")
    verdict.add_row(f"expected [bold]{_pct(sig['expected_return'])}[/] over "
                    f"[bold]{sig['horizon_days']}d[/] · P(up) [bold]"
                    f"{res['probability_up']:.1%}[/]")
    style = "green" if _dirval(sig["direction"]) > 0 else "red" if _dirval(sig["direction"]) < 0 else "yellow"
    console.print(Panel(verdict, title="SYSTEM SIGNAL", border_style=style))

    # ---- targets & bands
    tgt = Table(title="PRICE TARGETS", box=box.SIMPLE_HEAD, title_style="bold")
    tgt.add_column("horizon")
    tgt.add_column("target", justify="right")
    tgt.add_column("p10", justify="right", style="dim")
    tgt.add_column("p90", justify="right", style="dim")
    b = res["confidence_band"]
    for k, v in res["targets"].items():
        tgt.add_row(k, _num(v, 2), _num(b.get("p10")), _num(b.get("p90")))
    console.print(tgt)

    # ---- regime
    reg = res["regime"]
    reg_tbl = Table(title="MARKET REGIME", box=box.SIMPLE_HEAD, title_style="bold")
    reg_tbl.add_column("regime")
    reg_tbl.add_column("Hurst", justify="right")
    reg_tbl.add_column("ADX", justify="right")
    reg_tbl.add_column("vol", justify="right")
    reg_tbl.add_column("vol percentile", justify="right")
    reg_tbl.add_row(f"[bold cyan]{reg['regime']}[/]", f"{reg['hurst']:.2f}",
                    f"{reg['adx']:.1f}", _pct(reg["vol_annual"], 1),
                    f"{reg['vol_percentile']:.1%}")
    console.print(reg_tbl)

    # ---- models
    if res.get("models"):
        m_tbl = Table(title=f"MODEL BATTERY ({len(res['models'])} models)",
                      box=box.SIMPLE_HEAD, title_style="bold")
        m_tbl.add_column("model", style="cyan")
        m_tbl.add_column("dir", justify="center")
        m_tbl.add_column("expected", justify="right")
        m_tbl.add_column("conf", justify="right")
        m_tbl.add_column("detail", style="dim", overflow="fold")
        for m in res["models"]:
            m_tbl.add_row(m["model"], _dir(m["direction"]),
                          _pct(m["expected_return"]), f"{m['confidence']:.2f}",
                          m["detail"] or "")
        console.print(m_tbl)
        console.print(f"[dim]model agreement {sig.get('model_agreement', 0):.0%} · "
                      f"disagreement shrinkage ×{sig.get('shrinkage') or 1.0:.2f}[/]")

    # ---- strategies
    if res.get("top_strategies"):
        s_tbl = Table(title=f"STRATEGY CONSENSUS — {res['strategies_voted']}/100 voted "
                            f"(consensus {sig.get('strategy_consensus', 0):+.2f})",
                      box=box.SIMPLE_HEAD, title_style="bold")
        s_tbl.add_column("strategy", style="cyan")
        s_tbl.add_column("family", style="dim")
        s_tbl.add_column("dir", justify="center")
        s_tbl.add_column("strength", justify="right")
        s_tbl.add_column("rationale", overflow="fold")
        for s in res["top_strategies"]:
            s_tbl.add_row(s["name"], s["category"], _dir(s["direction"]),
                          f"{s['strength']:.2f}", s["rationale"])
        console.print(s_tbl)

    # ---- patterns + factors
    if res.get("patterns"):
        console.print(Text("PATTERNS: ", style="bold") +
                      Text(", ".join(res["patterns"]), style="magenta"))
    if res.get("factor_context"):
        fc = res["factor_context"]
        console.print(Text("PSX FACTORS: ", style="bold") + Text(
            f"1m momentum {_num(fc.get('momentum_1m_z'))} · 3m {_num(fc.get('momentum_3m_z'))} · "
            f"value {_num(fc.get('value_z'))} · quality {_num(fc.get('quality_z'))} · "
            f"composite rank #{fc.get('composite_rank')}", style="magenta"))

    # ---- risk
    r = res["risk"]
    risk_tbl = Table(title="RISK & SIZING", box=box.SIMPLE_HEAD, title_style="bold")
    for col in ("metric", "value"):
        risk_tbl.add_column(col)
    risk_tbl.add_row("VaR 95% (1d)", _pct(r["var_95_daily"]))
    risk_tbl.add_row("CVaR 95% (1d)", _pct(r["cvar_95_daily"]))
    risk_tbl.add_row("annual vol", _pct(r["annual_vol"], 1))
    risk_tbl.add_row("Sharpe / Sortino", f"{r['sharpe']:.2f} / {r['sortino']:.2f}")
    risk_tbl.add_row("max drawdown", _pct(r["max_drawdown"], 1))
    risk_tbl.add_row("Kelly fraction (capped 25%)", f"{r['kelly_fraction']:.3f}")
    risk_tbl.add_row("[bold]suggested position weight[/]",
                     f"[bold]{_pct(r['suggested_weight'])}[/]")
    console.print(risk_tbl)
    if r.get("notes"):
        console.print(f"[dim]{r['notes']}[/]")

    # ---- event risk / KSE regime
    ev = res.get("event_risk") or {}
    if ev.get("position_multiplier", 1.0) < 1.0:
        console.print(Panel(
            f"[yellow]EVENT RISK — next: {ev.get('next_event', '?')} "
            f"({ev.get('next_date', '?')}, impact {ev.get('next_impact', '?')})\n"
            f"position multiplier ×{ev['position_multiplier']:.2f}[/]",
            border_style="yellow"))
    km = res.get("kse_regime_multiplier")
    if km is not None and km != 1.0:
        console.print(f"[dim]KSE-100 regime multiplier ×{km:.2f} applied to PSX sizing[/]")

    # ---- council
    if council and res.get("council", {}).get("ok"):
        c = res["council"]
        cons = c.get("consensus") or {}
        view = str(cons.get("view", "neutral"))
        color = {"bullish": "green", "bearish": "red"}.get(view, "yellow")
        body = Text()
        body.append(f"COUNCIL VIEW: {view.upper()}", style=f"bold {color}")
        body.append(f" · P(up) {cons.get('probability_up', '?')}\n")
        drivers = c.get("drives") or cons.get("key_drivers") or []
        if isinstance(drivers, list) and drivers:
            body.append("drivers: " + "; ".join(str(d) for d in drivers[:5]) + "\n")
        risks = c.get("risks") or cons.get("risks") or []
        if isinstance(risks, list) and risks:
            body.append("risks: " + "; ".join(str(x) for x in risks[:4]) + "\n", style="red")
        thesis = cons.get("thesis") or c.get("thesis")
        if thesis:
            body.append(f"\n{thesis}", style="italic")
        console.print(Panel(body, title="LLM QUANT COUNCIL", border_style=color))
    console.print(Rule(style="dim"))


# ---------------------------------------------------------------------------
# Scan leaderboard
# ---------------------------------------------------------------------------
def render_scan(res: dict[str, Any], console: Console | None = None,
                top: int = 20) -> None:
    console = console or Console()
    if not res.get("ok"):
        console.print(Panel(f"[red]{res.get('error', 'scan failed')}",
                            title=f"{BRAND} PSX SCAN", border_style="red"))
        return
    tbl = Table(title=f"{BRAND} — PSX CROSS-SECTION · {res['universe']} stocks · "
                      f"{res['as_of'][:19].replace('T', ' ')}",
                box=box.ROUNDED, title_style="bold blue")
    cols = [("rank", "center"), ("ticker", "left"), ("price", "right"),
            ("momentum 1m", "right"), ("momentum 3m", "right"),
            ("value", "right"), ("quality", "right"), ("composite", "right")]
    for name, just in cols:
        tbl.add_column(name, justify=just)
    for row in res["top"][:top]:
        tbl.add_row(
            f"[bold]{row.get('rank', '?')}[/]", f"[cyan]{row.get('name', '?')}[/]",
            _num(row.get("close")),
            _num(row.get("momentum_1m")), _num(row.get("momentum_3m")),
            _num(row.get("value")), _num(row.get("quality")),
            f"[bold]{_num(row.get('composite_score'), 3)}[/]")
    console.print(tbl)
    console.print("[dim]composite = w·z(momentum, value, quality, low-vol) — "
                  "top decile is the candidate long basket[/]")


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
def render_portfolio(res: dict[str, Any], console: Console | None = None) -> None:
    console = console or Console()
    if not res.get("ok"):
        console.print(Panel(f"[red]{res.get('error', 'optimizer failed')}",
                            title=f"{BRAND} PORTFOLIO", border_style="red"))
        return
    tbl = Table(title=f"{BRAND} QUANTUM-ASSISTED PORTFOLIO · capital "
                      f"{res['capital']:,.0f}",
                box=box.ROUNDED, title_style="bold blue")
    for col in ("instrument", "weight", "allocation"):
        tbl.add_column(col, justify="left" if col == "instrument" else "right")
    for sym, w in sorted(res["weights"].items(), key=lambda kv: -abs(kv[1])):
        alloc = res["allocations"].get(sym, 0.0)
        style = "green" if w > 0.02 else "dim"
        tbl.add_row(f"[cyan]{sym}[/]", f"[{style}]{_pct(w)}[/]", f"{alloc:,.0f}")
    console.print(tbl)
    stats = Table(box=box.SIMPLE_HEAD, title_style="bold")
    for col in ("expected return", "expected vol", "Sharpe", "cardinality",
                "solver", "coherence"):
        stats.add_column(col, justify="right")
    qc = res.get("quantum_check") or {}
    coh = qc.get("coherence", "?")
    stats.add_row(_pct(res["expected_return"]), _pct(res["expected_vol"]),
                  f"{res['sharpe']:.2f}", str(res["cardinality"]),
                  res.get("solver", "annealer"), str(coh))
    console.print(stats)


# ---------------------------------------------------------------------------
# Backtest / paper session
# ---------------------------------------------------------------------------
def render_backtest(results: dict[str, Any], console: Console | None = None) -> None:
    console = console or Console()
    summary = results.get("summary", {})
    console.print(Panel(
        f"instruments [bold]{summary.get('instruments', 0)}[/] · "
        f"[green]profitable {summary.get('profitable', 0)}[/] · "
        f"[yellow]marginal {summary.get('marginal', 0)}[/] · "
        f"[red]unprofitable {summary.get('unprofitable', 0)}[/]",
        title=f"{BRAND} PAPER VALIDATION — WALK-FORWARD", border_style="blue"))
    tbl = Table(box=box.ROUNDED, title_style="bold")
    for col, just in (("instrument", "left"), ("verdict", "center"), ("return", "right"),
                      ("benchmark", "right"), ("excess", "right"), ("sharpe", "right"),
                      ("hit rate", "right"), ("max dd", "right"), ("trades", "right")):
        tbl.add_column(col, justify=just)
    for sym, r in sorted(results.get("results", {}).items(),
                         key=lambda kv: -(kv[1].get("return") or -9)):
        if "error" in r:
            tbl.add_row(sym, "[red]ERROR[/]", r["error"], "", "", "", "", "", "")
            continue
        excess = (r.get("return", 0) - r.get("benchmark", 0))
        tbl.add_row(f"[cyan]{sym}[/]",
                    f"[{_verdict_style(r['verdict'])}]{r['verdict']}[/]",
                    _pct(r.get("return")), _pct(r.get("benchmark")),
                    f"[{'green' if excess >= 0 else 'red'}]{_pct(excess)}[/]",
                    _num(r.get("sharpe")), _pct(r.get("hit_rate"), 1),
                    _pct(r.get("max_dd"), 1), str(r.get("trades", 0)))
    console.print(tbl)

    demo = results.get("demo_paper_order")
    if demo:
        console.print(Panel(
            f"executed demo order → [cyan]{demo.get('symbol')}[/] "
            f"{demo.get('side')} {demo.get('quantity')} @ {demo.get('price')} "
            f"[dim](fill {demo.get('fill_price')}, commission {demo.get('commission')}, "
            f"slippage {demo.get('slippage')})[/]",
            title="EXECUTION PATH PROOF — gate → broker → journal", border_style="green"))
    acct = results.get("account")
    if acct:
        console.print(f"[dim]paper account: equity {acct.get('equity', 0):,.0f} "
                      f"· cash {acct.get('cash', 0):,.0f} · "
                      f"fills {acct.get('fills', '?')}[/]")
