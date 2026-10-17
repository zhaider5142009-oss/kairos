#!/usr/bin/env python
"""KAIROS — institutional-grade market prediction & execution system.

Command line entry point. Run `python kairos.py --help` for the command list.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*starting (autoregressive|MA) parameters.*")
warnings.filterwarnings("ignore", message=".*failed to converge.*")
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization.*")

for _stream in (sys.stdout, sys.stderr):
    if _stream and _stream.encoding and _stream.encoding.lower().replace("-", "") != "utf8":
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

BANNER = r"""
  _  __          _     _
 | |/ /__  _   _| |__ | | ___   ___  _ __ ___
 | ' / _ \| | | | '_ \| |/ _ \ / _ \| '_ ` _ \
 | . \ (_) | |_| | |_) | | (_) | (_) | | | | | |
 |_|\_\___/ \__,_|_.__/|_|\___/ \___/|_| |_| |_|
  market intelligence engine · v2 institutional
"""


def _console():
    from rich.console import Console
    return Console(highlight=False)


def cmd_predict(args) -> int:
    from jarvis.engine import ENGINE
    from jarvis.report import render_prediction
    console = _console()
    console.print(BANNER, style="bold blue")
    for i, sym in enumerate(args.symbols):
        if i:
            time.sleep(0.2)
        res = ENGINE.predict(sym, horizon=args.horizon, use_llm=args.council,
                             use_wsl=not args.no_wsl)
        render_prediction(res, console)
        if args.json:
            out = ROOT / "reports" / f"predict_{res.get('symbol', 'x')}.json"
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(res, indent=2, default=str))
            console.print(f"[dim]saved → {out}[/]")
    return 0


def cmd_scan(args) -> int:
    from jarvis.engine import ENGINE
    from jarvis.report import render_scan
    console = _console()
    res = ENGINE.scan(top=args.top)
    render_scan(res, console, top=args.top)
    if args.json:
        out = ROOT / "reports" / "psx_scan.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(res, indent=2, default=str))
    return 0 if res.get("ok") else 1


def cmd_portfolio(args) -> int:
    from jarvis.engine import ENGINE
    from jarvis.report import render_portfolio
    res = ENGINE.portfolio(args.symbols, capital=args.capital, k=args.cardinality)
    render_portfolio(res, _console())
    return 0 if res.get("ok") else 1


def cmd_backtest(args) -> int:
    from jarvis.backtest import run_paper_session
    from jarvis.engine import ENGINE
    from jarvis.report import render_backtest
    console = _console()
    console.print(BANNER, style="bold blue")
    res = run_paper_session(ENGINE.md, args.symbols, ENGINE, days=args.days)
    render_backtest(res, console)
    out = ROOT / "reports" / "paper_session.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    console.print(f"[dim]saved → {out}[/]")
    return 0


def cmd_news(args) -> int:
    from rich.panel import Panel
    from rich.table import Table
    from jarvis.engine import ENGINE
    console = _console()
    items = ENGINE.news_items()
    if not items:
        console.print(Panel("[red]no news fetched — feeds unreachable[/]", title="KAIROS NEWS"))
        return 1
    tbl = Table(title=f"KAIROS NEWS RADAR — {len(items)} items", box="ROUNDED")
    for col in ("time", "source", "headline"):
        tbl.add_column(col, overflow="fold" if col == "headline" else "truncate")
    for it in items[:args.limit]:
        tbl.add_row(str(getattr(it, "published", ""))[:16],
                    getattr(it, "source", ""),
                    getattr(it, "title", ""))
    console.print(tbl)
    return 0


def cmd_watch(args) -> int:
    """Continuous loop: refresh predictions + show one-line signals."""
    from jarvis.engine import ENGINE
    console = _console()
    console.print(BANNER, style="bold blue")
    while True:
        console.clear()
        from rich.table import Table
        tbl = Table(title=f"KAIROS WATCH — {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    box="ROUNDED")
        for col in ("symbol", "price", "signal", "conf", "exp 5d", "P(up)", "weight"):
            tbl.add_column(col, justify="right" if col != "symbol" else "left")
        for sym in args.symbols:
            try:
                r = ENGINE.predict(sym, horizon=args.horizon, include_news=False)
            except Exception as exc:
                tbl.add_row(sym, "-", f"[red]ERR[/]", str(exc)[:40], "", "", "")
                continue
            if not r.get("ok"):
                tbl.add_row(sym, "-", "[dim]no data[/]", "", "", "", "")
                continue
            s = r["signal"]
            color = "green" if s["direction"] > 0 else "red" if s["direction"] < 0 else "yellow"
            tbl.add_row(f"[cyan]{r['symbol']}[/]", f"{r['price']:,.2f}",
                        f"[{color}]{s['direction']:+d}[/]", f"{s['confidence']:.2f}",
                        f"{s['expected_return']:+.2%}", f"{r['probability_up']:.0%}",
                        f"{r['risk']['suggested_weight']:.0%}")
        console.print(tbl)
        console.print(f"[dim]refreshing every {args.interval}s — Ctrl+C to exit[/]")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


def cmd_serve(args) -> int:
    import uvicorn
    from api.server import app
    console = _console()
    console.print(BANNER, style="bold blue")
    console.print(f"[bold]KAIROS control tower → http://127.0.0.1:{args.port}[/]")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_symbols(args) -> int:
    from rich.table import Table
    from jarvis.universe import universe
    console = _console()
    tbl = Table(title=f"KAIROS UNIVERSE — {len(universe())} instruments",
                box="ROUNDED")
    for col in ("key", "name", "class", "market", "yahoo"):
        tbl.add_column(col, overflow="fold" if col == "name" else "truncate")
    for inst in universe():
        tbl.add_row(f"[cyan]{inst.key}[/]", inst.name, inst.asset_class,
                    inst.market, inst.yahoo or "")
    console.print(tbl)
    return 0


def cmd_health(args) -> int:
    from rich.panel import Panel
    from rich.table import Table
    from jarvis.engine import ENGINE
    from jarvis.llm import health_probe
    from jarvis.wsl_bridge import WSL
    console = _console()
    h = ENGINE.md.health()
    tbl = Table(title="KAIROS SYSTEM HEALTH", box="ROUNDED")
    tbl.add_column("subsystem")
    tbl.add_column("status")
    for name, state in h.items():
        ok = state in ("ok", "up", "degraded")
        tbl.add_row(name, f"[{'green' if ok else 'red'}]{state}[/]")
    llm = health_probe()
    tbl.add_row("llm-router", f"[{'green' if llm.get('ok') else 'red'}]"
                f"{llm.get('provider', 'down')} ({llm.get('models', 0)} models)[/]")
    tbl.add_row("wsl-bridge", f"[{'green' if WSL.available() else 'yellow'}]"
                f"{'ready' if WSL.models_ready() else 'available (models not installed)'}[/]")
    console.print(tbl)
    console.print(Panel(
        "Full health: `python kairos.py predict GOLD` exercises data → models → "
        "strategies → fusion → risk in one shot.", border_style="dim"))
    return 0


def cmd_setup_wsl(args) -> int:
    from jarvis.wsl_bridge import WSL
    console = _console()
    console.print("[bold]Installing foundation models into WSL "
                  "(Chronos-Bolt, TimesFM 2.5, qlib LightGBM)…[/]")
    ok = WSL.setup(verbose=True)
    console.print(f"[{'green' if ok else 'red'}]"
                  f"{'WSL foundation models ready.' if ok else 'setup failed — see log above'}[/]")
    return 0 if ok else 1


def cmd_paper(args) -> int:
    """Place one paper order through the full execution stack."""
    from jarvis.engine import ENGINE
    from jarvis.execution import OrderIntent, PaperBroker, RiskGate
    from rich.panel import Panel
    console = _console()
    r = ENGINE.predict(args.symbol, horizon=5, include_news=False)
    if not r.get("ok"):
        console.print(Panel(f"[red]{r.get('error')}", title="KAIROS PAPER"))
        return 1
    s = r["signal"]
    if abs(s["score"]) < 0.18:
        console.print(Panel(f"[yellow]signal too weak to trade "
                            f"(score {s['score']:+.2f}) — standing down[/]",
                            title="KAIROS PAPER"))
        return 0
    side = "BUY" if s["score"] > 0 else "SELL"
    qty = max(1.0, round(args.notional / r["price"], 2))
    broker = PaperBroker()
    intent = OrderIntent(symbol=r["symbol"], side=side, quantity=qty, venue="PAPER")
    allowed, why = RiskGate().check(intent, broker.account(), broker.open_positions())
    if not allowed:
        console.print(Panel(f"[red]risk gate blocked: {why}", title="KAIROS PAPER"))
        return 1
    fill = broker.submit(intent, r["price"])
    console.print(Panel(
        f"[green]FILLED[/] {fill.symbol} {fill.side} {fill.quantity} @ "
        f"{fill.fill_price} (slip {fill.slippage:.4f}, comm {fill.commission:.2f})\n"
        f"[dim]client_order_id {fill.client_order_id} · journaled[/]",
        title="KAIROS PAPER EXECUTION"))
    return 0


def cmd_mt5(args) -> int:
    """MT5 bridge connection + order latency benchmark."""
    from jarvis.execution import Mt5Bridge
    from rich.panel import Panel
    from rich.table import Table
    console = _console()
    br = Mt5Bridge()
    if not br.connect():
        console.print(Panel("[red]MT5 terminal unreachable — is MetaTrader 5 running "
                            "with algo trading enabled?[/]", title="KAIROS MT5"))
        return 1
    console.print(Panel("[green]connected to MT5 terminal[/]", title="KAIROS MT5"))
    tbl = Table(box="ROUNDED")
    tbl.add_column("symbol")
    tbl.add_column("bid", justify="right")
    tbl.add_column("ask", justify="right")
    tbl.add_column("spread", justify="right")
    for sym in (args.symbols or ["XAUUSD", "XAGUSD", "EURUSD"]):
        tick = br.tick(sym)
        if tick:
            tbl.add_row(f"[cyan]{sym}[/]", f"{tick['bid']}", f"{tick['ask']}",
                        f"{tick['spread']}")
    console.print(tbl)
    if args.benchmark:
        console.print("[bold]order latency benchmark (3 round-trips on GOLD micro)…[/]")
        lat = br.benchmark_orders("XAUUSD", n=3)
        console.print(Panel(
            f"submit→result latency ms: {lat.get('latencies_ms')}\n"
            f"success rate: {lat.get('success_rate'):.0%} · "
            f"mean {lat.get('mean_ms'):.0f}ms", title="MT5 EXECUTION BENCHMARK"))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        prog="kairos", description="KAIROS — market prediction & execution engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("predict", help="full dossier for one or more symbols")
    sp.add_argument("symbols", nargs="+")
    sp.add_argument("--horizon", type=int, default=5)
    sp.add_argument("--council", action="store_true", help="add LLM quant council")
    sp.add_argument("--no-wsl", action="store_true", help="skip foundation models")
    sp.add_argument("--json", action="store_true", help="also save JSON report")
    sp.set_defaults(fn=cmd_predict)

    sp = sub.add_parser("scan", help="PSX cross-sectional factor leaderboard")
    sp.add_argument("--top", type=int, default=20)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_scan)

    sp = sub.add_parser("portfolio", help="quantum-assisted portfolio weights")
    sp.add_argument("symbols", nargs="+")
    sp.add_argument("--capital", type=float, default=1_000_000.0)
    sp.add_argument("--cardinality", type=int, default=None)
    sp.set_defaults(fn=cmd_portfolio)

    sp = sub.add_parser("backtest", help="paper trading validation session")
    sp.add_argument("symbols", nargs="+")
    sp.add_argument("--days", type=int, default=60)
    sp.set_defaults(fn=cmd_backtest)

    sp = sub.add_parser("news", help="news radar")
    sp.add_argument("--limit", type=int, default=25)
    sp.set_defaults(fn=cmd_news)

    sp = sub.add_parser("watch", help="live watchlist loop")
    sp.add_argument("symbols", nargs="+")
    sp.add_argument("--horizon", type=int, default=5)
    sp.add_argument("--interval", type=int, default=300)
    sp.set_defaults(fn=cmd_watch)

    sp = sub.add_parser("serve", help="launch web control tower")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8600)
    sp.set_defaults(fn=cmd_serve)

    sp = sub.add_parser("symbols", help="list the tradable universe")
    sp.set_defaults(fn=cmd_symbols)

    sp = sub.add_parser("health", help="subsystem health check")
    sp.set_defaults(fn=cmd_health)

    sp = sub.add_parser("setup-wsl", help="install Chronos/TimesFM/qlib in WSL")
    sp.set_defaults(fn=cmd_setup_wsl)

    sp = sub.add_parser("paper", help="place one gated paper order")
    sp.add_argument("symbol")
    sp.add_argument("--notional", type=float, default=10_000.0)
    sp.set_defaults(fn=cmd_paper)

    sp = sub.add_parser("mt5", help="MT5 bridge check + latency benchmark")
    sp.add_argument("--symbols", nargs="*")
    sp.add_argument("--benchmark", action="store_true")
    sp.set_defaults(fn=cmd_mt5)

    args = p.parse_args()
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
