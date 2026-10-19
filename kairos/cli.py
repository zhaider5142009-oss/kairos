#!/usr/bin/env python
"""KAIROS — institutional-grade market prediction & execution system.

Command line entry point. Run `python -m kairos --help` (or the `kairos`
console command after installation) for the full command list.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

from rich import box

from .config import CFG

# BLAS/OpenMP default to one thread per LOGICAL core; on a small box the fitting
# storm then oversubscribes and a refit takes minutes instead of seconds.
# Must be set before numpy/scipy/torch load, so it lives here at the entry point.
_THREADS = str(min(4, os.cpu_count() or 1))
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)
# joblib spawns loky workers that get fresh warning-filter state and printed
# ~2900 sklearn UserWarnings into the daemon log during one ML stack. Setting
# this now only affects the children — the parent's filters stay as below.
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*propagate the scikit-learn configuration.*")
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
    from .engine import ENGINE
    from .report import render_prediction
    console = _console()
    console.print(BANNER, style="bold blue")
    for i, sym in enumerate(args.symbols):
        if i:
            time.sleep(0.2)
        res = ENGINE.predict(sym, horizon=args.horizon, use_llm=args.council,
                             use_wsl=not args.no_wsl)
        render_prediction(res, console)
        if args.json:
            out = CFG.reports_dir / f"predict_{res.get('symbol', 'x')}.json"
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(res, indent=2, default=str))
            console.print(f"[dim]saved → {out}[/]")
    return 0


def cmd_scan(args) -> int:
    from .engine import ENGINE
    from .report import render_scan
    console = _console()
    res = ENGINE.scan(top=args.top)
    render_scan(res, console, top=args.top)
    if args.json:
        out = CFG.reports_dir / "psx_scan.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(res, indent=2, default=str))
    return 0 if res.get("ok") else 1


def cmd_portfolio(args) -> int:
    from .engine import ENGINE
    from .report import render_portfolio
    res = ENGINE.portfolio(args.symbols, capital=args.capital, k=args.cardinality)
    render_portfolio(res, _console())
    return 0 if res.get("ok") else 1


def cmd_backtest(args) -> int:
    from .backtest import run_paper_session
    from .engine import ENGINE
    from .report import render_backtest
    console = _console()
    console.print(BANNER, style="bold blue")
    res = run_paper_session(ENGINE.md, args.symbols, ENGINE, days=args.days)
    render_backtest(res, console)
    out = CFG.reports_dir / "paper_session.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    console.print(f"[dim]saved → {out}[/]")
    return 0


def cmd_news(args) -> int:
    from rich.panel import Panel
    from rich.table import Table
    from .engine import ENGINE
    console = _console()
    items = ENGINE.news_items()
    if not items:
        console.print(Panel("[red]no news fetched — feeds unreachable[/]", title="KAIROS NEWS"))
        return 1
    tbl = Table(title=f"KAIROS NEWS RADAR — {len(items)} items", box=box.ROUNDED)
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
    from .engine import ENGINE
    console = _console()
    console.print(BANNER, style="bold blue")
    while True:
        console.clear()
        from rich.table import Table
        tbl = Table(title=f"KAIROS WATCH — {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    box=box.ROUNDED)
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
            from .report import _dirval
            dv = _dirval(s["direction"])
            color = "green" if dv > 0 else "red" if dv < 0 else "yellow"
            tbl.add_row(f"[cyan]{r['symbol']}[/]", f"{r['price']:,.2f}",
                        f"[{color}]{s['direction']}[/]", f"{s['confidence']:.2f}",
                        f"{s['expected_return']:+.2%}", f"{r['probability_up']:.0%}",
                        f"{r['risk']['suggested_weight']:.0%}")
        console.print(tbl)
        console.print(f"[dim]refreshing every {args.interval}s — Ctrl+C to exit[/]")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


def _rotating_stdout(path, max_bytes: int = 4_000_000, backups: int = 1):
    """Own the daemon log from inside the process. cmd's `>>` redirect holds
    the file handle open forever, so an external truncate only punches holes;
    RotatingFileHandler rotates its own handle and the size really drops."""
    import logging
    from logging.handlers import RotatingFileHandler

    handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups,
                                  encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S"))
    box = logging.getLogger("kairos.stdout")
    box.handlers = [handler]
    box.setLevel(logging.INFO)
    box.propagate = False

    class _Writer:
        encoding = "utf-8"

        def __init__(self) -> None:
            self._buf = ""

        def write(self, text: str) -> int:
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    box.info(line)
            return len(text)

        def flush(self) -> None:
            if self._buf.strip():
                box.info(self._buf)
                self._buf = ""

        def isatty(self) -> bool:
            return False

    return _Writer()


def cmd_serve(args) -> int:
    import uvicorn
    from .api.server import app
    if args.logfile:
        writer = _rotating_stdout(args.logfile)
        sys.stdout = sys.stderr = writer
    console = _console()
    console.print(BANNER, style="bold blue")
    console.print(f"[bold]KAIROS control tower → http://127.0.0.1:{args.port}[/]")
    if args.public_url:
        console.print(f"[bold green]public → {args.public_url}[/]")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_symbols(args) -> int:
    from rich.table import Table
    from .universe import universe
    console = _console()
    tbl = Table(title=f"KAIROS UNIVERSE — {len(universe())} instruments",
                box=box.ROUNDED)
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
    from .health import system_health
    console = _console()
    h = system_health()
    tbl = Table(title="KAIROS SYSTEM HEALTH", box=box.ROUNDED)
    tbl.add_column("subsystem")
    tbl.add_column("status")
    for name, state in h["market_data"].items():
        ok = state in ("ok", "up", "degraded")
        tbl.add_row(name, f"[{'green' if ok else 'red'}]{state}[/]")
    llm = h["llm"]
    tbl.add_row("llm-router", f"[{'green' if llm['ok'] else 'yellow'}]"
                f"{llm['provider']} ({llm['models']} council models)[/]")
    wsl = h["wsl"]
    if wsl["models_ready"]:
        cell = "[green]models ready[/]"
    elif wsl["available"]:
        cell = "[yellow]available (models not installed)[/]"
    else:
        cell = "[dim]offline[/]"
    tbl.add_row("wsl-bridge", cell)
    tbl.add_row("universe", f"[cyan]{h['universe_size']} instruments[/]")
    console.print(tbl)
    console.print(Panel(
        "Full health: `python -m kairos predict GOLD` exercises data → models → "
        "strategies → fusion → risk in one shot.", border_style="dim"))
    return 0


def cmd_setup_wsl(args) -> int:
    from .wsl_bridge import WSL
    console = _console()
    console.print("[bold]Installing foundation models into WSL "
                  "(Chronos-Bolt, TimesFM 2.5, qlib LightGBM)…[/]")
    result = WSL.setup()
    ok = result.startswith("SETUP COMPLETE")
    console.print(result)
    console.print("[green]WSL foundation models ready."
                  if ok else "[red]setup failed — see log above")
    return 0 if ok else 1


def cmd_paper(args) -> int:
    """Place one paper order through the full execution stack."""
    from .engine import ENGINE
    from .execution import OrderIntent, PaperBroker, RiskGate
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
        f"{fill.price} (slip {fill.slippage:.4f}, comm {fill.commission:.2f})\n"
        f"[dim]client_order_id {fill.client_order_id} · journaled[/]",
        title="KAIROS PAPER EXECUTION"))
    return 0


def cmd_mt5(args) -> int:
    """MT5 bridge connection + order latency benchmark."""
    from .execution import get_mt5
    from rich.panel import Panel
    from rich.table import Table
    console = _console()
    br = get_mt5()
    if not br.ensure():
        console.print(Panel("[red]MT5 terminal unreachable — is MetaTrader 5 running "
                            "with algo trading enabled?[/]", title="KAIROS MT5"))
        return 1
    console.print(Panel("[green]connected to MT5 terminal[/]", title="KAIROS MT5"))
    tbl = Table(box=box.ROUNDED)
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
        console.print("[bold]quote latency benchmark (3 round-trips, no orders sent)…[/]")
        lat = br.benchmark_latency("XAUUSD", n=3)
        console.print(Panel(
            f"quote round-trip latency ms: {lat.get('latencies_ms')}\n"
            f"success rate: {lat.get('success_rate'):.0%} · "
            f"mean {lat.get('mean_ms'):.0f}ms", title="MT5 LATENCY BENCHMARK"))
    return 0


def cmd_autopilot(args) -> int:
    """Start/stop/status the autonomous scan-and-trade loop."""
    from .autopilot import AUTOPILOT
    from rich.panel import Panel
    from rich.table import Table
    console = _console()
    what = args.what
    if what == "start":
        if args.symbols:
            AUTOPILOT.configure(symbols=args.symbols, interval_s=args.interval)
        out = AUTOPILOT.start(mode=args.mode, confirm_live=args.yes_live)
        if not out.get("ok"):
            console.print(Panel(f"[red]{out.get('error')}[/]", title="KAIROS AUTOPILOT"))
            return 1
        console.print(Panel(
            f"[green]autopilot running[/] · mode [bold]{out['mode']}[/] · "
            f"every {out['interval_s']}s\nwatchlist: {', '.join(out['symbols'])}"
            + (f"\n[yellow]{out['note']}[/]" if out.get("note") else ""),
            title="KAIROS AUTOPILOT"))
        what = "status"
    elif what == "stop":
        AUTOPILOT.stop()
        console.print("[yellow]autopilot stopped; live loop never survives a restart[/]")
        return 0
    elif what == "configure":
        AUTOPILOT.configure(args.symbols, args.interval,
                            args.threshold, args.max_trades)
        console.print("[green]configuration saved[/]")
    if what == "cycle":
        r = AUTOPILOT.cycle()
        console.print(f"[green]one cycle done[/] · trades={r['trades']} "
                      f"signals={r['signals']}")
        what = "status"
    if what in ("status", "configure"):
        st = AUTOPILOT.status()
        mode = st["mode"] + (" [bold red]LIVE[/]" if st.get("armed_live") else " paper")
        state = "[green]running[/]" if st["running"] else "[dim]idle[/]"
        console.print(Panel(
            f"state: {state} · mode: {mode} · cycles: {st['cycles']}\n"
            f"last cycle: {st['last_cycle'] or '—'} · interval: {st['interval_s']}s\n"
            f"entry threshold: {st['entry_threshold']} · "
            f"kill switch: {'ENGAGED' if st['kill_switch'] else 'clear'}",
            title="KAIROS AUTOPILOT"))
        if st.get("signals"):
            tbl = Table(box=box.ROUNDED)
            for c in ("symbol", "price", "call", "score", "conf", "weight", "action"):
                tbl.add_column(c, justify="right" if c not in ("symbol", "call") else "left")
            for srow in st["signals"]:
                sc = srow.get("score") or 0
                col = "green" if sc > 0 else "red" if sc < 0 else "yellow"
                tbl.add_row(srow["symbol"], f"{srow.get('price', 0):,.2f}",
                            srow.get("direction", ""), f"[{col}]{sc:+.3f}[/]",
                            f"{(srow.get('confidence') or 0) * 100:.0f}%",
                            f"{(srow.get('weight') or 0) * 100:.1f}%",
                            srow.get("action", ""))
            console.print(tbl)
        if st.get("actions"):
            cons = Table(box=box.ROUNDED, title="recent actions")
            cons.add_column("ts")
            cons.add_column("symbol")
            cons.add_column("action")
            cons.add_column("detail")
            for a in st["actions"][:10]:
                cons.add_row(a.get("ts", ""), a.get("symbol", ""), a.get("action", ""),
                             str(a.get("why") or a.get("coid") or ""))
            console.print(cons)
        acct = st.get("paper_account") or {}
        rp = st.get("risk_posture") or {}
        console.print(f"paper equity: {acct.get('equity', 0):,.0f} · "
                      f"day P&L: {acct.get('day_pnl', 0):+,.0f} · "
                      f"positions: {len(st.get('paper_positions') or [])}")
        console.print(f"exposure: {rp.get('exposure_pct', 0)}% · "
                      f"consec losses: {rp.get('consec_losses', 0)} · "
                      + ("[yellow]breakers: " + "; ".join(rp.get("blockers") or [])
                         + "[/]" if rp.get("block_entries")
                         else "[green]breakers: clear[/]"))
    return 0


def cmd_rules(args) -> int:
    """Show the effective trading doctrine (ported JARVIS v1 rules)."""
    from rich.panel import Panel
    from rich.table import Table
    from . import rules as R
    console = _console()
    console.print(Panel(R.describe(), title="KAIROS DOCTRINE", style="blue"))
    tbl = Table(box=box.ROUNDED, title="effective rules (config/rules.json override)")
    tbl.add_column("rule")
    tbl.add_column("value", justify="right")
    for k, v in R.rules().items():
        tbl.add_row(k, str(v))
    console.print(tbl)
    if args.wisdom:
        for name, corpus in (("WYCKOFF", R.WYCKOFF_PHASES), ("DOW", R.DOW_THEORY),
                             ("CANSLIM", R.CANSLIM), ("BUFFETT", R.BUFFETT_FILTERS),
                             ("DALIO", R.DALIO_PRINCIPLES), ("LYNCH", R.LYNCH_CATEGORIES)):
            console.print(Panel("\n".join(f"• {k}: {v}" for k, v in corpus.items()),
                                title=name, style="dim"))
    return 0


def cmd_livebook(args) -> int:
    """Watch the REAL JS Global holdings with engine verdicts + exit tickets."""
    from rich.table import Table
    from . import livebook
    console = _console()
    if args.add:
        sym, qty, entry = args.add
        livebook.add(sym, float(qty), float(entry))
        console.print(f"[green]watching {sym.upper()}[/]")
    if args.remove:
        console.print("[yellow]removed[/]" if livebook.remove(args.remove)
                      else "[red]not in the live book[/]")
    if getattr(args, "sync", False):
        from . import jsgsync
        console.print("[bold blue]syncing from JS Global (Web Trade Cast)…[/]")
        try:
            rep = jsgsync.sync()
        except jsgsync.JsGlobalError as e:
            console.print(f"[red]sync failed:[/] {e}")
            return 2
        for i in rep["imported"]:
            ps = f" ({i['pending_sell']:g} pending sell)" if i.get("pending_sell") else ""
            console.print(f"[green]imported[/] {i['symbol']} "
                          f"{i['qty']}@{i['avg']:,.2f}{ps}")
        for o in rep["pending_orders"]:
            console.print(f"[dim]broker pending order:[/] {o['scrip']} "
                          f"{o['side']} {o['qty']}@{o['price']}")
        for s in rep["skipped_not_in_universe"]:
            console.print(f"[yellow]skipped (not in universe):[/] {s}")
        for s in rep.get("released_sold", []):
            console.print(f"[red]released (no longer at broker):[/] {s}")
    rows = livebook.review()
    if not rows:
        console.print("Live book is empty — add a holding: "
                      "[bold]kairos livebook --add OGDC 1000 124.50[/]")
        return 0
    tbl = Table(box=box.ROUNDED, title="JS GLOBAL LIVE BOOK")
    for col in ("symbol", "qty", "entry", "last", "P&L", "call", "stop", "status"):
        tbl.add_column(col)
    for r in rows:
        col = "green" if (r.get("pnl") or 0) >= 0 else "red"
        tbl.add_row(r["symbol"], f"{r['qty']:,}", f"{r['entry']:,.2f}",
                    f"{r.get('last', 0):,.2f}", f"[{col}]{r.get('pnl', 0):+,.0f}[/]",
                    r.get("direction", "—"),
                    f"{(r.get('stop') or r.get('rec_stop') or 0):,.2f}",
                    r.get("status", "—"))
    console.print(tbl)
    if args.ticket:
        t = livebook.make_sell_ticket(args.ticket)
        console.print(f"[cyan]ticket {t['ticket_id']} written — confirm it in the "
                      f"JS Global app[/]")
    return 0


def cmd_order(args) -> int:
    """JS Global orders — stage first, then ack with the broker's own PIN."""
    import getpass
    from . import jsgtrade
    from .jsgsync import JsGlobalError
    console = _console()

    def fail(msg: str) -> int:
        console.print(f"[red]{msg}[/]")
        return 2

    try:
        if args.list or not any((args.symbol, args.ack, args.drop,
                                 args.cancel, args.band)):
            st = jsgtrade.state()
            console.print(f"[bold]JS Global order desk[/]  "
                          f"allow_orders=[cyan]{st.get('allow_orders')}[/]  "
                          f"stored PIN=[cyan]{bool(st.get('has_pin'))}[/]  "
                          f"kill switch=[cyan]{st.get('kill_switch')}[/]  "
                          f"max notional={float(st.get('max_order_notional') or 0):,.0f}")
            rows = jsgtrade.pending()
            if not rows:
                console.print("nothing staged — stage with: "
                              "[bold]kairos order OGDC --side SELL --qty 50 "
                              "--price 124.50[/]")
            for r in rows:
                console.print(f"[yellow]{r['id']}[/] {r['ts']} {r['symbol']} "
                              f"{r['side']} {r['qty']}@{r['price']} "
                              f"{r['order_type']} {r['market']} — ack with: "
                              f"[bold]kairos order --ack {r['id']}[/]")
            return 0
        if args.band:
            b = jsgtrade.band(args.band, args.market)
            console.print(b if b.get("known") else
                          f"[yellow]no broker band returned for {args.band}[/]")
            return 0
        if args.cancel:
            out = jsgtrade.cancel(args.cancel)
            console.print(f"[{'green' if out['ok'] else 'red'}]cancel[/] "
                          f"{out['response'][:200]}")
            return 0 if out["ok"] else 2
        if args.drop:
            jsgtrade.drop(args.drop)
            console.print(f"[yellow]dropped[/] staged order {args.drop} "
                          "— nothing reached the broker")
            return 0
        if args.ack:
            pin = args.pin or getpass.getpass("Trading PIN: ")
            out = jsgtrade.ack(args.ack, pin)
        else:
            if not args.symbol or not args.qty or not args.price:
                return fail("need SYMBOL --qty N --price P")
            row = jsgtrade.stage(args.symbol, args.side, args.qty, args.price,
                                 args.order_type, args.market, note="cli")
            console.print(f"[cyan]staged[/] {row['id']} {row['symbol']} "
                          f"{row['side']} {row['qty']}@{row['price']} "
                          f"{row['order_type']} {row['market']}")
            if not args.now:
                console.print("nothing sent yet — "
                              f"[bold]kairos order --ack {row['id']}[/]")
                return 0
            out = jsgtrade.ack(row["id"])      # stored PIN path
        tag = "green" if out["ok"] else "red"
        console.print(f"[{tag}]{'ACCEPTED' if out['ok'] else 'NOT ACCEPTED'}[/] "
                      f"{out['symbol']} {out['side']} {out['qty']}@{out['price']} "
                      f"order_no={out.get('order_no')} acct={out.get('account')}")
        console.print(f"[dim]{out['response'][:200]}[/]")
        return 0 if out["ok"] else 2
    except JsGlobalError as e:
        return fail(str(e))


def cmd_fugu(args) -> int:
    """Fugu conductor: fused multi-expert verdicts with probability-weighted edge."""
    from rich.table import Table
    from . import fugu
    console = _console()
    rows = fugu.board([args.symbol.upper()] if args.symbol else None)
    tbl = Table(box=box.ROUNDED, title="FUGU CONDUCTOR")
    for col in ("symbol", "verdict", "call", "conviction", "bar", "p(win)",
                "edge", "R:R", "why"):
        tbl.add_column(col)
    for r in rows:
        col = "green" if r["edge_r"] > 0 else "red"
        tbl.add_row(r["symbol"],
                    f"[green]{r['action']}[/]" if r["action"] == "EXECUTE"
                    else r["action"],
                    r["direction"], f"{r['conviction']:.2f}", f"{r['threshold']:.2f}",
                    f"{r['p_hit']:.0%}", f"[{col}]{r['edge_r']:+.2f}R[/]",
                    f"{r['rr']:.1f}", "; ".join(r["reasons"][:2]))
    console.print(tbl)
    cal = rows[0]["calibration"] if rows else {}
    console.print(f"[dim]calibration anchor: hit-rate {cal.get('hit_rate') or '—'} "
                  f"over {cal.get('n_resolved') or 0} resolved calls · entry bar adapts "
                  f"with realized accuracy[/]")
    return 0


def cmd_share(args) -> int:
    """Render a shareable live signal card (WhatsApp/Instagram)."""
    from rich.panel import Panel
    from . import signalcard
    console = _console()
    console.print(f"[bold blue]building live signal card for {args.symbol}…[/]")
    path, cap = signalcard.card_for(args.symbol, args.out)
    console.print(Panel(cap, title="KAIROS LIVE SIGNAL", style="blue"))
    console.print(f"[green]card saved:[/] {path}")
    return 0


def cmd_feedback(args) -> int:
    """Resolve elapsed predictions and print the realized track record."""
    from rich.panel import Panel
    from rich.table import Table
    from .engine import ENGINE
    from .feedback import confidence_adjustment, resolve_outcomes, track_record
    console = _console()
    newly = resolve_outcomes(ENGINE.md)
    tr = track_record()
    console.print(BANNER, style="bold blue")
    if not tr.get("ok"):
        console.print(Panel(f"[yellow]{tr.get('reason', 'nothing recorded')}[/]",
                            title="KAIROS TRACK RECORD"))
        return 0
    tbl = Table(title=f"KAIROS TRACK RECORD — {tr['resolved']}/{tr['predictions']} "
                      f"predictions resolved ({newly} newly)", box=box.ROUNDED)
    for col in ("metric", "value"):
        tbl.add_column(col)
    tbl.add_row("hit rate (direction)", "n/a" if tr["hit_rate"] is None
                else f"{tr['hit_rate']:.1%}")
    tbl.add_row("strong-signal hit rate", "n/a" if tr.get("strong_signal_hit_rate") is None
                else f"{tr['strong_signal_hit_rate']:.1%}")
    tbl.add_row("mean abs return error", "n/a" if tr.get("mean_abs_error") is None
                else f"{tr['mean_abs_error']:.4f}")
    tbl.add_row("live confidence multiplier", f"{confidence_adjustment():.2f}×")
    cal = tr.get("calibration_by_confidence") or {}
    if cal:
        tbl.add_row("calibration by confidence",
                    " · ".join(f"{k}: {v:.0%}" for k, v in cal.items()))
    console.print(tbl)
    for e in (tr.get("recent") or [])[-5:]:
        mark = "[green]✓[/]" if e.get("correct") else "[red]✗[/]"
        console.print(f"  {mark} {e['symbol']:<8} {e['direction']:<12} "
                      f"exp {e['expected_return']:+.2%} → realized "
                      f"{e.get('realized_return', 0):+.2%}")
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
    sp.add_argument("--logfile", default=None,
                    help="route stdout/stderr here with size-capped rotation (daemon mode)")
    sp.add_argument("--public-url", default=None, help="print the tunnel URL on boot")
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

    sp = sub.add_parser("mt5", help="MT5 bridge check + quote latency benchmark")
    sp.add_argument("--symbols", nargs="*")
    sp.add_argument("--benchmark", action="store_true")
    sp.set_defaults(fn=cmd_mt5)

    sp = sub.add_parser("autopilot", help="autonomous scan-and-trade loop")
    sp.add_argument("what", nargs="?", default="status",
                    choices=["start", "stop", "status", "configure", "cycle"])
    sp.add_argument("--mode", default="paper", choices=["paper", "live"])
    sp.add_argument("--symbols", nargs="*")
    sp.add_argument("--interval", type=int)
    sp.add_argument("--threshold", type=float)
    sp.add_argument("--max-trades", type=int)
    sp.add_argument("--yes-live", action="store_true",
                    help="explicitly arm live MT5 execution (real money)")
    sp.set_defaults(fn=cmd_autopilot)

    sp = sub.add_parser("rules", help="show effective trading doctrine")
    sp.add_argument("--wisdom", action="store_true",
                    help="also print the Wyckoff/Dow/CANSLIM/Buffett/Dalio/Lynch corpus")
    sp.set_defaults(fn=cmd_rules)

    sp = sub.add_parser("share", help="shareable live signal card (PNG + caption)")
    sp.add_argument("symbol")
    sp.add_argument("-o", "--out",
                    help="output PNG path (default: data/cards/SYMBOL.png)")
    sp.set_defaults(fn=cmd_share)

    sp = sub.add_parser("fugu", help="Fugu conductor verdicts (fused experts + edge)")
    sp.add_argument("symbol", nargs="?", help="one instrument, or the watchlist board")
    sp.set_defaults(fn=cmd_fugu)

    sp = sub.add_parser("livebook",
                        help="watch your REAL JS Global holdings (verdicts + exit tickets)")
    sp.add_argument("--add", nargs=3, metavar=("SYMBOL", "QTY", "ENTRY"),
                    help="add a holding, e.g. --add OGDC 1000 124.50")
    sp.add_argument("--remove", metavar="SYMBOL")
    sp.add_argument("--sync", action="store_true",
                    help="log into Web Trade Cast (read-only) and import real holdings")
    sp.add_argument("--ticket", metavar="SYMBOL",
                    help="write a broker-ready JS Global SELL ticket")
    sp.set_defaults(fn=cmd_livebook)

    sp = sub.add_parser("order",
                        help="JS Global orders: stage, ack with the broker PIN, "
                             "cancel (real money — needs allow_orders + PIN)")
    sp.add_argument("symbol", nargs="?", help="scrip to stage, e.g. OGDC")
    sp.add_argument("--side", default="BUY", choices=["BUY", "SELL", "SHORT"])
    sp.add_argument("--qty", type=int, default=0, help="whole shares")
    sp.add_argument("--price", type=float, default=0.0, help="PKR limit price")
    sp.add_argument("--type", dest="order_type", default="LIMIT",
                    choices=["LIMIT", "MARKET", "STOP_LOSS", "FOK", "MIT"])
    sp.add_argument("--market", default=None,
                    help="PSX segment (default: js_global.market, i.e. REG)")
    sp.add_argument("--now", action="store_true",
                    help="ack straight away using the PIN stored in config")
    sp.add_argument("--ack", metavar="ID", help="release a staged order")
    sp.add_argument("--drop", metavar="ID", help="discard a staged order")
    sp.add_argument("--cancel", metavar="ORDER_NO",
                    help="cancel an order already live at the broker")
    sp.add_argument("--band", metavar="SYMBOL",
                    help="show the broker's upper cap / lower lock for a scrip")
    sp.add_argument("--pin", default=None,
                    help="trading PIN (omit to be prompted — keeps it out of "
                         "your shell history)")
    sp.add_argument("--list", action="store_true", help="gate status + staged orders")
    sp.set_defaults(fn=cmd_order)

    sp = sub.add_parser("feedback", help="resolve outcomes + realized track record")
    sp.set_defaults(fn=cmd_feedback)

    args = p.parse_args()
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
