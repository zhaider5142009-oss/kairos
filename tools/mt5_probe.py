"""Measure what the MT5 link actually costs, in microseconds and milliseconds.

KAIROS reaches a broker through the desktop terminal's inter-process pipe, not a
socket: the Python call goes to `terminal64.exe`, and the terminal talks to the
trade server. That means two different numbers people routinely confuse —

  * **ipc**: how long a KAIROS call takes to get an answer out of the local
    terminal (this is the one a strategy loop pays on every tick),
  * **server**: how far the terminal itself is from the broker (reported by the
    terminal as `ping`, in microseconds).

Only the first is ours to optimise, and neither is a millisecond-order claim
until it has been measured on the account it will run on. This tool prints both,
plus symbol discovery and history fetch timings, and exits non-zero if anything
refused — it reads, it never sends an order.

    python tools/mt5_probe.py            # attach + measure
    python tools/mt5_probe.py --symbols 50 --bars 500
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):        # redirected output is cp1252 on Windows
    sys.stdout.reconfigure(encoding="utf-8")

from kairos.config import CFG                 # noqa: E402
from kairos.execution import Mt5Bridge        # noqa: E402


def _ms(samples: list[float]) -> str:
    if not samples:
        return "no samples"
    out = f"median {statistics.median(samples) * 1000:7.3f} ms"
    if len(samples) > 4:
        s = sorted(samples)
        out += f" | p95 {s[int(0.95 * (len(s) - 1))] * 1000:7.3f} ms"
    return out + f" | n {len(samples)}"


def _time(fn, repeats: int):
    """Return (result, [durations]) — one entry per repeat, timings only."""
    spans: list[float] = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        spans.append(time.perf_counter() - t0)
    return result, spans


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="measure the MT5 link (read-only)")
    ap.add_argument("--symbols", type=int, default=20,
                    help="how many discovered symbols to probe for a tick")
    ap.add_argument("--bars", type=int, default=200,
                    help="how many candles to pull for the history measurement")
    ap.add_argument("--repeat", type=int, default=7,
                    help="samples per measurement (median/p95 are noise-prone below 5)")
    a = ap.parse_args(argv)

    m = CFG.section("mt5") or {}
    venue = str(m.get("venue") or "the configured MT5 broker")
    print(f"MT5 probe — venue '{venue}', terminal {m.get('terminal_path') or 'any running install'}")
    print(f"configured account: {'yes' if m.get('login') else 'no'} | "
          f"configured server: {m.get('server') or '(blank — attaches to whatever is signed in)'}")

    br = Mt5Bridge()
    t0 = time.perf_counter()
    ok = br.attach(login=m.get("login"), password=m.get("password"),
                   server=m.get("server"), terminal_path=m.get("terminal_path"))
    attach_s = time.perf_counter() - t0
    if not ok:
        print(f"\nATTACH REFUSED in {attach_s * 1000:.1f} ms: {br.last_error}")
        pf = br.preflight()
        for b in pf.get("blockers") or []:
            print(f"  • {b}")
        for t in pf.get("terminals") or []:
            print(f"  install: {t['id'][:8]} → {t['install'] or '?'}")
        print("nothing was measured. Open the terminal this bridge is configured for, "
              "sign the account in, enable Algo Trading, and run this again.")
        return 2
    print(f"attached in {attach_s * 1000:.0f} ms as account "
          f"{br.attached_account.get('login')} on {br.attached_account.get('server')}")

    mt5 = br._mt5
    info = None
    try:
        info = mt5.terminal_info()._asdict()
    except Exception:
        pass

    print("\n— what the terminal reports about its own link —")
    if info:
        ping = info.get("ping")
        for key in ("connected", "terminal_server_connected", "trade_allowed",
                    "numberOfJobs", "build"):
            if key in info:
                print(f"  {key:26} {info[key]}")
        if ping is not None:
            print(f"  server ping (terminal-reported, microseconds) {ping} "
                  f"= {ping / 1000:.2f} ms")
    else:
        print("  terminal_info() returned nothing — timings below are still ours")

    print("\n— KAIROS-side call cost (this is the loop's real overhead) —")
    _, spans = _time(lambda: mt5.account_info(), a.repeat)
    print(f"  account_info()   {_ms(spans)}")
    _, spans = _time(lambda: mt5.positions_get(), a.repeat)
    print(f"  positions_get()  {_ms(spans)}")
    _, spans = _time(lambda: mt5.orders_get(), a.repeat)
    print(f"  orders_get()     {_ms(spans)}")

    print("\n— symbol discovery —")
    allsyms, spans = _time(lambda: mt5.symbols_get(), 3)
    names = sorted({s.name for s in (allsyms or [])})
    print(f"  symbols_get() {_ms(spans)} — {len(names)} symbols visible to this account")
    if not names:
        print("  no symbols: the terminal shows none until the Market Watch list is "
              "populated or the contract codes are enabled")
    wanted = [n for n in names if any(k in n.upper() for k in
              ("GOLD", "SILVER", "CRUDE", "OIL", "USD", "INDEX", "US500", "BTC"))]
    print(f"  tradable candidates seen: {', '.join(wanted[:14]) or '(none matched)'}")

    ticks = []
    for n in names[:a.symbols]:
        t1 = time.perf_counter()
        tk = mt5.symbol_info_tick(n)
        span = time.perf_counter() - t1
        if tk is not None:
            ticks.append((n, span, getattr(tk, "ask", 0.0), getattr(tk, "bid", 0.0)))
    if ticks:
        print(f"\n— live ticks for {len(ticks)} symbols —")
        print(f"  symbol_info_tick() {_ms([s for _, s, _, _ in ticks])}")
        for n, _s, ask, bid in ticks[:8]:
            sp = (ask - bid) if ask and bid else 0.0
            print(f"  {n:14} ask {ask:<14g} bid {bid:<14g} spread {sp:.5g}")
    else:
        print("\nno tick could be read — the account sees no subscribed symbols")

    hist = names[:3] if names else []
    if hist:
        print("\n— history (candles) —")
        for n in hist:
            tf = getattr(mt5, "TIMEFRAME_H1", 16385)
            bars, spans = _time(
                lambda n=n, tf=tf: mt5.copy_rates_from(n, tf, time.time(), a.bars), 3)
            got = len(bars) if bars else 0
            print(f"  {n:14} H1 x{a.bars}: {_ms(spans)} — {got} bars returned")

    print("\nLatency here is the cost of our own calls. The round trip to the exchange "
          "belongs to the broker's servers and is printed above as the terminal's ping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
