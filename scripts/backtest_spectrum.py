"""Full-spectrum adaptive validation — writes reports/backtest_spectrum.json."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kairos.data import MarketData
from kairos.universe import resolve
from kairos.backtest import walk_forward

SYMBOLS = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "GOLD", "SILVER", "WTI", "BRENT", "NATGAS", "COPPER", "WHEAT",
    "EURUSD", "BTC", "KSE100", "OGDC", "HBL", "LUCK", "ENGRO", "MCB"]


def main() -> None:
    md = MarketData()
    out = {}
    for sym in SYMBOLS:
        try:
            df = md.history(resolve(sym))
            if df is None or len(df) < 140:
                out[sym] = {"error": f"insufficient history ({0 if df is None else len(df)} bars)"}
                print(f"{sym}: insufficient history", flush=True)
                continue
            t0 = time.time()
            bt = walk_forward(df, sym)
            out[sym] = {"verdict": bt.verdict, "return": bt.total_return,
                        "benchmark": bt.benchmark_return, "sharpe": bt.sharpe,
                        "hit_rate": bt.hit_rate, "trades": bt.trades,
                        "max_dd": bt.max_drawdown, "exposure": bt.exposure,
                        "params": bt.params, "calibrations": bt.calibrations}
            print(f"{sym}: {bt.verdict} ret={bt.total_return:+.1%} bench={bt.benchmark_return:+.1%} "
                  f"sharpe={bt.sharpe} hit={bt.hit_rate} trades={bt.trades} "
                  f"dd={bt.max_drawdown:.1%} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as exc:
            out[sym] = {"error": str(exc)}
            print(f"{sym}: ERROR {exc}", flush=True)

    profitable = sum(1 for r in out.values() if r.get("verdict") == "PROFITABLE")
    marginal = sum(1 for r in out.values() if r.get("verdict") == "MARGINAL")
    summary = {"instruments": len(out), "profitable": profitable,
               "marginal": marginal,
               "unprofitable": len(out) - profitable - marginal}
    dst = Path(__file__).resolve().parent.parent / "reports" / "backtest_spectrum.json"
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(json.dumps({"summary": summary, "results": out}, indent=2))
    print(f"\nsummary: {summary}", flush=True)


if __name__ == "__main__":
    main()
