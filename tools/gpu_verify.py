"""Measure the CPU/GPU split against the eager torch model it claims to reproduce.

`kairos/gpu_dml.py` carries a number in its docstring — a mean relative error
between the ONNX graph on the RX 560 and torch on the CPU. A number like that is
a claim about behaviour, and the split has since moved the input MLP from the CPU
into the graph, so the old measurement describes a model that no longer exists.
This tool produces the current one.

    python tools/gpu_verify.py [--batch 16] [--rows 4] [--keys KSE100,OGDC]

It reports three things and nothing else: the error between the graph and the
eager pipeline on real stored market data, the seconds per batch on each side,
and which execution provider answered. Memory is the limit on this box, so the
batch and row counts are arguments, not constants.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402


def _contexts(keys: list[str], ctx_len: int, rows: int) -> list[np.ndarray]:
    """Real closes from the local store, oldest first, trimmed to the model's window."""
    from kairos.data import STORE

    out: list[np.ndarray] = []
    for key in keys:
        df = STORE.load(key)
        if df is None or len(df) < 30 or "close" not in df.columns:
            print(f"  {key}: no usable stored history")
            continue
        out.append(np.asarray(df["close"], dtype=np.float32)[-ctx_len:])
        if len(out) >= rows:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rows", type=int, default=4,
                    help="series to compare (both sides hold a model in RAM)")
    ap.add_argument("--keys", default="BTC,ETH,SYS,FFC",
                    help="comma-separated store keys to read real closes from")
    args = ap.parse_args()

    import torch
    from chronos import ChronosBoltPipeline

    from kairos.gpu_dml import GpuForecaster

    print(f"loading amazon/chronos-bolt-{args.model}: the graph, then eager torch")
    t0 = time.time()
    f = GpuForecaster(model=args.model, batch=args.batch)
    t1 = time.time()
    ref = ChronosBoltPipeline.from_pretrained(
        f.repo, torch_dtype=torch.float32, local_files_only=True)
    ref.model.eval()
    t2 = time.time()
    print(f"  graph session : {t1 - t0:.1f} s  (provider {f.provider})")
    print(f"  eager torch   : {t2 - t1:.1f} s")
    print(f"  graph file    : {f.onnx_path.name}")

    ctxs = _contexts([k.strip().upper() for k in args.keys.split(",")],
                     f.ctx_len, args.rows)
    if not ctxs:
        print(f"no stored history for {args.keys}; run `kairos fetch` first")
        return 1
    print(f"  {len(ctxs)} real series x {f.ctx_len} closes, last = "
          f"{[round(float(c[-1]), 2) for c in ctxs]}")

    g = f.forecast(ctxs)                       # graph, in real units
    graph_s = f.timing["dml"] / max(f.timing["batches"], 1)

    # The library's own path is the reference, not `model.encode()`: that is the
    # half the split moved, and `predict()` is what a CPU-only KAIROS would have
    # answered for these same series. It returns (n, quantiles, horizon), the same
    # axis order `forecast()` promises.
    #
    # Both sides get the same row count: `forecast()` pads the batch up to `--batch`
    # because the exported graph has a static token count, so timing four series
    # against a sixteen-row graph would flatter the CPU and be reported as a
    # comparison of the two.
    padded = f._pad_contexts(ctxs)
    rows = padded.shape[0]

    e0 = time.time()
    with torch.no_grad():
        e = ref.predict([torch.from_numpy(r) for r in padded],
                        prediction_length=f.pred_len)
    eager_s = time.time() - e0
    e = e.detach().cpu().numpy()[:len(ctxs)]

    # The work the split removed from the CPU: the 12-layer T5 encoder running
    # eagerly so its output could be handed to a graph that ran it again.
    c0 = time.time()
    with torch.no_grad():
        f.model.encode(context=torch.from_numpy(np.ascontiguousarray(padded)),
                       mask=None)
    encode_s = time.time() - c0

    if e.shape != g.shape:
        print(f"SHAPE MISMATCH: graph {g.shape} vs eager {e.shape}")
        return 1
    rel = np.abs(g - e) / np.maximum(np.abs(e), 1e-6)
    print(f"\nmean relative error : {rel.mean():.3e}")
    print(f"max  relative error : {rel.max():.3e}")
    print(f"max  absolute error : {np.abs(g - e).max():.3e}")
    print(f"per batch of {rows} rows, graph : {graph_s:.3f} s  ({f.provider})")
    print(f"per batch of {rows} rows, eager : {eager_s:.3f} s  (torch CPU)")
    print(f"the encoder pass the CPU no longer runs : {encode_s:.3f} s")
    print(f"timing totals       : {f.health()['timing']}")

    ok = rel.mean() < 1e-3
    print(f"\n{'MATCH' if ok else 'DIVERGENT'} at {rel.mean():.3e} mean relative error")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
