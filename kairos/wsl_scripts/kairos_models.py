#!/usr/bin/env python3
"""KAIROS WSL-side foundation-model server (stdin JSON -> stdout JSON).

Runs inside a WSL distro under the venv set by `wsl.distro` / `wsl.venv_path`
(config.yaml; defaults Ubuntu + ~/.kairos-venv).
Models are imported lazily; failures degrade to {"ok": false} on stdout so the
Windows side can fall back to its native battery without crashing.
"""

import json
import sys


def _emit(payload):
    print(json.dumps(payload))
    sys.stdout.flush()


def run_chronos(series, horizon):
    import torch
    from chronos import ChronosPipeline
    pipe = ChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base", device_map="cpu", torch_dtype=torch.float32)
    ctx = torch.tensor(series, dtype=torch.float32)
    fcst = pipe.predict(ctx, prediction_length=horizon)
    path = fcst[0].numpy().tolist()
    q10 = float(sorted(path)[max(0, int(0.1 * len(path)) - 1)])
    q50 = float(sorted(path)[len(path) // 2])
    q90 = float(sorted(path)[min(len(path) - 1, int(0.9 * len(path)))])
    point = float(path[-1])
    sigma = max(abs(q90 - q10) / 2.0, 1e-9)
    conf = max(0.2, min(0.9, 0.5 + 0.25 * abs(q50 - series[-1]) / sigma))
    return {"ok": True, "model": "chronos", "point": point, "path": path,
            "q10": q10, "q50": q50, "q90": q90, "sigma": sigma,
            "confidence": conf, "backend": "wsl-chronos-bolt-base"}


def run_timesfm(series, horizon):
    import torch
    from timesfm.timesfm_base import TimesFmHparams
    from timesfm.timesfm_torch import TimesFmTorch
    tfm = TimesFmTorch(hparams=TimesFmHparams(
        backend="gpu" if torch.cuda.is_available() else "cpu",
        per_core_batch_size=32, horizon_len=horizon,
        context_len=512))
    tfm.load_from_checkpoint("google/timesfm-2.5-200m-pytorch")
    point, exp = tfm.forecast([series], freq=[0])
    path = list(map(float, point[0]))
    sigma = float(max(abs(exp[0][0].numpy().mean()), 1e-9)) if hasattr(exp[0][0], "numpy") else 0.02
    q_lo = [p - 1.28 * sigma for p in path]
    q_hi = [p + 1.28 * sigma for p in path]
    conf = max(0.2, min(0.9, 0.5 + 0.25 * abs(path[-1] - series[-1]) / sigma))
    return {"ok": True, "model": "timesfm", "point": path[-1], "path": path,
            "q10": q_lo[-1], "q50": path[-1], "q90": q_hi[-1], "sigma": sigma,
            "confidence": conf, "backend": "wsl-timesfm-2.5-200m"}


def run_qlib(series, horizon):
    """qlib-style LightGBM on engineered features from the raw series.
    Uses qlib's LGBModel when qlib is importable, else plain lightgbm."""
    return _lgb_forecast(series, horizon)


def _lgb_forecast(series, horizon):
    import numpy as np
    import pandas as pd
    s = pd.Series(series, dtype=float)
    if len(s) < 80:
        return {"ok": False, "error": "insufficient history"}
    df = pd.DataFrame({"close": s})
    for lag in (1, 2, 3, 5, 10, 21):
        df[f"ret_{lag}"] = s.pct_change(lag)
    df["ma5"] = s.rolling(5).mean() / s - 1
    df["ma20"] = s.rolling(20).mean() / s - 1
    df["vol20"] = s.pct_change().rolling(20).std()
    df["rsi14"] = _rsi(s)
    df["target"] = s.shift(-horizon) / s - 1
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feats = [c for c in df.columns if c not in ("close", "target")]
    train = df.iloc[:-horizon]
    X, y = train[feats].values, train["target"].values
    model = None
    try:
        from qlib.contrib.model.gbdt import LGBModel
        model = LGBModel(loss="mse", num_leaves=15, learning_rate=0.05)
        model.fit(X, y)
        pred = float(model.predict(df[feats].values[-1:])[0])
        backend = "wsl-qlib-lgbm"
    except Exception:
        pass
    if model is None:
        try:
            import lightgbm as lgb
            booster = lgb.train({"objective": "regression", "verbosity": -1,
                                 "learning_rate": 0.05, "num_leaves": 15},
                                lgb.Dataset(X, label=y), num_boost_round=150)
            pred = float(booster.predict(df[feats].values[-1:])[0])
            backend = "wsl-lightgbm"
        except Exception:
            return {"ok": False, "error": "neither qlib nor lightgbm in WSL"}
    sigma = float(np.std(y[-60:]) or 0.02)
    conf = max(0.2, min(0.85, 0.5 + 0.2 * abs(pred) / max(sigma, 1e-9)))
    return {"ok": True, "model": "qlib-lgbm", "point": float(series[-1] * (1 + pred)),
            "exp_return": pred, "sigma": sigma, "confidence": conf,
            "backend": backend}


def _rsi(s, n=14):
    import numpy as np
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n).mean()
    dn = (-d).clip(lower=0).ewm(alpha=1 / n).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50)


def main():
    try:
        payload = json.loads(sys.stdin.read())
    except Exception as exc:
        _emit({"ok": False, "error": f"stdin parse: {exc}"})
        return
    model = payload.get("model")
    series = [float(x) for x in payload.get("series", []) if x is not None]
    horizon = int(payload.get("horizon", 5))
    if len(series) < 40:
        _emit({"ok": False, "error": "series too short"})
        return
    try:
        if model == "chronos":
            _emit(run_chronos(series, horizon))
        elif model == "timesfm":
            _emit(run_timesfm(series, horizon))
        elif model == "qlib":
            _emit(run_qlib(series, horizon))
        else:
            _emit({"ok": False, "error": f"unknown model {model}"})
    except Exception as exc:
        _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
