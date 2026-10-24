"""ML model battery: tree ensembles + neural sequence models.

Philosophy: predict next-horizon *directional return*, validate with walk-forward
IC (information coefficient), and blend models by their validation skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import feature_panel
from .models_stat import ModelForecast, _direction


def _prepare_xy(df: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    panel = feature_panel(df)
    y_fwd = df["close"].shift(-horizon) / df["close"] - 1.0
    valid = y_fwd.notna() & panel.notna().all(axis=1)
    X = panel[valid].values.astype(np.float64)
    y = y_fwd[valid].values.astype(np.float64)
    return X, y, panel


def _walk_forward_splits(n: int, n_splits: int = 4, embargo: int = 5):
    n = max(n, 40)
    test_size = max(15, n // (n_splits + 2))
    for i in range(n_splits):
        end = n - (n_splits - i) * test_size
        if end - test_size - embargo < 30:
            continue
        yield 0, end - test_size - embargo, end - embargo, end


def _ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 8:
        return 0.0
    sy, sp = np.std(y_true), np.std(y_pred)
    if sy == 0 or sp == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _hit_rate(y_true: np.ndarray, y_pred_sign: np.ndarray) -> float:
    if len(y_true) < 8:
        return 0.5
    return float(np.mean(np.sign(y_true) == np.sign(y_pred_sign)))


# ---------------------------------------------------------------------------
# Tree/linear stack: directional classifiers with walk-forward validation.
# Classification avoids the degenerate constant-mean regression that kills
# tree models on noisy return targets; probability maps to expected return
# via the empirical up/down magnitude measured on validation folds.
# ---------------------------------------------------------------------------
def tree_stack_forecast(df: pd.DataFrame, horizon: int = 5) -> list[ModelForecast]:
    from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                                   ExtraTreesClassifier)
    from sklearn.linear_model import RidgeClassifier

    X, y, _ = _prepare_xy(df, horizon)
    if len(X) < 80:
        return []
    y_sign = np.sign(y)
    forecasts: list[ModelForecast] = []
    models = {
        "GBM": GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                          learning_rate=0.06, subsample=0.9,
                                          min_samples_leaf=10, random_state=7),
        "RandomForest": RandomForestClassifier(n_estimators=250, max_depth=6,
                                               min_samples_leaf=8, n_jobs=-1,
                                               random_state=7, class_weight="balanced"),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=250, max_depth=7,
                                           min_samples_leaf=8, n_jobs=-1,
                                           random_state=7, class_weight="balanced"),
        "Ridge": RidgeClassifier(alpha=1.0),
    }
    for name, model in models.items():
        try:
            hits, probs, mags = [], [], []
            for tr_s, tr_e, te_s, te_e in _walk_forward_splits(len(X)):
                ytr = y_sign[tr_s:tr_e]
                if len(np.unique(ytr)) < 2:
                    continue
                model.fit(X[tr_s:tr_e], ytr)
                p = model.predict(X[te_s:te_e])
                hits.append(_hit_rate(y[te_s:te_e], p))
                if hasattr(model, "predict_proba"):
                    probs.append(float(model.predict_proba(X[te_s:te_e])[:, 1].mean()))
                up_mask = y[tr_s:tr_e] > 0
                mags.append((float(np.mean(np.abs(y[tr_s:tr_e][up_mask]))) if up_mask.any() else 0.0,
                             float(np.mean(np.abs(y[tr_s:tr_e][~up_mask]))) if (~up_mask).any() else 0.0))
            if not hits:
                continue
            model.fit(X[:-horizon], y_sign[:-horizon])
            if hasattr(model, "predict_proba"):
                p_up = float(model.predict_proba(X[-1:].reshape(1, -1))[0, 1])
            else:
                d = float(model.decision_function(X[-1:].reshape(1, -1))[0])
                p_up = float(1 / (1 + np.exp(-d)))
            hit = float(np.mean(hits))
            up_mag = float(np.mean([m[0] for m in mags])) if mags else 0.0
            down_mag = float(np.mean([m[1] for m in mags])) if mags else 0.0
            # expected return = P(up)*E|up| - P(down)*E|down|
            exp_ret = float((2 * p_up - 1) * (up_mag + down_mag) / 2)
            edge = abs(hit - 0.5) * 2                       # 0..1 skill
            conf = float(np.clip(0.25 + edge * 0.9, 0.10, 0.90))
            forecasts.append(ModelForecast(
                f"ML-{name}", _direction(exp_ret), exp_ret, horizon, conf,
                f"walk-forward hit-rate={hit:.1%} P(up)={p_up:.2f} "
                f"E|up|={up_mag:.4f} E|down|={down_mag:.4f}"))
        except Exception:
            continue
    return forecasts


# ---------------------------------------------------------------------------
# Neural sequence models (torch, CPU, small & fast)
# ---------------------------------------------------------------------------
def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _make_sequences(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(lookback, len(values)):
        xs.append(values[i - lookback:i])
        ys.append(values[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _neural_forecast(df: pd.DataFrame, horizon: int, kind: str = "lstm") -> ModelForecast | None:
    if not _torch_available():
        return None
    import torch
    import torch.nn as nn

    close = df["close"].values.astype(float)
    if len(close) < 120:
        return None
    lookback = 30
    logp = np.log(close[-400:])
    rets = np.diff(logp)
    xs, ys = _make_sequences(rets, lookback)
    if len(xs) < 60:
        return None
    # scale
    sd = float(np.std(rets)) or 1e-6
    xs_t = torch.tensor(xs / sd).unsqueeze(-1)          # (N, L, 1)
    ys_t = torch.tensor(ys / sd).unsqueeze(-1)          # (N, 1)

    class Net(nn.Module):
        def __init__(self, hidden: int = 48, layers: int = 1):
            super().__init__()
            rnn = nn.LSTM if kind == "lstm" else nn.GRU
            self.rnn = rnn(input_size=1, hidden_size=hidden, num_layers=layers,
                           batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :])

    torch.manual_seed(7)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()
    n_train = int(len(xs_t) * 0.8)
    best_val, best_state, patience, bad = float("inf"), None, 12, 0
    for _ in range(80):
        net.train()
        opt.zero_grad()
        pred = net(xs_t[:n_train])
        loss = loss_fn(pred, ys_t[:n_train])
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        net.eval()
        with torch.no_grad():
            val = float(loss_fn(net(xs_t[n_train:]), ys_t[n_train:]))
        if val < best_val - 1e-6:
            best_val, best_state, bad = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    # recursive multi-step forecast with accumulated returns
    with torch.no_grad():
        seq_np = rets[-lookback:].copy()
        total = 0.0
        for _ in range(horizon):
            inp = torch.tensor((seq_np / sd).reshape(1, lookback, 1), dtype=torch.float32)
            nxt = float(net(inp)[0, 0]) * sd
            total += nxt
            seq_np = np.append(seq_np[1:], nxt)
    exp_ret = float(total)
    val_ic = 0.0
    with torch.no_grad():
        pv = net(xs_t[n_train:]).flatten().numpy() * sd
        tv = ys_t[n_train:].flatten().numpy() * sd
        val_ic = _ic(tv, pv)
    conf = float(np.clip(0.28 + abs(val_ic) * 1.5, 0.10, 0.90))
    label = "Neural-LSTM" if kind == "lstm" else "Neural-GRU"
    return ModelForecast(label, _direction(exp_ret), exp_ret, horizon, conf,
                         f"lookback=30 val_IC={val_ic:+.3f} val_loss={best_val:.5f}")


def neural_battery(df: pd.DataFrame, horizon: int = 5) -> list[ModelForecast]:
    out = []
    for kind in ("lstm", "gru"):
        try:
            f = _neural_forecast(df, horizon, kind)
            if f is not None:
                out.append(f)
        except Exception:
            continue
    return out
