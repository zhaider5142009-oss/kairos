"""
HTTP + WebSocket listener — FastAPI on localhost, ChatGPT-style UI served.

Endpoints:
    GET  /                        → SPA index.html (the ChatGPT-style UI)
    GET  /static/*                → CSS / JS assets
    GET  /health, /status, /pending, /universe, /news, /forces, /calendar, /models
    GET  /agents, /track, /log, /patterns, /quantum
    POST /confirm, /cancel, /cancel-all, /shutdown
    POST /chat                    → natural-language command (returns reply_html)
    POST /listen                  → one-shot voice capture (push-to-talk)
    WS   /ws                      → live event stream (all Dispatcher events)
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..chat import ChatProcessor
from ..symbols import universe as symbol_universe
from .dispatcher import Dispatcher, EventType

LOGGER = logging.getLogger("jarvis.http")

_UI_DIR = Path(__file__).resolve().parent.parent / "ui"


class _StaticCacheMiddleware:
    """Adds Cache-Control headers to everything under /static/."""

    def __init__(self, app, max_age: int = 86400):
        self.app = app
        self.max_age = max_age

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "").startswith("/static/"):
            async def _send(msg):
                if msg["type"] == "http.response.start":
                    existing = list(msg.get("headers", []))
                    if not any(k == b"cache-control" for k, _ in existing):
                        msg["headers"] = existing + [
                            [b"cache-control", f"public, max-age={self.max_age}".encode()]
                        ]
                await send(msg)
            await self.app(scope, receive, _send)
        else:
            await self.app(scope, receive, send)


class ConfirmBody(BaseModel):
    signal_id: str


class CancelBody(BaseModel):
    signal_id: str
    reason: str | None = None


class ChatBody(BaseModel):
    message: str


class ExecutionModeBody(BaseModel):
    mode: str
    confirm: bool = False


class HttpListener:
    """Runs FastAPI on a background thread + WebSocket fan-out."""

    def __init__(
        self,
        *,
        dispatcher: Dispatcher,
        enabled: bool = True,
        bind_host: str = "127.0.0.1",
        port: int = 8765,
        auth_token: str = "change-me-on-first-run",
        runtime=None,
    ) -> None:
        self.dispatcher = dispatcher
        self.enabled = enabled
        self.bind_host = bind_host
        self.port = int(port)
        self.auth_token = auth_token
        self.runtime = runtime

        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws_clients: set[WebSocket] = set()

        self.app = self._build_app()

    # --- public --------------------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            LOGGER.info("HTTP listener disabled in config")
            return
        if self._thread and self._thread.is_alive():
            return
        config = uvicorn.Config(
            self.app, host=self.bind_host, port=self.port,
            log_level="warning", lifespan="on",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._serve, name="jarvis-http", daemon=True)
        self._thread.start()
        LOGGER.info("HttpListener listening on http://%s:%d (UI + WS + REST)", self.bind_host, self.port)

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=3.0)

    def _serve(self) -> None:
        assert self._server is not None
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._server.serve())
        except Exception:
            LOGGER.exception("Uvicorn server crashed")

    def broadcast_event(self, event_type: EventType, payload: dict, source: str = "runtime") -> None:
        """Called from any thread (Dispatcher subscriber). Schedules WS push."""
        if not self._loop or not self._ws_clients:
            return
        msg = {"type": event_type.value, "payload": payload, "source": source}
        try:
            asyncio.run_coroutine_threadsafe(self._push_all(msg), self._loop)
        except Exception:
            LOGGER.debug("WS broadcast failed", exc_info=True)

    async def _push_all(self, msg: dict) -> None:
        dead: list[WebSocket] = []
        text = json.dumps(msg, default=str)
        for ws in list(self._ws_clients):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._ws_clients.discard(ws)

    # --- FastAPI app ---------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Jarvis Reborn — MT5 Algo Trader", version="2.0.0")
        token = self.auth_token

        def _check(authorization: str | None = None,
                   x_jarvis_token: str | None = Header(default=None)) -> None:
            supplied = x_jarvis_token or (authorization.split(" ", 1)[1] if authorization and authorization.lower().startswith("bearer ") else None)
            if supplied != token:
                raise HTTPException(status_code=401, detail="invalid token")

        # ----- UI (ChatGPT-style SPA) ----------------------------------------
        @app.get("/", response_class=HTMLResponse)
        def index() -> FileResponse:
            return FileResponse(_UI_DIR / "templates" / "index.html")

        # Mount static files (CSS, JS).
        app.mount("/static", StaticFiles(directory=str(_UI_DIR / "static")), name="static")

        # ----- WebSocket ------------------------------------------------------
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._ws_clients.add(websocket)
            LOGGER.info("WebSocket client connected (total=%d)", len(self._ws_clients))
            try:
                # Push initial status on connect (rich snapshot when available).
                if self.runtime is not None and hasattr(self.runtime, "status_snapshot"):
                    initial = self.runtime.status_snapshot()
                else:
                    initial = self.dispatcher.status()
                await websocket.send_text(json.dumps({
                    "type": "hello",
                    "payload": initial,
                    "source": "server",
                }, default=str))
                while True:
                    # Drain anything the client sends (heartbeats, etc.).
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            finally:
                self._ws_clients.discard(websocket)
                LOGGER.info("WebSocket client disconnected (total=%d)", len(self._ws_clients))

        # ----- REST endpoints (same as before + chat) ------------------------

        @app.get("/health")
        def health() -> dict[str, Any]:
            return {"ok": True}

        @app.get("/status")
        def status() -> dict[str, Any]:
            if self.runtime is not None and hasattr(self.runtime, "status_snapshot"):
                return self.runtime.status_snapshot()
            self.dispatcher.publish(EventType.STATUS, self.dispatcher.status(), source="http")
            return self.dispatcher.status()

        @app.get("/connectors")
        def connectors() -> dict[str, Any]:
            if self.runtime is None or not hasattr(self.runtime, "connectors_snapshot"):
                return {"total": 0, "categories": []}
            return self.runtime.connectors_snapshot()

        @app.get("/mt5")
        def mt5_endpoint() -> dict[str, Any]:
            if self.runtime is None or not hasattr(self.runtime, "mt5_snapshot"):
                return {"connected": False, "terminal": {}, "account": {}, "risk": {}}
            return self.runtime.mt5_snapshot()

        @app.get("/prices")
        def prices() -> dict[str, Any]:
            if self.runtime is None or not hasattr(self.runtime, "prices_snapshot"):
                return {"symbols": {}}
            return self.runtime.prices_snapshot()

        @app.get("/strategy")
        def strategy() -> dict[str, Any]:
            if self.runtime is None or not hasattr(self.runtime, "strategy_snapshot"):
                return {"enabled": False, "error": "strategy engine unavailable"}
            return self.runtime.strategy_snapshot()

        @app.post("/strategy/config")
        def strategy_config(body: dict = {},
                            x_jarvis_token: str | None = Header(default=None),
                            authorization: str | None = Header(default=None)) -> dict[str, Any]:
            _check(authorization, x_jarvis_token)
            if self.runtime is None or not hasattr(self.runtime, "strategy"):
                raise HTTPException(503, "runtime not ready")
            cfg = self.runtime.strategy.strategy_config
            keys = ("rsi_overbought", "rsi_oversold", "max_positions", "max_daily_loss",
                    "max_drawdown_pct", "hedge_enabled", "hedge_threshold_pct",
                    "confidence_weight")
            for k in keys:
                if k in body and body[k] is not None:
                    setattr(cfg, k, body[k])
            # map UI field names onto strategy config
            mapping = {
                "risk_per_trade": None, "mtf_enabled": None,  # handled below
            }
            if "risk_per_trade" in body:
                sc = self.runtime.strategy.position_manager.sizing_config
                sc.risk_per_trade_pct = float(body["risk_per_trade"])
            return {"ok": True, "config": self.runtime.strategy.get_status().get("config")}

        @app.get("/repos")
        def repos() -> dict[str, Any]:
            if self.runtime is None or not hasattr(self.runtime, "repo_snapshot"):
                return {"repos": [], "available": [], "probes": {}}
            return self.runtime.repo_snapshot()

        @app.get("/backtest")
        def backtest(bars: int = 400, symbol: str = "EURUSD") -> dict[str, Any]:
            """Run the built-in walk-forward backtester (no MT5 account needed)."""
            if self.runtime is None:
                raise HTTPException(503, "runtime not ready")
            try:
                import concurrent.futures
                from jarvis_predict.backtest import Backtester, BacktestConfig
                bt = BacktestConfig(bars=max(64, min(int(bars), 100000)),
                                    symbols=[str(symbol).upper()])
                def _run() -> dict:
                    return Backtester(self.runtime.config, bt).run()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    return ex.submit(_run).result(timeout=600)
            except Exception as e:
                LOGGER.exception("backtest failed")
                raise HTTPException(500, f"backtest failed: {e}")

        @app.get("/backtest/report")
        def backtest_report() -> dict[str, Any]:
            """Most recently saved backtest report (cache/backtest_report.json)."""
            try:
                from pathlib import Path
                cfg = self.runtime.config if self.runtime is not None else None
                cache_dir = cfg.section("paths")["cache_dir"] if cfg is not None \
                    else "cache"
                p = Path(cache_dir) / "backtest_report.json"
                if not p.is_file():
                    return {"report": None, "note": "no report yet — run GET /backtest"}
                import json
                return {"report": json.loads(p.read_text(encoding="utf-8", errors="replace"))}
            except Exception as e:
                raise HTTPException(500, f"report read failed: {e}")

        @app.get("/fleet")
        def fleet() -> dict[str, Any]:
            if self.runtime is None or not hasattr(self.runtime, "fleet_snapshot"):
                return {"models": [], "llm_default_cadence": 6}
            return self.runtime.fleet_snapshot()

        @app.get("/desk")
        def desk() -> dict[str, Any]:
            """Commodities research desk — live votes + step-by-step factors +
            crowd positioning read for each commodity."""
            if self.runtime is None or not hasattr(self.runtime, "desk_report"):
                return {"enabled": False}
            return self.runtime.desk_report()

        @app.get("/desk/validate")
        def desk_validate(days: int = 60, symbol: str = "XAUUSD") -> dict[str, Any]:
            """Historical, strictly no-lookahead edge check of the desk on the
            last N days (yfinance; pulls metal + force bars directly)."""
            if self.runtime is None or not hasattr(self.runtime, "desk"):
                raise HTTPException(503, "runtime not ready")
            try:
                import concurrent.futures
                def _run() -> dict:
                    return self.runtime.desk.desk_hit_rate(
                        days=max(30, min(int(days), 730)),
                        symbol=str(symbol).upper(),
                    )
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    return ex.submit(_run).result(timeout=600)
            except Exception as e:
                LOGGER.exception("desk validation failed")
                raise HTTPException(500, f"desk validation failed: {e}")

        @app.get("/desk/factors")
        def desk_factors(days: int = 250, symbol: str = "XAUUSD") -> dict[str, Any]:
            """Factor report card — per-research-factor 3d hit-rate over the
            last N days (strictly no-lookahead). Flags factors that are noise
            for this market so they can be disabled."""
            if self.runtime is None or not hasattr(self.runtime, "desk"):
                raise HTTPException(503, "runtime not ready")
            try:
                import concurrent.futures
                def _run() -> dict:
                    return self.runtime.desk.factor_attribution(
                        days=max(60, min(int(days), 730)),
                        symbol=str(symbol).upper(),
                    )
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    return ex.submit(_run).result(timeout=600)
            except Exception as e:
                LOGGER.exception("desk factor attribution failed")
                raise HTTPException(500, f"desk factor attribution failed: {e}")

        @app.get("/repos/lean")
        def repos_lean() -> dict[str, Any]:
            if self.runtime is None or not hasattr(self.runtime, "lean_snapshot"):
                return {"available": False}
            return self.runtime.lean_snapshot()

        @app.get("/repos/lean/backtests")
        def repos_lean_backtests() -> dict[str, Any]:
            try:
                from jarvis_predict.repos.lean_adapter import get_lean_engine
                return {"jobs": get_lean_engine().all_jobs()}
            except Exception:
                return {"jobs": []}

        @app.post("/repos/lean/backtest")
        def repos_lean_backtest(body: dict = {},
                                x_jarvis_token: str | None = Header(default=None),
                                authorization: str | None = Header(default=None)) -> dict[str, Any]:
            _check(authorization, x_jarvis_token)
            algorithm = body.get("algorithm") or body.get("algo") or "Main.py"
            try:
                from jarvis_predict.repos.lean_adapter import get_lean_engine
                jobs = get_lean_engine().launch_backtest(
                    algorithm=algorithm,
                    start=body.get("start"),
                    end=body.get("end"),
                    params=body.get("params"),
                )
                return {"job_id": jobs.job_id, "algorithm": algorithm, "status": jobs.status}
            except Exception as e:
                raise HTTPException(500, f"lean backtest launch failed: {e}")

        @app.get("/devices")
        def devices() -> dict[str, Any]:
            if self.runtime is None or not hasattr(self.runtime, "devices_snapshot"):
                return {"local": {}, "devices": [], "online": 0}
            return self.runtime.devices_snapshot()

        @app.post("/devices/register")
        def device_register(body: dict = {},
                            request: "Request" = None) -> dict[str, Any]:  # noqa: F821
            if self.runtime is None or not hasattr(self.runtime, "devices"):
                raise HTTPException(503, "runtime not ready")
            device_id = (body.get("device_id") or "").strip()[:64]
            if not device_id:
                raise HTTPException(422, "device_id required")
            client_host = ""
            if request is not None:
                client_host = getattr(request, "client", None).host if getattr(request, "client", None) else ""
            return self.runtime.devices.register(
                device_id=device_id,
                name=(body.get("name") or device_id)[:64],
                role=(body.get("role") or "peer")[:16],
                host=body.get("host") or client_host,
                port=int(body.get("port") or 0),
                user_agent=(body.get("user_agent") or "")[:128],
                capabilities=body.get("capabilities") or [],
            )

        @app.post("/devices/heartbeat")
        def device_heartbeat(body: dict = {}) -> dict[str, Any]:
            if self.runtime is None or not hasattr(self.runtime, "devices"):
                raise HTTPException(503, "runtime not ready")
            device_id = (body.get("device_id") or "").strip()[:64]
            ok = self.runtime.devices.heartbeat(device_id) if device_id else False
            return {"ok": ok, "online": self.runtime.devices.count_online()}

        @app.post("/devices/remove")
        def device_remove(body: dict = {},
                          x_jarvis_token: str | None = Header(default=None),
                          authorization: str | None = Header(default=None)) -> dict[str, Any]:
            _check(authorization, x_jarvis_token)
            if self.runtime is None or not hasattr(self.runtime, "devices"):
                raise HTTPException(503, "runtime not ready")
            device_id = (body.get("device_id") or "").strip()[:64]
            return {"removed": self.runtime.devices.remove(device_id) if device_id else False}

        @app.post("/kill-switch")
        def kill_switch(body: dict = {},
                        x_jarvis_token: str | None = Header(default=None),
                        authorization: str | None = Header(default=None)) -> dict[str, Any]:
            _check(authorization, x_jarvis_token)
            if self.runtime is None or not hasattr(self.runtime, "toggle_kill_switch"):
                raise HTTPException(503, "runtime not ready")
            enabled = body.get("enabled", body.get("paused"))
            return self.runtime.toggle_kill_switch(paused=enabled)

        @app.get("/execution/mode")
        def execution_mode(x_jarvis_token: str | None = Header(default=None),
                           authorization: str | None = Header(default=None)) -> dict[str, Any]:
            _check(authorization, x_jarvis_token)
            if self.runtime is None or not hasattr(self.runtime, "config"):
                raise HTTPException(503, "runtime not ready")
            mode = self.runtime.config.get_execution_mode()
            return {
                "mode": mode,
                "restarting": bool(getattr(self.runtime, "_restarting", False)),
                "supported": ["paper", "live"],
                "paper_active": mode == "paper",
            }

        @app.post("/execution/mode")
        def set_execution_mode(body: ExecutionModeBody,
                               x_jarvis_token: str | None = Header(default=None),
                               authorization: str | None = Header(default=None)) -> dict[str, Any]:
            _check(authorization, x_jarvis_token)
            if self.runtime is None or not hasattr(self.runtime, "config"):
                raise HTTPException(503, "runtime not ready")
            mode = (body.mode or "").lower()
            if mode not in ("paper", "live"):
                raise HTTPException(400, "mode must be 'paper' or 'live'")
            if mode == "live" and not body.confirm:
                raise HTTPException(400, "switching to LIVE execution requires confirm=true")
            current = self.runtime.config.get_execution_mode()
            if mode == current:
                return {"ok": True, "mode": mode, "restarting": False, "changed": False}
            # Persist, then bounce the process so the correct bridge boots cleanly.
            self.runtime.config.set_execution_mode(mode)
            self.runtime._restarting = True
            try:
                from ..restart import schedule_restart
                schedule_restart(
                    self.runtime.config.root,
                    self.runtime.config.cache_dir,
                    getattr(self.runtime.config, "path", None),
                )
            except Exception as e:
                self.runtime._restarting = False
                raise HTTPException(500, f"failed to schedule restart: {e}") from e
            LOGGER.warning("EXECUTION MODE -> %s (restart scheduled)", mode)
            return {"ok": True, "mode": mode, "restarting": True, "changed": True}

        @app.get("/pending")
        def pending() -> dict[str, Any]:
            return {"pending": self.dispatcher.pending_snapshot()}

        @app.get("/universe")
        def universe() -> dict[str, Any]:
            return {"universe": symbol_universe()}

        @app.get("/news")
        def news() -> dict[str, Any]:
            if self.runtime is None or not getattr(self.runtime, "news", None):
                return {"items": []}
            return {"items": self.runtime.news.snapshot()}

        @app.get("/forces")
        def forces() -> dict[str, Any]:
            if self.runtime is None or not getattr(self.runtime, "forces", None):
                return {"forces": {}}
            snaps = self.runtime.forces.snapshot()
            return {"forces": {k: v.to_dict() for k, v in snaps.items()}}

        @app.get("/calendar")
        def calendar() -> dict[str, Any]:
            if self.runtime is None or not getattr(self.runtime, "calendar", None):
                return {"upcoming": [], "blackouts": []}
            cal = self.runtime.calendar
            upcoming = [e.to_dict() for e in cal.upcoming(20)]
            blackouts = [
                {"start": w.start, "end": w.end, "event": w.event_name}
                for w in cal.blackouts()
            ]
            return {"upcoming": upcoming, "blackouts": blackouts}

        @app.get("/pipeline")
        def pipeline() -> dict[str, Any]:
            if self.runtime is None:
                return {"pipeline": {}}
            return {"pipeline": {
                "cycle_s": self.runtime.bridge.poll_seconds,
                "symbols": len(list(self.runtime.bridge.symbols)),
                "mtf_enabled": self.runtime.mtf is not None,
                "swarm_cache_ttl": self.runtime.swarm.cache_ttl,
                "learner_window": self.runtime.learner.window,
                "openrouter_keys": len(getattr(self.runtime.openrouter, "api_keys", []) or []),
            }}

        @app.get("/models")
        def models() -> dict[str, Any]:
            if self.runtime is None or not getattr(self.runtime, "fleet", None):
                return {"models": []}
            return {"models": self.runtime.fleet.status_table()}

        @app.get("/agents")
        def agents() -> dict[str, Any]:
            if self.runtime is None:
                return {"agents": [], "profile": {}}
            return self.runtime.agents_snapshot()

        @app.get("/track")
        def track() -> dict[str, Any]:
            if self.runtime is None:
                return {"track": {}}
            return self.runtime.learner_track()

        @app.get("/log")
        def log(lines: int = 80) -> dict[str, Any]:
            if self.runtime is None:
                return {"lines": []}
            n = max(10, min(500, int(lines)))
            return {"lines": list(getattr(self.runtime, "_log_ring", ()))[-n:]}

        @app.get("/patterns")
        def patterns(symbol: str | None = None) -> dict[str, Any]:
            if self.runtime is None:
                return {"symbol": symbol, "patterns": None}
            sym = symbol or ((self.runtime._last_swarm or {})).get("symbol")
            return {"symbol": sym,
                    "patterns": self.runtime.patterns_for(sym) if sym else None}

        @app.get("/quantum")
        def quantum(symbol: str | None = None) -> dict[str, Any]:
            if self.runtime is None:
                return {"symbol": symbol, "quantum": None}
            sym = symbol or ((self.runtime._last_swarm or {})).get("symbol")
            return {"symbol": sym,
                    "quantum": self.runtime.quantum_for(sym) if sym else None}

        @app.post("/confirm")
        def confirm(body: ConfirmBody,
                    x_jarvis_token: str | None = Header(default=None),
                    authorization: str | None = Header(default=None)) -> dict[str, Any]:
            _check(authorization, x_jarvis_token)
            r = self.dispatcher.confirm(body.signal_id, source="http")
            if r is None:
                raise HTTPException(404, "no such pending signal")
            return {"ok": r.success, "order_id": r.order_id, "comment": r.comment, "retcode": r.retcode}

        @app.post("/cancel")
        def cancel(body: CancelBody,
                   x_jarvis_token: str | None = Header(default=None),
                   authorization: str | None = Header(default=None)) -> dict[str, Any]:
            _check(authorization, x_jarvis_token)
            self.dispatcher.cancel(body.signal_id, source="http", reason=body.reason or "http:cancel")
            return {"ok": True}

        @app.post("/cancel-all")
        def cancel_all(x_jarvis_token: str | None = Header(default=None),
                       authorization: str | None = Header(default=None)) -> dict[str, Any]:
            _check(authorization, x_jarvis_token)
            n = self.dispatcher.cancel_all(source="http", reason="http:cancel-all")
            return {"cancelled": n}

        @app.post("/confirm-all")
        def confirm_all(x_jarvis_token: str | None = Header(default=None),
                        authorization: str | None = Header(default=None)) -> dict[str, Any]:
            _check(authorization, x_jarvis_token)
            n = self.dispatcher.confirm_all(source="http", reason="http:confirm-all")
            return {"confirmed": n}

        @app.post("/shutdown")
        def shutdown(x_jarvis_token: str | None = Header(default=None),
                     authorization: str | None = Header(default=None)) -> dict[str, Any]:
            _check(authorization, x_jarvis_token)
            self.dispatcher.publish(EventType.SHUTDOWN, {"source": "http"}, source="http")
            return {"ok": True}

        @app.post("/chat")
        def chat(body: ChatBody,
                 x_jarvis_token: str | None = Header(default=None),
                 authorization: str | None = Header(default=None)) -> dict[str, Any]:
            # Chat is intentionally lenient — no token required, but if a
            # token is configured, we accept it silently.
            cp = ChatProcessor(self.dispatcher, runtime=self.runtime)
            return cp.process(body.message)

        @app.post("/listen")
        async def listen() -> dict[str, Any]:
            """One-shot push-to-talk. Transcribes, runs the chat pipeline,
            returns the bubble HTML for the UI. Never raises."""
            if self.runtime is None:
                return {"text": "", "reply_html": "<p>Runtime not ready.</p>"}
            loop = asyncio.get_running_loop()
            try:
                text = await loop.run_in_executor(
                    None, self.runtime.voice.capture_once)
            except Exception as e:
                LOGGER.exception("one-shot listen failed")
                return {"text": "", "reply_html": f"<p>⚠ Voice capture failed: {e}</p>"}
            if not text:
                return {"text": "", "intent": "",
                        "reply_html": "<p class='hint'>🎙 Nothing heard (silence).</p>"}
            cp = ChatProcessor(self.dispatcher, runtime=self.runtime)
            result = cp.process(text)
            return {"text": text, **result}

        return _StaticCacheMiddleware(app)


__all__ = ["HttpListener"]
