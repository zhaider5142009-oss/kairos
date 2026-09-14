"""
Runtime orchestrator — async, parallel, fully wired.

Lifecycle:
    start() → connect MT5 → load every model → start listeners → main loop
    main loop: every cycle, in parallel:
        - tick stream + on-demand forecast
        - per-symbol: candles → indicators → patterns → all 12 models
                     → quantum → news → intermarket → calendar → ensemble
                     → signal → stage
    stop()  → graceful teardown of every subsystem
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal as posix_signal
import threading
import time
from collections import deque
from pathlib import Path
from typing import Sequence

import numpy as np

from .agents import AgentSwarm, ComputeRouter, ModelLearner, SwarmContext, scan
from .async_mt5 import AsyncMT5Bridge
from .calendar import EconomicCalendar
from .config import JarvisConfig, setup_logging
from .connectors import ConnectorRegistry
from .core import TimesFMEngine
from .data_pipeline import DataPipeline
from .ensemble import EnsembleFusion, MarketContext
from .executor import TradeExecutor
from .indicators import TechnicalIndicators
from .listener import Dispatcher, HttpListener, MT5EventListener, VoiceListener
from .listener.dispatcher import EventType
from .listener.hotkeys import GlobalHotkeys
from .market_forces import MarketForcesService
from .meta import MetaOrchestrator
from .models import build_fleet, specs_from_config
from .mt5_bridge import MT5Bridge, JarvisMT5Error
from .multi_timeframe import MultiTimeframePredictor, DEFAULT_TIMEFRAMES
from .strategy import Timeframe
from .news import NewsService
from .openrouter import OpenRouterClient
from .parallel import ParallelFleet
from .patterns import detect_patterns
from .quantum import QuantumDecisionEngine
from .signals import SignalGenerator
from .strategy import StrategyEngine, create_engine
from .tray import TrayUI

LOGGER = logging.getLogger("jarvis.runtime")


class _LogRingHandler(logging.Handler):
    """Appends formatted records to a deque — powers the dashboard log view."""

    def __init__(self, ring: deque) -> None:
        super().__init__()
        self._ring = ring

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._ring.append(self.format(record))
        except Exception:
            pass


def _fleet_consensus(model_forecasts) -> dict:
    """Signed-vote summary of the model fleet (feeds the FleetBarometer agent)."""
    ok = [f for f in model_forecasts if getattr(f, "status", "") == "ok"]
    if not ok:
        return {"signed_vote": 0.0, "agreement": 0.0, "ok": 0}
    votes = [float(f.direction) * float(f.confidence) for f in ok]
    pos = sum(1 for v in votes if v > 0)
    neg = sum(1 for v in votes if v < 0)
    total = max(1, len(votes))
    agreement = max(pos, neg) / total
    signed = sum(votes) / total
    return {"signed_vote": round(signed, 4), "agreement": round(agreement, 3), "ok": len(ok)}


class JarvisRuntime:
    """Top-level orchestrator. One per process. Async-first."""

    def __init__(self, config: JarvisConfig) -> None:
        self.config = config
        self._stop = asyncio.Event()

        # --- predictive stack ------------------------------------------------
        self.engine = TimesFMEngine.from_config(config)
        self.indicators = TechnicalIndicators.from_config(config)
        self.quantum = QuantumDecisionEngine(seed=config.section("quantum").get("seed"))
        self.signals = SignalGenerator.from_config(config)

        # --- data + market context ------------------------------------------
        self.data = DataPipeline(config=config)
        self.news = NewsService.from_config(config)
        self.forces = MarketForcesService.from_config(config)
        self.calendar = EconomicCalendar.from_config(config)

        # --- MT5 sync + async wrappers --------------------------------------
        self.bridge = MT5Bridge.from_config(config)
        self.async_bridge = AsyncMT5Bridge(self.bridge)

        # --- OpenRouter client (LLM reasoning over the model fleet) ---------
        or_cfg = config.section("openrouter")
        api_keys = list(or_cfg.get("api_keys", []))
        # Allow env-var override (preferred for security).
        env_keys = os.environ.get("OPENROUTER_API_KEYS", "")
        if env_keys:
            api_keys = [k.strip() for k in env_keys.split(",") if k.strip()] + api_keys
        self.openrouter = OpenRouterClient(
            api_keys=api_keys or None,
            base_url=or_cfg.get("base_url", "https://openrouter.ai/api/v1"),
            timeout=float(or_cfg.get("timeout", 60.0)),
            max_retries=int(or_cfg.get("max_retries", 4)),
            site_url=or_cfg.get("site_url", "https://jarvis.local"),
            site_name=or_cfg.get("site_name", "Jarvis Reborn — MT5 Algo Trader"),
        )

        # --- meta-orchestrator (OpenRouter-backed reasoning) -----------------
        meta_cfg = or_cfg.get("meta", {})
        self.meta = MetaOrchestrator(
            client=self.openrouter,
            model=meta_cfg.get("model", "deepseek/deepseek-chat"),
            fallback_models=meta_cfg.get("fallback_models"),
            max_tokens=int(meta_cfg.get("max_tokens", 2048)),
            prefer_openrouter=bool(meta_cfg.get("prefer_openrouter", True)
                                   and len(api_keys) > 0),
        )

        # --- 12-model fleet + parallel executor -----------------------------
        # The OpenRouter client is injected into LLM-backed adapters (deepseek).
        model_specs = specs_from_config(config.section("models"))
        self.fleet = build_fleet(model_specs, client=self.openrouter)
        self.parallel = ParallelFleet(self.fleet.models)

        # --- connector layer (120 integrations across 12 categories) --------
        self.connectors = ConnectorRegistry.from_config(config)

        # --- local quant/trading repo fabric (qlib, FinGPT, FinRobot, RD-Agent, LEAN, FinRL ...) --
        from .repos import RepoFabric
        repo_cfg = config.section("repos")
        self.repo_fabric = RepoFabric(
            enabled=list(repo_cfg.get("enabled_list", []) or []) or None,
            max_probe_workers=int(repo_cfg.get("max_probe_workers", 5)),
        )
        self.repo_probes = {}
        self._probe_repos_async()

        # --- cross-device registry (phones/tablets/other PCs on the LAN) ----
        from .devices import DeviceRegistry, default_state_path, local_device_info
        dev_cfg = config.section("devices")
        self.devices = DeviceRegistry(
            state_path=dev_cfg.get("state_path") or default_state_path()
        )
        self.device_info = local_device_info()
        self._register_local_device()

        # --- dispatcher (event bus) before strategy engine binds to it ---------
        self.dispatcher = Dispatcher()

        # --- strategy engine (pattern recognition, MTF, hedging, position mgmt) --
        strategy_cfg = config.section("strategy")
        self.strategy = create_engine(
            config={
                "rsi_overbought": strategy_cfg.get("rsi_overbought", 70),
                "rsi_oversold": strategy_cfg.get("rsi_oversold", 30),
                "max_positions": strategy_cfg.get("max_positions", 5),
                "max_daily_loss": strategy_cfg.get("max_daily_loss", 50.0),
                "max_drawdown_pct": strategy_cfg.get("max_drawdown_pct", 10.0),
                "hedge_enabled": strategy_cfg.get("hedge_enabled", True),
                "hedge_threshold_pct": strategy_cfg.get("hedge_threshold_pct", 5.0),
                "confidence_weight": strategy_cfg.get("confidence_weight", 0.6),
            },
            mt5_bridge=self.bridge,
            timesfm_engine=self.engine,
            dispatcher=self.dispatcher
        )

        # --- executor + ensemble fusion --------------------------------------
        self.executor = TradeExecutor.from_config(
            config, self.bridge,
            on_stage=self._on_stage,
            on_result=self._on_result,
        )
        self.dispatcher.executor = self.executor
        self.dispatcher.bridge = self.bridge
        self.dispatcher.runtime = self
        self.fusion = EnsembleFusion.from_config(
            config, news_service=self.news, forces_service=self.forces,
        )

        # --- multi-timeframe hierarchical predictor (after fusion) -----------
        mtf_cfg = config.section("multi_timeframe")
        self.mtf = MultiTimeframePredictor(
            bridge=self.bridge,
            engine=self.engine,
            indicators=self.indicators,
            quantum=self.quantum,
            fusion=self.fusion,
            parallel=self.parallel,
            news=self.news,
            forces=self.forces,
            calendar=self.calendar,
            executor=self.executor,
            data=self.data,
            timeframes=mtf_cfg.get("timeframes", DEFAULT_TIMEFRAMES),
        ) if mtf_cfg.get("enabled", True) else None

        # --- listeners / UI --------------------------------------------------
        http_cfg = config.section("listener", "http")
        voice_cfg = config.section("listener", "voice")
        mtev_cfg = config.section("listener", "mt5_events")
        ui_cfg = config.section("ui")
        self.voice = VoiceListener(
            dispatcher=self.dispatcher,
            enabled=bool(voice_cfg.get("enabled", True)),
            wake_word=voice_cfg.get("wake_word", "jarvis"),
            model_size=voice_cfg.get("model_size", "base"),
            language=voice_cfg.get("language", "en"),
            listen_seconds=voice_cfg.get("listen_seconds", 4.0),
            sample_rate=voice_cfg.get("sample_rate", 16000),
            silence_hold=voice_cfg.get("silence_hold", 1.2),
            poll_seconds=6.0,
        )
        self.mt5_events = MT5EventListener(
            dispatcher=self.dispatcher, bridge=self.bridge,
            enabled=bool(mtev_cfg.get("enabled", True)),
            tick_every=mtev_cfg.get("tick_every", 30),
            equity_drawdown_alert=mtev_cfg.get("equity_drawdown_alert", 0.05),
            alert_window_seconds=mtev_cfg.get("alert_window_seconds", 300),
            poll_seconds=self.bridge.poll_seconds,
        )
        # Public-LAN exposure override: devices.bind_public → 0.0.0.0
        bind_host = http_cfg.get("bind_host", "127.0.0.1")
        try:
            if dev_cfg.get("bind_public") and bind_host in ("127.0.0.1", "localhost"):
                bind_host = "0.0.0.0"
        except Exception:
            pass
        self.http = HttpListener(
            dispatcher=self.dispatcher,
            enabled=bool(http_cfg.get("enabled", True)),
            bind_host=bind_host,
            port=http_cfg.get("port", 8765),
            auth_token=http_cfg.get("auth_token", "change-me-on-first-run"),
            runtime=self,
        )
        self.tray = TrayUI(
            dispatcher=self.dispatcher, config=config,
            enabled=bool(ui_cfg.get("tray", True)),
            refresh_seconds=ui_cfg.get("status_refresh_seconds", 3.0),
        )

        # --- agent swarm + compute routing + learning + corpus ------------
        ag_cfg = config.section("agents")
        self.learner = ModelLearner(
            window=ag_cfg.get("learner_window", 120),
            min_samples=ag_cfg.get("learner_min_samples", 8),
        )
        self.router = ComputeRouter()
        self.swarm = AgentSwarm(
            learner=self.learner,
            cache_ttl=ag_cfg.get("cache_ttl", 4.0),
            max_workers=ag_cfg.get("max_workers", 8),
            min_confidence=ag_cfg.get("min_confidence", 0.20),
            on_report=self._on_agent_report,
        )
        for agent in self.swarm.agents:
            if getattr(agent, "name", "") == "RiskShield":
                agent.status_provider = self.dispatcher.status
        self.repos = scan(root=str(Path.home()))

        # --- global hotkey push-to-talk ------------------------------------
        hk_cfg = config.section("listener", "hotkey")
        self.hotkeys = None
        if hk_cfg.get("enabled", True):
            self.hotkeys = GlobalHotkeys(
                on_push_to_talk=self._on_hotkey_talk,
                combo=hk_cfg.get("combo", "ctrl+alt+j"),
            )

        # --- in-memory log ring (powers the dashboard log view) -------------
        self._log_ring: deque[str] = deque(maxlen=200)
        _ring_handler = _LogRingHandler(self._log_ring)
        _ring_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(_ring_handler)

        # --- cache of last decision per symbol (for /chat "why" queries) ---
        self._last_decision: dict[str, dict] = {}
        self._last_mtf: dict[str, dict] = {}
        self._last_swarm: dict = {}

        # --- short-TTL snapshot cache for the HTTP read endpoints -----------
        # MT5 sync calls (positions/account/terminal) are local-but-blocking;
        # the UI polls /status + /mt5 every couple of seconds. Serving a
        # ~1.5s-old snapshot makes every request cheap and consistent.
        self._snap_cache: dict[str, tuple[float, Any]] = {}
        self._snap_ttl: dict[str, float] = {
            "status": 1.5,
            "mt5": 1.5,
            "prices": 2.0,
            "connectors": 5.0,
            "repos": 5.0,
            "devices": 2.0,
            "agents": 3.0,
            "lean": 5.0,
        }

    def _cached_snapshot(self, key: str, producer) -> Any:
        """Return a cached snapshot if fresh, else recompute + store."""
        ttl = self._snap_ttl.get(key, 2.0)
        now = time.monotonic()
        cached = self._snap_cache.get(key)
        if cached is not None and (now - cached[0]) < ttl:
            return cached[1]
        try:
            value = producer()
        except Exception:
            LOGGER.exception("snapshot producer %r failed", key)
            raise
        self._snap_cache[key] = (now, value)
        return value

    def _invalidate_snapshot(self, key: str) -> None:
        """Force the next read of `key` to recompute (post-mutation)."""
        self._snap_cache.pop(key, None)

        # --- Dispatcher fan-out to WebSocket -------------------------------
        for ev in (
            EventType.STAGE, EventType.CONFIRMED, EventType.EXECUTED,
            EventType.FAILED, EventType.CANCELLED, EventType.EQUITY_ALERT,
            EventType.STATUS, EventType.NEWS, EventType.FORCES,
            EventType.CALENDAR, EventType.AGENTS, EventType.REPORT,
            EventType.RISK,
        ):
            self.dispatcher.subscribe(ev, self._ws_broadcast)

        # --- shutdown hooks -------------------------------------------------
        self.dispatcher.subscribe(EventType.SHUTDOWN, self._on_shutdown_event)
        self.dispatcher.subscribe(EventType.EQUITY_ALERT, self._on_equity_alert)

    # ----------------------------------------------------------- public API

    def start(self) -> None:
        LOGGER.info("=" * 70)
        LOGGER.info("Jarvis starting up — async + parallel + 12-model fleet")
        LOGGER.info("=" * 70)
        self._install_signal_handlers()
        # Eager-load every model so first cycle is fast.
        for m in self.fleet.models:
            try:
                m.load()
                LOGGER.info("  %-25s %-12s %s", m.name, m.family, m.status)
            except Exception:
                LOGGER.exception("Model %s failed to load — continuing", m.name)

        # Start data services + listeners (non-blocking threads).
        self.news.start()
        self.forces.start()
        self.calendar.start()
        self.voice.start()
        self.connectors.start()
        self.mt5_events.start()
        self.http.start()
        self.tray.start()
        if self.hotkeys is not None:
            self.hotkeys.start()
        LOGGER.info("Jarvis ready. WebSocket + REST + UI live on http://%s:%d",
                    "127.0.0.1", self.http.port)
        LOGGER.info("ChatGPT-style UI: http://%s:%d/", "127.0.0.1", self.http.port)

        # Eagerly load TimesFM in the background so boot is instant — the first
        # forecast that needs the engine blocks until it is ready (forecast
        # auto-loads), and the 400MB checkpoint streams in without stalling UI.
        def _load_engine():
            try:
                self.engine.load()
            except Exception:
                LOGGER.exception("TimesFM failed to load — will retry on demand")

        threading.Thread(target=_load_engine, name="jarvis-timesfm", daemon=True).start()

        try:
            asyncio.run(self._async_main())
        finally:
            self.stop()

    def stop(self) -> None:
        if self._stop.is_set():
            return
        LOGGER.info("Jarvis shutting down…")
        self._stop.set()
        for name in ("tray", "http", "voice", "mt5_events", "news", "forces", "calendar", "connectors", "strategy"):
            svc = getattr(self, name, None)
            if svc is None:
                continue
            try:
                svc.stop()
            except Exception:
                LOGGER.exception("%s stop error", name)
        try:
            self.bridge.shutdown()
        except Exception:
            LOGGER.exception("MT5 shutdown error")
        if self.hotkeys is not None:
            try:
                self.hotkeys.stop()
            except Exception:
                LOGGER.exception("hotkeys stop error")
        pool = getattr(self, "_hotkey_pool", None)
        if pool is not None:
            try:
                pool.shutdown(wait=False)
            except Exception:
                pass
        repo_pool = getattr(self, "_repo_pool", None)
        if repo_pool is not None:
            try:
                repo_pool.shutdown(wait=False)
            except Exception:
                pass
        LOGGER.info("Jarvis stopped cleanly.")

    # ----------------------------------------------------------- chat helpers

    def run_on_demand_forecast(self, symbol: str) -> dict | None:
        """Run the parallel fleet + ensemble for `symbol` (uses synthetic
        close if MT5 isn't connected). Returns a dict suitable for chat."""
        try:
            close = self._get_or_synth(symbol, n=200)
        except JarvisMT5Error as e:
            LOGGER.warning("run_on_demand: MT5 failed for %s: %s", symbol, e)
            return None
        return asyncio.run(self._forecast_async(symbol, close, horizon=16))

    def run_on_demand_timeframes(self, symbol: str) -> dict | None:
        """Run the multi-timeframe hierarchical predictor on demand."""
        if self.mtf is None:
            return None
        decision = self.mtf.predict(symbol)
        # Cache for chat.
        self._last_mtf[symbol] = decision.to_dict()
        return self._last_mtf[symbol]

    def explain_last_decision(self, symbol: str) -> str:
        """Returns HTML for the chat bubble when the user asks 'why SYM'."""
        d = self._last_decision.get(symbol)
        if not d:
            return f"<p>No recent decision cached for <code>{symbol}</code>.</p>"
        verdict = self.meta.explain(d)
        return (
            f"<p>Reasoning for last <b>{symbol}</b> decision "
            f"(<i>via {verdict.source}</i>):</p>"
            f"<p>→ {verdict.reasoning}</p>"
            f"<p>Verdict: <b>dir {verdict.direction:+d}</b> "
            f"conf <b>{verdict.confidence:.2f}</b></p>"
        )

    # ------------------------------------------------------------- main loop

    async def _async_main(self) -> None:
        poll = self.bridge.poll_seconds
        symbols = list(self.bridge.symbols)
        try:
            await self.async_bridge.prewarm()
        except Exception:
            LOGGER.debug("MT5 cache prewarm skipped at boot", exc_info=True)
        cycle = 0
        while not self._stop.is_set():
            cycle += 1
            if cycle % 10 == 0:
                try:
                    self.executor.expire_old()
                except Exception:
                    LOGGER.exception("expire_old error")
            # Keep the local hub visible on the LAN registry (heartbeat TTL).
            if cycle % 5 == 0:
                try:
                    did = f"hub-{self.device_info['name']}".lower().replace(" ", "-")
                    self.devices.heartbeat(did)
                except Exception:
                    LOGGER.debug("local heartbeat skipped")
            # Fan out per-symbol processing — bounded by semaphore so we
            # don't spawn 100 tasks when there are 100 symbols.
            sem = asyncio.Semaphore(4)
            async def run_one(sym):
                async with sem:
                    try:
                        await self._process_symbol_async(sym)
                    except Exception:
                        LOGGER.exception("error processing %s", sym)
            await asyncio.gather(*(run_one(s) for s in symbols))
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=poll)
            except asyncio.TimeoutError:
                pass

    async def _process_symbol_async(self, symbol: str) -> None:
        # 1. Async MT5 rates (non-blocking).
        try:
            df = await self.async_bridge.rates(symbol)
        except JarvisMT5Error as e:
            LOGGER.debug("rates(%s) failed: %s", symbol, e)
            return
        if df is None or df.empty or len(df) < 64:
            return

        # 2. Compute non-LLM features (CPU, fast — synchronous is fine).
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        tick_vol = df["tick_vol"].to_numpy(dtype=np.float64) if "tick_vol" in df.columns else None
        open_ = df["open"].to_numpy(dtype=np.float64)
        indicators_snap = self.indicators.compute(
            open_=open_, high=high, low=low, close=close, volume=tick_vol)
        pattern_snap = detect_patterns(
            open_=open_, high=high, low=low, close=close,
            lookback_chart=self.config.section("patterns").get("lookback_chart", 60))

        # 2b. Strategy engine analysis (MTF confluence, pattern recognition, hedging)
        strategy_signal = None
        try:
            # Build multi-timeframe data for strategy engine
            tf_data = {}
            for tf in [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]:
                try:
                    tf_df = await self.async_bridge.rates(symbol, timeframe=tf.value, count=100)
                    if tf_df is not None and not tf_df.empty:
                        tf_data[tf] = self.bridge.get_indicators(symbol, tf.value)
                        tf_data[tf]["candles"] = tf_df.to_dict("records")
                        tf_data[tf]["timestamp"] = time.time()
                except Exception:
                    pass

            # Get account info and positions
            account_info = self.bridge.account_info()
            mt5_positions = self.bridge.open_positions()

            # Run strategy engine
            if tf_data and account_info:
                strategy_signal = self.strategy.analyze_symbol(
                    symbol, tf_data, account_info, mt5_positions
                )
                # Emit strategy signal event
                self.dispatcher.publish(
                    EventType.AGENTS,
                    {"symbol": symbol, "strategy_signal": strategy_signal.__dict__},
                    source="strategy"
                )
                # Apply exit / trailing / hedge management through MT5.
                try:
                    applied = self.strategy.apply_management()
                    if applied:
                        self.dispatcher.publish(
                            EventType.PATTERN,
                            {"symbol": symbol, "strategy_actions": applied},
                            source="strategy"
                        )
                except Exception:
                    LOGGER.exception("apply_management failed for %s", symbol)
        except Exception:
            LOGGER.exception("Strategy engine failed for %s", symbol)

        # 3. Async TimesFM + parallel fleet.
        try:
            forecast_task = asyncio.get_running_loop().run_in_executor(
                None, self.engine.forecast, close,
            )
            fleet_task = self.parallel.forecast_all(close, horizon=16, symbol=symbol)
            forecast, model_forecasts = await asyncio.gather(forecast_task, fleet_task)
        except Exception:
            LOGGER.exception("Forecast/fleet failed for %s", symbol)
            return

        # 4. Quantum, news, forces, calendar (all synchronous, fast).
        q_decision = self.quantum.decide(
            per_asset_signal=[
                float(forecast.direction) * forecast.confidence,
                indicators_snap.composite_bias / 3.0,
                float(pattern_snap.direction_bias) * pattern_snap.confidence,
            ],
            per_asset_phase=[
                0.0 if forecast.direction >= 0 else 3.14159265,
                0.0 if indicators_snap.composite_bias >= 0 else 3.14159265,
                0.0 if pattern_snap.direction_bias >= 0 else 3.14159265,
            ],
            correlation_matrix=np.array([
                [1.0, 0.3, 0.5], [0.3, 1.0, 0.4], [0.5, 0.4, 1.0],
            ]),
        )
        news_sent = self.news.sentiment_for_symbol(symbol) if self.news.enabled else 0.0
        items = self.news.items_for_symbol(symbol) if self.news.enabled else []
        news_rel = min(1.0, len(items) / 5.0)
        i_hint = self.forces.force_direction_for(symbol) if self.forces.enabled else 0
        in_blackout, blackout_event = self.calendar.is_blackout_now()
        cal_impact = self.calendar.impact_for_symbol(symbol)

        # 5. Ensemble + signal.
        ctx = MarketContext(
            symbol=symbol, forecast=forecast, indicators=indicators_snap,
            quantum=q_decision, pattern=pattern_snap,
            news_sentiment=news_sent, news_relevance=news_rel,
            intermarket_hint=i_hint, calendar_impact=cal_impact,
            in_blackout=in_blackout, model_forecasts=model_forecasts,
        )
        fusion = self.fusion.fuse(ctx)
        current_price = float(close[-1])
        signal = self.signals.generate(symbol=symbol, fusion=fusion, ctx=ctx, current_price=current_price)

        # 5b. Agent swarm — coordinated review of the frozen snapshot.
        swarm_decision = None
        try:
            swarm_decision = await self.swarm.predict(SwarmContext(
                symbol=symbol,
                snapshot={
                    "ta": indicators_snap.to_dict(),
                    "patterns": pattern_snap.to_dict(),
                    "quantum": q_decision.to_dict(),
                    "news": {"sentiment": news_sent, "relevance": news_rel},
                    "forces": {"hint": i_hint, "regime": fusion.regime},
                    "calendar": {"in_blackout": in_blackout, "impact": cal_impact},
                    "model_consensus": _fleet_consensus(model_forecasts),
                },
                arrays={"open": open_, "high": high, "low": low, "close": close},
                action_type="forecast",
            ))
            self._last_swarm = {"symbol": symbol,
                                "decision": swarm_decision.to_dict()}
            self.dispatcher.publish(
                EventType.AGENTS,
                {"symbol": symbol, "direction": swarm_decision.direction,
                 "confidence": swarm_decision.confidence,
                 "summary": swarm_decision.summary,
                 "tier": swarm_decision.profile.tier if swarm_decision.profile else None},
                source="swarm",
            )
        except Exception:
            LOGGER.exception("swarm failed for %s", symbol)

        # Cache for /chat "why" queries.
        self._last_decision[symbol] = {
            "symbol": symbol,
            "models": [f.to_dict() for f in model_forecasts],
            "fusion": fusion.to_dict(),
            "ta": indicators_snap.to_dict(),
            "quantum": q_decision.to_dict(),
            "patterns": pattern_snap.to_dict(),
            "news": {"sentiment": news_sent, "relevance": news_rel},
            "forces": {"hint": i_hint},
            "calendar": {"in_blackout": in_blackout, "impact": cal_impact},
            "swarm": swarm_decision.to_dict() if swarm_decision is not None else None,
        }

        # 6. Stage if actionable + not in blackout.
        #    Strategy-engine veto: a confident, contradictory MTF/pattern read
        #    suppresses the fused signal (consultative layer protects the trade).
        strategy_veto = False
        if strategy_signal is not None and signal.is_actionable:
            sa = getattr(strategy_signal, "action", "hold")
            sc = float(getattr(strategy_signal, "confidence", 0.0) or 0.0)
            if sa == "hold" and sc >= 0.6:
                strategy_veto = True
            elif sa in ("buy", "sell"):
                sdir = 1 if sa == "buy" else -1
                if sdir != signal.direction and sc >= 0.6:
                    strategy_veto = True

        if signal.is_actionable and not in_blackout:
            if strategy_veto:
                self.dispatcher.publish(
                    EventType.AGENTS,
                    {"symbol": symbol, "veto": "strategy",
                     "strategy_action": getattr(strategy_signal, "action", "hold"),
                     "strategy_confidence": getattr(strategy_signal, "confidence", 0.0),
                     "fused_direction": signal.direction},
                    source="strategy"
                )
                LOGGER.info("%s: strategy veto suppressed direction=%s conf=%.2f",
                            symbol, signal.direction, signal.confidence)
            else:
                LOGGER.info("Signal %s %s conf=%.2f price=%.5f",
                            signal.symbol, "BUY" if signal.direction == 1 else "SELL",
                            signal.confidence, signal.entry_price)
                staged = self.executor.stage(signal)
                if staged is not None:
                    self.dispatcher.publish(
                        EventType.STAGE,
                        {"signal_id": staged.signal_id, "signal": signal.to_dict()},
                        source="runtime",
                    )
        elif in_blackout:
            self.dispatcher.publish(EventType.CALENDAR, {"symbol": symbol, "event": blackout_event}, source="runtime")

        # 7. Multi-timeframe hierarchical decision (cached for chat "timeframes").
        if self.mtf is not None:
            try:
                decision = await self.mtf.predict_async(symbol)
                self._last_mtf[symbol] = decision.to_dict()
            except Exception:
                LOGGER.debug("MTF prediction failed for %s", symbol)

    # --------------------------------------------------------- event handlers

    def _ws_broadcast(self, event) -> None:
        """Dispatcher subscriber — pushes every event to all WS clients."""
        self.http.broadcast_event(event.type, event.payload, source=event.source)

    def _on_stage(self, staged) -> None:
        LOGGER.info("Stage ready for confirmation: id=%s", staged.signal_id)

    def _on_result(self, staged, result) -> None:
        if result.success:
            self.dispatcher.publish(
                EventType.EXECUTED,
                {"signal_id": staged.signal_id, "result": {
                    "order_id": result.order_id, "retcode": result.retcode}},
                source="runtime",
            )
        else:
            self.dispatcher.publish(
                EventType.FAILED,
                {"signal_id": staged.signal_id, "comment": result.comment},
                source="runtime",
            )

    def _on_shutdown_event(self, event) -> None:
        LOGGER.info("Shutdown event from %s", event.source)
        # Schedule async stop.
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(self._stop.set)
        except RuntimeError:
            self._stop.set()
        self.stop()

    def _on_equity_alert(self, event) -> None:
        payload = event.payload
        LOGGER.warning("Equity drawdown %.2f%% — cancelling all pending", payload.get("drawdown_pct", 0))
        self.dispatcher.cancel_all(source="runtime", reason="equity drawdown")

    # --------------------------------------------------------- agent swarm API

    def _on_agent_report(self, report) -> None:
        """Swarm per-agent telemetry → WS live stream."""
        self.dispatcher.publish(EventType.REPORT, report.to_dict(), source="swarm")

    def agents_snapshot(self) -> dict:
        """GET /agents payload — swarm roster + router profile + learnt metrics."""
        summary = self.learner.summary()
        table = self.swarm.status_table()
        for a in table:
            key = f"agent:{a['name']}"
            m = summary.get(key, {})
            a["hit_rate"] = m.get("hit_rate")
            a["multiplier"] = round(self.learner.multiplier(key), 3)
        profile = self.router.evaluate(task_type="forecast", mode="quick")
        return {
            "agents": table,
            "profile": profile.to_dict(),
            "engaged": profile.agent_count,
            "total": len(table),
            "last_decision": self._last_swarm.get("symbol"),
            "learner": summary,
        }

    def learner_track(self) -> dict:
        """GET /track payload — every learnt source, its stats + multiplier."""
        out: dict[str, dict] = {}
        for key, m in self.learner.summary().items():
            out[key] = {**m, "multiplier": round(self.learner.multiplier(key), 3)}
        return {"track": out}

    def status_snapshot(self) -> dict:
        """Rich /status payload — core stats + live positions + connector layer."""

        def _build() -> dict:
            snapshot = dict(self.dispatcher.status())

            # Live positions detail (typified for the UI).
            positions = []
            try:
                for p in (self.bridge.open_positions() if self.bridge else []) or []:
                    positions.append({
                        "symbol": getattr(p, "symbol", "?"),
                        "type": "BUY" if getattr(p, "type", 1) == 0 else "SELL",
                        "volume": float(getattr(p, "volume", 0.0) or 0.0),
                        "price_open": float(getattr(p, "price_open", 0.0) or 0.0),
                        "profit": float(getattr(p, "profit", 0.0) or 0.0),
                        "ticket": getattr(p, "ticket", None),
                    })
            except Exception:
                positions = []
            snapshot["positions"] = positions

            try:
                snapshot["daily_pnl"] = round(float(self.bridge.daily_pnl()), 2) \
                    if self.bridge else 0.0
            except Exception:
                snapshot["daily_pnl"] = 0.0

            try:
                snapshot["connector_summary"] = self.connectors.summary()
            except Exception:
                pass
            return snapshot

        return self._cached_snapshot("status", _build)

    def connectors_snapshot(self) -> dict:
        """GET /connectors payload."""
        def _build() -> dict:
            try:
                return self.connectors.snapshot()
            except Exception:
                return {"total": 0, "categories": []}
        return self._cached_snapshot("connectors", _build)

    def invalidate_connectors(self) -> None:
        self._invalidate_snapshot("connectors")

    def mt5_snapshot(self) -> dict:
        """GET /mt5 payload — terminal, broker account, risk config + kill switch."""

        def _build() -> dict:
            account = {}
            terminal = {}
            connected = False
            try:
                if self.config.section("mt5").get("enabled", True) and self.bridge:
                    connected = bool(self.bridge.ensure_connected())
                    account = self.bridge.account_info()
                    terminal = self.bridge.terminal_info()
            except Exception:
                LOGGER.exception("mt5_snapshot failed")
            risk = {}
            if self.executor is not None:
                risk = {
                    "paused": bool(self.executor.risk_paused()),
                    "max_open_positions": self.executor.max_open_positions,
                    "max_daily_loss": self.executor.max_daily_loss,
                    "confirm_timeout": self.executor.confirm_timeout,
                }
            try:
                account.update({k: float(account[k] or 0.0) for k in ("balance", "equity") if k in account})
            except Exception:
                pass
            return {
                "connected": connected,
                "terminal": terminal,
                "account": account,
                "risk": risk,
                "symbols": list(self.bridge.symbols) if self.bridge is not None else [],
                "timeframe": self.bridge.timeframe_str if self.bridge is not None else "",
            }

        return self._cached_snapshot("mt5", _build)

    def prices_snapshot(self) -> dict:
        """GET /prices payload — live price + move% per watched symbol (ticker).

        Sources the strategy engine's latest per-symbol analysis (refreshed
        every main-loop cycle, zero MT5 IPC on the HTTP path). `change` is the
        move against the previously seen price (<=60 s window), so the ticker
        shows real per-poll movement without racing the MT5 driver.
        """
        def _build() -> dict:
            out: dict[str, dict] = {}
            now = time.time()
            tf = getattr(self.bridge, "timeframe_str", "M5")
            prev = dict(getattr(self, "_last_ticker", {}))
            prev_at = float(getattr(self, "_last_ticker_at", 0.0))
            trend = {}
            try:
                status = self.strategy.get_status()
                trend = status.get("signals", {}) or {}
            except Exception:
                LOGGER.debug("prices_snapshot: strategy status unavailable", exc_info=True)
            next_prices: dict[str, float] = {}
            for sym, sig in trend.items():
                try:
                    price = float(sig.get("entry_price") or 0.0)
                except Exception:
                    continue
                if price <= 0:
                    continue
                prev_price = prev.get(sym)
                change = (price - prev_price) if prev_price and (now - prev_at) < 60.0 else 0.0
                change_pct = (change / prev_price * 100.0) if prev_price else 0.0
                action = str(sig.get("action") or "hold")
                direction = 1 if action == "buy" else (-1 if action == "sell" else 0)
                out[sym] = {
                    "symbol": sym,
                    "price": round(price, 5),
                    "change": round(change, 5),
                    "change_pct": round(change_pct, 3),
                    "direction": direction,
                    "action": action,
                    "confidence": round(float(sig.get("confidence") or 0.0), 3),
                    "grade": sig.get("entry_grade"),
                    "timeframe": tf,
                }
                next_prices[sym] = price
            self._last_ticker = next_prices
            self._last_ticker_at = now
            return {"symbols": out, "as_of": now, "timeframe": tf}
        return self._cached_snapshot("prices", _build)

    def strategy_snapshot(self) -> dict:
        """GET /strategy payload — engine status, last signals, portfolio risk."""
        try:
            return self.strategy.get_status()
        except Exception:
            LOGGER.exception("strategy_snapshot failed")
            return {"enabled": False, "error": "strategy engine unavailable"}

    def _probe_repos_async(self) -> None:
        """Background-probe local repos so the UI never blocks on discovery."""
        if not hasattr(self, "_repo_pool"):
            import concurrent.futures
            self._repo_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        def _probe() -> None:
            try:
                self.repo_probes = self.repo_fabric.probe_all()
            except Exception:
                LOGGER.exception("repo probe failed")
        self._repo_pool.submit(_probe)

    def repo_snapshot(self) -> dict:
        """GET /repos payload — local quant repo integration status."""

        def _build() -> dict:
            try:
                return {
                    "repos": self.repo_fabric.summary(),
                    "available": self.repo_fabric.available(),
                    "probes": self.repo_probes,
                }
            except Exception:
                LOGGER.exception("repo_snapshot failed")
                return {"repos": [], "available": [], "probes": {}}

        return self._cached_snapshot("repos", _build)

    def lean_snapshot(self) -> dict:
        """GET /repos/lean payload — LEAN engine backtest status."""

        def _build() -> dict:
            try:
                from .repos.lean_adapter import get_lean_engine
                return get_lean_engine().summary()
            except Exception:
                LOGGER.exception("lean_snapshot failed")
                return {"available": False}

        return self._cached_snapshot("lean", _build)

    def _register_local_device(self) -> None:
        """Register this machine in its own registry so peers can find it."""
        try:
            info = dict(self.device_info)
            self.devices.register(
                device_id=f"hub-{info['name']}".lower().replace(" ", "-"),
                name=info["name"], role="hub", host=info.get("host", ""),
                user_agent=info.get("user_agent", ""),
                capabilities=info.get("capabilities", []),
            )
        except Exception:
            LOGGER.exception("local device register failed")

    def devices_snapshot(self) -> dict:
        """GET /devices payload — online devices + local device info."""

        def _build() -> dict:
            try:
                return {
                    "local": self.device_info,
                    "devices": self.devices.list_devices(),
                    "online": self.devices.count_online(),
                }
            except Exception:
                LOGGER.exception("devices_snapshot failed")
                return {"local": {}, "devices": [], "online": 0}

        return self._cached_snapshot("devices", _build)

    def toggle_kill_switch(self, paused: bool | None = None) -> dict:
        """Flip the algo kill switch. Returns the new risk state (WS-broadcast)."""
        if self.executor is None:
            return {"paused": False}
        current = self.executor.risk_paused()
        target = current if paused is None else bool(paused)
        self.executor.set_risk_pause(target)
        state = {
            "paused": target,
            "max_open_positions": self.executor.max_open_positions,
            "max_daily_loss": self.executor.max_daily_loss,
        }
        self.dispatcher.publish(EventType.RISK, state, source="runtime")
        return state

    def patterns_for(self, symbol: str) -> dict | None:
        """Real pattern snapshot for a symbol (chat/patterns view)."""
        try:
            df = self.bridge.rates(symbol)
        except JarvisMT5Error:
            return None
        if df is None or len(df) < 32:
            return None
        return detect_patterns(
            open_=df["open"].to_numpy(dtype=np.float64),
            high=df["high"].to_numpy(dtype=np.float64),
            low=df["low"].to_numpy(dtype=np.float64),
            close=df["close"].to_numpy(dtype=np.float64),
            lookback_chart=self.config.section("patterns").get("lookback_chart", 60),
        ).to_dict()

    def quantum_for(self, symbol: str) -> dict | None:
        """Real quantum layer read for a symbol (chat/quantum view)."""
        try:
            df = self.bridge.rates(symbol)
        except JarvisMT5Error:
            return None
        if df is None or len(df) < 32:
            return None
        open_ = df["open"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)
        volume = df["tick_vol"].to_numpy(dtype=np.float64) if "tick_vol" in df.columns else None
        ind = self.indicators.compute(open_=open_, high=high, low=low, close=close, volume=volume)
        pat = detect_patterns(open_=open_, high=high, low=low, close=close)
        return self.quantum.decide(
            per_asset_signal=[
                ind.composite_bias / 3.0,
                float(pat.direction_bias) * pat.confidence,
            ],
            per_asset_phase=[
                0.0 if ind.composite_bias >= 0 else 3.14159265,
                0.0 if pat.direction_bias >= 0 else 3.14159265,
            ],
            correlation_matrix=np.array([[1.0, 0.4], [0.4, 1.0]]),
        ).to_dict()

    # ------------------------------------------------------------- hotkey voice

    def _on_hotkey_talk(self) -> None:
        """Called on the hotkey hook thread — hand off to a worker."""
        import concurrent.futures
        if not hasattr(self, "_hotkey_pool"):
            self._hotkey_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._hotkey_pool.submit(self._handle_hotkey_talk)

    def _handle_hotkey_talk(self) -> None:
        try:
            text = self.voice.capture_once()
        except Exception:
            LOGGER.exception("push-to-talk capture failed")
            return
        if not text:
            return
        self._on_voice_transcript(text)

    def _on_voice_transcript(self, text: str) -> None:
        from .chat import ChatProcessor
        try:
            result = ChatProcessor(self.dispatcher, runtime=self).process(text)
        except Exception:
            LOGGER.exception("voice transcript handling failed")
            result = {}
        self.dispatcher.publish(EventType.VOICE_COMMAND,
                                {"text": text, **result}, source="voice")

    # ----------------------------------------------------- helpers / signal

    def _get_or_synth(self, symbol: str, n: int = 200) -> np.ndarray:
        try:
            return self.bridge.close_series(symbol, count=n)
        except JarvisMT5Error:
            return (np.arange(n, dtype=np.float64) * 0.05) + 100.0

    async def _forecast_async(self, symbol: str, close: np.ndarray, horizon: int = 16) -> dict:
        """Run the parallel fleet + ensemble for an on-demand chat query."""
        indicators_snap = self.indicators.compute(
            open_=close, high=close+0.05, low=close-0.05, close=close,
            volume=np.full(close.size, 2000.0),
        )
        pattern_snap = detect_patterns(
            open_=close, high=close+0.05, low=close-0.05, close=close)
        forecast_task = asyncio.get_running_loop().run_in_executor(
            None, self.engine.forecast, close,
        )
        fleet_task = self.parallel.forecast_all(close, horizon=horizon, symbol=symbol)
        forecast, model_forecasts = await asyncio.gather(forecast_task, fleet_task)
        q_decision = self.quantum.decide(
            per_asset_signal=[float(forecast.direction) * forecast.confidence,
                             indicators_snap.composite_bias / 3.0,
                             float(pattern_snap.direction_bias) * pattern_snap.confidence],
            per_asset_phase=[0, 0, 0],
            correlation_matrix=np.eye(3),
        )
        ctx = MarketContext(
            symbol=symbol, forecast=forecast, indicators=indicators_snap,
            quantum=q_decision, pattern=pattern_snap,
            news_sentiment=0.0, news_relevance=0.0, intermarket_hint=0,
            calendar_impact=0, in_blackout=False, model_forecasts=model_forecasts,
        )
        fusion = self.fusion.fuse(ctx)
        return {
            "symbol": symbol,
            "current_price": float(close[-1]),
            "models": model_forecasts,
            "fusion_direction": fusion.direction,
            "fusion_confidence": fusion.confidence,
            "regime": fusion.regime,
        }

    def _install_signal_handlers(self) -> None:
        def _handler(signum, frame):
            LOGGER.info("Signal %d received; initiating shutdown", signum)
            self.stop()
        try:
            posix_signal.signal(posix_signal.SIGINT, _handler)
            posix_signal.signal(posix_signal.SIGTERM, _handler)
        except ValueError:
            pass


__all__ = ["JarvisRuntime"]
