"""
Jarvis Reborn — MT5 Algorithmic Trading System
==============================================

Ultra-sophisticated algo trader on MetaTrader 5: TimesFM deep forecasting,
quant + LLM model fleet, agent swarm, risk shield, QuantConnect + 120
integrations, and a trading-terminal UI.

Intelligence stack (7 sources, weighted + adaptive):
    1. TimesFM    — deep time-series foundation model
    2. TA         — RSI / MACD / ATR / volume composite bias
    3. Quantum    — amplitude encoding + QAOA-inspired weight optimisation
                    + entanglement-inspired cross-asset coupling
    4. News       — RSS feeds + VADER sentiment + per-symbol relevance
    5. Intermarket— DXY, US10Y, VIX, SPX regime hints (yfinance)
    6. Patterns   — candlestick + chart pattern recognition
    7. Calendar   — economic calendar with blackout windows

Public surface:
    from jarvis_predict import (
        JarvisConfig, TimesFMEngine, TechnicalIndicators,
        EnsembleFusion, MarketContext, SignalGenerator, TradeSignal,
        MT5Bridge, TradeExecutor, JarvisRuntime,
        QuantumDecision, QuantumDecisionEngine,
        NewsService, MarketForcesService, EconomicCalendar,
        detect_patterns, DataPipeline, symbols,
    )

Author: Haider Bhatti (project owner), polished by Mavis (MiniMax Jarvis)
License: Personal use. TimesFM weights are subject to upstream license.
"""

from __future__ import annotations

__version__ = "1.1.0"
__all__ = [
    "JarvisConfig",
    "TimesFMEngine",
    "TechnicalIndicators",
    "EnsembleFusion",
    "MarketContext",
    "SignalGenerator",
    "TradeSignal",
    "MT5Bridge",
    "AsyncMT5Bridge",
    "ParallelFleet",
    "ChatProcessor",
    "MetaOrchestrator",
    "TradeExecutor",
    "JarvisRuntime",
    "QuantumDecision",
    "QuantumDecisionEngine",
    "NewsService",
    "MarketForcesService",
    "EconomicCalendar",
    "DataPipeline",
    "symbols",
]


def __getattr__(name: str):
    if name == "JarvisConfig":
        from .config import JarvisConfig
        return JarvisConfig
    if name == "TimesFMEngine":
        from .core import TimesFMEngine
        return TimesFMEngine
    if name == "TechnicalIndicators":
        from .indicators import TechnicalIndicators
        return TechnicalIndicators
    if name in ("EnsembleFusion", "MarketContext", "FusionResult"):
        from . import ensemble
        return getattr(ensemble, name)
    if name in ("SignalGenerator", "TradeSignal"):
        from . import signals
        return getattr(signals, name)
    if name == "MT5Bridge":
        from .mt5_bridge import MT5Bridge
        return MT5Bridge
    if name == "TradeExecutor":
        from .executor import TradeExecutor
        return TradeExecutor
    if name == "JarvisRuntime":
        from .runtime import JarvisRuntime
        return JarvisRuntime
    if name in ("QuantumDecision", "QuantumDecisionEngine"):
        from . import quantum
        return getattr(quantum, name)
    if name == "NewsService":
        from .news import NewsService
        return NewsService
    if name == "MarketForcesService":
        from .market_forces import MarketForcesService
        return MarketForcesService
    if name == "EconomicCalendar":
        from .calendar import EconomicCalendar
        return EconomicCalendar
    if name == "DataPipeline":
        from .data_pipeline import DataPipeline
        return DataPipeline
    if name == "ChatProcessor":
        from .chat import ChatProcessor
        return ChatProcessor
    if name == "MetaOrchestrator":
        from .meta import MetaOrchestrator
        return MetaOrchestrator
    if name == "AsyncMT5Bridge":
        from .async_mt5 import AsyncMT5Bridge
        return AsyncMT5Bridge
    if name == "ParallelFleet":
        from .parallel import ParallelFleet
        return ParallelFleet
    if name == "symbols":
        import importlib
        return importlib.import_module("jarvis_predict.symbols")
    raise AttributeError(f"module 'jarvis_predict' has no attribute {name!r}")
