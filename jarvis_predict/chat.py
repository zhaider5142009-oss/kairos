"""
Chat command processor — natural-language → action dispatcher.

Backbone of the Jarvis ChatGPT-style UI. Parses a user message, routes
to the right subsystem, and returns HTML for the chat bubble.

Supported intents (case-insensitive, punctuation-tolerant):
    status / equity / account           → /status snapshot
    models / fleet                      → /models status table
    news                                → /news snapshot
    forces / macro                      → /forces snapshot
    calendar / events                   → /calendar snapshot
    symbols / universe                  → /universe snapshot
    pending / signals                   → /pending list
    confirm [latest|<signal_id>]       → POST /confirm
    cancel  [latest|all|<signal_id>]   → POST /cancel or /cancel-all
    forecast <symbol>                   → run all 12 models + ensemble on synthetic / last
    run a model                          → "run timesfm" → forecast single model
    why <symbol>                        → meta-orchestrator: explain last decision
    shutdown / quit                     → POST /shutdown
    help                                → cheat sheet
    free text                           → falls through to "I heard you"
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .connectors import STATUS_LABEL

LOGGER = logging.getLogger("jarvis.chat")


# ----- helpers ---------------------------------------------------------------

def _escape(s: Any) -> str:
    return (str(s)
            .replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;"))


def _html_table(rows: list[list[Any]], headers: list[str]) -> str:
    out = ['<table><thead><tr>']
    for h in headers:
        out.append(f'<th>{_escape(h)}</th>')
    out.append('</tr></thead><tbody>')
    for r in rows:
        out.append('<tr>' + ''.join(f'<td>{_escape(c)}</td>' for c in r) + '</tr>')
    out.append('</tbody></table>')
    return ''.join(out)


# ----- intent parsing --------------------------------------------------------

@dataclass
class Intent:
    name: str
    args: dict


_INTENT_PATTERNS = [
    # (regex, intent_name, args_extractor)
    (re.compile(r"^(help|commands|what can you do)\b", re.I), "help", {}),
    (re.compile(r"^(status|equity|account|portfolio)\b", re.I), "status", {}),
    (re.compile(r"^(models?|fleet|predictors?)\b", re.I), "models", {}),
    (re.compile(r"^news\b", re.I), "news", {}),
    (re.compile(r"^(forces|macro)\b", re.I), "forces", {}),
    (re.compile(r"^(calendar|events?|blackouts?)\b", re.I), "calendar", {}),
    (re.compile(r"^(symbols|universe|instruments?)\b", re.I), "symbols", {}),
    (re.compile(r"^(pending|signals?|queue)\b", re.I), "pending", {}),
    (re.compile(r"^confirm(?:\s+(.+))?", re.I), "confirm",
        lambda m: {"arg": (m.group(1) or "latest").strip()}),
    (re.compile(r"^cancel(?:\s+(all|.+))?", re.I), "cancel",
        lambda m: {"arg": (m.group(1) or "latest").strip()}),
    (re.compile(r"^forecast\s+([A-Za-z0-9]{3,10})\b", re.I), "forecast",
        lambda m: {"symbol": m.group(1).upper()}),
    (re.compile(r"^run\s+([a-z0-9_]+)\b", re.I), "run_model",
        lambda m: {"model": m.group(1).lower()}),
    (re.compile(r"^why\s+([A-Za-z0-9]{3,10})\b", re.I), "why",
        lambda m: {"symbol": m.group(1).upper()}),
    (re.compile(r"^patterns?\s+([A-Za-z0-9]{3,10})\b", re.I), "patterns",
        lambda m: {"symbol": m.group(1).upper()}),
    (re.compile(r"^quantum\s+([A-Za-z0-9]{3,10})\b", re.I), "quantum",
        lambda m: {"symbol": m.group(1).upper()}),
    (re.compile(r"^timeframes?\s+([A-Za-z0-9]{3,10})\b", re.I), "timeframes",
        lambda m: {"symbol": m.group(1).upper()}),
    (re.compile(r"^usage\b", re.I), "usage", {}),
    (re.compile(r"^(agents?|swarm)\b", re.I), "agents", {}),
    (re.compile(r"^(connectors?|integrations?|connections?)\b", re.I), "connectors", {}),
    (re.compile(r"^(repos?|corpus|repositories)\b", re.I), "repos", {}),
    (re.compile(r"^(devices?|peers?|connect)\b", re.I), "devices", {}),
    (re.compile(r"^(shutdown|quit|exit|bye)\b", re.I), "shutdown", {}),
]


def parse_intent(message: str) -> Intent:
    """Return the parsed intent. Falls back to 'unknown'."""
    msg = message.strip()
    for pat, name, extractor in _INTENT_PATTERNS:
        m = pat.match(msg)
        if m:
            if callable(extractor):
                return Intent(name=name, args=extractor(m))
            return Intent(name=name, args=extractor or {})
    return Intent(name="unknown", args={"raw": msg})


# ----- handler ---------------------------------------------------------------

class ChatProcessor:
    """Routes intents to dispatcher + fleet + bridge. Holds no state of its own."""

    def __init__(self, dispatcher, runtime=None) -> None:
        self.dispatcher = dispatcher
        self.runtime = runtime

    def process(self, message: str) -> dict[str, Any]:
        intent = parse_intent(message)
        try:
            handler = getattr(self, f"_handle_{intent.name}", self._handle_unknown)
            html = handler(intent)
            return {"reply_html": html}
        except Exception as e:
            LOGGER.exception("chat handler failed")
            err_html = f'<p>⚠ Error: {_escape(e)}</p>'
            try:
                return {"reply_html": err_html, "error": str(e)}
            except UnicodeEncodeError:
                # Windows console fallback — strip non-ASCII for the log.
                return {"reply_html": "<p>⚠ Error (see logs).</p>", "error": "see logs"}

    # --- handlers -----------------------------------------------------------

    def _handle_help(self, intent: Intent) -> str:
        return (
            "<p>Available commands:</p>"
            "<pre>status | equity | account\n"
            "models | fleet\n"
            "agents | swarm\n"
            "connectors | integrations\n"
            "repos | corpus\n"
            "news | forces | macro | calendar | events\n"
            "symbols | universe\n"
            "pending | signals\n"
            "confirm [latest | &lt;signal_id&gt;]\n"
            "cancel  [latest | all | &lt;signal_id&gt;]\n"
            "forecast &lt;SYM&gt;     → run all 13 models + ensemble\n"
            "timeframes &lt;SYM&gt;   → multi-TF hierarchical (HTF + LTF)\n"
            "run &lt;model_name&gt;    → forecast a single model\n"
            "patterns &lt;SYM&gt; | quantum &lt;SYM&gt; | why &lt;SYM&gt;\n"
            "usage               → OpenRouter token usage\n"
            "shutdown | quit | help</pre>"
        )

    def _handle_status(self, intent: Intent) -> str:
        s = self.dispatcher.status()
        rows = [
            ["Equity", f"${s['equity']:.2f}"],
            ["Balance", f"${s['balance']:.2f}"],
            ["Open positions", s["open_positions"]],
            ["Pending signals", len(s["pending_signals"])],
            ["Uptime (s)", f"{s['uptime_s']:.1f}"],
        ]
        return "<p>Account snapshot:</p>" + _html_table(rows, ["Metric", "Value"])

    def _handle_models(self, intent: Intent) -> str:
        if self.runtime is None or not getattr(self.runtime, "fleet", None):
            return "<p>Fleet not bound to runtime.</p>"
        rows = []
        for m in self.runtime.fleet.status_table():
            rows.append([m["name"], m["family"], m["status"],
                        "✓" if m["available"] else "✗"])
        return "<p>12-model predictive fleet:</p>" + _html_table(
            rows, ["Name", "Family", "Status", "Available"])

    def _handle_agents(self, intent: Intent) -> str:
        if self.runtime is None or not getattr(self.runtime, "swarm", None):
            return "<p>Agent swarm not bound.</p>"
        snap = self.runtime.agents_snapshot()
        agents = snap.get("agents", [])
        prof = snap.get("profile", {})
        chips = " ".join(
            f"<span class='badge'>{_escape(k)}: <b>{v}</b></span>"
            for k, v in (
                ("tier", prof.get("tier")),
                ("complexity", prof.get("complexity")),
                ("engaged", f"{snap.get('engaged', '?')}/{snap.get('total', '?')}"),
                ("tokens", f"{prof.get('max_input_tokens', 0)}+{prof.get('max_output_tokens', 0)}"),
            ) if v not in (None, "")
        )
        rows = [[a["name"], a["role"], a.get("description", ""),
                 "✓" if a.get("hit_rate") is not None else "—",
                 f"{a.get('hit_rate', 0) * 100:.0f}%" if a.get("hit_rate") is not None else "—",
                 f"×{a.get('multiplier', 1.0):.2f}"] for a in agents]
        return (
            "<p>🛰 <b>Agent swarm</b> (token-aware orchestration):</p>"
            + f"<p>{chips}</p>"
            + _html_table(rows, ["Agent", "Role", "Focus", "Learnt", "Hit", "Weight"])
            + f"<p class='hint'>{_escape(prof.get('reason', ''))}</p>"
        )

    def _handle_connectors(self, intent: Intent) -> str:
        if self.runtime is None or not getattr(self.runtime, "connectors", None):
            return "<p>Connector layer not bound to runtime.</p>"
        snap = self.runtime.connectors_snapshot()
        summary = snap.get("summary", {})
        chips = " ".join(
            f"<span class='badge {st}'>{_escape(STATUS_LABEL.get(st, st))}: <b>{summary.get(st, 0)}</b></span>"
            for st in ("ok", "degraded", "unconfigured", "offline", "mock")
        )
        rows = []
        for cat in snap.get("categories", []):
            rows.append([cat["label"], f"{cat['ok']}/{cat['count']}",
                         " ".join(
                             _escape(f"{c['key']}·{STATUS_LABEL.get(c['status'], c['status'])}")
                             for c in cat.get("connectors", []))])
        return (
            f"<p>🔌 <b>Connector layer</b> — {summary.get('total', 0)} integrations "
            f"across {len(snap.get('categories', []))} categories:</p>"
            + f"<p>{chips}</p>"
            + _html_table(rows, ["Category", "Healthy", "Connectors"])
        )

    def _handle_repos(self, intent: Intent) -> str:
        if self.runtime is None or not getattr(self.runtime, "repo_fabric", None):
            return "<p>Repo fabric not bound.</p>"
        rows = []
        for a in self.runtime.repo_fabric.adapters.values():
            st = a.status()
            rows.append([a.name, a.kind, "✓" if st["available"] else "✗",
                         st["path"] or "absent"])
        if not rows:
            return "<p>No relevant AI/trading repos found on this machine.</p>"
        return ("<p>📦 Quant/trading repos wired into Jarvis Reborn:</p>"
                + _html_table(rows, ["Repo", "Kind", "Found", "Path"]))

    def _handle_devices(self, intent: Intent) -> str:
        if self.runtime is None or not getattr(self.runtime, "devices", None):
            return "<p>Device registry not bound.</p>"
        devs = self.runtime.devices.list_devices()
        if not devs:
            return "<p>📡 No devices registered yet. Open <code>/devices</code> to connect your phone/PC.</p>"
        rows = [[d["name"], d["role"], d["host"], "✓" if d["online"] else "✗"]
                for d in devs]
        return ("<p>📡 Devices connected to this Jarvis:</p>"
                + _html_table(rows, ["Device", "Role", "Host", "Online"]))

    def _handle_news(self, intent: Intent) -> str:
        if self.runtime is None or not getattr(self.runtime, "news", None):
            return "<p>News service not bound.</p>"
        items = self.runtime.news.snapshot()[:10]
        if not items:
            return "<p class='hint'>No news yet — first fetch in &lt;3 min.</p>"
        rows = []
        for n in items:
            rows.append([
                n["title"][:60],
                n["source"],
                f"{n['sentiment']:+.2f}",
                ", ".join(sorted(n.get("relevance", {}).keys())[:3]) or "—",
            ])
        return "<p>Latest news:</p>" + _html_table(
            rows, ["Title", "Source", "Sentiment", "Relevant"])

    def _handle_forces(self, intent: Intent) -> str:
        if self.runtime is None or not getattr(self.runtime, "forces", None):
            return "<p>Forces service not bound.</p>"
        snaps = self.runtime.forces.snapshot()
        rows = []
        for k, f in snaps.items():
            rows.append([k, f.kind, f.last, f"{f.change_pct_5d:+.2f}%",
                         f.regime, {1: "+", -1: "-", 0: "0"}.get(f.direction_hint, "0")])
        return "<p>Macro forces:</p>" + _html_table(
            rows, ["Name", "Kind", "Last", "5d %", "Regime", "Hint"])

    def _handle_calendar(self, intent: Intent) -> str:
        if self.runtime is None or not getattr(self.runtime, "calendar", None):
            return "<p>Calendar not bound.</p>"
        cal = self.runtime.calendar
        upcoming = cal.upcoming(10)
        blackouts = [w for w in cal.blackouts() if w.is_active()]
        rows = []
        for e in upcoming:
            rows.append([
                e.name[:50], e.currency, e.impact,
                __import__('datetime').datetime.fromtimestamp(e.timestamp).strftime("%Y-%m-%d %H:%M"),
            ])
        out = "<p>Upcoming events:</p>" + _html_table(
            rows, ["Name", "Ccy", "Impact", "Time (UTC)"])
        if blackouts:
            out += "<p class='hint'>⚠ Active blackouts: " + \
                   ", ".join(w.event_name for w in blackouts) + "</p>"
        return out

    def _handle_symbols(self, intent: Intent) -> str:
        from .symbols import by_sector
        out = "<p>Instrument universe by sector:</p>"
        for sector, items in [
            ("fx", by_sector("fx")), ("metals", by_sector("metals")),
            ("energy", by_sector("energy")), ("grains", by_sector("grains")),
            ("softs", by_sector("softs")), ("indices", by_sector("indices")),
            ("crypto", by_sector("crypto")), ("rates", by_sector("rates")),
        ]:
            out += f"<p><b>{sector}</b> ({len(items)}): " + \
                   ", ".join(i.symbol for i in items) + "</p>"
        return out

    def _handle_pending(self, intent: Intent) -> str:
        s = self.dispatcher.pending_snapshot()
        if not s:
            return "<p>No pending signals.</p>"
        rows = [[p["symbol"],
                 "BUY" if p["direction"] == 1 else "SELL",
                 f"{p['confidence']:.2f}",
                 f"{p['entry_price']:.5f}",
                 f"{p['stop_loss']:.5f}",
                 f"{p['take_profit']:.5f}",
                 f"{p['expires_in_s']:.0f}s"] for p in s]
        return "<p>Pending signals:</p>" + _html_table(
            rows, ["Symbol", "Dir", "Conf", "Entry", "SL", "TP", "Expires"])

    def _handle_confirm(self, intent: Intent) -> str:
        arg = intent.args.get("arg", "latest")
        if arg == "latest":
            pending = self.dispatcher.pending_snapshot()
            if not pending:
                return "<p>No pending signals to confirm.</p>"
            target = max(pending, key=lambda p: p["staged_at"])["signal_id"]
        else:
            target = arg
        result = self.dispatcher.confirm(target, source="chat")
        if result is None:
            return f"<p>Signal <code>{_escape(target)}</code> not found.</p>"
        return (f"<p>✅ Confirmed <code>{_escape(target)}</code> — "
                f"success={result.success}, retcode={result.retcode}, "
                f"comment={_escape(result.comment)}</p>")

    def _handle_cancel(self, intent: Intent) -> str:
        arg = intent.args.get("arg", "latest")
        if arg == "all":
            n = self.dispatcher.cancel_all(source="chat", reason="chat: cancel all")
            return f"<p>🚫 Cancelled {n} pending signal(s).</p>"
        if arg == "latest":
            pending = self.dispatcher.pending_snapshot()
            if not pending:
                return "<p>No pending signals to cancel.</p>"
            target = max(pending, key=lambda p: p["staged_at"])["signal_id"]
        else:
            target = arg
        self.dispatcher.cancel(target, source="chat", reason="chat: cancel")
        return f"<p>🚫 Cancelled <code>{_escape(target)}</code>.</p>"

    def _handle_forecast(self, intent: Intent) -> str:
        sym = intent.args.get("symbol", "EURUSD")
        if self.runtime is None:
            return "<p>Runtime not bound.</p>"
        out = self.runtime.run_on_demand_forecast(sym)
        if not out:
            return f"<p>Couldn't run forecast for <code>{sym}</code>.</p>"
        rows = [[r.model_name, r.family, r.direction, f"{r.expected_return_pct:+.3f}%",
                 f"{r.confidence:.3f}", r.status] for r in out["models"]]
        return (
            f"<p>📈 Forecast for <b>{sym}</b> "
            f"(current={out['current_price']:.5f}):</p>"
            + _html_table(rows, ["Model", "Family", "Dir", "Ret", "Conf", "Status"])
            + f"<p>Ensemble: <b>dir {out['fusion_direction']:+d}</b> "
            f"conf <b>{out['fusion_confidence']:.3f}</b> "
            f"regime <b>{out['regime']}</b></p>"
        )

    def _handle_run_model(self, intent: Intent) -> str:
        name = intent.args.get("model", "")
        if self.runtime is None or not getattr(self.runtime, "fleet", None):
            return "<p>Fleet not bound.</p>"
        model = self.runtime.fleet.get(name)
        if model is None:
            return (f"<p>Model <code>{name}</code> not found. "
                    f"Available: {', '.join(self.runtime.fleet.names())}</p>")
        # Generate a tiny synthetic series and forecast.
        n = 200
        close = (np.arange(n, dtype=np.float64) * 0.05) + 100.0
        result = model.forecast(close, horizon=16)
        return (f"<p>🧠 <b>{result.model_name}</b> ({result.family}) "
                f"dir <b>{result.direction:+d}</b> ret <b>{result.expected_return_pct:+.3f}%</b> "
                f"conf <b>{result.confidence:.3f}</b> "
                f"({result.inference_ms:.1f} ms, status={result.status})</p>")

    def _handle_patterns(self, intent: Intent) -> str:
        sym = intent.args.get("symbol", "")
        if not sym or self.runtime is None:
            return ("<p>Pattern detection runs continuously for every symbol. "
                    "Try <code>patterns EURUSD</code> for a live read.</p>")
        snap = self.runtime.patterns_for(sym)
        if not snap:
            return f"<p>No candle/price data for <code>{sym}</code> right now.</p>"
        return (
            f"<p>📐 Patterns for <b>{sym}</b>:</p>"
            + _html_table([
                ["Candles", ", ".join(snap.get("candle_patterns") or []) or "none"],
                ["Chart", ", ".join(snap.get("chart_patterns") or []) or "none"],
                ["Trend", snap.get("trend_structure", "neutral")],
                ["Direction bias", f"{snap.get('direction_bias', 0):+d}"],
                ["Confidence", f"{snap.get('confidence', 0):.2f}"],
            ], ["Feature", "Value"])
        )

    def _handle_quantum(self, intent: Intent) -> str:
        sym = intent.args.get("symbol", "")
        if not sym or self.runtime is None:
            return ("<p>Quantum-inspired layer runs continuously. "
                    "Try <code>quantum EURUSD</code> for a live read.</p>")
        snap = self.runtime.quantum_for(sym)
        if not snap:
            return f"<p>No market data for <code>{sym}</code> right now.</p>"
        return (
            f"<p>⚛ Quantum read for <b>{sym}</b>:</p>"
            + _html_table([
                ["Direction", f"{snap.get('direction', 0):+d}"],
                ["Confidence", f"{snap.get('confidence', 0):.2f}"],
                ["Magnitude", f"{snap.get('magnitude', 0):.3f}"],
            ], ["Signal", "Value"])
        )

    def _handle_why(self, intent: Intent) -> str:
        sym = intent.args.get("symbol", "")
        if self.runtime is None:
            return "<p>Runtime not bound.</p>"
        explanation = self.runtime.explain_last_decision(sym)
        if not explanation:
            return f"<p>No recent decision recorded for <code>{sym}</code>.</p>"
        return explanation

    def _handle_timeframes(self, intent: Intent) -> str:
        sym = intent.args.get("symbol", "EURUSD")
        if self.runtime is None or getattr(self.runtime, "mtf", None) is None:
            return "<p>Multi-timeframe predictor not enabled.</p>"
        decision = self.runtime.run_on_demand_timeframes(sym)
        if not decision:
            return f"<p>Couldn't run multi-TF for <code>{sym}</code>.</p>"
        rows = []
        for tf, info in decision["tf_forecasts"].items():
            rows.append([
                tf,
                "BUY" if info["fusion_direction"] == 1 else ("SELL" if info["fusion_direction"] == -1 else "HOLD"),
                f"{info['fusion_confidence']:.2f}",
                f"{info['expected_return_pct']:+.3f}%",
                str(info["horizon"]),
            ])
        out = (
            f"<p>📊 <b>Multi-timeframe hierarchical</b> for <b>{sym}</b>:</p>"
            + _html_table(rows, ["TF", "Dir", "Conf", "Ret", "Horizon"])
            + f"<p>Trend (HTF): dir <b>{decision['trend_direction']:+d}</b> "
            f"conf <b>{decision['trend_confidence']:.2f}</b></p>"
            f"<p>Timing (LTF): dir <b>{decision['timing_direction']:+d}</b></p>"
            f"<p>Agreement: <b>{'YES ✓' if decision['agreement'] else 'NO ✗'}</b> "
            f"→ final <b>dir {decision['final_direction']:+d}</b> "
            f"conf <b>{decision['final_confidence']:.2f}</b></p>"
        )
        return out

    def _handle_usage(self, intent: Intent) -> str:
        if self.runtime is None or getattr(self.runtime, "openrouter", None) is None:
            return "<p>OpenRouter not configured.</p>"
        u = self.runtime.openrouter.usage
        return (
            "<p>OpenRouter usage (this session):</p>"
            "<pre>"
            f"keys configured:    {u['keys']}\n"
            f"prompt tokens:      {u['prompt_tokens']:,}\n"
            f"completion tokens:  {u['completion_tokens']:,}\n"
            f"total tokens:       {u['total_tokens']:,}\n"
            f"estimated cost:     ${u['cost_usd']:.4f} USD\n"
            "</pre>"
        )

    def _handle_shutdown(self, intent: Intent) -> str:
        from .listener.dispatcher import EventType
        self.dispatcher.publish(EventType.SHUTDOWN, {"source": "chat"}, source="chat")
        return "<p>⏻ Shutdown initiated.</p>"

    def _handle_unknown(self, intent: Intent) -> str:
        raw = intent.args.get("raw", "")
        return (f"<p>I heard: <q>{_escape(raw)}</q></p>"
                "<p class='hint'>Type <code>help</code> to see what I understand.</p>")


__all__ = ["ChatProcessor", "parse_intent", "Intent"]
