/* =============================================================
   Jarvis — Predictive Trading UI client
   v2.0: WebSocket-driven, keyboard-first, trading cockpit.
         Dark/light/auto themes, focus-visible, system accent.
   ============================================================= */
"use strict";

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
));
const fmt = (n, d = 2) => (n ?? 0).toFixed(d);
const money = n => Number(n ?? 0).toLocaleString("en-US", { style: "currency", currency: "USD" });
const signedMoney = n => {
  const v = Number(n ?? 0);
  return (v >= 0 ? "+" : "−") + money(Math.abs(v));
};

const STATUS_LABEL = {
  ok: "Live", degraded: "Degraded", unconfigured: "Needs key",
  offline: "Offline", mock: "Simulated", unknown: "Checking…",
};

const state = {
  token: localStorage.getItem("jarvis_token") || "",
  view: "chat",
  chatHistory: [],
  chatIdx: -1,
  equitySamples: [],
  slashCmd: "",
  slashSel: 0,
  slashItems: [],
  agentBuffer: [],
  execBuffer: [],
  lastRefresh: {},
  lastStatus: null,
  mt5: null,
  kill: false,
  mode: "paper",
  restarting: false,
};

const VIEWS = ["chat", "dashboard", "positions", "signals", "agents", "models",
               "connectors", "patterns", "quantum", "strategy", "repos", "symbols", "news", "forces",
               "calendar", "devices", "settings"];

const SLASH_COMMANDS = [
  { cmd: "/forecast", hint: "Run the full fleet + swarm on a symbol", need: "SYM" },
  { cmd: "/positions", hint: "Live positions + pending signals" },
  { cmd: "/connectors", hint: "Connector layer status (120 integrations)" },
  { cmd: "/signals", hint: "Pending signals" },
  { cmd: "/confirm", hint: "Confirm the latest staged signal" },
  { cmd: "/confirm-all", hint: "Confirm every pending signal", danger: true },
  { cmd: "/cancel-all", hint: "Cancel every pending signal", danger: true },
  { cmd: "/agents", hint: "Inspect the agent swarm" },
  { cmd: "/models", hint: "Model fleet status" },
  { cmd: "/news", hint: "Latest news + sentiment" },
  { cmd: "/forces", hint: "Macro forces / regime" },
  { cmd: "/calendar", hint: "Economic calendar & blackouts" },
  { cmd: "/symbols", hint: "Instrument universe" },
  { cmd: "/patterns", hint: "Live pattern read for a symbol", need: "SYM" },
  { cmd: "/quantum", hint: "Live quantum-layer read for a symbol", need: "SYM" },
  { cmd: "/strategy", hint: "Full strategy engine analysis for a symbol", need: "SYM" },
  { cmd: "/repos", hint: "Local quant/trading repo integrations" },
  { cmd: "/devices", hint: "Cross-device connectivity" },
  { cmd: "/status", hint: "Account snapshot" },
  { cmd: "/usage", hint: "OpenRouter token usage" },
  { cmd: "/shutdown", hint: "Gracefully stop Jarvis" },
  { cmd: "/help", hint: "Full command list" },
];

function resolveCommand(cmd, full) {
  const viewRows = ["/agents", "/models", "/connectors", "/positions", "/news", "/forces", "/calendar", "/signals", "/symbols", "/strategy", "/repos", "/devices"];
  const inputRows = ["/forecast", "/timeframes", "/patterns", "/quantum"];
  if (cmd === "/confirm-all") {
    closeSlash();
    confirmAll();
    return;
  }
  if (viewRows.includes(cmd)) { closeSlash(); return switchView(cmd.slice(1)); }
  if (inputRows.includes(cmd)) {
    const arg = (full || "").split(/\s+/).slice(1).filter(Boolean);
    if (arg.length) { closeSlash(); return userSend(full); }
    closeSlash();
    const input = $("#chat-input");
    input.value = cmd + " ";
    input.focus();
    autoGrow();
    renderSlash(cmd + " ");
    return;
  }
  closeSlash();
  return userSend(cmd);
}

// ================= API helpers =================
async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
  if (state.token && !path.endsWith("/health")) headers["X-Jarvis-Token"] = state.token;
  const init = { ...opts, headers };
  if (opts.timeout) {
    const ctl = new AbortController();
    init.signal = ctl.signal;
    const t = setTimeout(() => ctl.abort(), opts.timeout);
    try {
      const r = await fetch(path, init);
      if (!r.ok) {
        let msg; try { msg = (await r.json()).detail || r.statusText; } catch { msg = r.statusText; }
        throw new Error(`${r.status} ${msg}`);
      }
      return r.json();
    } finally { clearTimeout(t); }
  }
  const r = await fetch(path, init);
  if (!r.ok) {
    let msg; try { msg = (await r.json()).detail || r.statusText; } catch { msg = r.statusText; }
    throw new Error(`${r.status} ${msg}`);
  }
  return r.json();
}

// ================= Theme (dark / light / auto) =================
const THEME_KEY = "jarvis_theme";
function getThemePref() { return localStorage.getItem(THEME_KEY) || "light"; }
function themeMode(pref) {
  return pref === "auto"
    ? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")
    : pref;
}
function applyTheme(pref) {
  document.documentElement.dataset.theme = themeMode(pref);
}
function renderThemeBtn() {
  const btn = $("#theme-toggle");
  if (!btn) return;
  const pref = getThemePref();
  btn.textContent = pref === "dark" ? "☀" : pref === "light" ? "☾" : "◐";
  btn.title = pref === "auto"
    ? "Theme: auto (follows system) — click: dark"
    : pref === "dark"
      ? "Theme: dark — click: light"
      : "Theme: light — click: auto";
  btn.setAttribute("aria-label", btn.title);
}
function cycleTheme() {
  const pref = getThemePref();
  const next = pref === "dark" ? "light" : pref === "light" ? "auto" : "dark";
  localStorage.setItem(THEME_KEY, next);
  applyTheme(next);
  renderThemeBtn();
  toast(`🎨 ${next[0].toUpperCase() + next.slice(1)} theme`);
}
$("#theme-toggle")?.addEventListener("click", cycleTheme);
if (matchMedia) {
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => {
    if (getThemePref() === "auto") applyTheme("auto");
  });
}

// ================= View switching =================
function switchView(name) {
  if (!VIEWS.includes(name)) return;
  state.view = name;
  $$(".nav-item").forEach(b => b.classList.toggle("active", b.dataset.view === name));
  $$(".view").forEach(v => v.classList.toggle("active", v.dataset.view === name));
  if (name === "chat") setTimeout(() => $("#chat-input").focus(), 30);
  refreshView(name);
}
$$(".nav-item").forEach(b => b.addEventListener("click", () => switchView(b.dataset.view)));

$("#collapse-toggle").addEventListener("click", () => $("#sidebar").classList.toggle("collapsed"));

// ================= Toasts =================
function toast(msg, kind = "") {
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.innerHTML = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transform = "translateX(20px)"; }, 3800);
  setTimeout(() => el.remove(), 4300);
}

// ================= Chat =================
function thinkingMsg() {
  const wrap = document.createElement("div");
  wrap.className = "msg jarvis thinking-msg";
  wrap.innerHTML = `<div class="avatar">J</div><div class="bubble"><div class="thinking"><i></i><i></i><i></i></div></div>`;
  $("#chat-log").appendChild(wrap);
  $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
  return wrap;
}

function appendMessage(role, html, cls = "") {
  const log = $("#chat-log");
  const wrap = document.createElement("div");
  wrap.className = `msg ${role} ${cls}`;
  const av = document.createElement("div");
  av.className = "avatar";
  av.textContent = role === "user" ? "H" : (role === "system" ? "S" : "J");
  const bub = document.createElement("div");
  bub.className = "bubble";
  if (typeof html === "string") bub.innerHTML = html; else bub.appendChild(html);
  wrap.append(av, bub);
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
  return wrap;
}

async function userSend(text) {
  if (!text.trim()) return;
  const history = $("#chat-history");
  if (history) history.remove();
  appendMessage("user", `<p>${esc(text)}</p>`);
  state.chatHistory.push(text);
  state.chatIdx = state.chatHistory.length;
  $("#chat-input").value = "";
  autoGrow();
  const t = thinkingMsg();
  try {
    const j = await api("/chat", { method: "POST", body: JSON.stringify({ message: text }) });
    t.remove();
    if (j.reply_html) appendMessage("jarvis", j.reply_html);
    else if (j.reply) appendMessage("jarvis", `<p>${esc(j.reply)}</p>`);
    else if (j.error) appendMessage("system", `<p>⚠ ${esc(j.error)}</p>`, "err");
    refreshStatus();
  } catch (e) {
    t.remove();
    appendMessage("system", `<p>⚠ ${esc(e.message)}</p>`, "err");
  }
}

function autoGrow() {
  const el = $("#chat-input");
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 200) + "px";
}
$("#chat-input").addEventListener("input", autoGrow);

$("#chat-send").addEventListener("click", () => userSend($("#chat-input").value));
$("#chat-input").addEventListener("keydown", e => {
  const input = e.target;
  if (slashOpen()) {
    if (e.key === "ArrowDown") { e.preventDefault(); slashMove(1); return; }
    if (e.key === "ArrowUp") { e.preventDefault(); slashMove(-1); return; }
    if (e.key === "Escape") { e.preventDefault(); closeSlash(); return; }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const items = state.slashItems;
      if (items.length) resolveCommand(items[state.slashSel].cmd, input.value);
      else userSend(input.value);
      return;
    }
  }
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    userSend(input.value);
    return;
  }
  if (e.key === "ArrowUp" && input.value === "") {
    e.preventDefault();
    if (state.chatIdx > 0) {
      state.chatIdx--;
      input.value = state.chatHistory[state.chatIdx];
      autoGrow();
    }
  } else if (e.key === "ArrowDown" && input.value === "") {
    e.preventDefault();
    if (state.chatIdx < state.chatHistory.length - 1) {
      state.chatIdx++;
      input.value = state.chatHistory[state.chatIdx];
    } else { state.chatIdx = state.chatHistory.length; input.value = ""; }
    autoGrow();
  }
});

$("#chat-input").addEventListener("input", () => {
  const v = $("#chat-input").value;
  if (v.startsWith("/") && !v.includes("\n")) renderSlash(v);
  else closeSlash();
});

// ---- Slash command menu ----
function slashOpen() { return !$("#slash-menu").hidden; }
function renderSlash(v) {
  const menu = $("#slash-menu");
  menu.hidden = false;
  const parts = v.trim().split(/\s+/);
  const query = (parts[0] || "").toLowerCase();
  let items = SLASH_COMMANDS.filter(c => c.cmd.startsWith(query));
  if (!items.length) items = SLASH_COMMANDS;
  state.slashSel = 0;
  state.slashItems = items;
  menu.innerHTML = items.map((c, i) => `
    <div class="sm-item ${i === 0 ? "sel" : ""} ${c.danger ? "danger" : ""}" data-i="${i}">
      <b>${c.cmd}${c.need ? ` <span style='color:var(--accent)'>${c.need}</span>` : ""}</b>
      <small>${c.hint}</small>
    </div>`).join("");
  $$(".sm-item", menu).forEach(el => el.addEventListener("click", () => {
    const it = items[Number(el.dataset.i)];
    resolveCommand(it.cmd, $("#chat-input").value);
  }));
}
function closeSlash() {
  $("#slash-menu").hidden = true;
  $("#slash-menu").innerHTML = "";
  state.slashItems = [];
}
function slashMove(dir) {
  const items = state.slashItems;
  if (!items.length) return;
  const sel = $$(".sm-item", $("#slash-menu"))[state.slashSel];
  if (sel) sel.classList.remove("sel");
  state.slashSel = (state.slashSel + dir + items.length) % items.length;
  const next = $$(".sm-item", $("#slash-menu"))[state.slashSel];
  if (next) next.classList.add("sel");
}

// ================= Keyboard shortcuts =================
const KBD_VIEWS = { 1: "chat", 2: "dashboard", 3: "positions", 4: "signals",
                    5: "agents", 6: "models", 7: "connectors", 8: "patterns",
                    9: "quantum", 0: "symbols", n: "news", f: "forces",
                    c: "calendar", s: "strategy", r: "repos", d: "devices",
                    g: "settings" };
document.addEventListener("keydown", e => {
  const tag = (e.target.tagName || "").toLowerCase();
  const typing = tag === "input" || tag === "textarea" || e.target.isContentEditable;

  // Global: Ctrl+K → focus chat & open slash
  if (e.key === "k" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    switchView("chat");
    const input = $("#chat-input");
    input.focus();
    renderSlash("/");
    return;
  }
  // Alt+T → cycle theme
  if (e.key === "t" && e.altKey && !e.ctrlKey && !e.metaKey) {
    e.preventDefault();
    cycleTheme();
    return;
  }
  // Ctrl+Shift+C → cancel all pending
  if (e.key === "c" && e.ctrlKey && e.shiftKey) {
    e.preventDefault();
    cancelAll();
    return;
  }
  // Ctrl+Shift+Enter → confirm all pending
  if (e.key === "Enter" && e.ctrlKey && e.shiftKey) {
    e.preventDefault();
    confirmAll();
    return;
  }
  if (typing) {
    return;
  }
  if (e.key === "[" && !e.metaKey && !e.ctrlKey && !e.altKey) {
    e.preventDefault();
    $("#sidebar").classList.toggle("collapsed");
    return;
  }
  if (e.key in KBD_VIEWS) {
    const v = KBD_VIEWS[e.key];
    switchView(v);
  } else if (e.key === "/") {
    e.preventDefault();
    switchView("chat");
    renderSlash("/");
    $("#chat-input").focus();
  }
});

// ================= Quick search =================
$("#quick-search").addEventListener("focus", () => {
  if (state.view !== "chat") switchView("chat");
});
$("#quick-search").addEventListener("keydown", e => {
  if (e.key === "Enter") {
    const q = e.target.value.trim();
    if (/^[A-Za-z]{3,12}$/.test(q)) {
      userSend(`forecast ${q.toUpperCase()}`);
      e.target.value = "";
    }
  }
});
$("#quick-search").addEventListener("input", e => {
  const q = e.target.value.trim().toLowerCase();
  if (!q) return;
  const match = VIEWS.find(v => v === q || v.startsWith(q));
  if (match && match !== state.view) switchView(match);
});

// ================= Live ticker =================
async function refreshTicker() {
  const el = $("#ticker");
  if (!el) return;
  try {
    const j = await api("/prices");
    const syms = j.symbols || {};
    const entries = Object.values(syms);
    if (!entries.length) {
      el.innerHTML = `<span class="ticker-item muted">No live quotes — MT5 offline</span>`;
      return;
    }
    const key = s => s.symbol.replace(".", "_");
    el.innerHTML = entries.map(s => `
      <span class="ticker-item ${s.direction === 1 ? "up" : (s.direction === -1 ? "down" : "flat")}"
            data-sym="${esc(key(s))}" title="${esc(s.symbol)} · ${esc(s.timeframe)} · last ${fmt(s.price, 5)}">
        <span class="tsym">${esc(s.symbol)}</span>
        <span class="tprice">${fmt(s.price, 5)}</span>
        <span class="tchg">${s.direction === 1 ? "▲" : (s.direction === -1 ? "▼" : "—")} ${s.change_pct >= 0 ? "+" : ""}${fmt(s.change_pct, 2)}%</span>
      </span>`).join("");
    el.querySelectorAll(".ticker-item[data-sym]").forEach(item => {
      item.addEventListener("click", () => {
        userSend(`forecast ${item.dataset.sym.toUpperCase()}`);
        switchView("chat");
      });
    });
  } catch (e) {
    el.innerHTML = `<span class="ticker-item muted">Ticker offline</span>`;
  }
}

function tickClock() {
  const c = $("#ticker-clock");
  if (c) c.textContent = new Date().toLocaleTimeString("en-GB");
}

// ================= Status + sparkline =================
async function refreshStatus() {
  try {
    const s = await api("/status");
    state.lastStatus = s;
    $("#status-equity").textContent = money(s.equity);
    $("#status-pnl").textContent = signedMoney(s.daily_pnl);
    $("#status-pnl").style.color = (s.daily_pnl >= 0) ? "var(--good)" : "var(--bad)";
    $("#status-open").textContent = s.open_positions ?? 0;
    $("#status-pending").textContent = (s.pending_signals || []).length;
    const online = (s.equity > 0 || s.balance > 0);
    $("#status-mt5").textContent = online ? "MT5 online" : "MT5 offline";
    $("#dot-mt5").className = "dot " + (online ? "ok" : "warn");
    $("#status-regime").textContent = s.regime || "neutral";
    $("#status-uptime").textContent = fmt(s.uptime_s / 60, 0) + "m";
    state.equitySamples.push(s.equity || 0);
    if (state.equitySamples.length > 160) state.equitySamples.shift();
    if (state.view === "dashboard") refreshDashboard();
    renderActionBar(s.pending_signals || []);
    drawSpark();
  } catch (e) { /* transient */ }
}

function drawSpark() {
  const cv = $("#equity-spark");
  if (!cv) return;
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 240, h = cv.clientHeight || 44;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const pts = state.equitySamples;
  if (pts.length < 4) { renderPnlHUD(); return; }
  const min = Math.min(...pts), max = Math.max(...pts);
  const span = (max - min) || 1;
  const up = pts[pts.length - 1] >= pts[0];
  const color = up ? "#2ecc71" : "#ef5350";
  ctx.beginPath();
  pts.forEach((p, i) => {
    const x = (i / (pts.length - 1)) * (w - 4) + 2;
    const y = h - 4 - ((p - min) / span) * (h - 8);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.8;
  ctx.lineJoin = "round";
  ctx.stroke();
  ctx.lineTo(w, h); ctx.lineTo(2, h); ctx.closePath();
  ctx.fillStyle = up ? "rgba(46,204,113,.12)" : "rgba(239,83,80,.12)";
  ctx.fill();
}

// ================= Sticky Action Bar =================
function renderActionBar(pending) {
  const bar = $("#action-bar");
  const refresh = list => {
    const items = list || [];
    if (!items.length) { bar.hidden = true; return; }
    bar.hidden = false;
    $("#ab-count").textContent = `${items.length} signal${items.length > 1 ? "s" : ""} awaiting action`;
    const chips = items.slice(0, 4).map(s => `
      <span class="chip ${s.direction === 1 ? "buy" : "sell"}"
            title="${esc(s.symbol)} · conf ${fmt(s.confidence)} · expires ${Math.ceil(s.expires_in_s)}s">
        ${esc(s.symbol)} ${s.direction === 1 ? "▲" : "▼"} ${fmt(s.confidence)}
      </span>`).join("");
    const extra = items.length > 4 ? `<span class="chip more">+${items.length - 4}</span>` : "";
    $("#ab-chips").innerHTML = chips + extra;
  };
  if (pending === undefined) {
    api("/status").then(s => {
      const list = s.pending_signals || [];
      refresh(list);
      if (state.view === "dashboard") refreshDashboard();
    }).catch(() => bar.hidden = true);
    return;
  }
  refresh(pending);
}

window.confirmAll = async () => {
  try {
    const j = await api("/confirm-all", { method: "POST" });
    toast(`✅ Confirmed ${j.confirmed} signal${j.confirmed === 1 ? "" : "s"}`);
    refreshStatus();
    if (state.view === "signals") refreshSignals();
    if (state.view === "positions") refreshPositions();
  } catch (e) { toast(`⚠ ${esc(e.message)}`, "error"); }
};
window.cancelAll = async () => {
  try {
    const j = await api("/cancel-all", { method: "POST" });
    toast(`🚫 Cancelled ${j.cancelled} pending`);
    refreshStatus();
    if (state.view === "signals") refreshSignals();
    if (state.view === "positions") refreshPositions();
  } catch (e) { toast(`⚠ ${esc(e.message)}`, "error"); }
};
$("#ab-confirm-all").addEventListener("click", confirmAll);
$("#ab-cancel-all").addEventListener("click", cancelAll);

// ================= Views =================
async function refreshView(name) {
  const now = Date.now();
  if (state.lastRefresh[name] && now - state.lastRefresh[name] < 1400) return;
  state.lastRefresh[name] = now;
  try {
    if (name === "dashboard") await refreshDashboard();
    else if (name === "positions") await refreshPositions();
    else if (name === "signals") await refreshSignals();
    else if (name === "connectors") await refreshConnectors();
    else if (name === "agents") await refreshAgents();
    else if (name === "models") await refreshModels();
    else if (name === "symbols") await refreshSymbols();
    else if (name === "news") await refreshNews();
    else if (name === "forces") await refreshForces();
    else if (name === "calendar") await refreshCalendar();
    else if (name === "patterns") await refreshPatterns();
    else if (name === "quantum") await refreshQuantum();
    else if (name === "strategy") await refreshStrategy();
    else if (name === "repos") await refreshRepos();
    else if (name === "devices") await refreshDevices();
  } catch (e) {
    if (name === "chat") return;
    toast(`⚠ ${esc(name)}: ${esc(e.message)}`, "error");
  }
}

function posTable(positions) {
  const list = positions || [];
  if (!list.length) return '<p class="hint">No open positions.</p>';
  return `<table class="tbl">
    <tr><th>Symbol</th><th>Side</th><th>Volume</th><th>Open price</th><th>Profit</th></tr>
    ${list.map(p => `<tr>
      <td><b>${esc(p.symbol)}</b></td>
      <td><span class="dir ${p.type === "BUY" ? "buy" : "sell"}">${p.type === "BUY" ? "▲ BUY" : "▼ SELL"}</span></td>
      <td>${fmt(p.volume, 2)}</td>
      <td>${fmt(p.price_open, 5)}</td>
      <td class="${p.profit >= 0 ? "pos" : "neg"}">${signedMoney(p.profit)}</td>
    </tr>`).join("")}
  </table>`;
}

function signalCardHTML(s) {
  const dir = s.direction === 1 ? "buy" : "sell";
  return `
    <div class="card signal-card">
      <div class="dir-badge ${dir}">${s.direction === 1 ? "▲ BUY" : "▼ SELL"}</div>
      <div>
        <h3 style="margin:0 0 4px">${esc(s.symbol)} · conf <b>${fmt(s.confidence)}</b></h3>
        <div class="meta">Entry ${s.entry_price} · SL ${s.stop_loss} · TP ${s.take_profit} · expires ${Math.ceil(s.expires_in_s)}s</div>
        <div class="meta dim">staged ${new Date((s.staged_at || 0) * 1000).toLocaleTimeString()} · ${esc(s.signal_id.slice(0, 8))}</div>
      </div>
      <div class="signal-actions">
        <button class="action primary" onclick="confirmSignal('${s.signal_id}')">✓ Confirm</button>
        <button class="action danger" onclick="cancelSignal('${s.signal_id}')">✗ Cancel</button>
      </div>
      <details class="wide"><summary>Provenance</summary><pre>${esc(JSON.stringify(s, null, 2))}</pre></details>
    </div>`;
}
window.confirmSignal = async id => {
  try {
    await api("/confirm", { method: "POST", body: JSON.stringify({ signal_id: id }) });
    toast(`✅ Confirmed <code>${esc(id)}</code>`);
    refreshStatus();
    if (state.view === "signals") refreshSignals();
    if (state.view === "positions") refreshPositions();
  } catch (e) { toast(`⚠ ${esc(e.message)}`, "error"); }
};
window.cancelSignal = async id => {
  try {
    await api("/cancel", { method: "POST", body: JSON.stringify({ signal_id: id, reason: "ui" }) });
    toast(`Cancelled <code>${esc(id)}</code>`);
    refreshStatus();
    if (state.view === "signals") refreshSignals();
    if (state.view === "positions") refreshPositions();
  } catch (e) { toast(`⚠ ${esc(e.message)}`, "error"); }
};

async function refreshDashboard() {
  const [s, m, pipe, lg] = await Promise.all([
    api("/status"), api("/models"),
    api("/pipeline").catch(() => ({ pipeline: {} })),
    api("/log?lines=60").catch(() => ({ lines: [] })),
  ]);

  const pnlCls = (s.daily_pnl >= 0) ? "pos" : "neg";
  $("#d-metrics").innerHTML = `
    <div class="metric"><span>Equity</span><b>${money(s.equity)}</b></div>
    <div class="metric"><span>Balance</span><b>${money(s.balance)}</b></div>
    <div class="metric"><span>Daily PnL</span><b class="${pnlCls}">${signedMoney(s.daily_pnl)}</b></div>
    <div class="metric"><span>Open positions</span><b>${s.open_positions ?? 0}</b></div>
    <div class="metric"><span>Regime</span><b>${esc(s.regime || "neutral")}</b></div>
    <div class="metric"><span>Uptime</span><b>${fmt((s.uptime_s || 0) / 60, 0)}m</b></div>`;

  if (!state.mt5) refreshMT5();
  $("#d-mt5").innerHTML = mt5CardHTML();
  $("#d-risk").innerHTML = riskCardHTML();

  $("#d-positions").innerHTML = posTable(s.positions || []);

  const okModels = (m.models || []).filter(x => x.available).length;
  $("#d-regime").innerHTML = `
    <div class="row"><span>Predictive fleet</span><b>${okModels}/${(m.models || []).length} ready</b></div>
    <div class="row"><span>Regime</span><b>${esc(s.regime || "neutral")}</b></div>
    <div class="row"><span>Equity</span><b>${money(s.equity)}</b></div>
    <div class="row"><span>Drawdown guard</span><b>5% HWM</b></div>`;

  const sigs = s.pending_signals || [];
  $("#d-signals").innerHTML = sigs.length
    ? sigs.map(signalCardHTML).join("")
    : '<p class="hint">No active signals — Jarvis scans every symbol in real time.</p>';

  const cs = s.connector_summary || {};
  const total = cs.total ?? 0;
  $("#d-connectors").innerHTML = total ? `
    <div class="row"><span>Connectors</span><b>${total}</b></div>
    <div class="row"><span>Live</span><b class="pos">${cs.ok ?? 0}</b></div>
    <div class="row"><span>Needs key</span><b>${cs.unconfigured ?? 0}</b></div>
    <div class="row"><span>Mock / offline</span><b>${(cs.mock ?? 0) + (cs.offline ?? 0)}</b></div>
    <div class="row"><a class="link" href="#" onclick="event.preventDefault(); switchView('connectors')">Open connector layer →</a></div>`
    : '<p class="hint">Connector layer loading…</p>';

  $("#d-pipeline").textContent = JSON.stringify(pipe.pipeline || {}, null, 2)
    .replace(/"([^"]+)":/g, "$1: ");
  $("#d-log").textContent = (lg.lines || []).join("\n") || "No logs yet.";
}

// ================= Positions =================
async function refreshPositions() {
  const s = await api("/status");
  $("#p-live").innerHTML = posTable(s.positions || []);
  const pending = s.pending_signals || [];
  $("#p-pending").innerHTML = pending.length
    ? '<div class="sig-stack">' + pending.map(signalCardHTML).join("") + "</div>"
    : '<p class="hint">No pending signals.</p>';
  const log = state.execBuffer.slice(-12);
  $("#p-log").innerHTML = (log.length
    ? '<table class="tbl"><tr><th>Time</th><th>Signal</th><th>Event</th><th>Order</th></tr>' +
      log.map(e => `<tr><td>${new Date(e.at * 1000).toLocaleTimeString()}</td>
        <td><b>${esc(e.signal)}</b></td>
        <td class="${e.ok ? "pos" : "neg"}">${esc(e.event)}</td>
        <td>${esc(e.order_id || "—")}</td></tr>`).join("")
      + "</table>"
    : '<p class="hint">Executions will stream here.</p>');
}

// ================= Connectors =================
function connCardHTML(c) {
  const st = c.status || "unknown";
  const lat = (c.latency_ms === undefined || c.latency_ms === null) ? null : Number(c.latency_ms);
  const bits = [];
  if (lat !== null && c.status === "ok") bits.push(`<span class="lat">${fmt(lat, 0)}ms</span>`);
  if (c.last_checked) bits.push(`<span class="lat">${new Date(c.last_checked * 1000).toLocaleTimeString()}</span>`);
  return `
    <div class="conn-card ${st}" data-key="${esc(c.key.toLowerCase())}" data-status="${st}">
      <div class="conn-top">
        <b>${esc(c.key)}</b>
        <span class="st-pill ${st}">${esc(STATUS_LABEL[st] || st)}</span>
      </div>
      <div class="conn-meta dim">${esc(c.description || "")}</div>
      <div class="conn-meta">${bits.length ? bits.join(" · ") : "not probed"}${c.config_hint ? ` · <span class="hint">${esc(c.config_hint)}</span>` : ""}</div>
      ${c.detail ? `<div class="conn-meta dim">${esc(c.detail)}</div>` : ""}
    </div>`;
}

async function refreshConnectors() {
  const j = await api("/connectors");
  const sum = j.summary || {};
  const total = sum.total ?? j.total ?? 0;
  const chips = [
    `<div class="chip">Total <b>${total}</b></div>`,
    `<div class="chip st-ok">Live <b>${sum.ok ?? 0}</b></div>`,
    `<div class="chip st-degraded">Degraded <b>${sum.degraded ?? 0}</b></div>`,
    `<div class="chip st-unconfigured">Needs key <b>${sum.unconfigured ?? 0}</b></div>`,
    `<div class="chip st-offline">Offline <b>${sum.offline ?? 0}</b></div>`,
    `<div class="chip st-mock">Simulated <b>${sum.mock ?? 0}</b></div>`,
    `<div class="chip">Sweep <b>${j.last_sweep ? new Date(j.last_sweep * 1000).toLocaleTimeString() : "—"}</b></div>`,
  ];
  $("#c-summary").innerHTML = chips.join("");
  $("#c-filter").innerHTML = `
    <input id="conn-search" type="text" placeholder="Filter connectors…  (e.g. binance, telegram, openrouter)" autocomplete="off" spellcheck="false" />
    <select id="conn-status-filter">
      <option value="">All statuses</option>
      <option value="ok">Live</option>
      <option value="degraded">Degraded</option>
      <option value="unconfigured">Needs key</option>
      <option value="offline">Offline</option>
      <option value="mock">Simulated</option>
    </select>`;
  const list = j.categories || [];
  $("#c-list").innerHTML = list.length
    ? list.map(cat => `
      <details open class="cat-block">
        <summary><b>${esc(cat.label)}</b><span class="badge">${cat.ok}/${cat.count} live</span></summary>
        <div class="conn-grid">${(cat.connectors || []).map(connCardHTML).join("")}</div>
      </details>`).join("")
    : '<p class="hint">No connectors yet.</p>';
  applyConnFilter();
  bindConnFilter();
}

$("#c-filter").addEventListener("input", applyConnFilter);
$("#c-filter").addEventListener("change", applyConnFilter);
function bindConnFilter() { /* re-bound via delegation above; no-op guard */ }
function applyConnFilter() {
  const q = ($("#conn-search")?.value || "").toLowerCase();
  const stQ = ($("#conn-status-filter")?.value || "");
  $$("#c-list .conn-card").forEach(el => {
    const show = (!q || (el.dataset.key || "").includes(q)) && (!stQ || el.dataset.status === stQ);
    el.style.display = show ? "" : "none";
  });
}

// ================= Signals =================
async function refreshSignals() {
  const j = await api("/pending");
  const list = j.pending || [];
  const el = $("#d-signals-list");
  if (!list.length) { el.innerHTML = '<p class="hint">No pending signals. Jarvis scans every symbol in real time.</p>'; return; }
  el.innerHTML = '<div class="sig-stack">' + list.map(signalCardHTML).join("") + "</div>";
}

async function refreshModels() {
  const [j, track] = await Promise.all([
    api("/models"),
    api("/track").catch(() => ({ track: {} })),
  ]);
  const el = $("#d-models");
  el.innerHTML = "";
  (j.models || []).forEach(m => {
    const cls = m.status === "ready" ? "ready" : (m.status === "mock" ? "mock" : (m.status === "uninitialised" ? "loading" : "bad"));
    const t = (track.track || {})[m.name] || {};
    const card = document.createElement("div");
    card.className = "model-card " + cls;
    card.innerHTML = `
      <h4>${esc(m.name)} <span class="badge">${esc(m.family)}</span></h4>
      <div class="row"><span>Status</span><b>${esc(m.status)}</b></div>
      <div class="row"><span>Available</span><b>${m.available ? "✓" : "✗"}</b></div>
      ${t.hit_rate !== undefined ? `<div class="track">Track: hit <b>${(t.hit_rate * 100).toFixed(0)}%</b> · n=${Math.round(t.n || 0)} · factor <b>${t.profit_factor ?? 0}</b></div>` : ""}
      ${m.error ? `<div class="meta">${esc(m.error)}</div>` : ""}`;
    el.appendChild(card);
  });
}

async function refreshAgents() {
  const j = await api("/agents");
  const agents = j.agents || [];
  const profile = j.profile || {};
  const chips = [
    `<div class="chip">Tier <b>${esc(profile.tier || "—")}</b></div>`,
    `<div class="chip">Complexity <b>${profile.complexity ?? "—"}</b></div>`,
    `<div class="chip">Agents <b>${(j.engaged ?? agents.length)}/${(profile.agent_count ?? agents.length)}</b></div>`,
    `<div class="chip">Token budget <b>${profile.max_input_tokens || 0}+${profile.max_output_tokens || 0}</b></div>`,
    `<div class="chip">Last decision <b>${esc(j.last_decision || "—")}</b></div>`,
  ];
  $("#d-agents-summary").innerHTML = chips.join("");
  const el = $("#d-agents");
  el.innerHTML = agents.map(a => `
    <div class="agent-card ${(a.direction_hint || 0) === -1 ? "error" : ""}">
      <h4>${esc(a.name)} <span class="badge">${esc(a.role)}</span></h4>
      <div class="meta">${esc(a.description || "")}</div>
      ${a.direction_hint ? `<div class="row"><span>Vote</span><b>${a.direction_hint > 0 ? "BUY" : "SELL"}</b></div>` : ""}
      ${a.hit_rate !== undefined ? `<div class="meta">Learned: hit <b>${(a.hit_rate * 100).toFixed(0)}%</b> ×${a.multiplier ?? 1}</div>` : ""}
    </div>`).join("") || '<p class="hint">Swarm idle — run <code>/forecast SYM</code> to activate.</p>';
  renderAgentStream();
}

function renderAgentStream() {
  const el = $("#d-agent-stream");
  if (!el) return;
  const rows = state.agentBuffer.slice(-14).map(r => {
    const t = new Date(r.started_at * 1000).toLocaleTimeString();
    return `<div class="${r.status === "error" ? "err" : "live"}">${t} · ${esc(r.agent)} · ${esc(r.note || r.status)} · ${r.elapsed_ms.toFixed(0)}ms</div>`;
  });
  el.innerHTML = rows.join("") || '<div class="hint">Agent activity will stream here.</div>';
}

async function refreshSymbols() {
  const j = await api("/universe");
  const bySector = {};
  for (const [sym, meta] of Object.entries(j.universe || {})) {
    (bySector[meta.sector] = bySector[meta.sector] || []).push({ sym, ...meta });
  }
  $("#d-universe").innerHTML = Object.entries(bySector).map(([sector, list]) => `
    <details open>
      <summary><b>${esc(sector)}</b> (${list.length})</summary>
      <table>
        <tr><th>Symbol</th><th>Name</th><th>Sector</th><th>Daily %</th><th>Correlated</th></tr>
        ${list.map(x => `<tr>
          <td><b>${esc(x.sym)}</b></td><td>${esc(x.name)}</td><td>${esc(x.sector)}</td>
          <td>${x.avg_daily_range_pct}</td>
          <td>${(x.correlated_to || []).slice(0, 3).map(c => `<span class="badge">${esc(c)}</span>`).join("")}</td>
        </tr>`).join("")}
      </table>
    </details>`).join("");
}

async function refreshNews() {
  const j = await api("/news");
  $("#d-news").innerHTML = (j.items || []).slice(0, 30).map(n => `
    <div class="card feed-card">
      <h3>${esc(n.title)}</h3>
      <div class="meta">${esc(n.source)} · ${new Date(n.published * 1000).toLocaleString()} ·
        sentiment <b class="${n.sentiment >= 0 ? "pos" : "neg"}">${fmt(n.sentiment)}</b></div>
      <p class="sum">${esc(n.summary || "")}</p>
      ${Object.entries(n.relevance || {}).sort((a, b) => b[1] - a[1]).slice(0, 4)
        .map(([s, v]) => `<span class="badge">${esc(s)}: ${fmt(v)}</span>`).join("")}
    </div>`).join("") || '<p class="hint">No news yet — first fetch in &lt;3 min.</p>';
}

async function refreshForces() {
  const j = await api("/forces");
  const entries = Object.entries(j.forces || {});
  $("#d-forces").innerHTML = entries.length ? `<div class="grid-2">` + entries.map(([k, f]) => `
    <div class="card">
      <h3>${esc(k)} <span class="badge">${esc(f.kind)}</span></h3>
      <pre>last=${f.last}\n5d=${f.change_5d_pct}% · 20d=${f.change_20d_pct}%\nregime=${f.regime}\nhint=${f.direction_hint > 0 ? "+" : (f.direction_hint < 0 ? "-" : "0")}</pre>
    </div>`).join("") + "</div>" : '<p class="hint">Forces fetch in progress…</p>';
}

async function refreshCalendar() {
  const j = await api("/calendar");
  const upcoming = j.upcoming || [], blackouts = j.blackouts || [];
  $("#d-calendar").innerHTML = `
    <div class="card">
      <h3>Active Blackouts</h3>
      <pre style="color:${blackouts.length ? "var(--bad)" : "var(--text-2)"}">${blackouts.length
        ? blackouts.map(b => `${b.event}: ${new Date(b.start * 1000).toLocaleString()} → ${new Date(b.end * 1000).toLocaleString()}`).join("\n")
        : "No active blackouts."}</pre>
    </div>
    <div class="card wide">
      <h3>Upcoming Events</h3>
      <table>
        <tr><th>Time</th><th>Ccy</th><th>Impact</th><th>Name</th></tr>
        ${upcoming.map(e => `<tr>
          <td>${new Date(e.timestamp * 1000).toLocaleString()}</td><td>${esc(e.currency)}</td>
          <td><span class="badge ${e.impact === 3 ? "neg" : ""}">${"✸".repeat(e.impact || 0) || "•"}</span></td>
          <td>${esc(e.name)}</td></tr>`).join("")}
      </table>
    </div>`;
}

async function refreshPatterns() {
  let sym = null;
  try {
    const a = await api("/agents");
    sym = a.last_decision || null;
  } catch { /* swarm not ready */ }
  if (!sym) {
    $("#d-patterns").innerHTML = '<p class="hint">No symbol analysed yet — run <code>/patterns EURUSD</code> in chat.</p>';
    return;
  }
  try {
    const j = await api(`/patterns?symbol=${encodeURIComponent(sym)}`);
    const p = j.patterns;
    if (!p) {
      $("#d-patterns").innerHTML = `<p class="hint">No pattern data for ${esc(sym)} yet.</p>`;
      return;
    }
    const candles = (p.candle_patterns || []).slice(0, 6).map(c => `<span class="badge">${esc(c)}</span>`).join("") || '<span class="hint">none</span>';
    const charts = (p.chart_patterns || []).slice(0, 6).map(c => `<span class="badge">${esc(c)}</span>`).join("") || '<span class="hint">none</span>';
    const bias = p.direction_bias > 0 ? `+${p.direction_bias}` : p.direction_bias;
    $("#d-patterns").innerHTML = `
      <div class="grid-2">
        <div class="card"><h3>${esc(sym)} — Trend Structure</h3>
          <div class="row"><b>${esc(p.trend_structure || "neutral")}</b></div>
          <div class="meta">Direction bias <b>${bias ?? 0}</b> · confidence <b>${fmt(p.confidence)}</b></div>
        </div>
        <div class="card"><h3>Candles</h3>${candles}</div>
        <div class="card wide"><h3>Chart Patterns</h3>${charts}</div>
      </div>`;
  } catch (e) {
    $("#d-patterns").innerHTML = `<p class="hint">⚠ ${esc(e.message)}</p>`;
  }
}

async function refreshQuantum() {
  let sym = null;
  try {
    const a = await api("/agents");
    sym = a.last_decision || null;
  } catch { /* swarm not ready */ }
  if (!sym) {
    $("#d-quantum").innerHTML = '<p class="hint">No symbol analysed yet — run <code>/quantum EURUSD</code> in chat.</p>';
    return;
  }
  try {
    const j = await api(`/quantum?symbol=${encodeURIComponent(sym)}`);
    const q = j.quantum;
    if (!q) {
      $("#d-quantum").innerHTML = `<p class="hint">No quantum data for ${esc(sym)} yet.</p>`;
      return;
    }
    const dir = q.direction === 1 ? "▲ BUY" : (q.direction === -1 ? "▼ SELL" : "• HOLD");
    $("#d-quantum").innerHTML = `
      <div class="grid-2">
        <div class="card"><h3>${esc(sym)} — Quantum Decision</h3>
          <div class="row"><span>Verdict</span><b>${dir}</b></div>
          <div class="row"><span>Confidence</span><b>${fmt(q.confidence)}</b></div>
          <div class="row"><span>Magnitude</span><b>${fmt(q.magnitude)}</b></div>
        </div>
        <div class="card"><h3>Layer</h3>
          <p class="hint">Amplitude → QAOA → entanglement coupling. Every decision streams into each signal's provenance.</p>
        </div>
      </div>`;
  } catch (e) {
    $("#d-quantum").innerHTML = `<p class="hint">⚠ ${esc(e.message)}</p>`;
  }
}

// ================= Strategy Engine =================
async function refreshStrategy() {
  try {
    const j = await api("/strategy");
    const cfg = j.config || {};
    const s = $("#st-assessment"), c = $("#st-confluence"), p = $("#st-patterns"),
          h = $("#st-hedge"), pos = $("#st-positions");

    if (j.enabled && cfg && Object.keys(cfg).length) {
      $("#st-hedge-enable").checked = !!cfg.hedge_enabled;
      $("#st-hedge-thresh").value = cfg.hedge_threshold_pct ?? 5;
      $("#st-risk-trade").value = cfg.risk_per_trade ?? 1;
      $("#st-max-pos").value = cfg.max_positions ?? j.positions ?? 5;
      $("#st-conf-weight").value = cfg.confidence_weight ?? 0.6;
      $("#st-max-loss").value = cfg.max_daily_loss ?? 50;
      $("#st-max-dd").value = cfg.max_drawdown_pct ?? 10;
      $("#st-mtf-enable").checked = j.enabled;
    }

    s.innerHTML = `<div class="row"><span>Engine</span><b class="${j.enabled ? "pos" : "neg"}">${j.enabled ? "● LIVE" : "○ OFF"}</b></div>
      <div class="row"><span>Symbols analyzed</span><b>${j.symbols_analyzed ?? 0}</b></div>
      <div class="row"><span>Last run</span><b>${j.last_run ? new Date(j.last_run * 1000).toLocaleTimeString() : "—"}</b></div>`;

    const signals = j.signals || {};
    const sigKeys = Object.keys(signals);
    c.innerHTML = sigKeys.length
      ? sigKeys.slice(0, 12).map(sym => {
          const sig = signals[sym];
          const cf = sig.confluence || {};
          const dir = sig.action === "buy" ? "▲ BUY" : sig.action === "sell" ? "▼ SELL" : sig.action === "hedge" ? "🛡 HEDGE" : "• HOLD";
          return `<div class="row"><span>${esc(sym)}</span><b>${dir}</b><span class="meta dim">conf ${fmt(sig.confidence)} · ${sig.entry_grade}</span></div>`;
        }).join("")
      : '<p class="hint">No symbol analyzed yet — run <code>/strategy EURUSD</code> in chat.</p>';

    p.innerHTML = sigKeys.length
      ? sigKeys.slice(0, 6).map(sym => {
          const ps = (signals[sym].pattern_summary || {});
          return `<div class="row"><span>${esc(sym)}</span><b>${ps.total ?? 0} patterns</b></div>
          <div class="row"><span>Bullish</span><b class="pos">${ps.bullish ?? 0}</b></div>
          <div class="row"><span>Bearish</span><b class="neg">${ps.bearish ?? 0}</b></div>`;
        }).join("")
      : '<p class="hint">Pattern recognition idle.</p>';

    h.innerHTML = `<div class="row"><span>Hedge enabled</span><b>${j.positions != null ? "monitoring" : "config loaded"}</b></div>
      <div class="row"><span>Max drawdown</span><b>${cfg.max_drawdown_pct ?? 10}%</b></div>
      <div class="row"><span>Risk / trade</span><b>${cfg.risk_per_trade ?? 1}%</b></div>`;

    pos.innerHTML = sigKeys.length
      ? sigKeys.slice(0, 10).map(sym => {
          const sig = signals[sym];
          return `<div class="row"><span>${esc(sym)}</span><b>${fmt(sig.entry_price)}</b><span class="meta dim">conf ${fmt(sig.confidence)}</span></div>`;
        }).join("")
      : '<p class="hint">No positions tracked in strategy engine yet.</p>';
  } catch (e) {
    $("#st-assessment").innerHTML = `<p class="hint">⚠ ${esc(e.message)}</p>`;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const save = $("#st-save-config");
  if (save) save.addEventListener("click", async () => {
    try {
      const body = {
        hedge_enabled: $("#st-hedge-enable").checked,
        mtf_enabled: $("#st-mtf-enable").checked,
        hedge_threshold_pct: parseFloat($("#st-hedge-thresh").value) || 5,
        risk_per_trade: parseFloat($("#st-risk-trade").value) || 1,
        max_positions: parseInt($("#st-max-pos").value, 10) || 5,
        confidence_weight: parseFloat($("#st-conf-weight").value) || 0.6,
        max_daily_loss: parseFloat($("#st-max-loss").value) || 50,
        max_drawdown_pct: parseFloat($("#st-max-dd").value) || 10,
      };
      await api("/strategy/config", { method: "POST", body: JSON.stringify(body) });
      toast("✅ Strategy config saved");
      refreshStrategy();
    } catch (e) { toast(`⚠ ${esc(e.message)}`, "error"); }
  });
});

// ================= Repo Integrations =================
async function refreshRepos() {
  try {
    const j = await api("/repos");
    const probes = j.probes || {};
    const repos = j.repos || [];
    $("#rp-list").innerHTML = repos.length
      ? '<div class="grid-2">' + repos.map(r => {
          const p = probes[r.name] || {};
          const ok = p.python_ok;
          return `<div class="card">
            <h3>${esc(r.name)} <span class="dim">· ${esc(r.kind)}</span></h3>
            <div class="row"><span>Home</span><b>${esc(r.home || "absent")}</b></div>
            <div class="row"><span>Ready</span><b class="${ok ? "pos" : "neg"}">${ok ? "● python ready" : "○ not importable"}</b></div>
            ${p.detail ? `<div class="meta dim">${esc(p.detail)}</div>` : ""}
            ${r.capabilities ? `<div class="row"><span>Caps</span><span class="meta dim">${r.capabilities.join(", ")}</span></div>` : ""}
          </div>`;
        }).join("") + "</div>"
      : '<p class="hint">No repos discovered. Clone <code>LEAN</code>/<code>FinRL</code>/<code>qlib</code> into <code>~/</code> to unlock them.</p>';
  } catch (e) {
    $("#rp-list").innerHTML = `<p class="hint">⚠ ${esc(e.message)}</p>`;
  }
  try {
    const l = await api("/repos/lean");
    $("#rp-lean").innerHTML = `<div class="row"><span>Engine</span><b>${l.available ? "available" : "not installed"}</b></div>
      <div class="row"><span>Version</span><b>${esc(l.version || "—")}</b></div>
      <div class="row"><span>Launcher</span><b>${l.available === false || !l.launcher ? "absent" : esc(l.launcher)}</b></div>
      <div class="row"><span>Projects</span><b>${l.projects || 0}</b></div>
      <div class="row"><span>Backtests</span><b>${l.backtests || 0}</b></div>`;
  } catch (e) {
    $("#rp-lean").innerHTML = `<p class="hint">⚠ ${esc(e.message)}</p>`;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const leanBtn = $("#rp-lean-refresh");
  if (leanBtn) leanBtn.addEventListener("click", refreshRepos);
});

// ================= Devices =================
async function refreshDevices() {
  try {
    const j = await api("/devices");
    $("#dv-local").innerHTML = `<div class="row"><span>Name</span><b>${esc(j.local.name || "—")}</b></div>
      <div class="row"><span>Role</span><b>${esc(j.local.role || "hub")}</b></div>
      <div class="row"><span>Host</span><b>${esc(j.local.host || "—")}</b></div>
      <div class="row"><span>Online</span><b class="pos">${j.online ?? 0} peer(s)</b></div>`;
    const devs = j.devices || [];
    $("#dv-devices").innerHTML = devs.length
      ? devs.map(d => `<div class="row">
          <span>${esc(d.name)} <span class="meta dim">${esc(d.device_id || "")}</span></span>
          <b class="${d.online ? "pos" : "neg"}">${d.online ? "● online" : "○ offline"}</b>
          <span class="meta dim">${esc(d.role)} · ${esc(d.host || "")}:${d.port || ""}</span>
        </div>`).join("")
      : '<p class="hint">No devices registered yet.</p>';
    const proto = location.protocol === "https:" ? "https://" : "http://";
    const url = j.local.host ? `${proto}${j.local.host}:${location.port}` : location.origin;
    $("#dv-url").textContent = url;
  } catch (e) {
    $("#dv-devices").innerHTML = `<p class="hint">⚠ ${esc(e.message)}</p>`;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const addBtn = $("#dv-add");
  if (addBtn) addBtn.addEventListener("click", async () => {
    const id = ($("#dv-id").value || "").trim();
    const name = ($("#dv-name").value || "").trim();
    if (!id) { toast("⚠ device_id required", "error"); return; }
    try {
      const r = await api("/devices/register", { method: "POST", body: JSON.stringify({ device_id: id, name, role: "peer" }) });
      toast(`✅ Registered ${esc(r.name)}`);
      $("#dv-id").value = ""; $("#dv-name").value = "";
      refreshDevices();
    } catch (e) { toast(`⚠ ${esc(e.message)}`, "error"); }
  });
});

// ================= WebSocket =================
let ws, wsRetry = 0;
function connectWS() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen = () => {
    $("#dot-ws").className = "dot ok"; $("#status-ws").textContent = "WS live";
    wsRetry = 0;
  };
  ws.onclose = () => {
    $("#dot-ws").className = "dot warn"; $("#status-ws").textContent = "WS reconnecting";
    setTimeout(connectWS, Math.min(2000 * (wsRetry + 1), 10000));
    wsRetry++;
  };
  ws.onerror = () => { $("#dot-ws").className = "dot bad"; };
  ws.onmessage = e => {
    try { handleEvent(JSON.parse(e.data)); } catch { /* ignore non-JSON */ }
  };
}

function pushExec(event, signal, order_id, ok) {
  state.execBuffer.push({ at: Date.now() / 1000, signal, event, order_id, ok });
  if (state.execBuffer.length > 60) state.execBuffer.splice(0, state.execBuffer.length - 60);
  if (state.view === "positions") refreshPositions();
}

function handleEvent(evt) {
  const t = evt.type;
  const p = evt.payload || {};
  if (t === "stage") {
    const sig = p.signal || {};
    const dir = sig.direction === 1 ? "▲ BUY" : "▼ SELL";
    toast(`🎯 New signal staged: <b>${esc(sig.symbol)}</b> ${dir} conf=${fmt(sig.confidence)}`);
    if (state.view === "chat") appendMessage("jarvis", `<p>🎯 New signal staged: <code>${esc(sig.symbol)}</code> ${dir} conf=${fmt(sig.confidence)}</p>`);
    pushExec("staged", sig.symbol, null, true);
    refreshStatus();
  }
  else if (t === "executed") {
    toast(`✅ Executed ${esc(p.signal_id || "")} order_id=${esc((p.result || {}).order_id)}`);
    pushExec("executed", (p.result || {}).signal_id || p.signal_id || "?", (p.result || {}).order_id, true);
    refreshStatus();
  }
  else if (t === "failed") {
    toast(`❌ Failed: ${esc(p.comment || "")}`, "error");
    pushExec("failed", p.signal_id || "?", null, false);
    refreshStatus();
  }
  else if (t === "cancelled") {
    if (!p.bulk) { toast(`Cancelled ${esc(p.signal_id || "")}`); pushExec("cancelled", p.signal_id || "?", null, false); }
    refreshStatus();
  }
  else if (t === "equity_alert") {
    toast(`⚠ Equity drawdown ${fmt(p.drawdown_pct)}% — pending cancelled`, "error");
  }
  else if (t === "risk") {
    state.kill = !!p.paused;
    renderKillButton(state.kill);
    toast(state.kill ? "⛔ Kill switch ON — algo staging paused" : "✅ Kill switch OFF — algo trading live",
          state.kill ? "error" : "ok");
    refreshStatus();
  }
  else if (t === "agents") {
    if (state.view === "agents") { refreshAgents(); }
    else if (state.view === "chat" && p.votes && p.symbol) {
      const d = p.direction;
      appendMessage("jarvis", `<p>🛰 Swarm for <b>${esc(p.symbol)}</b>: <b>${d > 0 ? "BUY" : d < 0 ? "SELL" : "HOLD"}</b> conf ${fmt(p.confidence)} · ${esc(p.summary || "")}</p>`);
    }
  }
  else if (t === "report") {
    state.agentBuffer.push({ agent: p.agent, role: p.role, status: p.status, note: p.note, elapsed_ms: p.elapsed_ms, started_at: p.started_at });
    if (state.agentBuffer.length > 60) state.agentBuffer.splice(0, state.agentBuffer.length - 60);
    if (state.view === "agents") renderAgentStream();
  }
  else if (t === "news" && state.view === "news") refreshNews();
  else if (t === "forces" && state.view === "forces") refreshForces();
  else if (t === "calendar" && state.view === "calendar") refreshCalendar();
  else if (t === "voice") {
    toast(`🎙 Voice: ${esc(p.text || p.intent || "")}`);
    if (p.reply_html && state.view === "chat") {
      appendMessage("user", `<p>🎙 ${esc(p.text || "")}</p>`);
      appendMessage("jarvis", p.reply_html);
    }
  }
  else if (t === "hello" && state.view === "dashboard" && p.pending_signal_events === undefined) {
    refreshDashboard();
  }
  else if (t === "status" && state.view === "dashboard") refreshDashboard();
}

// ================= Voice (push to talk) =================
$("#btn-voice").addEventListener("click", async e => {
  const btn = e.currentTarget;
  btn.classList.add("listening");
  btn.title = "Listening…";
  toast("🎙 Listening — speak after the beep…");
  try {
    const j = await api("/listen", { method: "POST", body: JSON.stringify({}) });
    if (j.text) {
      toast(`🎙 Heard: <b>${esc(j.text)}</b>`);
      if (state.view === "chat") {
        appendMessage("user", `<p>🎙 ${esc(j.text)}</p>`);
        if (j.reply_html) appendMessage("jarvis", j.reply_html);
        else appendMessage("jarvis", `<p>${esc(j.reply || "Command received.")}</p>`);
      }
    } else {
      toast("🎙 Nothing heard (silence)");
    }
  } catch (err) {
    toast(`🎙 Voice error: ${esc(err.message)}`, "error");
  } finally {
    btn.classList.remove("listening");
    btn.title = "Push to talk (Ctrl+Alt+J)";
  }
});

// ================= Settings / actions =================
$$("button.action").forEach(b => b.addEventListener("click", async () => {
  const a = b.dataset.action;
  if (a === "cancel-all") {
    cancelAll();
  } else if (a === "kill") {
    toggleKillSwitch();
  } else if (a === "shutdown") {
    if (!confirm("Shut down Jarvis?")) return;
    try { await api("/shutdown", { method: "POST" }); toast("⏻ Shutdown initiated."); }
    catch (e) { toast(`⚠ ${esc(e.message)}`, "error"); }
  } else if (a === "usage") {
    userSend("usage");
    switchView("chat");
  }
}));
$("#toggle-token").addEventListener("click", () => {
  const input = $("#set-token");
  input.type = input.type === "password" ? "text" : "password";
});
$("#set-token").addEventListener("change", e => {
  state.token = e.target.value;
  localStorage.setItem("jarvis_token", state.token);
  toast("🔑 Token saved.");
  refreshStatus();
});

// ================= MT5 terminal + kill switch =================
window.toggleKillSwitch = async target => {
  try {
    const want = target === undefined ? !state.kill : target;
    const j = await api("/kill-switch", { method: "POST", body: JSON.stringify({ paused: want }) });
    state.kill = !!j.paused;
    renderKillButton(state.kill);
    toast(state.kill ? "⛔ Kill switch ON — algo staging paused" : "✅ Kill switch OFF — algo trading live");
  } catch (e) { toast(`⚠ ${esc(e.message)}`, "error"); }
};

function renderKillButton(paused) {
  state.kill = !!paused;
  const tiny = $("#btn-kill");
  if (tiny) {
    tiny.textContent = state.kill ? "⛔ Kill switch: ON" : "⛔ Kill switch: OFF";
    tiny.classList.toggle("paused", state.kill);
  }
  const dash = $("#btn-dash-kill");
  if (dash) {
    dash.textContent = state.kill ? "▶ Resume algo" : "⛔ Kill switch";
    dash.classList.toggle("paused", state.kill);
  }
  const broker = $("#status-broker");
  if (broker && !broker._override) {
    broker.textContent = state.kill ? "Algo: PAUSED" : (broker.dataset.live ? "Algo: LIVE" : broker.textContent);
  }
}

// ================= Execution mode (paper / live) =================
function renderExecutionMode() {
  const m = state.mode === "live" ? "LIVE" : "PAPER";
  const pill = $("#status-mode");
  if (pill) {
    pill.textContent = state.restarting ? "RESTARTING…" : m;
    pill.className = "mode-pill " + (state.restarting
      ? "restarting"
      : state.mode === "live" ? "live" : "paper");
  }
  for (const b of $$("[data-mode-btn]")) b.disabled = state.restarting;
  const liveBtn = $("#btn-mode-live");
  const paperBtn = $("#btn-mode-paper");
  if (liveBtn) liveBtn.classList.toggle("active", state.mode === "live" && !state.restarting);
  if (paperBtn) paperBtn.classList.toggle("active", state.mode === "paper" && !state.restarting);
  const banner = $("#mode-banner");
  if (banner) {
    banner.className = "mode-banner " + (state.restarting ? "restarting" : state.mode);
    banner.innerHTML = state.restarting
      ? `⚠ Execution <b>${state.mode.toUpperCase()}</b> applied — restarting Jarvis…`
      : state.mode === "live"
        ? `🔴 <b>LIVE execution</b> — real orders go to the broker account logged into MetaTrader`
        : `🟢 <b>Paper Money</b> — simulated fills, journaled in <code>cache/paper/</code>`;
  }
}

async function switchExecutionMode(mode) {
  if (mode === state.mode) return;
  const toLive = mode === "live";
  if (state.restarting) {
    toast("Restart in progress — the UI returns automatically", "error");
    return;
  }
  if (toLive) {
    const ok = confirm(
      "Switch to LIVE execution?\n\n" +
      "Real market orders will be sent to the broker account currently logged " +
      "into MetaTrader 5.\n\n" +
      "• Verify the terminal is logged into your REAL broker account — not demo.\n" +
      "• Jarvis restarts automatically; trading resumes in ~40 seconds.\n\n" +
      "Cancel to stay on Paper Money."
    );
    if (!ok) return;
  }
  try {
    const r = await api("/execution/mode", {
      method: "POST",
      body: JSON.stringify({ mode, confirm: toLive }),
    });
    state.mode = r.mode;
    state.restarting = !!r.restarting;
    renderExecutionMode();
    toast(r.restarting
      ? `🚀 ${toLive ? "LIVE" : "Paper"} mode set — restarting (~40 s)`
      : `✔ Execution mode: ${mode}`);
    if (r.restarting) pollAfterRestart();
  } catch (e) {
    toast(`⚠ ${esc(e.message)}`, "error");
  }
}

function pollAfterRestart() {
  const t0 = Date.now();
  const iv = setInterval(async () => {
    const dead = Date.now() - t0 > 120000;
    try {
      const m = state.mode;
      const r = await api("/execution/mode", { timeout: 4000 });
      if (r.mode === m && !r.restarting) {
        clearInterval(iv);
        toast("✔ Restart complete — execution mode applied");
        await refreshStatus();
        await refreshMT5();
        renderExecutionMode();
        if (state.view === "dashboard") refreshDashboard();
      } else if (dead) {
        clearInterval(iv);
        toast("⚠ Restart still pending — refresh the page", "error");
      }
    } catch {
      if (dead) { clearInterval(iv); toast("⚠ Jarvis offline after restart", "error"); }
    }
  }, 3000);
}

window.toggleAutopilot = async () => {
  try {
    const cur = (state.autopilot || {}).enabled;
    const want = !cur;
    const j = await api("/autopilot", { method: "POST", body: JSON.stringify({ enabled: want }) });
    state.autopilot = j;
    toast(want ? "Autopilot ON — auto-confirming staged signals" : "Autopilot OFF — manual confirm required");
    if (state.view === "dashboard") refreshDashboard();
  } catch (e) {
    toast("Failed to toggle autopilot: " + String(e).slice(0, 60), "error");
  }
};

async function refreshMT5() {
  try {
    const j = await api("/mt5");
    const t0 = Date.now();
    state.mt5 = j;
    if (j.execution_mode && !j.restarting) state.mode = j.execution_mode;
    if (typeof j.restarting === "boolean") state.restarting = j.restarting;
    state.autopilot = j.autopilot || {};
    renderExecutionMode();
    const acc = j.account || {};
    const term = j.terminal || {};
    if (j.connected) {
      $("#acct-line").textContent = `#${acc.login} @ ${acc.server} · ${acc.currency} · 1:${acc.leverage}`;
      const live = !!term.trade_allowed;
      $("#dot-broker").className = "dot " + (live ? "ok" : "warn");
      const broker = $("#status-broker");
      broker.dataset.live = live ? "1" : "";
      broker.textContent = state.kill ? "Algo: PAUSED" : (live ? "Algo: LIVE" : "Algo: BLOCKED");
    } else {
      $("#acct-line").textContent = "No broker session — open MetaTrader 5 and log in";
      $("#dot-broker").className = "dot warn";
      const broker = $("#status-broker");
      broker.dataset.live = "";
      broker.textContent = state.kill ? "Algo: PAUSED" : "Algo: awaiting broker";
    }
    renderKillButton((j.risk || {}).paused);
    if (state.view === "dashboard") refreshDashboard();
  } catch (e) { /* keep last known */ }
}

function mt5CardHTML() {
  const j = state.mt5;
  if (!j) return '<p class="hint">Connecting to MetaTrader 5…</p>';
  const term = j.terminal || {};
  const modeRow = `
    <div class="row"><span>Execution</span>
      <b><span class="mode-value ${state.mode === "live" ? "live" : "paper"}">${state.mode === "live" ? "LIVE" : "Paper Money"}</span></b>
    </div>`;
  if (!j.connected) return `
    <div id="mode-banner"></div>
    <div class="conn-flag off"><b>MetaTrader 5 not connected</b></div>
    <div class="meta">Run <code>terminal64.exe</code> and log into your broker to activate algo execution.</div>
    <div class="row"><span>Terminal</span><b>${esc((term.terminal_path || "default path"))}</b></div>
    <div class="row"><span>Symbols watched</span><b>${(j.symbols || []).length}</b></div>
    <div class="row"><span>Timeframe</span><b>${esc(j.timeframe || "—")}</b></div>
    ${modeRow}`;
  const acc = j.account || {};
  const risk = j.risk || {};
  return `
    <div id="mode-banner"></div>
    <div class="row"><span>Account</span><b>#${acc.login} @ ${esc(acc.server)}</b></div>
    <div class="row"><span>Broker</span><b>${esc(acc.company || "—")}</b></div>
    <div class="row"><span>Leverage</span><b>1:${acc.leverage}</b></div>
    <div class="row"><span>Free margin</span><b>${money(acc.margin_free)}</b></div>
    <div class="row"><span>Margin level</span><b>${fmt(acc.margin_level, 0)}%</b></div>
    <div class="row"><span>Algo trading</span><b class="${term.trade_allowed ? "pos" : "neg"}">${term.trade_allowed ? "ALLOWED" : "BLOCKED"}</b></div>
    ${modeRow}
    <div class="row" style="margin-top:8px"><button class="action ${risk.paused ? "primary" : "danger"}" id="btn-dash-kill" onclick="toggleKillSwitch()">${risk.paused ? "▶ Resume algo" : "⛔ Kill switch"}</button></div>`;
}

function riskCardHTML() {
  const s = state.lastStatus || {};
  const j = state.mt5;
  const risk = (j && j.risk) || {};
  const openCount = s.open_positions ?? (s.positions && s.positions.length) ?? 0;
  const drawdown = (s.daily_pnl !== undefined && s.daily_pnl < 0) ? s.daily_pnl : 0;
  return `
    <div class="row"><span>Status</span><b class="${risk.paused ? "neg" : "pos"}">${risk.paused ? "PAUSED (kill switch)" : "LIVE"}</b></div>
    <div class="row"><span>Max open positions</span><b>${risk.max_open_positions ?? "—"}</b></div>
    <div class="row"><span>Now open</span><b>${openCount}</b></div>
    <div class="row"><span>Max daily loss</span><b>${money(risk.max_daily_loss ?? 0)}</b></div>
    <div class="row"><span>Open float PnL</span><b class="${(s.daily_pnl ?? 0) >= 0 ? "pos" : "neg"}">${signedMoney(s.daily_pnl)}</b></div>
    <div class="row"><span>Drawdown guard</span><b>5% HWM (auto-cancel)</b></div>
    <div class="row"><span>Execution mode</span><b class="${state.mode === "live" ? "neg" : "pos"}">${state.mode === "live" ? "LIVE" : "Paper Money"}</b></div>
    <div class="row" style="margin-top:10px">
      <div class="seg">
        <button type="button" id="btn-mode-paper"  data-mode-btn onclick="switchExecutionMode('paper')">Paper Money</button>
        <button type="button" id="btn-mode-live"   data-mode-btn onclick="switchExecutionMode('live')">Live</button>
      </div>
    </div>
    <div class="row" style="margin-top:12px"><span>Autopilot</span><b class="${(state.autopilot || {}).enabled ? "pos" : "neg"}">${(state.autopilot || {}).enabled ? "ON" : "OFF"}</b></div>
    <div class="row"><span>Auto-confirm threshold</span><b>${((state.autopilot || {}).min_confidence ?? 0.52).toFixed(2)} conf / ${((state.autopilot || {}).min_agreement ?? 0.50).toFixed(2)} agree / ${(state.autopilot || {}).min_models ?? 3} models</b></div>
    <div class="row"><span>Auto-orders executed</span><b>${(state.autopilot || {}).auto_confirmed_total ?? 0}</b></div>
    <div class="row" style="margin-top:6px">
      <button type="button" class="action ${(state.autopilot || {}).enabled ? "danger" : ""}" onclick="toggleAutopilot()" style="width:100%">${(state.autopilot || {}).enabled ? "Pause Autopilot" : "Enable Autopilot"}</button>
    </div>
    <p class="hint" style="margin-top:8px">Switching mode applies after a short restart. Live sends real orders — verify the broker account in MT5.</p>`;
}

// ================= Boot =================
renderThemeBtn();
connectWS();
refreshStatus();
refreshMT5();
refreshTicker();
tickClock();
renderActionBar();
setInterval(refreshStatus, 3500);
setInterval(refreshMT5, 12000);
setInterval(refreshTicker, 5000);
setInterval(tickClock, 1000);
setInterval(() => {
  const active = $$(".view.active").map(v => v.dataset.view).find(v => v !== "chat");
  if (active && active !== "agents") refreshView(active);
}, 8000);
window.addEventListener("resize", drawSpark);