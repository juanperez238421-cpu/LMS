const $ = (id) => document.getElementById(id);

function clamp01(x) { return Math.max(0, Math.min(1, x)); }
function fmtPct(x) { return (x === null || x === undefined) ? "-" : `${Number(x).toFixed(1)}%`; }
function fmtNum(x) { return (x === null || x === undefined) ? "-" : `${Number(x).toFixed(1)}`; }
function safeText(v) { return (v === null || v === undefined || v === "") ? "-" : String(v); }
function numOrNull(v) { const n = Number(v); return Number.isFinite(n) ? n : null; }
function clampPct(v) { const n = numOrNull(v); return n === null ? null : Math.max(0, Math.min(100, n)); }

let paused = false;
let frozenByDeath = false;
let freezeEnabled = true;
let rxCount = 0;
let lastRxT = performance.now();
let lastFrameTs = 0;

const events = [];
const maxEvents = 120;
let eventFilter = "all";

const maxHist = 600;
const tHist = [];
const hpHist = [];
const stHist = [];
const dinHist = [];
const doutHist = [];
let hpDisplayLast = null;
let lastEnemyTrack = null;

function pushHist(t, hp, st, din, dout) {
  tHist.push(t);
  hpHist.push(hp);
  stHist.push(st);
  dinHist.push(din);
  doutHist.push(dout);
  if (tHist.length > maxHist) {
    tHist.shift();
    hpHist.shift();
    stHist.shift();
    dinHist.shift();
    doutHist.shift();
  }
}

function setBar(el, pct) {
  const p = clamp01((pct ?? 0) / 100.0);
  el.style.width = `${(p * 100).toFixed(1)}%`;
}

function inferTeam(entity) {
  const team = String(entity?.team ?? "").toLowerCase();
  if (team === "ally" || team === "enemy") return team;
  const kind = String(entity?.kind ?? "").toLowerCase();
  if (kind === "bot") return "ally";
  if (kind === "player") return "enemy";
  return "unknown";
}

function collectRadarTracks(frame) {
  const tracks = { allies: [], enemies: [] };
  const entities = Array.isArray(frame.entities) ? frame.entities : [];
  for (const e of entities) {
    const p = e?.anchor_xy;
    if (!Array.isArray(p) || p.length !== 2) continue;
    const x = clamp01(Number(p[0]));
    const y = clamp01(Number(p[1]));
    const dx = x - 0.5;
    const dy = y - 0.5;
    const dist = Math.min(1, Math.hypot(dx, dy) / 0.70710678118);
    const item = {
      name: safeText(e?.name),
      conf: Number(e?.conf ?? 0),
      x,
      y,
      dist,
    };
    const team = inferTeam(e);
    if (team === "ally") tracks.allies.push(item);
    if (team === "enemy") tracks.enemies.push(item);
  }

  const enemyVisible = !!frame.enemy_visible;
  const enemyXY = frame.enemy_xy;
  if (enemyVisible && Array.isArray(enemyXY) && enemyXY.length === 2 && tracks.enemies.length === 0) {
    const x = clamp01(Number(enemyXY[0]));
    const y = clamp01(Number(enemyXY[1]));
    const dx = x - 0.5;
    const dy = y - 0.5;
    const dist = Math.min(1, Math.hypot(dx, dy) / 0.70710678118);
    lastEnemyTrack = {
      name: "Enemy",
      conf: Number(frame.enemy_conf ?? 0),
      x,
      y,
      dist,
      ts: Date.now(),
    };
    tracks.enemies.push(lastEnemyTrack);
  } else if (lastEnemyTrack && (Date.now() - lastEnemyTrack.ts) <= 1500) {
    tracks.enemies.push(lastEnemyTrack);
  }

  return tracks;
}

function radarDraw(radarState) {
  const canvas = $("radar");
  const ctx = canvas.getContext("2d");
  const w = (canvas.width = canvas.clientWidth);
  const h = (canvas.height = Number(canvas.getAttribute("height") || 180));

  ctx.clearRect(0, 0, w, h);
  const cx = w / 2;
  const cy = h / 2;
  const r = Math.min(w, h) * 0.42;

  ctx.globalAlpha = 0.9;
  ctx.lineWidth = 1;
  for (const k of [0.25, 0.5, 0.75, 1.0]) {
    ctx.beginPath();
    ctx.arc(cx, cy, r * k, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(cx - r, cy);
  ctx.lineTo(cx + r, cy);
  ctx.moveTo(cx, cy - r);
  ctx.lineTo(cx, cy + r);
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(cx, cy, 4, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(248,250,252,0.95)";
  ctx.fill();

  const allies = radarState?.allies ?? [];
  const enemies = radarState?.enemies ?? [];

  for (const a of allies) {
    const px = cx + ((a.x - 0.5) * 2 * r);
    const py = cy + ((a.y - 0.5) * 2 * r);
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(78,161,255,0.88)";
    ctx.fill();
  }

  for (const e of enemies) {
    const px = cx + ((e.x - 0.5) * 2 * r);
    const py = cy + ((e.y - 0.5) * 2 * r);
    const sz = 6;
    ctx.beginPath();
    ctx.moveTo(px, py - sz);
    ctx.lineTo(px + sz, py + sz);
    ctx.lineTo(px - sz, py + sz);
    ctx.closePath();
    ctx.fillStyle = "rgba(255,94,87,0.92)";
    ctx.fill();
  }
}

let vitPlot = null;
let dmgPlot = null;

function makePlot(el, series, yRange) {
  const opts = {
    width: el.clientWidth,
    height: 180,
    scales: { x: { time: false }, y: { range: yRange } },
    axes: [
      { stroke: "rgba(255,255,255,0.35)", grid: { stroke: "rgba(255,255,255,0.06)" } },
      { stroke: "rgba(255,255,255,0.35)", grid: { stroke: "rgba(255,255,255,0.06)" } },
    ],
    series,
  };
  return new uPlot(opts, [[], ...series.slice(1).map(() => [])], el);
}

function ensurePlots() {
  if (!vitPlot) {
    vitPlot = makePlot(
      $("plotVitals"),
      [{}, { label: "HP", stroke: "rgba(122,162,255,0.9)" }, { label: "Stam", stroke: "rgba(157,255,176,0.85)" }],
      () => [0, 100]
    );
  }
  if (!dmgPlot) {
    dmgPlot = makePlot(
      $("plotDamage"),
      [{}, { label: "IN", stroke: "rgba(255,211,107,0.90)" }, { label: "OUT", stroke: "rgba(122,162,255,0.85)" }],
      () => {
        let m = 10;
        for (const v of dinHist) m = Math.max(m, v);
        for (const v of doutHist) m = Math.max(m, v);
        return [0, m * 1.25];
      }
    );
  }
}

function syncPlots() {
  if (vitPlot) vitPlot.setData([tHist, hpHist, stHist]);
  if (dmgPlot) dmgPlot.setData([tHist, dinHist, doutHist]);
}

function isDeathFrame(frame) {
  const hp = Number(frame.hp_pct);
  if (Number.isFinite(hp) && hp <= 0) return true;

  if (frame.death && frame.death.active === true) return true;

  const s = `${safeText(frame.decision)} ${safeText(frame.action_last)} ${safeText(frame.event)} ${safeText(frame.stop_reason)}`.toLowerCase();
  return s.includes("death") || s.includes("dead") || s.includes("eliminat");
}

function pushEvent(kind, text) {
  events.unshift({ kind, text });
  if (events.length > maxEvents) events.length = maxEvents;
}

function renderEvents() {
  let rows = events;
  if (eventFilter === "actions") rows = events.filter((e) => e.kind === "action");
  else if (eventFilter === "damage") rows = events.filter((e) => e.kind === "damage");
  $("events").textContent = rows.map((e) => e.text).join("\n");
}

function deriveHpForDisplay(frame) {
  const raw = clampPct(frame.hp_pct);
  const src = String(frame.hp_src ?? "unknown").toLowerCase();
  const dmgInTick = Math.max(0, Number(frame.dmg_in_tick ?? 0));
  let hp = raw;

  if (hp === null) hp = hpDisplayLast;
  if (hpDisplayLast !== null && (src === "fallback" || src === "unknown" || src === "heur")) {
    if (dmgInTick > 0.01 && (raw === null || Math.abs(raw - hpDisplayLast) < 0.05)) {
      hp = Math.max(0, hpDisplayLast - dmgInTick);
    } else if (raw !== null && raw > (hpDisplayLast + 12.0)) {
      hp = Math.min(raw, hpDisplayLast + 2.0);
    }
  }

  hp = clampPct(hp);
  if (hp !== null) hpDisplayLast = hp;
  return hp;
}

function renderFrame(frame) {
  const ts = frame.ts_ms ? Number(frame.ts_ms) : 0;
  if (ts && ts === lastFrameTs) return;
  lastFrameTs = ts;

  const hp = deriveHpForDisplay(frame);
  $("hp").textContent = fmtPct(hp);
  $("hpConf").textContent = Number(frame.hp_conf ?? 0).toFixed(2);
  $("hpSrc").textContent = safeText(frame.hp_src);
  setBar($("hpBar"), hp);

  const st = frame.stamina_pct;
  $("stam").textContent = fmtPct(st);
  $("stamConf").textContent = Number(frame.stamina_conf ?? 0).toFixed(2);
  $("stamSrc").textContent = safeText(frame.stamina_src);
  setBar($("stamBar"), st);

  const din = Number(frame.dmg_in_tick ?? 0);
  const dout = Number(frame.dmg_out_tick ?? 0);
  $("dinTick").textContent = fmtNum(din);
  $("dinTotal").textContent = fmtNum(frame.dmg_in_total ?? 0);
  $("doutTick").textContent = fmtNum(dout);
  $("doutTotal").textContent = fmtNum(frame.dmg_out_total ?? 0);

  $("enemyVis").textContent = String(!!frame.enemy_visible);
  $("enemyConf").textContent = Number(frame.enemy_conf ?? 0).toFixed(2);
  $("enemyDir").textContent = safeText(frame.enemy_dir_deg);
  $("enemyXY").textContent = frame.enemy_xy ? JSON.stringify(frame.enemy_xy) : "-";
  const radarState = collectRadarTracks(frame);
  radarDraw(radarState);
  const enemyDist = numOrNull(frame.enemy_dist_norm);
  const inferredDist = (radarState.enemies.length > 0) ? radarState.enemies[0].dist : null;
  const distNorm = enemyDist ?? inferredDist;
  $("enemyDist").textContent = (distNorm === null || !Number.isFinite(distNorm)) ? "-" : `${(distNorm * 100).toFixed(1)}%`;
  const enemyAgeMs = numOrNull(frame.enemy_age_ms);
  $("enemyAge").textContent = (enemyAgeMs === null) ? "-" : `${Math.round(enemyAgeMs)} ms`;
  $("allyCount").textContent = String(Number(frame.ally_count ?? radarState.allies.length ?? 0));
  $("enemyCount").textContent = String(Number(frame.enemy_count ?? radarState.enemies.length ?? 0));

  $("decision").textContent = safeText(frame.decision);
  $("action").textContent = safeText(frame.action_last);
  $("actionOk").textContent = safeText(frame.action_ok);

  $("zoneOutside").textContent = String(!!frame.zone_outside);
  $("zoneToxic").textContent = String(!!frame.zone_toxic);
  $("zoneCd").textContent = safeText(frame.zone_countdown_s);

  const ents = frame.entities ?? [];
  $("entities").textContent = ents.slice(0, 20).map((e) =>
    `${safeText(e.name).padEnd(16)} ${safeText(e.kind).padEnd(7)} team=${safeText(e.team).padEnd(6)} hp=${safeText(e.hp_pct).toString().padStart(5)} conf=${Number(e.conf ?? 0).toFixed(2)} dist=${(Number.isFinite(Number(e.distance_norm)) ? (Number(e.distance_norm) * 100).toFixed(1) + '%' : '-')}`
  ).join("\n");

  const tLabel = ts ? new Date(ts).toLocaleTimeString() : "";
  const action = safeText(frame.action_last);
  if (action !== "-") pushEvent("action", `[${tLabel}] action: ${action} ok=${safeText(frame.action_ok)}`);
  if (din > 0.01) pushEvent("damage", `[${tLabel}] dmg IN +${din.toFixed(1)} total=${fmtNum(frame.dmg_in_total)}`);
  if (dout > 0.01) pushEvent("damage", `[${tLabel}] dmg OUT +${dout.toFixed(1)} total=${fmtNum(frame.dmg_out_total)}`);
  renderEvents();

  const t = ts ? (ts / 1000.0) : (performance.now() / 1000.0);
  pushHist(t, Number(hp ?? 0), Number(st ?? 0), din, dout);
  ensurePlots();
  syncPlots();

  const raw = JSON.stringify(frame, null, 2);
  $("raw").textContent = raw.length > 20000 ? `${raw.slice(0, 20000)}\n...(truncated)` : raw;
  $("ts").textContent = ts ? `ts: ${new Date(ts).toLocaleTimeString()}` : "-";
}

function updateRate() {
  const now = performance.now();
  const dt = (now - lastRxT) / 1000;
  if (dt >= 1) {
    const hz = rxCount / dt;
    $("rate").textContent = `rx: ${hz.toFixed(1)} Hz`;
    rxCount = 0;
    lastRxT = now;
  }
}

function setWsStatus(txt, cls) {
  const el = $("wsStatus");
  el.textContent = txt;
  el.className = `pill ${cls}`;
}

function connectWs() {
  const wsUrl = `ws://${location.host}/ws`;
  let ws = null;
  let backoff = 250;

  const loop = () => {
    setWsStatus("WS: connecting", "warn");
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      backoff = 250;
      setWsStatus("WS: connected", "ok");
    };

    ws.onclose = () => {
      setWsStatus("WS: reconnecting", "warn");
      setTimeout(loop, backoff);
      backoff = Math.min(4000, backoff * 1.6);
    };

    ws.onerror = () => {
      try { ws.close(); } catch (_) {}
    };

    ws.onmessage = (ev) => {
      updateRate();
      rxCount += 1;
      try {
        const msg = JSON.parse(ev.data);
        if (msg && msg._type === "ping") return;

        if (!paused && !frozenByDeath) {
          renderFrame(msg);
          if (freezeEnabled && isDeathFrame(msg)) {
            frozenByDeath = true;
            $("freezeState").textContent = "frozen: death";
          }
        }
      } catch (_) {}
    };
  };

  loop();
}

function setupTabs() {
  const tabs = Array.from(document.querySelectorAll(".tab"));
  const panes = Array.from(document.querySelectorAll(".tab-pane"));

  const activate = (key) => {
    for (const t of tabs) t.classList.toggle("active", t.dataset.tab === key);
    for (const p of panes) p.classList.toggle("active", p.dataset.pane === key);
    if (vitPlot && key === "vitals") vitPlot.setSize({ width: $("plotVitals").clientWidth, height: 180 });
    if (dmgPlot && key === "combat") dmgPlot.setSize({ width: $("plotDamage").clientWidth, height: 180 });
  };

  for (const t of tabs) {
    t.onclick = () => activate(t.dataset.tab);
  }
}

async function loadConfig() {
  try {
    const r = await fetch("/config", { cache: "no-store" });
    if (!r.ok) return;
    const cfg = await r.json();
    const wsHz = Number(cfg.ws_hz ?? 0).toFixed(1);
    const hist = Number(cfg.history ?? 0);
    $("cfg").textContent = `cfg: ${wsHz}Hz hist=${hist}`;
  } catch (_) {}
}

function boot() {
  $("btnPause").onclick = () => {
    paused = !paused;
    $("btnPause").textContent = paused ? "Resume" : "Pause";
  };

  $("btnUnfreeze").onclick = () => {
    frozenByDeath = false;
    $("freezeState").textContent = "live";
  };

  $("btnClear").onclick = () => {
    events.length = 0;
    renderEvents();
  };

  $("freezeOnDeath").onchange = (ev) => {
    freezeEnabled = !!ev.target.checked;
  };

  $("eventFilter").onchange = (ev) => {
    eventFilter = String(ev.target.value || "all");
    renderEvents();
  };

  window.addEventListener("resize", () => {
    if (vitPlot) vitPlot.setSize({ width: $("plotVitals").clientWidth, height: 180 });
    if (dmgPlot) dmgPlot.setSize({ width: $("plotDamage").clientWidth, height: 180 });
  });

  setupTabs();
  loadConfig();
  connectWs();
  setInterval(updateRate, 200);
}

boot();
