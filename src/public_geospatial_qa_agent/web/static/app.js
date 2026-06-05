/* Conversation-style UI for the public geospatial Q&A agent.
 *
 * Each user turn is one cycle of the six-stage agent. The previous
 * turns stay visible above; the map accumulates layers across turns
 * so the conversation builds up a picture of the area being explored.
 *
 * Sections, in order:
 *   - state + DOM helpers
 *   - map (Leaflet) and per-turn layer groups
 *   - chat feed and per-turn rendering
 *   - SSE consumer for /api/ask
 *   - boot
 */

(() => {
  /* -------------------------------------------------------------- */
  /* State                                                          */
  /* -------------------------------------------------------------- */
  const STAGE_LABELS = {
    parse_datetime:   "parse date range",
    geocode:          "geocode",
    collections_rag:  "search collections",
    select_collection: "select collection",
    stac_search:      "STAC search",
    compute_stats:    "compute stats",
    build_viz_tiles:  "build viz",
  };

  // Color palette for per-turn map overlays. The first user turn is
  // blue, the next is teal, then orange, then magenta, then green —
  // looping. Keeps adjacent turns visually distinct on the map.
  const TURN_COLORS = ["#4a7bd1", "#1ca7a0", "#e07b39", "#b34cb3", "#3b9d3b"];

  let map;
  let itemsLayers = [];   // per-turn L.FeatureGroup of STAC items
  let geoLayers = [];     // per-turn L.GeoJSON of geocoded areas
  let geocodeBboxByTurn = []; // per-turn AOI bbox, used to clip items
  let turnCount = 0;
  let inFlight = false;
  let pendingClarification = null; // {originalQuery, question} when agent asked back
  // Per-session cache namespace. The server uses this as the
  // prompt_cache_key suffix so each Reset starts the next cycle on a
  // cold cache. Rotated on Reset; persists across turns inside a
  // session so cache warm-up across turns is real.
  let sessionKey = newSessionKey();

  function newSessionKey() {
    // 8 hex chars from crypto if available, otherwise Math.random.
    if (window.crypto && crypto.getRandomValues) {
      const a = new Uint8Array(4);
      crypto.getRandomValues(a);
      return Array.from(a, (b) => b.toString(16).padStart(2, "0")).join("");
    }
    return Math.random().toString(16).slice(2, 10);
  }

  const $ = (id) => document.getElementById(id);
  const el = (tag, attrs = {}, children = []) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") e.className = v;
      else if (k === "html") e.innerHTML = v;
      else if (k === "text") e.textContent = v;
      else e.setAttribute(k, v);
    }
    for (const c of children) e.appendChild(c);
    return e;
  };

  /* -------------------------------------------------------------- */
  /* Map                                                            */
  /* -------------------------------------------------------------- */
  function setupMap() {
    map = L.map("map", { zoomControl: true }).setView([20, 0], 2);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      attribution: "© OpenStreetMap",
    }).addTo(map);
  }

  function colorForTurn(idx) {
    return TURN_COLORS[idx % TURN_COLORS.length];
  }

  function renderGeocodeForTurn(turnIdx, geocode) {
    if (!geocode || !geocode.geometry) return;
    const color = colorForTurn(turnIdx);
    const layer = L.geoJSON(geocode.geometry, {
      style: { color, weight: 2, fillOpacity: 0.06 },
    }).bindTooltip(`turn ${turnIdx + 1}: ${geocode.place || ""}`);
    layer.addTo(map);
    geoLayers[turnIdx] = layer;
    if (geocode.bbox && geocode.bbox.length === 4) {
      geocodeBboxByTurn[turnIdx] = geocode.bbox;
      const [w, s, e, n] = geocode.bbox;
      map.fitBounds([[s, w], [n, e]], { padding: [20, 20] });
    }
  }

  function renderStacItemsForTurn(turnIdx, items, aoiBbox) {
    if (!items || items.length === 0) return;
    const color = colorForTurn(turnIdx);
    let group = itemsLayers[turnIdx];
    if (!group) {
      group = L.featureGroup().addTo(map);
      itemsLayers[turnIdx] = group;
    }
    // Some STAC collections (e.g. Sentinel-5P) carry hemispheric per-
    // item bboxes; drawing all of them at any fill opacity paints
    // the whole map. Skip items whose bbox is much larger than the
    // AOI, and cap the rest to the first few with outline-only.
    const aoiArea = aoiBbox ? bboxArea(aoiBbox) : null;
    let drawn = 0;
    for (const it of items) {
      if (!it.bbox || it.bbox.length !== 4) continue;
      if (aoiArea && bboxArea(it.bbox) > aoiArea * 20) continue;
      const [w, s, e, n] = it.bbox;
      const rect = L.rectangle([[s, w], [n, e]], {
        color, weight: 1, fill: false, opacity: 0.8,
      }).bindTooltip(`${it.id}<br/>${it.datetime || ""}`);
      group.addLayer(rect);
      if (++drawn >= 5) break;
    }
    // If nothing fit (everything was too large), drop a centroid dot
    // at the AOI so the user sees the turn registered on the map.
    if (drawn === 0 && aoiBbox) {
      const [w, s, e, n] = aoiBbox;
      const cx = (w + e) / 2;
      const cy = (s + n) / 2;
      const dot = L.circleMarker([cy, cx], {
        radius: 6, color, weight: 2, fillOpacity: 0.4,
      }).bindTooltip(`${items.length} items in AOI`);
      group.addLayer(dot);
    }
  }

  function bboxArea(b) {
    const [w, s, e, n] = b;
    return Math.max(0, e - w) * Math.max(0, n - s);
  }

  function appendLegendRow(turnIdx, label) {
    const host = $("map-legend");
    const swatch = el("span", { class: "legend-swatch" });
    swatch.style.background = colorForTurn(turnIdx);
    swatch.style.opacity = "0.4";
    const row = el("div", { class: "legend-row" }, [
      swatch,
      el("span", { text: `turn ${turnIdx + 1}: ${label}` }),
    ]);
    host.appendChild(row);
  }

  /* -------------------------------------------------------------- */
  /* Health                                                         */
  /* -------------------------------------------------------------- */
  async function loadHealth() {
    try {
      const r = await fetch("/api/health");
      const j = await r.json();
      const e = $("health");
      const left = j.budget_remaining_usd.toFixed(4);
      const cap = j.budget_cap_usd.toFixed(2);
      const keyPart = j.has_api_key ? "" : " · ⚠ no API key";
      const backendPart = j.backend ? ` · backend ${j.backend}` : "";
      e.textContent = `v${j.version} · $${left}/$${cap} left${backendPart}${keyPart}`;
      e.style.color = j.has_api_key ? "" : "#ffb547";
    } catch (e) {
      $("health").textContent = "health check failed";
    }
  }

  /* -------------------------------------------------------------- */
  /* Archetype suggestions                                          */
  /* -------------------------------------------------------------- */
  async function loadArchetypes() {
    try {
      const r = await fetch("/api/archetypes");
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      const host = $("archetypes");
      j.archetypes.slice(0, 5).forEach((a) => {
        const b = el("button", { type: "button", title: a.query });
        b.textContent = a.id.replace(/_/g, " ");
        b.addEventListener("click", () => {
          $("query").value = a.query;
          $("query").focus();
        });
        host.appendChild(b);
      });
    } catch (e) {
      console.warn("archetypes failed to load:", e);
    }
  }

  /* -------------------------------------------------------------- */
  /* Chat                                                           */
  /* -------------------------------------------------------------- */
  function appendUserBubble(text) {
    const bubble = el("div", { class: "bubble" });
    bubble.appendChild(el("p", { text }));
    const wrap = el("div", { class: "msg user" }, [bubble]);
    $("chat").appendChild(wrap);
    scrollChatToBottom();
    return wrap;
  }

  function appendAgentBubble(turnIdx) {
    const bubble = el("div", { class: "bubble" });
    bubble.appendChild(el("p", { class: "agent-headline",
                                  text: "Walking the six stages…" }));
    const stages = el("div", { class: "stages" });
    bubble.appendChild(stages);
    const summary = el("div", { class: "turn-summary", style: "display:none" });
    bubble.appendChild(summary);
    const wrap = el("div", { class: "msg agent" }, [bubble]);
    wrap.dataset.turnIdx = String(turnIdx);
    $("chat").appendChild(wrap);
    scrollChatToBottom();
    return { wrap, bubble, stages, summary,
             headline: bubble.querySelector(".agent-headline") };
  }

  function renderStage(parts, ev) {
    let row = parts.stages.querySelector(`[data-stage="${ev.name}"]`);
    if (!row) {
      row = el("div", { class: "stage-row done", "data-stage": ev.name });
      row.appendChild(el("span", { class: "stage-icon", text: "✓" }));
      row.appendChild(el("span", { class: "stage-name",
                                    text: STAGE_LABELS[ev.name] || ev.name }));
      row.appendChild(el("span", { class: "stage-detail", text: "" }));
      row.appendChild(el("span", { class: "stage-cost", text: "" }));
      parts.stages.appendChild(row);
    }
    const detail = stageDetail(ev);
    row.children[2].textContent = detail;
    const pct = (ev.cache_ratio * 100).toFixed(0);
    row.children[3].textContent = `${pct}% cached · $${ev.call_cost_usd.toFixed(6)}`;
    scrollChatToBottom();
  }

  function stageDetail(ev) {
    const p = ev.map_payload || {};
    if (ev.name === "parse_datetime") return "";
    if (ev.name === "geocode" && p.geocode) return p.geocode.place || "";
    if (ev.name === "collections_rag" && p.collections) {
      return `${p.collections.length} match${p.collections.length === 1 ? "" : "es"}`;
    }
    if (ev.name === "select_collection") return "";
    if (ev.name === "stac_search" && p.stac_items) {
      return `${p.stac_items.length} item${p.stac_items.length === 1 ? "" : "s"}`;
    }
    if (ev.name === "compute_stats" && p.stats) {
      return `${p.stats.length} row${p.stats.length === 1 ? "" : "s"}`;
    }
    return "";
  }

  function renderStatsTable(parts, perItem) {
    if (!perItem || perItem.length === 0) return;
    if (parts.bubble.querySelector(".stats-table")) return; // idempotent
    // Pick the columns from the first row. Excludes item_id (long, noisy).
    const sample = perItem[0];
    const keys = Object.keys(sample).filter(
      (k) => k !== "item_id" && sample[k] !== null && sample[k] !== undefined
    );
    if (keys.length === 0) return;
    const wrap = el("div", { class: "stats-table" });
    wrap.appendChild(el("div", { class: "stats-caption",
                                  text: `compute_stats — ${perItem.length} rows` }));
    const table = el("table");
    const thead = el("thead");
    const headRow = el("tr");
    headRow.appendChild(el("th", { text: "#" }));
    keys.forEach((k) => headRow.appendChild(el("th", { text: k })));
    thead.appendChild(headRow);
    table.appendChild(thead);
    const tbody = el("tbody");
    perItem.slice(0, 5).forEach((row, i) => {
      const tr = el("tr");
      tr.appendChild(el("td", { text: String(i + 1) }));
      keys.forEach((k) => {
        const v = row[k];
        tr.appendChild(el("td", {
          text: typeof v === "number" ? formatNumber(v) : (v == null ? "—" : String(v)),
        }));
      });
      tbody.appendChild(tr);
    });
    if (perItem.length > 5) {
      const tr = el("tr", { class: "stats-more" });
      const td = el("td", { text: `+ ${perItem.length - 5} more` });
      td.colSpan = keys.length + 1;
      tr.appendChild(td);
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    parts.bubble.appendChild(wrap);
    scrollChatToBottom();
  }

  function formatNumber(n) {
    if (!Number.isFinite(n)) return String(n);
    const abs = Math.abs(n);
    if (abs >= 1000) return n.toFixed(0);
    if (abs >= 1) return n.toFixed(2);
    if (abs >= 0.001) return n.toFixed(4);
    return n.toExponential(2);
  }

  function renderTurnSummary(parts, trace) {
    parts.headline.textContent = "Done.";
    const ratio = (trace.cache_ratio * 100).toFixed(1);
    parts.summary.style.display = "";
    parts.summary.innerHTML = `
      <div><b>prompt</b> ${fmt(trace.total_prompt_tokens)} tok</div>
      <div><b>cached</b> ${fmt(trace.total_cached_tokens)} (${ratio}%)</div>
      <div><b>output</b> ${fmt(trace.total_completion_tokens)} tok</div>
      <div><b>cost</b> $${trace.total_cost_usd.toFixed(6)}</div>
      <div><b>server-side state</b> ${fmt(trace.final_state_size_chars)} chars</div>
      <div><b>kept from LLM</b> by templating</div>
    `;
    scrollChatToBottom();
  }

  function renderClarification(parts, question, costUsd, tokens) {
    parts.headline.textContent = question;
    parts.headline.classList.add("clarification");
    parts.stages.style.display = "none";
    if (typeof costUsd === "number" && costUsd > 0) {
      parts.summary.style.display = "";
      parts.summary.innerHTML = `
        <div><b>clarify</b> ${tokens || 0} tok</div>
        <div><b>cost</b> $${costUsd.toFixed(6)}</div>
      `;
    } else {
      parts.summary.style.display = "none";
    }
    scrollChatToBottom();
  }

  function renderStageError(parts, msg) {
    parts.headline.textContent = "Stopped on error.";
    const row = el("div", { class: "stage-row error" });
    row.appendChild(el("span", { class: "stage-icon", text: "✗" }));
    row.appendChild(el("span", { class: "stage-name", text: "error" }));
    row.appendChild(el("span", { class: "stage-detail", text: msg }));
    row.appendChild(el("span", { text: "" }));
    parts.stages.appendChild(row);
    scrollChatToBottom();
  }

  function scrollChatToBottom() {
    const c = $("chat");
    c.scrollTop = c.scrollHeight;
  }

  function fmt(n) {
    return (n || 0).toLocaleString();
  }

  /* -------------------------------------------------------------- */
  /* Ask                                                            */
  /* -------------------------------------------------------------- */
  async function askQuestion(ev) {
    if (ev) ev.preventDefault();
    if (inFlight) return;
    const text = $("query").value.trim();
    if (!text) return;

    inFlight = true;
    $("ask").disabled = true;

    // If the agent just asked a clarifying question, fold this answer
    // into the original query and clear the pending state.
    let effectiveQuery = text;
    if (pendingClarification) {
      effectiveQuery = `${pendingClarification.originalQuery} — ${text}`;
      pendingClarification = null;
    }

    appendUserBubble(text);
    $("query").value = "";

    const clarifyEnabled = $("clarify").checked;
    const turnIdx = turnCount++;
    const parts = appendAgentBubble(turnIdx);

    try {
      // Mode + pattern come from one of three sources, in priority:
      //   1. Driver-set window vars (the Playwright corpus runner).
      //   2. Header dropdowns (manual user testing).
      //   3. Server defaults (templated, single-turn).
      const sessionIdHint = window.PGQA_SESSION_ID || undefined;
      const modeHint = window.PGQA_MODE || $("mode-select")?.value || undefined;
      const patternHint = window.PGQA_PATTERN || $("pattern-select")?.value || undefined;
      const body = {
        query: effectiveQuery,
        clarify: clarifyEnabled,
        cache_namespace: sessionKey,
      };
      if (sessionIdHint) body.session_id = sessionIdHint;
      if (modeHint) body.mode = modeHint;
      if (patternHint) body.pattern = patternHint;
      // Reflect the cell in the visible header so the human watching
      // can see which row of the matrix is being measured.
      const indicator = $("cell-indicator");
      if (modeHint || patternHint) {
        indicator.hidden = false;
        $("mode-label").textContent = modeHint || "templated";
        $("pattern-label").textContent = patternHint || "single-turn";
      }
      const r = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const errBody = await r.json().catch(() => ({ detail: r.statusText }));
        throw new Error(errBody.detail || `HTTP ${r.status}`);
      }
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf("\n\n")) !== -1) {
          const raw = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 2);
          if (!raw.startsWith("data:")) continue;
          const evt = JSON.parse(raw.slice(5).trim());
          if (evt.type === "stage") {
            renderStage(parts, evt);
            const p = evt.map_payload || {};
            if (p.geocode) {
              renderGeocodeForTurn(turnIdx, p.geocode);
              appendLegendRow(turnIdx, p.geocode.place || "(area)");
            }
            if (p.stac_items) {
              renderStacItemsForTurn(
                turnIdx, p.stac_items, geocodeBboxByTurn[turnIdx]
              );
            }
            if (p.stats) {
              renderStatsTable(parts, p.stats);
            }
          } else if (evt.type === "done") {
            renderTurnSummary(parts, evt.trace);
          } else if (evt.type === "clarification") {
            renderClarification(
              parts, evt.question,
              evt.clarify_cost_usd, evt.clarify_tokens
            );
            pendingClarification = {
              originalQuery: evt.pending_query,
              question: evt.question,
            };
          } else if (evt.type === "error") {
            renderStageError(parts, evt.message);
          }
        }
      }
    } catch (e) {
      renderStageError(parts, e.message || String(e));
    } finally {
      inFlight = false;
      $("ask").disabled = false;
      $("query").focus();
      loadHealth();
    }
  }

  /* -------------------------------------------------------------- */
  /* Reset                                                          */
  /* -------------------------------------------------------------- */
  function resetSession() {
    // Force-clear the in-flight guard. If a previous turn's SSE
    // stream is hung (network glitch, server stall), this lets the
    // next turn proceed instead of leaving Send disabled forever.
    inFlight = false;
    $("ask").disabled = false;
    // Drop every chat message except the original welcome bubble.
    const chat = $("chat");
    const welcome = chat.firstElementChild;
    chat.innerHTML = "";
    if (welcome) chat.appendChild(welcome);
    // Clear all map overlays.
    geoLayers.forEach((l) => l && map.removeLayer(l));
    itemsLayers.forEach((g) => g && map.removeLayer(g));
    geoLayers = [];
    itemsLayers = [];
    geocodeBboxByTurn = [];
    $("map-legend").innerHTML = "";
    // Reset conversation state.
    turnCount = 0;
    pendingClarification = null;
    sessionKey = newSessionKey();
    map.setView([20, 0], 2);
    $("query").focus();
  }

  /* -------------------------------------------------------------- */
  /* Boot                                                           */
  /* -------------------------------------------------------------- */
  window.addEventListener("DOMContentLoaded", () => {
    setupMap();
    loadHealth();
    loadArchetypes();
    $("composer").addEventListener("submit", askQuestion);
    $("reset").addEventListener("click", resetSession);
    $("query").addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") askQuestion(e);
    });
  });
})();
