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
    stats:            "compute stats",
    viz:              "build viz",
  };

  // Color palette for per-turn map overlays. The first user turn is
  // blue, the next is teal, then orange, then magenta, then green —
  // looping. Keeps adjacent turns visually distinct on the map.
  const TURN_COLORS = ["#4a7bd1", "#1ca7a0", "#e07b39", "#b34cb3", "#3b9d3b"];

  let map;
  let itemsLayers = [];   // per-turn L.FeatureGroup of STAC items
  let geoLayers = [];     // per-turn L.GeoJSON of geocoded areas
  let turnCount = 0;
  let inFlight = false;

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
      style: { color, weight: 2, fillOpacity: 0.12 },
    }).bindTooltip(`turn ${turnIdx + 1}: ${geocode.place || ""}`);
    layer.addTo(map);
    geoLayers[turnIdx] = layer;
    if (geocode.bbox && geocode.bbox.length === 4) {
      const [w, s, e, n] = geocode.bbox;
      map.fitBounds([[s, w], [n, e]], { padding: [20, 20] });
    }
  }

  function renderStacItemsForTurn(turnIdx, items) {
    if (!items || items.length === 0) return;
    const color = colorForTurn(turnIdx);
    let group = itemsLayers[turnIdx];
    if (!group) {
      group = L.featureGroup().addTo(map);
      itemsLayers[turnIdx] = group;
    }
    items.forEach((it) => {
      if (!it.bbox || it.bbox.length !== 4) return;
      const [w, s, e, n] = it.bbox;
      const rect = L.rectangle([[s, w], [n, e]], {
        color, weight: 1, fillOpacity: 0.05,
      }).bindTooltip(`${it.id}<br/>${it.datetime || ""}`);
      group.addLayer(rect);
    });
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
    if (ev.name === "stats" && p.stats) {
      return `${p.stats.length} row${p.stats.length === 1 ? "" : "s"}`;
    }
    return "";
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
    appendUserBubble(text);
    $("query").value = "";

    const turnIdx = turnCount++;
    const parts = appendAgentBubble(turnIdx);

    try {
      const r = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text }),
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
              renderStacItemsForTurn(turnIdx, p.stac_items);
            }
          } else if (evt.type === "done") {
            renderTurnSummary(parts, evt.trace);
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
  /* Boot                                                           */
  /* -------------------------------------------------------------- */
  window.addEventListener("DOMContentLoaded", () => {
    setupMap();
    loadHealth();
    loadArchetypes();
    $("composer").addEventListener("submit", askQuestion);
    $("query").addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") askQuestion(e);
    });
  });
})();
