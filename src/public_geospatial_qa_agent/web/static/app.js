/* Public Geospatial Q&A Agent — single-page UI.
 *
 * Three responsibilities, kept in three small named functions so the
 * code is easy to skim during review:
 *
 *   - setupMap()      builds the Leaflet map and tile layer
 *   - loadHealth()    polls /api/health, fills the header
 *   - askQuestion()   POSTs /api/ask and consumes the SSE stream,
 *                     rendering each stage as it arrives and updating
 *                     the map when geocode / stac_search payloads land
 *
 * No frameworks, no bundler. Read top-to-bottom.
 */

(() => {
  let map;
  let geoLayer;     // L.GeoJSON of the geocode polygon
  let itemsLayer;   // L.FeatureGroup of STAC item markers
  let inFlight = false;
  // Track the currently selected archetype id so /api/ask receives the
  // right one when the user clicks a quick-button. Defaults to the
  // single-dataset archetype; updated when the user clicks an option.
  let selectedArchetypeId = "single_dataset_viz";

  /* ---------------------------------------------------------------- */
  /* Map                                                               */
  /* ---------------------------------------------------------------- */
  function setupMap() {
    map = L.map("map", { zoomControl: true }).setView([34.0, -118.0], 3);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      attribution: "© OpenStreetMap",
    }).addTo(map);
    itemsLayer = L.featureGroup().addTo(map);
  }

  function renderGeocodeOnMap(geocode) {
    // Geocode payload: { place, bbox, geometry }
    if (geoLayer) { map.removeLayer(geoLayer); geoLayer = null; }
    if (!geocode || !geocode.geometry) return;
    geoLayer = L.geoJSON(geocode.geometry, {
      style: { color: "#4a7bd1", weight: 2, fillOpacity: 0.15 },
    }).addTo(map);
    if (geocode.bbox && geocode.bbox.length === 4) {
      const [w, s, e, n] = geocode.bbox;
      map.fitBounds([[s, w], [n, e]], { padding: [20, 20] });
    }
  }

  function renderStacItemsOnMap(items) {
    itemsLayer.clearLayers();
    if (!items) return;
    items.forEach((it) => {
      if (!it.bbox || it.bbox.length !== 4) return;
      const [w, s, e, n] = it.bbox;
      const rect = L.rectangle([[s, w], [n, e]], {
        color: "#c0392b",
        weight: 1,
        fillOpacity: 0.08,
      }).bindTooltip(`${it.id}<br/>${it.datetime || ""}`);
      itemsLayer.addLayer(rect);
    });
  }

  /* ---------------------------------------------------------------- */
  /* Health                                                            */
  /* ---------------------------------------------------------------- */
  async function loadHealth() {
    try {
      const r = await fetch("/api/health");
      const j = await r.json();
      const el = document.getElementById("health");
      const left = j.budget_remaining_usd.toFixed(4);
      const cap = j.budget_cap_usd.toFixed(2);
      const keyPart = j.has_api_key ? "" : " · ⚠ no API key";
      el.textContent = `v${j.version} · budget $${left} / $${cap} left${keyPart}`;
      if (!j.has_api_key) el.style.color = "#ffb547";
    } catch (e) {
      document.getElementById("health").textContent = "health check failed";
    }
  }

  /* ---------------------------------------------------------------- */
  /* Archetype quick-buttons                                           */
  /* ---------------------------------------------------------------- */
  async function loadArchetypes() {
    try {
      const r = await fetch("/api/archetypes");
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      const host = document.getElementById("archetypes");
      j.archetypes.forEach((a) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = a.id.replace(/_/g, " ");
        b.title = a.query;
        b.dataset.archetypeId = a.id;
        b.dataset.query = a.query;
        b.addEventListener("click", () => {
          document.getElementById("query").value = a.query;
          selectedArchetypeId = a.id;
        });
        host.appendChild(b);
      });
    } catch (e) {
      // Quick-buttons are optional; the textbox still works without
      // them. Don't throw — just log.
      console.warn("archetypes failed to load:", e);
    }
  }

  /* ---------------------------------------------------------------- */
  /* Ask                                                               */
  /* ---------------------------------------------------------------- */
  function appendStage(ev) {
    const host = document.getElementById("stages");
    const card = document.createElement("div");
    card.className = "stage";
    card.dataset.stageName = ev.name;

    const ratioPct = (ev.cache_ratio * 100).toFixed(1);
    card.innerHTML = `
      <h3>stage ${ev.idx} · ${ev.name}</h3>
      <div class="meta">
        <span><b>prompt</b> ${ev.prompt_tokens.toLocaleString()}</span>
        <span><b>cached</b> ${ev.cached_tokens.toLocaleString()} (${ratioPct}%)</span>
        <span><b>output</b> ${ev.completion_tokens.toLocaleString()}</span>
        <span><b>cost</b> $${ev.call_cost_usd.toFixed(6)}</span>
        <span><b>to LLM</b> ${ev.tool_message_chars.toLocaleString()} chars</span>
        <span><b>server-side</b> ${ev.state_size_chars.toLocaleString()} chars</span>
      </div>
      <pre></pre>
    `;
    card.querySelector("pre").textContent = ev.tool_response_preview || "(empty)";
    host.appendChild(card);
    host.scrollTop = host.scrollHeight;

    // Update the map for stages that produce visible artefacts
    if (ev.map_payload?.geocode) renderGeocodeOnMap(ev.map_payload.geocode);
    if (ev.map_payload?.stac_items) renderStacItemsOnMap(ev.map_payload.stac_items);
  }

  function showSummary(trace) {
    const el = document.getElementById("summary");
    const ratio = (trace.cache_ratio * 100).toFixed(1);
    el.hidden = false;
    el.innerHTML = `
      <h3>Cycle complete</h3>
      <table>
        <tr><td>Total prompt tokens</td><td class="stat">${trace.total_prompt_tokens.toLocaleString()}</td></tr>
        <tr><td>Total cached</td><td class="stat">${trace.total_cached_tokens.toLocaleString()} (${ratio}%)</td></tr>
        <tr><td>Total output</td><td class="stat">${trace.total_completion_tokens.toLocaleString()}</td></tr>
        <tr><td>Per-cycle cost</td><td class="stat">$${trace.total_cost_usd.toFixed(6)}</td></tr>
        <tr><td>Server-side state size</td><td class="stat">${trace.final_state_size_chars.toLocaleString()} chars</td></tr>
      </table>
    `;
  }

  function showError(msg) {
    const host = document.getElementById("stages");
    const card = document.createElement("div");
    card.className = "stage error";
    card.innerHTML = `<h3>error</h3><pre></pre>`;
    card.querySelector("pre").textContent = msg;
    host.appendChild(card);
  }

  async function askQuestion() {
    if (inFlight) return;
    const query = document.getElementById("query").value.trim();
    if (!query) return;

    inFlight = true;
    const askBtn = document.getElementById("ask");
    askBtn.disabled = true;
    document.getElementById("stages").innerHTML = "";
    document.getElementById("summary").hidden = true;
    if (geoLayer) { map.removeLayer(geoLayer); geoLayer = null; }
    itemsLayer.clearLayers();

    try {
      const r = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, archetype_id: selectedArchetypeId }),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({ detail: r.statusText }));
        throw new Error(err.detail || `HTTP ${r.status}`);
      }
      // Consume the SSE stream
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
          const json = raw.slice(5).trim();
          const ev = JSON.parse(json);
          if (ev.type === "stage") appendStage(ev);
          else if (ev.type === "done") showSummary(ev.trace);
          else if (ev.type === "error") showError(ev.message);
        }
      }
    } catch (e) {
      showError(e.message || String(e));
    } finally {
      inFlight = false;
      askBtn.disabled = false;
      loadHealth();
    }
  }

  /* ---------------------------------------------------------------- */
  /* Boot                                                              */
  /* ---------------------------------------------------------------- */
  window.addEventListener("DOMContentLoaded", () => {
    setupMap();
    loadHealth();
    loadArchetypes();
    document.getElementById("ask").addEventListener("click", askQuestion);
    document.getElementById("query").addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") askQuestion();
    });
  });
})();
