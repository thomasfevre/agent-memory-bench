const state = {
  registry: null,
  activeView: "overview",
};

const colors = ["#77a9ff", "#7de2c6", "#f3aa68", "#d58cff", "#ff7e88"];

function byId(id) {
  return document.getElementById(id);
}

function run(id) {
  return state.registry.runs.find((item) => item.id === id);
}

function percent(value, digits = 1) {
  return `${(value * 100).toFixed(digits).replace(".0", "")}%`;
}

function number(value) {
  return new Intl.NumberFormat("en-US").format(value);
}

function setView(name) {
  state.activeView = name;
  document.querySelectorAll("[data-view-panel]").forEach((panel) => {
    const active = panel.dataset.viewPanel === name;
    panel.hidden = !active;
    panel.classList.toggle("is-visible", active);
  });
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.view === name);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderMetricStrip() {
  const manifestPromise = fetch("data/raw-evidence-manifest.json")
    .then((response) => (response.ok ? response.json() : null))
    .catch(() => null);

  manifestPromise.then((manifest) => {
    const controlled = state.registry.runs.filter(
      (item) => item.evidence_level === "controlled",
    ).length;
    const official = state.registry.runs.filter(
      (item) => item.evidence_level === "official-data",
    ).length;
    const artifacts = manifest?.artifact_count ?? 0;
    const metrics = [
      ["Public run records", state.registry.runs.length, "bounded configurations"],
      ["Controlled runs", controlled, "aligned local protocols"],
      ["Official-data runs", official, "public benchmark slices"],
      ["Raw evidence artifacts", artifacts, "hashes plus reviewed public copies"],
    ];
    byId("metric-strip").innerHTML = metrics
      .map(
        ([label, value, note]) => `
          <div class="metric">
            <span class="metric-label">${label}</span>
            <strong class="metric-value">${number(value)}</strong>
            <span class="metric-note">${note}</span>
          </div>`,
      )
      .join("");
  });
}

function renderWorkflow() {
  byId("workflow").innerHTML = state.registry.workflow
    .map(
      (item, index) => `
        <article class="workflow-step" style="--coverage:${55 + index * 6}%">
          <span class="workflow-index">${String(index + 1).padStart(2, "0")}</span>
          <h3>${item.label}</h3>
          <p>${item.description}</p>
        </article>`,
    )
    .join("");
}

function renderOverview() {
  byId("headline").textContent = state.registry.campaign.headline;
  byId("updated-at").textContent = state.registry.updated_at;
  byId("finding-list").innerHTML = state.registry.findings
    .slice(0, 8)
    .map((item) => `<li>${item}</li>`)
    .join("");
  byId("limitation-list").innerHTML = state.registry.limitations
    .map((item) => `<li>${item}</li>`)
    .join("");
  renderMetricStrip();
  renderWorkflow();
}

function addSvg(tag, attributes, text) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  if (text !== undefined) element.textContent = text;
  return element;
}

function renderScatter() {
  const svg = byId("retrieval-scatter");
  const series = run("common-retrieval-20260729").metrics.series;
  const width = 860;
  const height = 430;
  const margin = { left: 72, right: 48, top: 38, bottom: 64 };
  const x = (value) =>
    margin.left + ((value - 0.84) / 0.17) * (width - margin.left - margin.right);
  const y = (value) =>
    height -
    margin.bottom -
    (value / 0.45) * (height - margin.top - margin.bottom);

  [...svg.querySelectorAll(":scope > *:not(title):not(desc)")].forEach((node) =>
    node.remove(),
  );

  [0, 0.1, 0.2, 0.3, 0.4].forEach((tick) => {
    svg.appendChild(
      addSvg("line", {
        x1: margin.left,
        x2: width - margin.right,
        y1: y(tick),
        y2: y(tick),
        class: "grid",
      }),
    );
    svg.appendChild(
      addSvg(
        "text",
        {
          x: margin.left - 12,
          y: y(tick) + 5,
          "text-anchor": "end",
          class: "tick",
        },
        percent(tick, 0),
      ),
    );
  });

  [0.85, 0.9, 0.95, 1].forEach((tick) => {
    svg.appendChild(
      addSvg("line", {
        x1: x(tick),
        x2: x(tick),
        y1: margin.top,
        y2: height - margin.bottom,
        class: "grid",
      }),
    );
    svg.appendChild(
      addSvg(
        "text",
        {
          x: x(tick),
          y: height - margin.bottom + 26,
          "text-anchor": "middle",
          class: "tick",
        },
        percent(tick, 0),
      ),
    );
  });

  svg.appendChild(
    addSvg(
      "text",
      {
        x: (margin.left + width - margin.right) / 2,
        y: height - 12,
        "text-anchor": "middle",
        class: "axis-label",
      },
      "Evidence recall",
    ),
  );
  svg.appendChild(
    addSvg(
      "text",
      {
        transform: `translate(20 ${(margin.top + height - margin.bottom) / 2}) rotate(-90)`,
        "text-anchor": "middle",
        class: "axis-label",
      },
      "Context precision",
    ),
  );

  const offsets = {
    "Long context": [-96, 20],
    BM25: [-48, -16],
    "MiniLM dense": [10, 24],
    Hybrid: [10, 6],
    "Dated facts": [-82, 24],
    "Dated graph": [-88, 24],
    "Type router": [-28, -18],
    "Parallel fusion": [-70, -16],
  };

  series.forEach((item, index) => {
    const circle = addSvg("circle", {
      cx: x(item.recall),
      cy: y(item.context_precision),
      r: 8 + item.temporal_correctness * 7,
      fill: colors[index % colors.length],
      class: "point",
    });
    circle.appendChild(
      addSvg(
        "title",
        {},
        `${item.method}: ${percent(item.recall)} recall, ${percent(item.context_precision)} precision, ${percent(item.temporal_correctness, 0)} temporal`,
      ),
    );
    svg.appendChild(circle);
    const [dx, dy] = offsets[item.method] ?? [10, -10];
    svg.appendChild(
      addSvg(
        "text",
        {
          x: x(item.recall) + dx,
          y: y(item.context_precision) + dy,
          class: "point-label",
        },
        item.method,
      ),
    );
  });
}

function renderBars(target, series, valueKey, options = {}) {
  const max = options.max ?? Math.max(...series.map((item) => item[valueKey]));
  target.innerHTML = series
    .map((item) => {
      const value = item[valueKey];
      const label = item.method ?? item.policy ?? item.system ?? item.mode;
      const display = options.format ? options.format(value) : percent(value);
      const submetric = options.submetric ? options.submetric(item) : "";
      return `
        <div class="bar-row">
          <span class="bar-name">${label}${submetric ? `<span class="submetric">${submetric}</span>` : ""}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.max(1, (value / max) * 100)}%"></div></div>
          <span class="bar-value">${display}</span>
        </div>`;
    })
    .join("");
}

function renderResultCharts() {
  renderScatter();
  const generation = run("common-generation-qwen8-20260729").metrics.series;
  renderBars(byId("generation-chart"), generation, "accuracy", {
    max: 1,
    submetric: (item) => `${number(item.tokens)} tokens`,
  });
  const topology = run("topology-qwen17-20260729").metrics.series;
  renderBars(byId("topology-chart"), topology, "accuracy", {
    max: 1,
    submetric: (item) => `${item.calls} calls · ${number(item.tokens)} tokens`,
  });

  const locomo = run("locomo-retrieval-20260729");
  const longmem = run("longmemeval-retrieval-20260729");
  const conflict = run("memoryagentbench-conflict-6k-20260729");
  const memgym = run("memgym-dr-retrieval-20260729");
  const cards = [
    {
      title: "LoCoMo turn evidence",
      value: percent(locomo.metrics.series.find((item) => item.method === "Hybrid").top5_recall),
      body: `Hybrid top-5 recall falls to ${percent(locomo.metrics.hybrid_multihop_recall)} on multi-hop.`,
    },
    {
      title: "LongMemEval sessions",
      value: percent(longmem.metrics.series.find((item) => item.method === "Hybrid").top5_recall),
      body: "The unit is a full session, so the result is not directly comparable to LoCoMo turns.",
    },
    {
      title: "MemoryAgentBench multi-hop",
      value: percent(
        conflict.metrics.series.find(
          (item) => item.task === "multi-hop" && item.method === "BM25",
        ).substring,
      ),
      body: "BM25 beats full context at 1%, but both remain weak on contradictions and relation chains.",
    },
    {
      title: "MemGym-DR lexical proxy",
      value: percent(memgym.metrics.bm25_proxy_recall.find((item) => item.top_k === 2).value),
      body: "At top-10 the proxy saturates at 100% with roughly 3,800 words and no final reader.",
    },
  ];
  byId("public-comparisons").innerHTML = cards
    .map(
      (card) => `
        <article class="comparison-card">
          <strong>${card.title}</strong>
          <span class="large-value">${card.value}</span>
          <p>${card.body}</p>
        </article>`,
    )
    .join("");

  const jcode = run("jcode-common-20260729").metrics.series;
  const mem0 = run("mem0-common-20260729").metrics.series;
  const jcodeInputCount = Math.max(
    ...jcode.map((item) => item.memories_after_ingestion),
  );
  const rows = [
    ...jcode.map((item) => [
      "jcode",
      item.mode,
      `${item.memories_after_ingestion} / ${jcodeInputCount}`,
      percent(item.recall),
      percent(item.temporal_correctness),
      item.diagnostic,
    ]),
    ...mem0.map((item) => [
      "Mem0",
      item.mode,
      `${item.memories} memories`,
      percent(item.recall),
      percent(item.temporal_correctness),
      item.diagnostic,
    ]),
  ];
  byId("ingestion-table").innerHTML = `
    <table>
      <thead><tr><th>System</th><th>Mode</th><th>Stored</th><th>Recall</th><th>Temporal</th><th>Observed failure</th></tr></thead>
      <tbody>${rows
        .map(
          (row) => `<tr>${row
            .map((cell, index) => `<td class="${index >= 2 && index <= 4 ? "number" : ""}">${cell}</td>`)
            .join("")}</tr>`,
        )
        .join("")}</tbody>
    </table>`;
}

function renderSystems() {
  const list = byId("system-list");
  list.innerHTML = state.registry.systems
    .map(
      (system, index) => `
        <button class="system-button ${index === 0 ? "is-active" : ""}" type="button" role="listitem"
          data-system="${system.name}" data-status="${system.status}">
          <span>${system.name}</span>
        </button>`,
    )
    .join("");

  function showSystem(name) {
    const system = state.registry.systems.find((item) => item.name === name);
    byId("system-detail").innerHTML = `
      <span class="status">${system.status}</span>
      <h1>${system.name}</h1>
      <dl>
        <dt>Tested</dt><dd>${system.tested}</dd>
        <dt>Finding</dt><dd>${system.headline}</dd>
        <dt>Boundary</dt><dd>${system.status === "not-reproduced" ? "Research only" : "See matching public registry records"}</dd>
      </dl>`;
    document.querySelectorAll(".system-button").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.system === name);
    });
  }

  list.addEventListener("click", (event) => {
    const button = event.target.closest("[data-system]");
    if (button) showSystem(button.dataset.system);
  });
  showSystem(state.registry.systems[0].name);

  const graph = run("cognee-graphiti-common-20260729").metrics.series;
  byId("graph-comparison").innerHTML = graph
    .map(
      (item) => `
        <article class="comparison-card">
          <strong>${item.system}</strong>
          <span class="large-value">${percent(item.recall)}</span>
          <p>Mean source recall · ${item.documents_ingested}/8 documents ingested · ${number(Math.round(item.retrieval_ms))} ms retrieval</p>
          <p>${number(Math.round(item.ingestion_s_mean))} s mean ingestion · ${percent(item.temporal_correctness)} temporal correctness</p>
        </article>`,
    )
    .join("");

  const harnessRun = run("priority3-coding-harness-qwen25-14b-20260730");
  const harnessLabel = (value) =>
    value === "tool_protocol_incompatible"
      ? "Tool protocol incompatible"
      : value === "provider_incompatible"
        ? "Provider incompatible"
        : value;
  byId("harness-comparison").innerHTML = harnessRun.metrics.harnesses
    .map(
      (item) => `
        <article class="comparison-card">
          <strong>${item.harness}</strong>
          <span class="large-value">${item.tasks_completed}/${item.distinct_tasks}</span>
          <p>Tasks completed · ${harnessLabel(item.classification)}</p>
          <p>${item.known_total_tokens === null ? "Tokens unavailable" : `${number(item.known_total_tokens)} known tokens`} · ${number(Math.round(item.known_wall_time_seconds))} s retained wall time</p>
        </article>`,
    )
    .join("");
  byId("harness-comparison").insertAdjacentHTML(
    "afterend",
    '<p class="chart-note">This is a pinned-model compatibility result and not a quality ranking of the harnesses with their recommended models.</p>',
  );
}

function renderEvidence() {
  const phases = [...new Set(state.registry.runs.map((item) => item.phase))].sort();
  const levels = [
    ...new Set(state.registry.runs.map((item) => item.evidence_level)),
  ].sort();
  byId("phase-filter").insertAdjacentHTML(
    "beforeend",
    phases.map((item) => `<option value="${item}">${item}</option>`).join(""),
  );
  byId("level-filter").insertAdjacentHTML(
    "beforeend",
    levels.map((item) => `<option value="${item}">${item}</option>`).join(""),
  );

  function update() {
    const phase = byId("phase-filter").value;
    const level = byId("level-filter").value;
    const search = byId("run-search").value.trim().toLowerCase();
    const filtered = state.registry.runs.filter((item) => {
      const haystack = `${item.id} ${item.dataset} ${item.task} ${item.method} ${item.reader ?? ""}`.toLowerCase();
      return (
        (!phase || item.phase === phase) &&
        (!level || item.evidence_level === level) &&
        (!search || haystack.includes(search))
      );
    });
    byId("result-count").textContent = `${filtered.length} of ${state.registry.runs.length} public run records`;
    byId("run-list").innerHTML = filtered
      .map(
        (item) => `
          <details class="run-card">
            <summary>
              <span class="run-id">${item.id}</span>
              <strong>${item.dataset} · ${item.task}</strong>
              <span class="evidence-level">${item.evidence_level}</span>
            </summary>
            <div class="run-body">
              <div>
                <h3>Configuration</h3>
                <p>${item.method}</p>
                <p><strong>Reader:</strong> ${item.reader ?? "none"}<br>
                <strong>Sample:</strong> ${item.sample}<br>
                <strong>Repetitions:</strong> ${item.repetitions}</p>
              </div>
              <div>
                <h3>Interpretation</h3>
                <p>${item.conclusion}</p>
                <p><strong>Limit:</strong> ${item.limitation}</p>
                <p>${item.evidence_files.map((file) => `<code>${file}</code>`).join("<br>")}</p>
              </div>
            </div>
          </details>`,
      )
      .join("");
  }

  ["phase-filter", "level-filter", "run-search"].forEach((id) => {
    byId(id).addEventListener(id === "run-search" ? "input" : "change", update);
  });
  update();
}

async function initialize() {
  const response = await fetch("data/registry.json");
  if (!response.ok) throw new Error(`registry request failed: ${response.status}`);
  state.registry = await response.json();
  renderOverview();
  renderResultCharts();
  renderSystems();
  renderEvidence();

  document.querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.view));
  });
  document.querySelectorAll("[data-open-view]").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.openView));
  });
}

initialize().catch((error) => {
  console.error(error);
  byId("headline").textContent =
    "The public result registry could not be loaded. Open the repository for the versioned evidence.";
});
