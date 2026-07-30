const state = {
  campaigns: [],
  activeCampaignId: null,
  itemIndex: 0,
  itemStartedAt: Date.now(),
  saveTimer: null,
  saving: false,
  queuedSave: false,
  reviewingCompleteCampaign: false,
};

const scoreLabels = {
  "0": "Incorrecte",
  "0.3": "Faible",
  "0.5": "Partielle",
  "0.7": "Presque juste",
  "1": "Correcte",
};

const decisionLabels = {
  approved: ["Approuver", "Mémoire suffisamment fiable"],
  rejected: ["Rejeter", "Incorrecte ou trop fragile"],
  deferred: ["Reporter", "Davantage de preuves nécessaires"],
};

const scopeLabels = {
  personal: "Personnel",
  team: "Équipe",
  task: "Tâche précise",
};

const injectionLabels = {
  always_on: "Toujours injecter",
  task_specific: "Seulement si pertinent",
  never: "Ne jamais injecter",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function campaign() {
  return state.campaigns.find(
    (item) => item.campaign_id === state.activeCampaignId,
  );
}

function item() {
  return campaign().items[state.itemIndex];
}

function annotation() {
  return campaign().annotations[item().item_id];
}

function itemIsComplete(targetCampaign, targetItem) {
  const row = targetCampaign.annotations[targetItem.item_id];
  if (targetCampaign.task_type === "semantic_answer_judge") {
    return row.score !== "" && row.confidence !== "";
  }
  return (
    row.decision !== "" &&
    row.scope !== "" &&
    row.injection !== "" &&
    row.confidence !== ""
  );
}

function campaignCompleted(targetCampaign) {
  return targetCampaign.items.filter((targetItem) =>
    itemIsComplete(targetCampaign, targetItem),
  ).length;
}

function commitElapsed() {
  if (!state.activeCampaignId || !campaign()?.items.length) return;
  const elapsed = Math.max(0, Math.round((Date.now() - state.itemStartedAt) / 1000));
  if (elapsed > 0) {
    const row = annotation();
    const previous = Number(row.time_seconds || 0);
    row.time_seconds = String(previous + elapsed);
  }
  state.itemStartedAt = Date.now();
}

function resetTimer() {
  state.itemStartedAt = Date.now();
}

function setSaveStatus(message) {
  document.getElementById("save-status").textContent = message;
}

async function saveActiveCampaign() {
  if (state.saving) {
    state.queuedSave = true;
    return;
  }
  commitElapsed();
  state.saving = true;
  setSaveStatus("Enregistrement…");
  const active = campaign();
  try {
    const response = await fetch(`/api/save/${active.campaign_id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ annotations: active.annotations }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Échec de sauvegarde");
    const index = state.campaigns.findIndex(
      (entry) => entry.campaign_id === payload.campaign_id,
    );
    state.campaigns[index] = payload;
    setSaveStatus("Sauvegardé localement");
  } catch (error) {
    setSaveStatus(`Non sauvegardé : ${error.message}`);
  } finally {
    state.saving = false;
    resetTimer();
    if (state.queuedSave) {
      state.queuedSave = false;
      await saveActiveCampaign();
    }
  }
}

function scheduleSave() {
  window.clearTimeout(state.saveTimer);
  setSaveStatus("Modification locale…");
  state.saveTimer = window.setTimeout(saveActiveCampaign, 350);
}

function renderOverall() {
  const completed = state.campaigns.reduce(
    (total, entry) => total + campaignCompleted(entry),
    0,
  );
  const total = state.campaigns.reduce(
    (sum, entry) => sum + entry.total_items,
    0,
  );
  const percent = total ? Math.round((completed / total) * 100) : 0;
  document.getElementById("overall-progress").innerHTML = `
    <div class="overall">
      <div class="overall-line">
        <span>Progression globale</span>
        <strong>${completed}/${total}</strong>
      </div>
      <div class="progress-track" aria-label="${percent} % terminé">
        <div class="progress-fill" style="width:${percent}%"></div>
      </div>
    </div>`;
}

function renderCampaignNav() {
  document.getElementById("campaign-nav").innerHTML = state.campaigns
    .map((entry) => {
      const completed = campaignCompleted(entry);
      return `
        <button class="campaign-button" type="button"
          data-campaign="${escapeHtml(entry.campaign_id)}"
          ${entry.campaign_id === state.activeCampaignId ? 'aria-current="page"' : ""}>
          <span class="campaign-name">${escapeHtml(entry.label)}</span>
          <strong>${completed}/${entry.total_items}</strong>
          <span>${entry.task_type === "semantic_answer_judge" ? "Qualité des réponses" : "Décisions de mémoire"}</span>
          <span>${completed === entry.total_items ? "Terminé" : "En cours"}</span>
        </button>`;
    })
    .join("");
}

function optionMarkup(options, labels, selected, placeholder) {
  return `
    <option value="">${escapeHtml(placeholder)}</option>
    ${options
      .map(
        (value) =>
          `<option value="${escapeHtml(value)}" ${selected === value ? "selected" : ""}>${escapeHtml(labels[value] || value)}</option>`,
      )
      .join("")}`;
}

function confidenceMarkup(selected) {
  return optionMarkup(
    ["0.5", "0.8", "1.0"],
    {
      "0.5": "Faible · 50 %",
      "0.8": "Moyenne · 80 %",
      "1.0": "Haute · 100 %",
    },
    selected,
    "Choisir une confiance",
  );
}

function renderSemantic(targetItem, row) {
  const scoreOptions = targetItem.score_options || [0, 0.3, 0.5, 0.7, 1];
  return `
    <section class="question-block">
      <span class="answer-label">Question</span>
      <h3>${escapeHtml(targetItem.question)}</h3>
    </section>
    <div class="answer-comparison">
      <section class="answer">
        <span class="answer-label">Réponse de référence</span>
        <p>${escapeHtml(targetItem.gold_answer)}</p>
      </section>
      <section class="answer">
        <span class="answer-label">Réponse du système</span>
        <p>${escapeHtml(targetItem.predicted_answer)}</p>
      </section>
    </div>
    <section class="evaluation-form" aria-label="Évaluation de la réponse">
      <div>
        <span class="field-label">Quelle note donnerais-tu à la réponse du système ?</span>
        <div class="segment-grid" role="group" aria-label="Score">
          ${scoreOptions
            .map((score) => {
              const value = String(score);
              return `
                <button class="segment" type="button" data-field="score" data-value="${value}"
                  aria-pressed="${String(row.score) === value}">
                  <strong>${value}</strong>
                  <span>${scoreLabels[value] || ""}</span>
                </button>`;
            })
            .join("")}
        </div>
      </div>
      <div class="field">
        <label class="field-label" for="confidence">Ton niveau de confiance</label>
        <select id="confidence" data-input-field="confidence">
          ${confidenceMarkup(row.confidence)}
        </select>
      </div>
      ${notesMarkup(row)}
    </section>`;
}

function notesMarkup(row) {
  return `
    <div class="field">
      <label class="field-label" for="notes">Note facultative</label>
      <textarea id="notes" data-input-field="notes" placeholder="Explique surtout les scores partiels ou les décisions difficiles.">${escapeHtml(row.notes)}</textarea>
    </div>`;
}

function renderShard(targetItem, row) {
  const evidence = (targetItem.evidence || [])
    .map(
      (source) => `
        <article class="evidence">
          <div class="evidence-meta">
            <strong>${escapeHtml(source.source_label)}</strong>
            <span>${escapeHtml(source.timestamp || "Date non fournie")}</span>
          </div>
          <p>${escapeHtml(source.text)}</p>
        </article>`,
    )
    .join("");
  return `
    <section class="candidate-block">
      <span class="answer-label">Mémoire candidate</span>
      <h3 class="candidate-text">${escapeHtml(targetItem.candidate_text)}</h3>
      <div class="evidence-list" aria-label="Éléments de preuve">${evidence}</div>
    </section>
    <section class="evaluation-form" aria-label="Décision de mémoire">
      <div>
        <span class="field-label">Que faire de cette mémoire ?</span>
        <div class="segment-grid three" role="group" aria-label="Décision">
          ${(targetItem.decision_options || ["approved", "rejected", "deferred"])
            .map((value) => {
              const [label, detail] = decisionLabels[value];
              return `
                <button class="segment" type="button" data-field="decision" data-value="${value}"
                  aria-pressed="${row.decision === value}">
                  <strong>${label}</strong>
                  <span>${detail}</span>
                </button>`;
            })
            .join("")}
        </div>
      </div>
      <div class="form-grid">
        <div class="field">
          <label class="field-label" for="scope">Périmètre</label>
          <select id="scope" data-input-field="scope">
            ${optionMarkup(targetItem.scope_options || [], scopeLabels, row.scope, "Choisir un périmètre")}
          </select>
        </div>
        <div class="field">
          <label class="field-label" for="injection">Mode d’injection</label>
          <select id="injection" data-input-field="injection">
            ${optionMarkup(targetItem.injection_options || [], injectionLabels, row.injection, "Choisir un mode")}
          </select>
        </div>
      </div>
      <div class="field">
        <label class="field-label" for="confidence">Ton niveau de confiance</label>
        <select id="confidence" data-input-field="confidence">
          ${confidenceMarkup(row.confidence)}
        </select>
      </div>
      ${notesMarkup(row)}
    </section>`;
}

function firstIncompleteIndex(targetCampaign) {
  const index = targetCampaign.items.findIndex(
    (targetItem) => !itemIsComplete(targetCampaign, targetItem),
  );
  return index === -1 ? 0 : index;
}

function renderReview() {
  const active = campaign();
  const completed = campaignCompleted(active);
  const content = document.getElementById("review-content");
  if (
    completed === active.total_items &&
    !state.reviewingCompleteCampaign
  ) {
    const next = state.campaigns.find(
      (entry) => campaignCompleted(entry) < entry.total_items,
    );
    content.innerHTML = `
      <section class="complete-state">
        <span class="complete-mark" aria-hidden="true">✓</span>
        <h2>${escapeHtml(active.label)} est terminée</h2>
        <p class="empty-copy">Les ${active.total_items} réponses sont sauvegardées localement et prêtes pour le scorer mono-utilisateur.</p>
        ${
          next
            ? `<button class="button primary" type="button" data-campaign="${escapeHtml(next.campaign_id)}">Continuer avec ${escapeHtml(next.label)}</button>`
            : '<p><strong>Les 64 éléments sont terminés.</strong> Tu peux revenir ici à tout moment pour les relire.</p>'
        }
        <button class="button" type="button" data-action="review">Relire cette campagne</button>
      </section>`;
    return;
  }

  const targetItem = item();
  const row = annotation();
  const complete = itemIsComplete(active, targetItem);
  content.innerHTML = `
    <div class="section-header">
      <div>
        <p class="item-position">${escapeHtml(active.label)} · Élément ${state.itemIndex + 1} sur ${active.total_items}</p>
        <h2>${active.task_type === "semantic_answer_judge" ? "Comparer les deux réponses" : "Décider si cette mémoire mérite d’être conservée"}</h2>
      </div>
      <p class="timer">Temps actif : <span id="timer-value">${Number(row.time_seconds || 0)} s</span></p>
    </div>
    ${
      active.task_type === "semantic_answer_judge"
        ? renderSemantic(targetItem, row)
        : renderShard(targetItem, row)
    }
    <p class="validation-message" id="validation-message" ${complete ? "hidden" : ""}>
      ${active.task_type === "semantic_answer_judge" ? "Choisis un score et ta confiance pour valider." : "Choisis une décision, un périmètre, un mode d’injection et ta confiance."}
    </p>
    <nav class="navigation" aria-label="Navigation entre les éléments">
      <div class="navigation-group">
        <button class="button" type="button" data-action="previous" ${state.itemIndex === 0 ? "disabled" : ""}>Précédent</button>
        <button class="button ghost" type="button" data-action="skip">Passer pour l’instant</button>
      </div>
      <button class="button primary" type="button" data-action="next" ${complete ? "" : "disabled"}>
        ${state.itemIndex === active.total_items - 1 ? "Terminer la campagne" : "Enregistrer et continuer"}
      </button>
    </nav>`;
}

function render() {
  renderOverall();
  renderCampaignNav();
  renderReview();
  resetTimer();
}

async function switchCampaign(campaignId) {
  if (campaignId === state.activeCampaignId) return;
  await saveActiveCampaign();
  state.activeCampaignId = campaignId;
  state.itemIndex = firstIncompleteIndex(campaign());
  state.reviewingCompleteCampaign = false;
  render();
  document.getElementById("review-main").focus();
}

function move(direction, requireComplete = false) {
  const active = campaign();
  if (requireComplete && !itemIsComplete(active, item())) return;
  commitElapsed();
  const nextIndex = state.itemIndex + direction;
  if (nextIndex >= 0 && nextIndex < active.total_items) {
    state.itemIndex = nextIndex;
  } else if (direction > 0) {
    const incomplete = firstIncompleteIndex(active);
    if (campaignCompleted(active) === active.total_items) {
      state.reviewingCompleteCampaign = false;
    } else if (incomplete !== state.itemIndex) {
      state.itemIndex = incomplete;
    }
  }
  render();
  scheduleSave();
  document.getElementById("review-main").focus();
}

function setField(field, value) {
  commitElapsed();
  annotation()[field] = value;
  render();
  scheduleSave();
}

function handleClick(event) {
  const campaignButton = event.target.closest("[data-campaign]");
  if (campaignButton) {
    switchCampaign(campaignButton.dataset.campaign);
    return;
  }
  const segment = event.target.closest("[data-field]");
  if (segment) {
    setField(segment.dataset.field, segment.dataset.value);
    return;
  }
  const action = event.target.closest("[data-action]")?.dataset.action;
  if (action === "previous") move(-1);
  if (action === "skip") move(1);
  if (action === "next") move(1, true);
  if (action === "review") {
    state.reviewingCompleteCampaign = true;
    state.itemIndex = 0;
    render();
  }
}

function handleInput(event) {
  const field = event.target.dataset.inputField;
  if (!field) return;
  annotation()[field] = event.target.value;
  const complete = itemIsComplete(campaign(), item());
  const next = document.querySelector('[data-action="next"]');
  next.disabled = !complete;
  document.getElementById("validation-message").hidden = complete;
  scheduleSave();
}

function handleKeyboard(event) {
  if (
    event.target.matches("textarea, select, input") ||
    event.metaKey ||
    event.ctrlKey ||
    event.altKey
  ) {
    return;
  }
  if (campaign().task_type === "semantic_answer_judge") {
    const values = ["0", "0.3", "0.5", "0.7", "1"];
    const index = Number(event.key) - 1;
    if (index >= 0 && index < values.length) setField("score", values[index]);
  }
}

function startTimerDisplay() {
  window.setInterval(() => {
    const target = document.getElementById("timer-value");
    if (!target || !state.activeCampaignId) return;
    const existing = Number(annotation().time_seconds || 0);
    const current = Math.max(
      0,
      Math.round((Date.now() - state.itemStartedAt) / 1000),
    );
    target.textContent = `${existing + current} s`;
  }, 1000);
}

async function initialize() {
  const response = await fetch("/api/state", { cache: "no-store" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Impossible de charger les campagnes");
  state.campaigns = payload;
  const firstOpen =
    state.campaigns.find(
      (entry) => campaignCompleted(entry) < entry.total_items,
    ) || state.campaigns[0];
  state.activeCampaignId = firstOpen.campaign_id;
  state.itemIndex = firstIncompleteIndex(firstOpen);
  const template = document.getElementById("app-template");
  const app = document.getElementById("app");
  app.replaceChildren(template.content.cloneNode(true));
  app.addEventListener("click", handleClick);
  app.addEventListener("input", handleInput);
  document.addEventListener("keydown", handleKeyboard);
  window.addEventListener("beforeunload", commitElapsed);
  render();
  setSaveStatus("Prêt · sauvegarde automatique");
  startTimerDisplay();
}

initialize().catch((error) => {
  document.getElementById("app").innerHTML = `
    <div class="error-state">
      <strong>Impossible de lancer la revue.</strong>
      <p>${escapeHtml(error.message)}</p>
      <p>Vérifie que le serveur local est toujours ouvert dans le Terminal.</p>
    </div>`;
  setSaveStatus("Erreur de chargement");
});
