import { createConfidenceBar } from "./ConfidenceBar.js";

const INFO_CONFIG = [
  { key: "description", label: "Description" },
  { key: "temperament", label: "Temperament" },
  { key: "size", label: "Size" },
  { key: "life_span", label: "Life span" },
];

function getText(value) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function createInfoItem(labelText, valueText) {
  const item = document.createElement("div");
  item.className = "info-item";

  const label = document.createElement("p");
  label.className = "info-item-label";
  label.textContent = labelText;

  const value = document.createElement("p");
  value.className = "info-item-value";
  value.textContent = valueText || "Not available";

  item.append(label, value);
  return item;
}

function createSummaryChip(text) {
  const chip = document.createElement("span");
  chip.className = "prediction-summary-chip";
  chip.textContent = text;
  return chip;
}

function createId(prefix) {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

function getConfidencePercent(confidence) {
  const numeric = Number(confidence);
  if (!Number.isFinite(numeric)) {
    return 0;
  }

  if (numeric <= 1) {
    return Math.max(0, Math.min(100, numeric * 100));
  }

  return Math.max(0, Math.min(100, numeric));
}

export function createPredictionCard(prediction, rank) {
  const breed = getText(prediction?.breed) || "Unknown breed";
  const confidence = getConfidencePercent(prediction?.confidence);
  const info = prediction?.info && typeof prediction.info === "object" ? prediction.info : {};
  const meta = prediction?.meta && typeof prediction.meta === "object" ? prediction.meta : {};
  const isTopMatch = rank === 1;

  const card = document.createElement("article");
  card.className = "prediction-card";
  card.dataset.rank = String(rank);
  if (isTopMatch) {
    card.dataset.featured = "true";
  }

  const header = document.createElement("div");
  header.className = "prediction-card-header";

  const rankBadge = document.createElement("div");
  rankBadge.className = "rank-badge";
  rankBadge.textContent = `#${rank}`;

  const titleGroup = document.createElement("div");
  titleGroup.className = "prediction-title-group";

  const title = document.createElement("h3");
  title.className = "prediction-breed";
  title.textContent = breed;

  const rankLabel = document.createElement("p");
  rankLabel.className = "prediction-rank-label";
  rankLabel.textContent = isTopMatch ? "Top match" : `Rank ${rank} prediction`;

  titleGroup.append(title, rankLabel);
  header.append(rankBadge, titleGroup);

  const confidenceBar = createConfidenceBar(confidence);

  const narrative = document.createElement("p");
  narrative.className = "prediction-meta-line";
  narrative.textContent = isTopMatch
    ? "This is the strongest match based on the model output."
    : "This prediction adds diversity to the ranked shortlist.";

  const summaryChipRow = document.createElement("div");
  summaryChipRow.className = "prediction-summary-chip-row";
  summaryChipRow.append(
    createSummaryChip(`${confidence.toFixed(1)}% confidence`),
    createSummaryChip(isTopMatch ? "Primary match" : "Supporting candidate"),
  );

  if (getText(meta?.source)) {
    summaryChipRow.append(createSummaryChip(meta.source));
  }

  const detailsToggle = document.createElement("button");
  detailsToggle.type = "button";
  detailsToggle.className = "details-toggle";
  detailsToggle.textContent = "Show breed details";

  const detailsId = createId(`details-${rank}`);
  detailsToggle.setAttribute("aria-expanded", "false");
  detailsToggle.setAttribute("aria-controls", detailsId);

  const detailsPanel = document.createElement("div");
  detailsPanel.id = detailsId;
  detailsPanel.className = "details-panel";

  const detailsGrid = document.createElement("div");
  detailsGrid.className = "details-grid";

  let hasAnyInfo = false;
  INFO_CONFIG.forEach(({ key, label }) => {
    const value = getText(info[key]);
    if (value) {
      hasAnyInfo = true;
    }
    detailsGrid.append(createInfoItem(label, value));
  });

  if (!hasAnyInfo) {
    const fallback = document.createElement("p");
    fallback.className = "details-empty";
    fallback.textContent = "No additional breed metadata is available for this result yet.";
    detailsPanel.append(fallback);
  } else {
    detailsPanel.append(detailsGrid);
  }

  if (getText(meta?.note)) {
    const note = document.createElement("p");
    note.className = "prediction-meta-line";
    note.textContent = meta.note;
    detailsPanel.append(note);
  }

  detailsToggle.addEventListener("click", () => {
    const expanded = card.classList.toggle("is-expanded");
    detailsToggle.setAttribute("aria-expanded", String(expanded));
    detailsToggle.textContent = expanded ? "Hide breed details" : "Show breed details";
  });

  card.append(header, confidenceBar, narrative, summaryChipRow, detailsToggle, detailsPanel);
  return card;
}
