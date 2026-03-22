function normalizeConfidence(confidence) {
  const numericConfidence = Number(confidence);
  if (!Number.isFinite(numericConfidence)) {
    return 0;
  }

  if (numericConfidence <= 1) {
    return Math.max(0, Math.min(100, numericConfidence * 100));
  }

  return Math.max(0, Math.min(100, numericConfidence));
}

function getToneClass(percent) {
  if (percent >= 75) {
    return "is-high";
  }

  if (percent >= 45) {
    return "is-medium";
  }

  return "is-low";
}

export function createConfidenceBar(confidence) {
  const percent = normalizeConfidence(confidence);
  const roundedPercent = Number(percent.toFixed(1));

  const block = document.createElement("div");
  block.className = "confidence-block";

  const row = document.createElement("div");
  row.className = "confidence-row";

  const label = document.createElement("span");
  label.className = "confidence-label";
  label.textContent = "Confidence";

  const value = document.createElement("span");
  value.className = "confidence-value";
  value.textContent = `${roundedPercent}%`;

  row.append(label, value);

  const track = document.createElement("div");
  track.className = "confidence-track";
  track.setAttribute("role", "progressbar");
  track.setAttribute("aria-label", "Prediction confidence");
  track.setAttribute("aria-valuemin", "0");
  track.setAttribute("aria-valuemax", "100");
  track.setAttribute("aria-valuenow", String(Math.round(roundedPercent)));

  const fill = document.createElement("div");
  fill.className = `confidence-fill ${getToneClass(percent)}`;
  fill.style.width = "0%";

  track.append(fill);
  block.append(row, track);

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      fill.style.width = `${percent}%`;
    });
  });

  return block;
}
