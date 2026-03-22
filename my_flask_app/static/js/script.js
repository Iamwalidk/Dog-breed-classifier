import { createUploadCardController } from "./components/UploadCard.js";
import { showLoader, hideLoader } from "./components/Loader.js";
import { createPredictionCard } from "./components/PredictionCard.js";

const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const previewImage = document.getElementById("previewImage");
const uploadPreview = document.getElementById("uploadPreview");
const uploadPlaceholder = document.getElementById("uploadPlaceholder");
const predictBtn = document.getElementById("predictBtn");
const resetBtn = document.getElementById("resetBtn");
const statusText = document.getElementById("statusText");
const errorBanner = document.getElementById("errorBanner");
const loaderContainer = document.getElementById("loaderContainer");
const emptyState = document.getElementById("emptyState");
const resultsContainer = document.getElementById("resultsContainer");
const resultsSummary = document.getElementById("resultsSummary");
const fileMeta = document.getElementById("fileMeta");
const initialPredictionsScript = document.getElementById("initialPredictionsData");
const initialPredictionMetaScript = document.getElementById("initialPredictionMetaData");

function setStatus(message) {
  if (statusText) {
    statusText.textContent = message;
  }
}

function showError(message) {
  if (!errorBanner) {
    return;
  }

  if (!message) {
    errorBanner.textContent = "";
    errorBanner.hidden = true;
    return;
  }

  errorBanner.textContent = message;
  errorBanner.hidden = false;
}

function setBusy(isBusy) {
  if (resultsContainer) {
    resultsContainer.setAttribute("aria-busy", String(Boolean(isBusy)));
  }
}

function hideResults() {
  if (resultsContainer) {
    resultsContainer.replaceChildren();
    resultsContainer.hidden = true;
  }
  if (emptyState) {
    emptyState.hidden = true;
  }
  if (resultsSummary) {
    resultsSummary.hidden = true;
    resultsSummary.classList.remove("is-populated");
    resultsSummary.replaceChildren();
  }
}

function createSummaryItem(labelText, valueText) {
  const item = document.createElement("div");
  item.className = "info-item";

  const label = document.createElement("p");
  label.className = "info-item-label";
  label.textContent = labelText;

  const value = document.createElement("p");
  value.className = "info-item-value";
  value.textContent = valueText;

  item.append(label, value);
  return item;
}

function formatMetaValue(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }

  if (Array.isArray(value)) {
    const joined = value
      .map((item) => (item === null || item === undefined ? "" : String(item).trim()))
      .filter(Boolean)
      .join(", ");
    return joined || null;
  }

  if (typeof value === "object") {
    return null;
  }

  return String(value);
}

function formatConfidence(confidence) {
  const numeric = Number(confidence);
  if (!Number.isFinite(numeric)) {
    return "0.0%";
  }
  const percent = numeric <= 1 ? numeric * 100 : numeric;
  return `${Math.max(0, Math.min(100, percent)).toFixed(1)}%`;
}

function renderSummary(predictions, meta) {
  if (!resultsSummary) {
    return;
  }

  const safePredictions = Array.isArray(predictions) ? predictions : [];
  const topPrediction = safePredictions[0];
  const count = safePredictions.length;

  resultsSummary.replaceChildren();

  if (!topPrediction) {
    resultsSummary.hidden = true;
    resultsSummary.classList.remove("is-populated");
    return;
  }

  const title = document.createElement("p");
  title.className = "results-summary-title";
  title.textContent = "Best match";

  const value = document.createElement("p");
  value.className = "results-summary-value";
  value.textContent = topPrediction.breed || "Unknown breed";

  const copy = document.createElement("p");
  copy.className = "results-summary-copy";
  copy.textContent = `The model returned ${count} ranked prediction${count === 1 ? "" : "s"} with ${formatConfidence(topPrediction.confidence)} confidence for the top result.`;

  resultsSummary.append(title, value, copy);

  if (meta && typeof meta === "object") {
    const metaEntries = Object.entries(meta)
      .map(([label, valueText]) => [label, formatMetaValue(valueText)])
      .filter(([, valueText]) => valueText)
      .slice(0, 3);

    if (metaEntries.length) {
      metaEntries.forEach(([label, valueText]) => {
        resultsSummary.append(createSummaryItem(label.replace(/_/g, " "), String(valueText)));
      });
    }
  }

  resultsSummary.hidden = false;
  resultsSummary.classList.add("is-populated");
}

function renderPredictions(predictions, meta) {
  const safePredictions = Array.isArray(predictions) ? predictions.slice(0, 3) : [];
  resultsContainer.replaceChildren();

  if (!safePredictions.length) {
    resultsContainer.hidden = true;
    if (emptyState) {
      emptyState.hidden = false;
    }
    renderSummary([], null);
    setStatus("No predictions were returned for this image.");
    setBusy(false);
    return;
  }

  safePredictions.forEach((prediction, index) => {
    resultsContainer.append(createPredictionCard(prediction, index + 1));
  });

  resultsContainer.hidden = false;
  if (emptyState) {
    emptyState.hidden = true;
  }
  renderSummary(safePredictions, meta);
  setStatus("Top 3 predictions ready.");
  setBusy(false);
}

function readJsonFromScript(scriptElement) {
  if (!scriptElement) {
    return null;
  }

  try {
    const parsed = JSON.parse(scriptElement.textContent || "null");
    return parsed === undefined ? null : parsed;
  } catch {
    return null;
  }
}

if (
  form &&
  fileInput &&
  previewImage &&
  uploadPreview &&
  uploadPlaceholder &&
  predictBtn &&
  resetBtn &&
  loaderContainer &&
  emptyState &&
  resultsContainer &&
  resultsSummary
) {
  const uploadController = createUploadCardController({
    fileInput,
    previewImage,
    previewWrapper: uploadPreview,
    placeholder: uploadPlaceholder,
    analyzeButton: predictBtn,
    resetButton: resetBtn,
    fileMeta,
    onSelectionChange: (file) => {
      if (file) {
        showError("");
        setStatus(`Selected ${file.name || "an image"} and ready to analyze.`);
        return;
      }

      showError("");
      setStatus("Ready to analyze a dog image.");
    },
    onError: (message) => {
      showError(message);
      setStatus("Choose a valid image file to continue.");
    },
  });

  const initialPredictions = readJsonFromScript(initialPredictionsScript) || [];
  const initialMeta = readJsonFromScript(initialPredictionMetaScript);

  if (Array.isArray(initialPredictions) && initialPredictions.length) {
    renderPredictions(initialPredictions, initialMeta);
  } else {
    resultsContainer.hidden = true;
    if (resultsSummary) {
      resultsSummary.hidden = true;
      resultsSummary.classList.remove("is-populated");
      resultsSummary.replaceChildren();
    }
    if (emptyState) {
      emptyState.hidden = false;
    }
    setBusy(false);
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const selectedFile = uploadController.getSelectedFile();
    if (!selectedFile) {
      showError("Please select an image before running analysis.");
      setStatus("Select a dog image to begin.");
      return;
    }

    showError("");
    hideResults();
    setBusy(true);
    showLoader(loaderContainer, "Running inference and arranging the shortlist.");
    setStatus("Analyzing breed characteristics...");
    uploadController.setLoading(true);

    try {
      const formData = new FormData(form);
      if (!formData.get("file") && selectedFile) {
        formData.set("file", selectedFile);
      }
      const response = await fetch(form.action || "/predict", {
        method: "POST",
        body: formData,
        headers: {
          Accept: "application/json",
          "X-Requested-With": "XMLHttpRequest",
        },
      });

      let payload = {};
      try {
        payload = await response.json();
      } catch {
        payload = {};
      }

      if (!response.ok) {
        throw new Error(payload.error || "Prediction failed. Please try again.");
      }

      renderPredictions(payload.predictions || [], payload.meta || null);
    } catch (error) {
      hideResults();
      showError(error.message || "Prediction failed. Please try again.");
      setStatus("Unable to complete analysis.");
      setBusy(false);
    } finally {
      hideLoader(loaderContainer);
      uploadController.setLoading(false);
    }
  });
}
