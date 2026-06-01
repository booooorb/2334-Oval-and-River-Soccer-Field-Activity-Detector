const rows = window.LABELER_ROWS || [];
let counts = window.LABELER_COUNTS || {};
let index = Math.max(0, rows.findIndex((row) => !row.label));

if (index < 0) {
  index = 0;
}

const image = document.querySelector("#image");
const foregroundImage = document.querySelector("#foregroundImage");
const foregroundEmpty = document.querySelector("#foregroundEmpty");
const meta = document.querySelector("#meta");
const notes = document.querySelector("#notes");
const list = document.querySelector("#list");
const stats = document.querySelector("#stats");
const updateDataButton = document.querySelector("#updateData");
const experimentStatus = document.querySelector("#experimentStatus");
const modelResults = document.querySelector("#modelResults");
const predictionTable = document.querySelector("#predictionTable");
let selectedModelReport = null;
const imageCacheBuster = Date.now();

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function rowDate(row) {
  const source = row.timestamp_local || row.timestamp_utc || row.timestamp || row.image_id || "";
  const match = source.match(/\d{4}-\d{2}-\d{2}/);
  if (match) {
    return match[0];
  }
  return (row.image_id || "unknown").split("/", 1)[0] || "unknown";
}

function rowTime(row) {
  const source = row.timestamp_local || row.timestamp || row.timestamp_utc || row.image_id || "";
  const match = source.match(/T?(\d{2}:\d{2})(?::\d{2})?/);
  if (match) {
    return match[1];
  }
  const nameMatch = source.match(/T(\d{2})(\d{2})(\d{2})/);
  if (nameMatch) {
    return `${nameMatch[1]}:${nameMatch[2]}`;
  }
  return source.split("/").pop() || "unknown";
}

function refreshStats() {
  counts = { unlabeled: 0, active: 0, inactive: 0, discard: 0 };
  for (const row of rows) {
    counts[row.label || "unlabeled"] += 1;
  }
  stats.innerHTML = Object.entries(counts)
    .map(([key, value]) => `<span class="stat">${key}: ${value}</span>`)
    .join("");
}

function renderList() {
  const groups = new Map();
  rows.forEach((row, rowIndex) => {
    const date = rowDate(row);
    if (!groups.has(date)) {
      groups.set(date, []);
    }
    groups.get(date).push({ row, rowIndex });
  });

  list.innerHTML = [...groups.entries()]
    .map(([date, entries]) => {
      const body = entries
        .map(({ row, rowIndex }) => {
          const label = row.label || "unlabeled";
          const current = rowIndex === index ? " current" : "";
          const modelPrediction = selectedModelReport?.predictions?.find((prediction) => prediction.image_id === row.image_id);
          const predictionClass = modelPrediction ? (modelPrediction.correct ? " correct" : " incorrect") : "";
          const predictionText = modelPrediction ? modelPrediction.prediction : "";
          return `<button type="button" class="item${current}${predictionClass}" data-index="${rowIndex}">
            <span class="item-time">${escapeHtml(rowTime(row))}</span>
            <span class="item-label ${label}">${escapeHtml(label)}</span>
            <span class="item-prediction">${escapeHtml(predictionText)}</span>
          </button>`;
        })
        .join("");
      return `<section class="date-group">
        <div class="date-header">
          <span>${escapeHtml(date)}</span>
          <strong>${entries.length}</strong>
        </div>
        <div class="date-items">${body}</div>
      </section>`;
    })
    .join("");

  requestAnimationFrame(() => {
    const currentItem = list.querySelector(".item.current");
    if (currentItem) {
      currentItem.scrollIntoView({ block: "nearest" });
    }
  });
}

function show() {
  if (!rows.length) {
    meta.textContent = "No rows found. Run update_data.py first.";
    image.removeAttribute("src");
    return;
  }

  const row = rows[index];
  image.src = `/image?path=${encodeURIComponent(row.masked_path)}&v=${imageCacheBuster}`;
  const previous = row.previous_image_id || "none";
  const modelPrediction = selectedModelReport?.predictions?.find((prediction) => prediction.image_id === row.image_id);
  const predictionText = modelPrediction
    ? ` - prediction: ${modelPrediction.prediction} (${modelPrediction.correct ? "correct" : "incorrect"})`
    : "";
  if (modelPrediction?.foreground_path) {
    foregroundImage.src = `/image?path=${encodeURIComponent(modelPrediction.foreground_path)}&v=${imageCacheBuster}`;
    foregroundImage.hidden = false;
    foregroundEmpty.hidden = true;
  } else {
    foregroundImage.removeAttribute("src");
    foregroundImage.hidden = true;
    foregroundEmpty.hidden = false;
  }
  meta.textContent = `${index + 1} / ${rows.length} - ${row.timestamp || row.image_id} - ${row.label || "unlabeled"} - previous: ${previous}${predictionText}`;
  notes.value = row.notes || "";
  document.querySelectorAll("[data-label]").forEach((button) => {
    button.classList.toggle("selected", button.dataset.label === (row.label || ""));
  });
  refreshStats();
  renderList();
}

async function save(label = rows[index].label || "") {
  const row = rows[index];
  const response = await fetch("/label", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_id: row.image_id, label, notes: notes.value }),
  });

  if (!response.ok) {
    alert(await response.text());
    return;
  }

  row.label = label;
  row.notes = notes.value;
  show();
}

function nextUnlabeled() {
  const found = rows.findIndex((row, rowIndex) => rowIndex > index && !row.label);
  index = found >= 0 ? found : Math.max(0, rows.findIndex((row) => !row.label));
  show();
}

document.querySelector("#prev").addEventListener("click", () => {
  index = Math.max(0, index - 1);
  show();
});

document.querySelector("#next").addEventListener("click", nextUnlabeled);

document.querySelector("#nextAny").addEventListener("click", () => {
  index = Math.min(rows.length - 1, index + 1);
  show();
});

document.querySelectorAll("[data-label]").forEach((button) => {
  button.addEventListener("click", async () => {
    await save(button.dataset.label);
    if (button.dataset.label) {
      nextUnlabeled();
    }
  });
});

notes.addEventListener("change", () => save(rows[index].label || ""));

list.addEventListener("click", (event) => {
  const button = event.target.closest("[data-index]");
  if (!button) {
    return;
  }
  index = Number(button.dataset.index);
  show();
});

window.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea")) {
    return;
  }

  if (event.key === "1") {
    save("inactive").then(nextUnlabeled);
  }
  if (event.key === "2") {
    save("active").then(nextUnlabeled);
  }
  if (event.key === "3") {
    save("discard").then(nextUnlabeled);
  }
  if (event.key === "ArrowRight") {
    nextUnlabeled();
  }
  if (event.key === "ArrowLeft") {
    index = Math.max(0, index - 1);
    show();
  }
});

function selectedModels() {
  const models = [...document.querySelectorAll('input[name="model"]:checked')]
    .map((input) => input.value)
    .filter(Boolean);
  if (models.length) {
    return models;
  }

  const fallback = document.querySelector('input[name="model"][value="balanced_previous_diff_blur_lumps_deploy"]');
  if (fallback) {
    fallback.checked = true;
    return [fallback.value];
  }
  return [];
}

function percent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function numberList(id) {
  return document
    .querySelector(id)
    .value.split(",")
    .map((value) => value.trim())
    .filter(Boolean);
}

function detectorOptions() {
  return {
    reference_update_mode: document.querySelector("#referenceUpdateMode").value,
    reference_strategy: document.querySelector("#referenceStrategy").value,
    min_lump_area: Number(document.querySelector("#minLumpArea").value),
    max_lump_area: Number(document.querySelector("#maxLumpArea").value),
    min_density: Number(document.querySelector("#minLumpDensity").value),
    max_aspect_ratio: Number(document.querySelector("#maxLumpAspect").value),
    artifact_penalty: Number(document.querySelector("#artifactPenalty").value),
    high_confidence_inactive: Number(document.querySelector("#highConfidenceInactive").value),
    hysteresis_margin: Number(document.querySelector("#hysteresisMargin").value),
    rgb_color_weight: Number(document.querySelector("#rgbColorWeight").value),
    auto_tune_lumps: document.querySelector("#autoTuneLumps").checked,
  };
}

function renderMetric(label, value) {
  return `<div class="metric-row"><span>${label}</span><strong>${value}</strong></div>`;
}

function renderModelReport(report, reportIndex) {
  const train = report.train;
  const validation = report.validation;
  const best = reportIndex === 0 ? " best" : "";
  return `<button type="button" class="model-card${best}" data-report-index="${reportIndex}">
    <h3>${report.model}</h3>
    ${renderMetric("threshold", report.threshold)}
    ${renderMetric("cutoff", Number(report.cutoff).toFixed(3))}
    ${renderMetric("window", report.window)}
    ${renderMetric("blur", Number(report.blur_radius).toFixed(2))}
    ${report.config ? renderMetric("config", report.config) : ""}
    ${renderMetric("train accuracy", percent(train.accuracy))}
    ${renderMetric("validation accuracy", percent(validation.accuracy))}
    ${renderMetric("balanced activity", percent(validation.balanced_activity))}
    ${renderMetric("active F1", percent(validation.f1_active))}
    ${renderMetric("active precision", percent(validation.precision_active))}
    ${renderMetric("active recall", percent(validation.recall_active))}
    ${renderMetric("inactive recall", percent(validation.recall_inactive))}
    ${renderMetric("discard F1", percent(validation.f1_discard))}
    ${renderMetric("discard recall", percent(validation.recall_discard))}
  </button>`;
}

function renderPredictions(report) {
  selectedModelReport = report;
  if (report.predictions.length) {
    const firstPredictionIndex = rows.findIndex((row) => row.image_id === report.predictions[0].image_id);
    if (firstPredictionIndex >= 0) {
      index = firstPredictionIndex;
    }
  }
  predictionTable.innerHTML = report.predictions
    .map((prediction) => {
      const status = prediction.correct ? "correct" : "incorrect";
      return `<button type="button" class="prediction-row ${status}" data-image-id="${prediction.image_id}">
        <span>${prediction.timestamp || prediction.image_id}</span>
        <span>actual: ${prediction.label}</span>
        <span>pred: ${prediction.prediction}</span>
        <span>${status}</span>
        <span class="prediction-score">score ${Number(prediction.score).toFixed(4)} - conf ${percent(prediction.confidence)} - changed ${percent(prediction.changed_ratio)} - largest ${percent(prediction.largest_blob_ratio)} - blobs ${prediction.blob_count} - shift ${prediction.camera_shift_x},${prediction.camera_shift_y}</span>
      </button>`;
    })
    .join("");
  show();
}

async function runModels() {
  const runButton = document.querySelector("#runModels");
  runButton.disabled = true;
  runButton.textContent = "Running...";
  experimentStatus.textContent = "Running detector comparison...";
  modelResults.innerHTML = "";
  predictionTable.innerHTML = "";

  try {
    const response = await fetch("/api/run-models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        models: selectedModels(),
        validation_percent: Number(document.querySelector("#validationPercent").value),
        thresholds: numberList("#thresholds"),
        cutoffs: numberList("#cutoffs"),
        windows: numberList("#windows"),
        min_blob_area: Number(document.querySelector("#minBlobArea").value),
        max_shift_pixels: Number(document.querySelector("#maxShiftPixels").value),
        blur_radius: Number(document.querySelector("#blurRadius").value),
        detector_options: detectorOptions(),
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      experimentStatus.textContent = data.error || "Model comparison failed.";
      return;
    }

    experimentStatus.textContent = `Non-validation frames: ${data.train_count}. Validation frames: ${data.validation_count}. Best settings balance active detection with keeping inactive frames inactive.`;
    window.LAST_MODEL_REPORTS = [];
    modelResults.innerHTML = data.models.map(renderModelReport).join("");
    if (data.models.length) {
      renderPredictions(data.models[0]);
    }
  } finally {
    runButton.disabled = false;
    runButton.textContent = "Run";
  }
}

document.querySelector("#runModels").addEventListener("click", runModels);

updateDataButton.addEventListener("click", async () => {
  updateDataButton.disabled = true;
  updateDataButton.textContent = "Updating...";
  try {
    const response = await fetch("/api/update-data", { method: "POST" });
    const data = await response.json();
    if (!response.ok) {
      alert(data.error || "Update failed.");
      return;
    }
    alert(`Synced ${data.synced} image(s), processed ${data.processed}, added ${data.added_labels} label row(s).`);
    window.location.reload();
  } finally {
    updateDataButton.disabled = false;
    updateDataButton.textContent = "Update Data";
  }
});

modelResults.addEventListener("click", (event) => {
  const button = event.target.closest("[data-report-index]");
  if (!button) {
    return;
  }
  const reports = [...modelResults.querySelectorAll("[data-report-index]")];
  reports.forEach((item) => item.classList.remove("best"));
  button.classList.add("best");
  const reportIndex = Number(button.dataset.reportIndex);
  const report = window.LAST_MODEL_REPORTS?.[reportIndex];
  if (report) {
    renderPredictions(report);
  }
});

predictionTable.addEventListener("click", (event) => {
  const button = event.target.closest("[data-image-id]");
  if (!button) {
    return;
  }
  const rowIndex = rows.findIndex((row) => row.image_id === button.dataset.imageId);
  if (rowIndex >= 0) {
    index = rowIndex;
    show();
  }
});

const originalRenderModelReport = renderModelReport;
renderModelReport = function rememberReport(report, reportIndex) {
  if (!window.LAST_MODEL_REPORTS) {
    window.LAST_MODEL_REPORTS = [];
  }
  window.LAST_MODEL_REPORTS[reportIndex] = report;
  return originalRenderModelReport(report, reportIndex);
};

show();
