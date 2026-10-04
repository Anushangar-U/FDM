const form = document.getElementById("prediction-form");
const sampleButton = document.getElementById("sample-button");
const realSampleButton = document.getElementById("real-sample-button");
const sampleStatus = document.getElementById("sample-status");
const predictionModal = document.getElementById("prediction-modal");
const modalClose = document.getElementById("modal-close");
const modalDone = document.getElementById("modal-done");
const occupancyNumber = document.getElementById("occupancy-number");
const predictionMessage = document.getElementById("prediction-message");
const predictionDetail = document.getElementById("prediction-detail");
const probabilityPanel = document.getElementById("probability-panel");
const submitButton = form.querySelector('button[type="submit"]');

const numericFields = [
  "S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp",
  "S1_Light", "S2_Light", "S3_Light", "S4_Light",
  "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
  "S5_CO2", "S5_CO2_Slope", "S6_PIR", "S7_PIR",
];

let inputRanges = {};

function closePredictionModal() {
  if (predictionModal.open) {
    predictionModal.close();
  }
}

function renderProbabilities(probabilities) {
  const hasProbabilities = probabilities && typeof probabilities === "object";
  probabilityPanel.hidden = !hasProbabilities;

  for (let classId = 0; classId <= 3; classId += 1) {
    const value = hasProbabilities ? Number(probabilities[String(classId)] ?? 0) : 0;
    const percent = Math.max(0, Math.min(100, value * 100));
    document.getElementById(`probability-${classId}-text`).textContent = `${percent.toFixed(1)}%`;
    document.getElementById(`probability-${classId}-bar`).style.width = `${percent}%`;
  }
}

function showResult({ prediction = "!", message, detail = "", probabilities = null, isError = false }) {
  occupancyNumber.textContent = prediction;
  predictionMessage.textContent = message;
  predictionDetail.textContent = detail;
  predictionModal.classList.toggle("error", isError);
  renderProbabilities(isError ? null : probabilities);

  if (!predictionModal.open) {
    predictionModal.showModal();
  }
}

function collectPayload() {
  const payload = { time: document.getElementById("time").value };

  for (const field of numericFields) {
    const rawValue = document.getElementById(field).value;
    const value = Number(rawValue);
    if (rawValue === "" || !Number.isFinite(value)) {
      throw new Error(`${field} must contain a valid number.`);
    }
    payload[field] = value;
  }

  return payload;
}

function formatRangeValue(field, value) {
  const number = Number(value);
  if (field.includes("Sound") || field === "S5_CO2_Slope") {
    return number.toFixed(3);
  }
  if (field.includes("Temp")) {
    return number.toFixed(2);
  }
  return number.toFixed(0);
}

function getWarningElement(field) {
  const input = document.getElementById(field);
  if (!input) {
    return null;
  }

  const label = input.closest("label");
  if (!label) {
    return null;
  }

  let warning = label.querySelector(".field-warning");
  if (!warning) {
    warning = document.createElement("small");
    warning.className = "field-warning";
    warning.hidden = true;
    label.appendChild(warning);
  }

  return warning;
}

function setFieldWarning(field, message = "") {
  const input = document.getElementById(field);
  const warning = getWarningElement(field);
  if (!input || !warning) {
    return;
  }

  const hasWarning = Boolean(message);
  warning.textContent = message;
  warning.hidden = !hasWarning;
  input.closest("label")?.classList.toggle("has-range-warning", hasWarning);
}

function updateFieldWarning(field) {
  const range = inputRanges[field];
  if (!range) {
    return false;
  }

  const input = document.getElementById(field);
  const value = Number(input.value);
  if (input.value === "" || !Number.isFinite(value)) {
    setFieldWarning(field, "");
    return false;
  }

  const low = Number(range.typical_low);
  const high = Number(range.typical_high);
  if (value < low || value > high) {
    setFieldWarning(
      field,
      `Unusual for this dataset. Typical range: ${formatRangeValue(field, low)}–${formatRangeValue(field, high)}. Prediction is still allowed.`,
    );
    return true;
  }

  setFieldWarning(field, "");
  return false;
}

function updateAllRangeWarnings() {
  let count = 0;
  for (const field of Object.keys(inputRanges)) {
    if (updateFieldWarning(field)) {
      count += 1;
    }
  }
  return count;
}

function applyServerWarnings(warnings = []) {
  updateAllRangeWarnings();
  for (const warning of warnings) {
    const field = warning.field;
    if (!field || !document.getElementById(field)) {
      continue;
    }
    setFieldWarning(
      field,
      `Unusual for this dataset. Typical range: ${formatRangeValue(field, warning.typical_low)}–${formatRangeValue(field, warning.typical_high)}. Prediction is still allowed.`,
    );
  }
}

function fillForm(sample) {
  for (const [field, value] of Object.entries(sample)) {
    const input = document.getElementById(field);
    if (input) {
      input.value = value;
    }
  }

  updateAllRangeWarnings();
  closePredictionModal();
}

function setAutofillBusy(isBusy) {
  sampleButton.disabled = isBusy;
  realSampleButton.disabled = isBusy;
}

async function loadInputRanges() {
  try {
    const response = await fetch("/api/input-ranges");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Could not load input ranges.");
    }

    inputRanges = data.ranges || {};
    updateAllRangeWarnings();
  } catch (error) {
    console.warn("Range guidance unavailable:", error);
  }
}

async function autofillRandomSample() {
  setAutofillBusy(true);
  sampleButton.textContent = "Generating…";
  sampleStatus.classList.remove("status-error");

  try {
    const response = await fetch("/api/random-input");
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Could not generate random sensor values.");
    }

    fillForm(data.sample);
    sampleStatus.textContent = "Generated new values within dataset-informed realistic limits";
  } catch (error) {
    sampleStatus.textContent = error.message;
    sampleStatus.classList.add("status-error");
  } finally {
    setAutofillBusy(false);
    sampleButton.textContent = "🎲 Generate random values";
  }
}

async function autofillRealSample() {
  setAutofillBusy(true);
  realSampleButton.textContent = "Loading…";
  sampleStatus.classList.remove("status-error");

  try {
    const response = await fetch("/api/real-input");
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Could not load a real dataset row.");
    }

    fillForm(data.sample);
    sampleStatus.textContent = "";
  } catch (error) {
    sampleStatus.textContent = error.message;
    sampleStatus.classList.add("status-error");
  } finally {
    setAutofillBusy(false);
    realSampleButton.textContent = "📋 Load real dataset row";
  }
}

for (const field of numericFields) {
  const input = document.getElementById(field);
  input?.addEventListener("input", () => updateFieldWarning(field));
  input?.addEventListener("change", () => updateFieldWarning(field));
}

sampleButton.addEventListener("click", autofillRandomSample);
realSampleButton.addEventListener("click", autofillRealSample);
modalClose.addEventListener("click", closePredictionModal);
modalDone.addEventListener("click", closePredictionModal);

predictionModal.addEventListener("click", (event) => {
  if (event.target === predictionModal) {
    closePredictionModal();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  // Hard validation catches missing values, invalid numbers, negative values for
  // sensors that cannot be negative, and PIR values outside 0/1. Soft range
  // warnings do not block prediction.
  if (!form.reportValidity()) {
    return;
  }

  let payload;
  try {
    payload = collectPayload();
  } catch (error) {
    showResult({ message: error.message, detail: "Please correct the input and try again.", isError: true });
    return;
  }

  updateAllRangeWarnings();
  submitButton.disabled = true;
  submitButton.textContent = "Predicting…";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction request failed.");
    }

    applyServerWarnings(data.warnings || []);
    const warningCount = (data.warnings || []).length;
    const detail = warningCount > 0
      ? `Prediction generated. ${warningCount} input value${warningCount === 1 ? " is" : "s are"} outside the typical dataset range, so interpret this result with extra caution.`
      : "Based on the current sensor readings.";

    showResult({
      prediction: data.prediction,
      message: data.message,
      detail,
      probabilities: data.probabilities,
    });
  } catch (error) {
    showResult({
      message: error.message,
      detail: "The input was rejected or the prediction service could not complete the request.",
      isError: true,
    });
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Predict occupancy";
  }
});

loadInputRanges();
