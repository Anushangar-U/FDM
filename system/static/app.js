const form = document.getElementById("prediction-form");
const sampleButton = document.getElementById("sample-button");
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

async function autofillRandomSample() {
  sampleButton.disabled = true;
  sampleButton.textContent = "Randomizing…";

  try {
    const response = await fetch("/api/random-input");
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Could not generate random sensor values.");
    }

    for (const [field, value] of Object.entries(data.sample)) {
      document.getElementById(field).value = value;
    }

    sampleStatus.textContent = `Random real sensor record loaded · ${data.available_records.toLocaleString()} records available`;
    closePredictionModal();
  } catch (error) {
    sampleStatus.textContent = error.message;
  } finally {
    sampleButton.disabled = false;
    sampleButton.textContent = "🎲 Randomize sensor values";
  }
}

sampleButton.addEventListener("click", autofillRandomSample);
modalClose.addEventListener("click", closePredictionModal);
modalDone.addEventListener("click", closePredictionModal);

predictionModal.addEventListener("click", (event) => {
  if (event.target === predictionModal) {
    closePredictionModal();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  // Browser validation catches required fields, negative values where prohibited,
  // and invalid PIR selections before any request is sent to the backend.
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

    showResult({
      prediction: data.prediction,
      message: data.message,
      detail: "Based on the current sensor readings.",
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
