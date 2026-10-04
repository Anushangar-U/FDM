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

// Five fixed, realistic demo records taken from the project dataset.
// The target class is intentionally not stored here because the frontend should
// send only sensor inputs to the backend and let the saved model predict it.
const samplePresets = [
  {
    name: "Demo sample 1",
    values: {
      time: "10:49:41",
      S1_Temp: 24.94, S2_Temp: 24.75, S3_Temp: 24.56, S4_Temp: 25.38,
      S1_Light: 121, S2_Light: 34, S3_Light: 53, S4_Light: 40,
      S1_Sound: 0.08, S2_Sound: 0.19, S3_Sound: 0.06, S4_Sound: 0.06,
      S5_CO2: 390, S5_CO2_Slope: 0.769230769231, S6_PIR: 0, S7_PIR: 0,
    },
  },
  {
    name: "Demo sample 2",
    values: {
      time: "11:42:16",
      S1_Temp: 25.5, S2_Temp: 25.56, S3_Temp: 24.88, S4_Temp: 25.81,
      S1_Light: 155, S2_Light: 237, S3_Light: 71, S4_Light: 55,
      S1_Sound: 1.87, S2_Sound: 0.56, S3_Sound: 0.23, S4_Sound: 0.14,
      S5_CO2: 500, S5_CO2_Slope: 1.99615384615, S6_PIR: 1, S7_PIR: 1,
    },
  },
  {
    name: "Demo sample 3",
    values: {
      time: "13:50:56",
      S1_Temp: 26.13, S2_Temp: 27.06, S3_Temp: 26, S4_Temp: 26.31,
      S1_Light: 165, S2_Light: 254, S3_Light: 274, S4_Light: 71,
      S1_Sound: 1.13, S2_Sound: 1.4, S3_Sound: 1.51, S4_Sound: 1.36,
      S5_CO2: 910, S5_CO2_Slope: 1.39230769231, S6_PIR: 1, S7_PIR: 1,
    },
  },
  {
    name: "Demo sample 4",
    values: {
      time: "20:00:36",
      S1_Temp: 26.13, S2_Temp: 26.31, S3_Temp: 25.81, S4_Temp: 26.19,
      S1_Light: 0, S2_Light: 0, S3_Light: 0, S4_Light: 0,
      S1_Sound: 0.07, S2_Sound: 0.05, S3_Sound: 0.06, S4_Sound: 0.06,
      S5_CO2: 1055, S5_CO2_Slope: -4.78461538462, S6_PIR: 0, S7_PIR: 0,
    },
  },
  {
    name: "Demo sample 5",
    values: {
      time: "13:51:27",
      S1_Temp: 26.06, S2_Temp: 27.13, S3_Temp: 26, S4_Temp: 26.31,
      S1_Light: 165, S2_Light: 256, S3_Light: 279, S4_Light: 71,
      S1_Sound: 0.55, S2_Sound: 0.14, S3_Sound: 0.85, S4_Sound: 0.25,
      S5_CO2: 910, S5_CO2_Slope: 1.35, S6_PIR: 1, S7_PIR: 1,
    },
  },
];

let lastSampleIndex = -1;

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

function randomSampleIndex() {
  if (samplePresets.length === 1) {
    return 0;
  }

  let index;
  do {
    index = Math.floor(Math.random() * samplePresets.length);
  } while (index === lastSampleIndex);

  return index;
}

function autofillRandomSample() {
  const index = randomSampleIndex();
  const sample = samplePresets[index];
  lastSampleIndex = index;

  for (const [field, value] of Object.entries(sample.values)) {
    document.getElementById(field).value = value;
  }

  sampleStatus.textContent = `${sample.name} loaded · ${samplePresets.length} saved samples`;
  closePredictionModal();
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
