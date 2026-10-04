const form = document.getElementById("prediction-form");
const sampleButton = document.getElementById("sample-button");
const resultCard = document.getElementById("result-card");
const occupancyNumber = document.getElementById("occupancy-number");
const predictionMessage = document.getElementById("prediction-message");
const predictionDetail = document.getElementById("prediction-detail");
const submitButton = form.querySelector('button[type="submit"]');

const numericFields = [
  "S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp",
  "S1_Light", "S2_Light", "S3_Light", "S4_Light",
  "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
  "S5_CO2", "S5_CO2_Slope", "S6_PIR", "S7_PIR",
];

const sample = {
  time: "10:49:41",
  S1_Temp: 24.94,
  S2_Temp: 24.75,
  S3_Temp: 24.56,
  S4_Temp: 25.38,
  S1_Light: 121,
  S2_Light: 34,
  S3_Light: 53,
  S4_Light: 40,
  S1_Sound: 0.08,
  S2_Sound: 0.19,
  S3_Sound: 0.06,
  S4_Sound: 0.06,
  S5_CO2: 390,
  S5_CO2_Slope: 0.769230769231,
  S6_PIR: 0,
  S7_PIR: 0,
};

function showResult({ prediction = "!", message, detail = "", isError = false }) {
  occupancyNumber.textContent = prediction;
  predictionMessage.textContent = message;
  predictionDetail.textContent = detail;
  resultCard.classList.toggle("error", isError);
  resultCard.hidden = false;
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
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

sampleButton.addEventListener("click", () => {
  for (const [field, value] of Object.entries(sample)) {
    document.getElementById(field).value = value;
  }
  resultCard.hidden = true;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!form.reportValidity()) {
    return;
  }

  let payload;
  try {
    payload = collectPayload();
  } catch (error) {
    showResult({ message: error.message, isError: true });
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
      detail: "Prediction generated using the saved final Random Forest pipeline.",
    });
  } catch (error) {
    showResult({
      message: error.message,
      detail: "Check that the backend is running and the trained model can be loaded.",
      isError: true,
    });
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Predict occupancy";
  }
});
