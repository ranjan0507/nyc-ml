// NYC Taxi Fare Predictor — Frontend Logic

const API_URL = "http://127.0.0.1:8000/predict";

const form        = document.getElementById("predict-form");
const resultPanel = document.getElementById("result-panel");
const submitBtn   = document.getElementById("submit-btn");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  // Collect form values
  const tripDistance = parseFloat(document.getElementById("trip_distance").value);
  const passengerCount = parseInt(document.getElementById("passenger_count").value, 10);
  const pickupDatetime = document.getElementById("pickup_datetime").value.replace("T", " ");
  const puLocationID = parseInt(document.getElementById("PULocationID").value, 10);
  const doLocationID = parseInt(document.getElementById("DOLocationID").value, 10);
  const vendorID = parseInt(document.getElementById("VendorID").value, 10);
  const ratecodeID = parseInt(document.getElementById("RatecodeID").value, 10);

  // Basic validation
  if (
    isNaN(tripDistance) || isNaN(passengerCount) || !pickupDatetime ||
    isNaN(puLocationID) || isNaN(doLocationID) || isNaN(vendorID) || isNaN(ratecodeID)
  ) {
    showError("Please fill in all fields with valid values.");
    return;
  }

  const payload = {
    trip_distance: tripDistance,
    passenger_count: passengerCount,
    tpep_pickup_datetime: pickupDatetime,
    PULocationID: puLocationID,
    DOLocationID: doLocationID,
    VendorID: vendorID,
    RatecodeID: ratecodeID,
  };

  showLoading();
  submitBtn.disabled = true;

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }

    const data = await response.json();
    showResult(data.predicted_fare);
  } catch (err) {
    console.error("Prediction error:", err);
    showError("Prediction failed. Check inputs or ensure the API is running.");
  } finally {
    submitBtn.disabled = false;
  }
});

// --- UI State Renderers ---

function showLoading() {
  resultPanel.innerHTML = `
    <div class="result-card loading">
      <div class="spinner"></div>
      Generating prediction…
    </div>`;
}

function showResult(fare) {
  const formatted = Number(fare).toFixed(2);
  resultPanel.innerHTML = `
    <div class="result-card success">
      <div class="result-label">Predicted Fare</div>
      <div class="result-value">$${formatted}</div>
    </div>`;
}

function showError(message) {
  resultPanel.innerHTML = `
    <div class="result-card error">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/>
        <line x1="15" y1="9" x2="9" y2="15"/>
        <line x1="9" y1="9" x2="15" y2="15"/>
      </svg>
      ${message}
    </div>`;
}
