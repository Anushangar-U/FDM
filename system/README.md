# Stage 9 and 10 prediction system

This folder integrates the saved final Random Forest pipeline into a simple Flask backend and browser frontend.

## What the system does

1. The user enters the current time and raw temperature, light, sound, CO2 and PIR sensor values.
2. The backend validates the values.
3. `feature_engineering.py` recreates the PE1 engineered features required by the final model: time-of-day minutes, sensor averages, PIR activity, temperature/light spreads and the fixed CO2 level code.
4. The saved pipeline recreates `Sound_Level_Code` internally through `SoundTertileEncoder` and generates the occupancy prediction.
5. The frontend displays the predicted class as 0, 1, 2 or 3 people.

## Run locally

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe system\app.py
```

Open:

```text
http://127.0.0.1:5000
```

The **Use sample values** button fills the form with the first observation from the original dataset so the end-to-end flow can be demonstrated quickly.

## API

### Health check

```text
GET /api/health
```

### Prediction

```text
POST /api/predict
Content-Type: application/json
```

Required fields:

- `time` (`HH:MM` or `HH:MM:SS`)
- `S1_Temp` to `S4_Temp`
- `S1_Light` to `S4_Light`
- `S1_Sound` to `S4_Sound`
- `S5_CO2`
- `S5_CO2_Slope`
- `S6_PIR` and `S7_PIR` (0 or 1)

The backend intentionally does not ask the user to enter engineered features manually. It recreates them using the same definitions used during PE1.

## Tests

Run the feature-engineering checks from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_system_feature_engineering
```

The tests verify that the sample raw observation recreates the same engineered values used in the PE1 training export, as well as CO2 boundary and input-validation behaviour.
