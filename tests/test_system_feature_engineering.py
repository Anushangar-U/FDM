import json
from pathlib import Path
import unittest

from system.feature_engineering import InputValidationError, build_model_frame


ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "models" / "room_occupancy_final_model_metadata.json").open(
    "r", encoding="utf-8"
) as handle:
    EXPECTED_FEATURES = json.load(handle)["input_features"]


SAMPLE = {
    "time": "10:49:41",
    "S1_Temp": 24.94,
    "S2_Temp": 24.75,
    "S3_Temp": 24.56,
    "S4_Temp": 25.38,
    "S1_Light": 121,
    "S2_Light": 34,
    "S3_Light": 53,
    "S4_Light": 40,
    "S1_Sound": 0.08,
    "S2_Sound": 0.19,
    "S3_Sound": 0.06,
    "S4_Sound": 0.06,
    "S5_CO2": 390,
    "S5_CO2_Slope": 0.769230769231,
    "S6_PIR": 0,
    "S7_PIR": 0,
}


class FeatureEngineeringTests(unittest.TestCase):
    def test_sample_matches_first_training_row_engineering(self):
        frame = build_model_frame(SAMPLE, EXPECTED_FEATURES)
        row = frame.iloc[0]

        self.assertEqual(frame.columns.tolist(), EXPECTED_FEATURES)
        self.assertAlmostEqual(row["Time_of_Day_Minutes"], 649.6833333333333)
        self.assertAlmostEqual(row["Avg_Temp"], 24.9075)
        self.assertAlmostEqual(row["Avg_Light"], 62.0)
        self.assertAlmostEqual(row["Avg_Sound"], 0.0975)
        self.assertEqual(row["PIR_Activity"], 0)
        self.assertAlmostEqual(row["Temp_Spread"], 0.82)
        self.assertAlmostEqual(row["Light_Spread"], 87.0)
        self.assertEqual(row["CO2_Level_Code"], 0)

    def test_co2_boundaries_match_pe1_rules(self):
        payload = dict(SAMPLE)
        payload["S5_CO2"] = 800
        self.assertEqual(
            build_model_frame(payload, EXPECTED_FEATURES).iloc[0]["CO2_Level_Code"], 1
        )

        payload["S5_CO2"] = 1000
        self.assertEqual(
            build_model_frame(payload, EXPECTED_FEATURES).iloc[0]["CO2_Level_Code"], 1
        )

        payload["S5_CO2"] = 1000.1
        self.assertEqual(
            build_model_frame(payload, EXPECTED_FEATURES).iloc[0]["CO2_Level_Code"], 2
        )

    def test_invalid_pir_is_rejected(self):
        payload = dict(SAMPLE)
        payload["S6_PIR"] = 2
        with self.assertRaises(InputValidationError):
            build_model_frame(payload, EXPECTED_FEATURES)

    def test_missing_sensor_is_rejected(self):
        payload = dict(SAMPLE)
        del payload["S1_Temp"]
        with self.assertRaises(InputValidationError):
            build_model_frame(payload, EXPECTED_FEATURES)


if __name__ == "__main__":
    unittest.main()
