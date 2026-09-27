# Room Occupancy Estimation

IT3051 Fundamentals of Data Mining: predict the number of people in a room
as one of four classes (0, 1, 2 or 3).

The supplied `data/raw/Occupancy_Estimation.csv` contains 10,129 timestamped
temperature, light, sound, CO2 and motion-sensor observations.

## Setup and notebooks

Tested with Python 3.14.2. In PowerShell, create the environment if needed,
then install the project dependencies:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Open the folder in VS Code with the Python and Jupyter extensions. Select
the `.venv` Python kernel and run notebooks in order:

1. `01_dataset_validation_PE1_completed.ipynb` ? dataset checks.
2. `02_eda.ipynb` ? exploratory analysis.
3. `03_preprocessing_PE1_fixed.ipynb` ? feature preparation and session split.
4. `04_pe2_model_development.ipynb` ? four models and imbalance experiments.
5. `05_pe2_model_optimization.ipynb` ? tuning, feature comparison and final evaluation.

Prepared PE1 files are included, so PE2 review can start at notebook 04.
Notebook 05 reuses the completed search CSVs and saved final model when present.
Rerunning it verifies the fixed result; do not tune based on the test scores.

## Project files

- `data/raw/` ? original sensor data.
- `data/pe1_preprocessing/` ? authoritative PE1 features and train/test exports.
- `notebooks/` ? the experimental record and viva explanations.
- `results/pe2/` ? essential comparison/search tables, selection, test results and figures.
- `models/` ? final pipeline and its input/parameter metadata.
- `src/pe2_transformers.py` ? shared sound encoder required to load the pipeline.

The final model is an 800-tree Random Forest with the extended sensor feature set
(24 inputs plus one fold-safe sound category). It was selected using four-fold,
30-minute-block CV before evaluation on the held-out **2017/12/23** session.
Final test accuracy is **0.9489** and macro F1 is **0.8586**.

Keep `src/pe2_transformers.py` importable from the project root when loading
`models/room_occupancy_final_model.joblib` with joblib.
