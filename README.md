# Public Transport Delay Analysis & Prediction (v2)

Data Analytics and Visualization mini project: Flask web app + scikit-learn models.

## How to run
```
pip install -r requirements.txt
python app.py                  # open http://127.0.0.1:5000   (trained models + charts are already included)
```
Optional: `python generate_dataset.py` re-creates the dataset, `python train.py` re-runs cleaning, EDA, model tuning and training (about 3 minutes).
Windows: double-click `run.bat`. Linux/Mac: `bash run.sh`.

## Dataset (summary)
| Item | Value |
|---|---|
| Source | Simulated by `generate_dataset.py` (seed 42) with realistic rules; replaceable by a real CSV with the same columns |
| Period / scope | 2025-01-01 to 2025-12-31, 12 routes, 4 modes (Bus, Tram, Metro, Train) |
| Rows BEFORE cleaning | **6,060** rows x 17 columns (`data/transport_delays_raw.csv`) |
| Rows AFTER cleaning | **6,002** rows x 20 columns (`data/transport_delays_clean.csv`) |
| Removed | 58 duplicate rows |
| Missing values | 729 missing cells in the raw file, all imputed (718 after duplicates) -> 0 missing after cleaning |
| Other fixes | 40 inconsistent weather labels standardised, 26 extreme delay outliers capped at 30.2 min, 3 new features added |
| Targets | `delay_minutes` (regression) and `is_delayed` = delay > 5 min (classification) |

The same figures appear in the app on the **Dataset** page, with a full data dictionary.

## What is in the web app
| Page | What it shows |
|---|---|
| Overview | key numbers, rows before -> after, best model |
| Dataset | rows before/after, cleaning steps, missing values per column, data dictionary, preview, statistics |
| Dashboard | **interactive** charts with filters (mode, weather, day, peak) and hover tooltips |
| Exploration | 15 charts (heatmaps, violin, route x hour, weather x mode ...) |
| Models | tuned models, metrics, residuals, ROC, confusion matrices, permutation importance |
| Predict | delay in minutes from 3 models, likely range, probability of delay, **why** (factor contributions) |
| Batch | upload a CSV of trips, get predictions for every row, download results |
| API | `POST /api/predict` with JSON (route_id, hour, weather ...) returns the prediction as JSON |

## ML integration
* `src/preprocessing.py` is shared by training and the app, so inputs are transformed identically.
* Pipelines (scaling + one-hot encoding + model) are saved with joblib in `models/` and loaded by `app.py`.
* Regression: Linear Regression, Random Forest (GridSearchCV), Gradient Boosting (RandomizedSearchCV).
* Classification: Logistic Regression, Random Forest (GridSearchCV), Gradient Boosting.
* Evaluation: MAE, RMSE, R2, 5-fold CV, accuracy, precision, recall, F1, ROC-AUC, residual analysis, permutation importance.

## Folder structure
```
app.py  train.py  generate_dataset.py  requirements.txt
data/       raw + cleaned CSV
src/        config, preprocessing, data_report, eda, modeling
models/     saved models + results.json
static/     style.css, charts.js, charts/
templates/  HTML pages
```
