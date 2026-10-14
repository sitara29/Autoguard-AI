# AI-Powered Vehicle Failure Risk Prediction and Intelligent Maintenance Decision Support System

A machine-learning backend for a vehicle diagnostic decision-support system. It takes nine vehicle
parameters, predicts the dataset's `failure_risk` label, explains the prediction with SHAP, grades
each vehicle subsystem, and returns transparent maintenance advice. A FastAPI service exposes all of
it to the React / 3D-vehicle frontend.

| Project goal | Where it lives |
|---|---|
| Take vehicle parameters | `POST /predict` (validated by Pydantic) |
| Predict failure risk | tuned XGBoost classifier → `risk_level`, `risk_index`, `predicted_failure_risk` |
| Explain why | SHAP contributions for **all nine** features (`explainability.contributions`) |
| Maintenance decision support | model band + transparent rules with dataset evidence (`recommendations`, `maintenance_priority`) |
| Interactive 3D vehicle | per-component status for the hotspots (`components`) — rendered by `frontend/` |

## Important interpretation limits

- The dataset is **synthetic**: `failure_risk` was generated from a rule-based risk score (credit the original
  GitHub project that supplied `sensor_data.csv` in the report). Metrics describe how well the model learns
  that label, **not** real-fleet performance.
- The model score is **not a calibrated probability** (the model is class-weighted). Show `risk_level` /
  `risk_index`, not the raw percentage.
- The dataset has no time-to-failure target: the system does not predict *when* something fails.
- `vehicle_health_index` is a display value (100 − `risk_index`), not a separately trained health model.
- Recommendations are decision support, not diagnoses. Rule thresholds come from general maintenance
  practice; the API reports how strongly each rule is associated with the label in this dataset.

## Quick start

Python 3.11–3.13, ideally in a virtual environment.

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1      macOS/Linux:  source .venv/bin/activate
python -m pip install -r requirements.txt

uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000   # API  →  http://127.0.0.1:8000/docs
python -m pytest -q tests                                          # 27 tests
```

The trained model (`backend/models/vehicle_failure_model.joblib`) and reports are already included, so the
API runs without retraining.

## Reproduce the ML pipeline

```bash
python train_model.py              # ~5 min on 2 CPU cores; rewrites the model + reports/
python generate_report_figures.py  # ~10 s; writes reports/figures/*.png and INDEX.md
```

`train_model.py` performs, in order (all selection uses the training split only):

1. Stratified 80/20 split (12,000 train / 3,000 test, `random_state=42`).
2. Untuned baselines — Logistic Regression, Random Forest, XGBoost — with stratified 5-fold CV.
3. Hyperparameter tuning of **all three** models with the same folds and metric (average precision):
   grid search (Logistic Regression: 7 settings; Random Forest: 18) and randomized search (XGBoost: 40).
4. Final model: XGBoost (highest tuned CV average precision, and compatible with SHAP TreeExplainer).
5. Decision threshold = F1-maximising cut-off on out-of-fold training predictions (0.825).
6. One evaluation of the final model on the untouched test split.
7. Reference data for the API (feature ranges, SHAP grading cut-offs, risk-band evidence, rule evidence,
   demo vehicles) stored inside the model bundle.

Class imbalance (4.55 % positives) is handled with class weights (`class_weight` / `scale_pos_weight`),
stratified splits and folds, average precision, and threshold selection — **no resampling / SMOTE**.

## Folder layout

```text
vehicle_failure_project/
├── data/sensor_data.csv
├── train_model.py                 training, tuning, evaluation, saves model + reports
├── generate_report_figures.py     12 report/PPT figures from the saved model and data
├── backend/
│   ├── app.py                     FastAPI service
│   ├── reference.py               feature meta, components, risk index/bands, reference-data builder
│   ├── rules.py                   transparent maintenance rules (shared by training and API)
│   └── models/vehicle_failure_model.joblib
├── reports/
│   ├── model_metrics.json         full results (comparison, tuning, threshold, test, bands, rule evidence)
│   ├── baseline_cv_comparison.csv untuned vs tuned CV metrics
│   ├── dataset_audit.json
│   └── figures/                   fig01 … fig12 + INDEX.md
├── frontend/                      React + Three.js interface (npm run dev)
├── tests/                         API, extended-API and unit tests
├── requirements.txt
├── start_backend.bat
└── start_frontend.bat
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | service / model status |
| `GET /metadata` | per-feature label, unit, training range and percentiles; components; risk-band definitions; rules with dataset evidence; decision threshold |
| `GET /samples` | real held-out dataset rows (healthy, moderate, high-risk, overheating) to load as demo vehicles |
| `GET /model-info` | model comparison, tuning, threshold and test results (the contents of `model_metrics.json`) |
| `POST /predict` | prediction + explanation + component status + advice for one vehicle |

Example request body for `POST /predict` (all nine fields required, unknown fields rejected):

```json
{ "mileage_km": 160000, "engine_temp_c": 105, "rpm": 2200, "vibration_level": 0.08,
  "oil_quality_pct": 55, "brake_pad_mm": 3.2, "battery_voltage": 12.1,
  "coolant_level_pct": 70, "vehicle_age_years": 11 }
```

Key fields of the `/predict` response:

| Field | Meaning |
|---|---|
| `risk_level` | `LOW` / `MODERATE` / `HIGH`; `HIGH` ⇔ `predicted_failure_risk == 1` |
| `risk_index` | 0–100 display scale, monotone in the score; the decision threshold maps to 50 (LOW < 30 ≤ MODERATE < 50 ≤ HIGH) |
| `vehicle_health_index` | 100 − `risk_index` (display only) |
| `band_context` | how often the dataset label was high-risk for vehicles in this band (out-of-fold, training data) |
| `maintenance_priority` | `HIGH` if the model flags the vehicle or a high-severity rule fires; `MEDIUM` for the moderate band, a medium rule, or a notable SHAP component; else `LOW` |
| `explainability.contributions` | all 9 features, sorted by impact: `contribution` (SHAP, log-odds), `level` (`HIGH`/`MEDIUM`/`LOW`/`NONE`), `input_percentile`, `component` |
| `explainability.top_features` | the first five of the above |
| `components` | six subsystems (`engine`, `cooling`, `lubrication`, `brakes`, `battery`, `general_wear`), each with `status` (`ok` / `watch` / `critical`), its features and AI contribution |
| `recommendations` | model-based advice plus rule-based advisories, each with severity, component and dataset `evidence` |
| `input_warnings` | inputs outside the range seen in training (predictions there may be unreliable) |
| `model_risk_score` | raw uncalibrated model score — keep for transparency, do not show as a probability |

The React / Three.js frontend lives in `frontend/` (see `frontend/README.md` for the page list and for how the reference mockup was adapted to what the model supports; `start_frontend.bat` on Windows).

Frontend notes: colour the 3D hotspots from `components[].status`; the engine panel can read
`components[engine].features` for temperature / RPM / vibration contributions; build sliders from
`/metadata` feature ranges (the dataset's vibration values run 0.001–0.36, not 0–10). Allowed browser
origins default to `http://localhost:5173` and `http://127.0.0.1:5173`; override with the
`FRONTEND_ORIGINS` environment variable (comma-separated).

## Experiment results

15,000 rows, 9 features, no missing values, no duplicates; 682 high-risk rows (4.55 %).

Stratified 5-fold CV on the 12,000-row training split (mean scores), candidates ranked by average precision:

| Model | Stage | Configs tried | ROC-AUC | Average precision | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | baseline | – | 0.817 | 0.422 | 0.153 | 0.696 | 0.250 |
| Logistic Regression | tuned | 7 | 0.818 | 0.423 | 0.154 | 0.698 | 0.252 |
| Random Forest | baseline | – | 0.814 | 0.411 | 0.663 | 0.353 | 0.460 |
| Random Forest | tuned | 18 | 0.824 | 0.434 | 0.229 | 0.615 | 0.333 |
| XGBoost | baseline | – | 0.791 | 0.412 | 0.266 | 0.566 | 0.362 |
| XGBoost | tuned | 40 | 0.823 | **0.438** | 0.209 | 0.645 | 0.315 |

Precision / recall / F1 here use a fixed 0.5 cut-off and depend on each model's class weighting; compare
models on the threshold-free columns. The three tuned models differ by less than the fold-to-fold standard
deviation (≈ 0.04 average precision), so they are statistically close — XGBoost is selected because it scored
highest and works with SHAP TreeExplainer. Selected parameters: `max_depth=3, n_estimators=100,
learning_rate=0.03, min_child_weight=10, subsample=0.9, colsample_bytree=0.8, reg_lambda=5` (shallow, heavily
regularised — consistent with noisy labels).

**Held-out test (3,000 vehicles, used once).** ROC-AUC **0.851**, average precision **0.420** (no-skill 0.045).

| Threshold | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.825 (selected, max OOF F1) | 0.448 | 0.412 | 0.429 | 2795 | 69 | 80 | 56 |
| 0.5 (high-recall watch level) | 0.201 | 0.662 | 0.308 | 2506 | 358 | 46 | 90 |

A production threshold should be chosen from the cost of a missed failure versus an unnecessary inspection.

**Risk bands vs the dataset label** (out-of-fold, training split): LOW 1.8 % high-risk (10,249 vehicles),
MODERATE 8.8 % (1,321), HIGH 56.5 % (430) — against 4.55 % overall.

**Maintenance rules vs the dataset label** (training split):

| Rule | Vehicles | High-risk rate | Lift | Strength |
|---|---:|---:|---:|---|
| brake pad ≤ 3.0 mm | 489 | 51.1 % | 11.2× | strong |
| age ≥ 10 yr or mileage ≥ 150,000 km | 1,005 | 30.3 % | 6.7× | strong |
| oil quality < 60 % | 877 | 30.1 % | 6.6× | strong |
| engine temperature ≥ 110 °C | 590 | 11.4 % | 2.5× | moderate |
| battery voltage < 12.0 V | 2,722 | 6.5 % | 1.4× | weak |
| coolant level < 60 % | 502 | 5.4 % | 1.2× | weak |

## Feature findings (use in the report)

- Strongest single features: mileage (AUC 0.81), vehicle age (0.81), brake pad (0.80, lower = riskier),
  oil quality (0.79, lower = riskier). Engine temperature and battery voltage matter only at the extremes;
  **RPM, vibration and coolant level show almost no association** with the label (AUC 0.51–0.52).
  Removing those three left cross-validated average precision unchanged (0.437 vs 0.436); they are kept as
  inputs for the diagnostic display, not because they improve the model.
- Mileage, vehicle age, brake wear and oil quality are heavily intercorrelated (|r| up to 0.97), so SHAP and
  importance rankings split credit among them; do not claim one of them "matters more" than another.
- Mean |SHAP| on the test set: mileage 0.50, brake pad 0.27, vehicle age 0.21, engine temperature 0.19; all
  other features ≤ 0.06.

## Report figures

`reports/figures/` (see `INDEX.md` for each figure's title and caption): class balance, feature distributions,
failure rate by feature, correlation heatmap, model comparison, threshold trade-off, confusion matrices,
ROC / precision–recall curves, SHAP importance and beeswarm, risk-band validation, rule evidence.
