# Frontend: Vehicle Risk Console

React + Three.js (`@react-three/fiber`) interface for the FastAPI backend, styled after the project's reference dashboard.

```bash
# terminal 1 (project root)
uvicorn backend.app:app --host 127.0.0.1 --port 8000

# terminal 2
cd frontend
npm install
npm run dev          # http://127.0.0.1:5173
```

The backend address defaults to `http://127.0.0.1:8000`; override it with `VITE_API_URL`.
The backend already allows the `localhost:5173` / `127.0.0.1:5173` origins.

## Pages (left sidebar)

| Page | What it shows | Backend data |
|---|---|---|
| Home | Health-score ring, 3D vehicle (Exterior / X-ray / Component map) with live status labels, component status table, model-vs-rules notice | `/predict`, `/samples` |
| Vehicle overview | Nine sliders from the training ranges, demo vehicles, and where each reading sits among the training vehicles | `/metadata`, `/predict` |
| Diagnostics | Per-component X-ray view with camera fly-to, key indicators, SHAP insights, sensor positions, recommended action | `components`, `explainability`, `recommendations` |
| Explainability | SHAP contributions, what-if sensitivity curve (21 live predictions), model insights | `explainability`, `/predict`, `/model-info` |
| Service insights | Urgency wording, subsystems to check, advice with dataset evidence, demo service-request summary, read-aloud assistant | `recommendations`, `maintenance_priority` |
| Model & evaluation | Comparison, tuning, test results, risk bands, rule evidence, limitations | `/model-info` |

Predictions refresh 300 ms after the last change to an input.

## Deliberate differences from the reference mockup

The mockup shows things the model and dataset cannot support. They were replaced, not faked:

| Reference shows | Built instead | Why |
|---|---|---|
| Per-component failure % (89%, 61%…) | Status (Normal / Watch / Critical) plus the component's SHAP contribution | The model predicts one overall label, not one probability per part |
| Sensor trend charts over May 1–29, repair history | "Sensor positions": where the reading sits among training vehicles | The dataset has one reading per vehicle and no time series |
| "Failure in next 500–700 km", 3-month predictive timeline | What-if sensitivity: risk index vs one input, others held fixed | The dataset has no time-to-failure target |
| ₹ cost estimate, "within 7 days" | Urgency wording of the maintenance priority | No cost or scheduling data exists |
| Transmission, interior / infotainment view | Not included | No such sensors in the dataset |
| Voice assistant with a person and microphone | Read-aloud of a summary built from the model output (browser text-to-speech) | No conversational model is part of the system |
| Book Service Appointment | "Prepare service request (demo)": copyable summary, nothing is sent | There is no booking backend |

## Design notes

- The raw model score is never shown as a probability: the UI shows the risk index, the band, and how often the dataset's
  label was high-risk in that band.
- The car is built from Three.js primitives (an extruded sedan profile plus status-coloured internals), so it works offline.
  To use a `.glb` model, replace `CarBody` and `Internals` in `src/components/Car3D.jsx` and adjust the `HOT` coordinates.
- The page states that the data is synthetic and the advice is decision support.

## Files

```text
src/App.jsx                      state, live prediction, sidebar and routing
src/api.js                       backend client
src/lib.js                       colours, labels, urgency wording, read-aloud text
src/pages/                       Home, Overview, Diagnostics, Explain, Service
src/components/Car3D.jsx         3D scene: sedan, X-ray internals, hotspots, camera
src/components/ComponentMap.jsx  top-down component map (SVG)
src/components/InputPanel.jsx    sliders and demo vehicles
src/components/ResultParts.jsx   SHAP bars, recommendations, callouts
src/components/widgets.jsx       icons, health ring, range strip
src/components/ModelInfo.jsx     model & evaluation page
```
