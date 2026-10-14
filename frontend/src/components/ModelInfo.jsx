import { Fragment, useEffect, useState } from "react";
import { getModelInfo } from "../api.js";

const f3 = (x) => x.toFixed(3);
const pct = (x) => `${(x * 100).toFixed(1)}%`;
const MODELS = ["Logistic Regression", "Random Forest", "XGBoost"];

function ConfusionMatrix({ m }) {
  const [[tn, fp], [fn, tp]] = m.confusion_matrix;
  return (
    <div className="cm">
      <div className="h" />
      <div className="h">pred. no</div>
      <div className="h">pred. yes</div>
      <div className="h">actual no</div>
      <div>{tn}</div>
      <div>{fp}</div>
      <div className="h">actual yes</div>
      <div>{fn}</div>
      <div>{tp}</div>
    </div>
  );
}

function ComparisonTable({ info }) {
  const rows = MODELS.flatMap((name) => [
    { name, stage: "baseline", m: info.baseline_5_fold_cv[name], tried: "–" },
    { name, stage: "tuned", m: info.tuned_5_fold_cv[name].cv_metrics, tried: info.tuned_5_fold_cv[name].configurations_tried },
  ]);
  const bestAp = Math.max(...rows.map((r) => r.m.average_precision));
  return (
    <div className="scroll-x">
      <table>
        <thead>
          <tr>
            <th>Model</th><th>Stage</th><th>Configs</th><th>ROC-AUC</th><th>Avg. precision</th>
            <th>Precision*</th><th>Recall*</th><th>F1*</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name + r.stage}>
              <td>{r.name}</td><td>{r.stage}</td><td>{r.tried}</td>
              <td>{f3(r.m.roc_auc)}</td>
              <td className={r.m.average_precision === bestAp ? "best" : ""}>{f3(r.m.average_precision)}</td>
              <td>{f3(r.m.precision)}</td><td>{f3(r.m.recall)}</td><td>{f3(r.m.f1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ModelInfo() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getModelInfo().then(setInfo).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="info-page"><div className="banner err">{error}</div></div>;
  if (!info) return <div className="info-page"><p className="placeholder">Loading…</p></div>;

  const ds = info.dataset;
  const sel = info.model_selection;
  const std = info.tuned_5_fold_cv.XGBoost.cv_average_precision_std;
  const thr = info.threshold;
  const tests = Object.values(info.test_metrics).sort(
    (a, b) => Math.abs(a.threshold - info.threshold.threshold) - Math.abs(b.threshold - info.threshold.threshold)
  );
  const selectedT = tests.find((t) => Math.abs(t.threshold - thr.threshold) < 1e-9) || tests[0];

  return (
    <main className="info-page">
      <div className="grid2">
        <div className="card">
          <h2 style={{ marginBottom: 10 }}>Dataset &amp; protocol</h2>
          <dl className="kv">
            <dt>Rows</dt><dd>{ds.n_rows.toLocaleString()} (synthetic), {ds.n_features} features</dd>
            <dt>High-risk label</dt><dd>{ds.positive_count} rows ({pct(ds.positive_rate)})</dd>
            <dt>Missing / duplicates</dt><dd>{ds.missing_total} / {ds.duplicate_rows}</dd>
            <dt>Split</dt><dd>{info.split.train_rows.toLocaleString()} train / {info.split.test_rows.toLocaleString()} test, stratified</dd>
            <dt>Cross-validation</dt><dd>{info.cross_validation.n_splits}-fold stratified, seed {info.cross_validation.random_state}</dd>
            <dt>Imbalance handling</dt><dd>class weights, stratification, threshold tuning (no SMOTE)</dd>
          </dl>
        </div>
        <div className="card">
          <h2 style={{ marginBottom: 10 }}>Selected model</h2>
          <p><b>{sel.selected_model}</b> · CV average precision {f3(sel.five_fold_cv_average_precision)}</p>
          <p className="disclaimer" style={{ marginTop: 6 }}>{sel.reason}</p>
          <dl className="kv" style={{ marginTop: 10 }}>
            {Object.entries(sel.best_params).map(([k, v]) => (
              <Fragment key={k}><dt>{k.replace("model__", "")}</dt><dd>{String(v)}</dd></Fragment>
            ))}
          </dl>
        </div>
      </div>

      <div className="card">
        <h2 style={{ marginBottom: 4 }}>Model comparison (5-fold CV on the training split)</h2>
        <p className="disclaimer" style={{ marginBottom: 8 }}>
          Same folds and the same tuning metric (average precision) for all three models. The tuned models differ by
          less than the fold-to-fold standard deviation (≈ {std.toFixed(3)}), so they are statistically close;
          XGBoost is selected because it scored highest and works with SHAP TreeExplainer.
        </p>
        <ComparisonTable info={info} />
        <p className="axis-note">
          *Precision / recall / F1 use a fixed 0.5 cut-off and depend on each model's class weighting; compare
          models on ROC-AUC and average precision.
        </p>
      </div>

      <div className="grid2">
        <div className="card">
          <h2 style={{ marginBottom: 4 }}>Held-out test results (3,000 vehicles, used once)</h2>
          <p className="disclaimer" style={{ marginBottom: 8 }}>
            ROC-AUC {f3(selectedT.roc_auc)} · average precision {f3(selectedT.average_precision)} (no-skill 0.045)
          </p>
          {tests.map((t) => (
            <div key={t.threshold} style={{ marginBottom: 14 }}>
              <h3 style={{ marginBottom: 6 }}>
                Threshold {t.threshold} {Math.abs(t.threshold - thr.threshold) < 1e-9 ? "(selected, max out-of-fold F1)" : "(high-recall watch level)"}
              </h3>
              <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
                <ConfusionMatrix m={t} />
                <dl className="kv">
                  <dt>Precision</dt><dd>{f3(t.precision)}</dd>
                  <dt>Recall</dt><dd>{f3(t.recall)}</dd>
                  <dt>F1</dt><dd>{f3(t.f1)}</dd>
                  <dt>Accuracy</dt><dd>{f3(t.accuracy)}</dd>
                </dl>
              </div>
            </div>
          ))}
        </div>
        <div className="card">
          <h2 style={{ marginBottom: 10 }}>Risk bands vs the dataset label</h2>
          <table>
            <thead><tr><th>Band</th><th>Out-of-fold train</th><th>Test</th></tr></thead>
            <tbody>
              {["LOW", "MODERATE", "HIGH"].map((b) => (
                <tr key={b}>
                  <td><span className={`badge ${b}`}>{b}</span></td>
                  <td>{pct(info.risk_bands.out_of_fold_train[b].dataset_failure_rate)} of {info.risk_bands.out_of_fold_train[b].n_vehicles.toLocaleString()}</td>
                  <td>{pct(info.risk_bands.test[b].dataset_failure_rate)} of {info.risk_bands.test[b].n_vehicles.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="axis-note">Share of vehicles in each band that carry the dataset's high-risk label (overall 4.5%).</p>
        </div>
      </div>

      <div className="card">
        <h2 style={{ marginBottom: 10 }}>Maintenance rules vs the dataset label</h2>
        <div className="scroll-x">
          <table>
            <thead><tr><th>Rule</th><th>Vehicles</th><th>High-risk rate when it fires</th><th>Lift</th><th>Strength</th></tr></thead>
            <tbody>
              {Object.entries(info.maintenance_rule_evidence.rules).map(([id, e]) => (
                <tr key={id}>
                  <td>{id}</td><td>{e.n_triggered.toLocaleString()}</td>
                  <td>{pct(e.failure_rate_when_triggered)}</td><td>{e.lift.toFixed(1)}×</td><td>{e.strength}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h2>Limitations</h2>
        <ul className="plain">
          {info.caveats.map((c) => <li key={c}>{c}</li>)}
        </ul>
      </div>
    </main>
  );
}
