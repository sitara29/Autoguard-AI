import { useEffect, useState } from "react";
import { getModelInfo, predict } from "../api.js";
import { ShapBars, Callouts } from "../components/ResultParts.jsx";
import { BAND_COLOR, signed, topDrivers } from "../lib.js";

function Sensitivity({ metadata, inputs, result }) {
  const [name, setName] = useState("brake_pad_mm");
  const [pts, setPts] = useState(null);
  const [err, setErr] = useState("");
  const f = metadata.features.find((x) => x.name === name);

  useEffect(() => {
    const ctrl = new AbortController();
    setPts(null);
    setErr("");
    const xs = Array.from({ length: 21 }, (_, i) => f.min + ((f.max - f.min) * i) / 20);
    Promise.all(xs.map((x) => predict({ ...inputs, [name]: x }, ctrl.signal).then((r) => ({ x, y: r.risk_index }))))
      .then(setPts)
      .catch((e) => e.name !== "AbortError" && setErr(e.message));
    return () => ctrl.abort();
  }, [name, inputs, f]);

  const W = 600, H = 250, L = 42, R = 12, T = 12, B = 32;
  const sx = (v) => L + ((v - f.min) / (f.max - f.min)) * (W - L - R);
  const sy = (v) => T + (1 - v / 100) * (H - T - B);
  const cur = Math.min(Math.max(inputs[name], f.min), f.max);

  return (
    <div>
      <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 10, flexWrap: "wrap" }}>
        <label htmlFor="wi">Vary:</label>
        <select id="wi" className="select" value={name} onChange={(e) => setName(e.target.value)}>
          {metadata.features.map((x) => <option key={x.name} value={x.name}>{x.label} ({x.unit})</option>)}
        </select>
        <span className="disclaimer">all other inputs stay at the current values</span>
      </div>
      {err && <div className="callout warn">{err}</div>}
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`Risk index as ${f.label} varies`}>
        <rect x={L} y={sy(30)} width={W - L - R} height={sy(0) - sy(30)} fill={BAND_COLOR.LOW} opacity=".10" />
        <rect x={L} y={sy(50)} width={W - L - R} height={sy(30) - sy(50)} fill={BAND_COLOR.MODERATE} opacity=".12" />
        <rect x={L} y={sy(100)} width={W - L - R} height={sy(50) - sy(100)} fill={BAND_COLOR.HIGH} opacity=".10" />
        {[0, 30, 50, 100].map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={sy(v)} y2={sy(v)} stroke="#1b3560" />
            <text x={L - 6} y={sy(v) + 4} textAnchor="end" fill="#8ea3c4" fontSize="11">{v}</text>
          </g>
        ))}
        {[f.min, (f.min + f.max) / 2, f.max].map((v, i) => (
          <text key={i} x={sx(v)} y={H - 10} textAnchor={i === 0 ? "start" : i === 2 ? "end" : "middle"} fill="#8ea3c4" fontSize="11">
            {Number(v.toPrecision(3)).toLocaleString()}
          </text>
        ))}
        {pts && (
          <polyline fill="none" stroke="#3b9bff" strokeWidth="2.5" strokeLinejoin="round"
                    points={pts.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")} />
        )}
        <line x1={sx(cur)} x2={sx(cur)} y1={T} y2={H - B} stroke="#fff" strokeDasharray="4 3" opacity=".7" />
        <circle cx={sx(cur)} cy={sy(result.risk_index)} r="5.5" fill="#fff" />
        <text x={L + 4} y={T + 12} fill="#8ea3c4" fontSize="11">risk index</text>
      </svg>
      {!pts && !err && <p className="disclaimer">Computing…</p>}
      <p className="axis-note">
        This shows how the model reacts to one input, not a forecast over time. Mileage, age, brake wear and oil quality
        normally move together, so changing one alone can create combinations that rarely occur in the data.
      </p>
    </div>
  );
}

function ModelInsights() {
  const [info, setInfo] = useState(null);
  useEffect(() => { getModelInfo().then(setInfo).catch(() => setInfo(false)); }, []);
  if (info === false) return <p className="placeholder">Model information is unavailable.</p>;
  if (!info) return <p className="placeholder">Loading…</p>;
  const t = Object.values(info.test_metrics).find((x) => Math.abs(x.threshold - info.threshold.threshold) < 1e-9);
  return (
    <div>
      <dl className="kv">
        <dt>Model</dt><dd>{info.model_selection.selected_model} (tuned, class-weighted)</dd>
        <dt>Selection metric</dt><dd>{info.model_selection.selection_metric.replace("_", " ")}, 5-fold stratified CV</dd>
        <dt>CV average precision</dt><dd>{info.model_selection.five_fold_cv_average_precision.toFixed(3)}</dd>
        <dt>Test ROC-AUC</dt><dd>{t.roc_auc.toFixed(3)}</dd>
        <dt>Test average precision</dt><dd>{t.average_precision.toFixed(3)} (no-skill 0.045)</dd>
        <dt>Decision threshold</dt><dd>{info.threshold.threshold} (max out-of-fold F1)</dd>
        <dt>Test precision / recall</dt><dd>{t.precision.toFixed(3)} / {t.recall.toFixed(3)}</dd>
      </dl>
      <p className="axis-note">
        LR, Random Forest and XGBoost were tuned on identical folds and ended within the fold-to-fold noise of each other;
        XGBoost was chosen for the highest score and SHAP compatibility. See the Model &amp; evaluation page for the full
        comparison.
      </p>
    </div>
  );
}

export default function Explain({ result, metadata, inputs }) {
  const [tab, setTab] = useState("contrib");
  if (!result || !metadata || !inputs) return <div className="card"><p className="placeholder">Loading…</p></div>;
  const drivers = topDrivers(result, 3);
  return (
    <div className="cols-2" style={{ gridTemplateColumns: "minmax(0,1.4fr) minmax(0,1fr)" }}>
      <div className="card">
        <div className="card-head"><h2>Why this prediction?</h2></div>
        <div className="tabs" role="tablist">
          {[["contrib", "Feature contribution"], ["whatif", "What-if sensitivity"], ["model", "Model insights"]].map(([id, label]) => (
            <button key={id} className="tab" role="tab" aria-selected={tab === id} onClick={() => setTab(id)}>{label}</button>
          ))}
        </div>
        {tab === "contrib" && <ShapBars contributions={result.explainability.contributions} />}
        {tab === "whatif" && <Sensitivity metadata={metadata} inputs={inputs} result={result} />}
        {tab === "model" && <ModelInsights />}
      </div>
      <div className="stack">
        <div className="card">
          <div className="card-head"><h2>Interpretation</h2></div>
          <p>
            The model rates this vehicle <b style={{ color: BAND_COLOR[result.risk_level] }}>{result.risk_level}</b> (risk index{" "}
            {Math.round(result.risk_index)}/100).
          </p>
          <p style={{ marginTop: 8 }}>
            {drivers.length
              ? <>The largest contributions raising the score are {drivers.map((d, i) => (
                  <span key={d.feature}>{i ? ", " : ""}<b>{d.label}</b> ({signed(d.contribution)})</span>
                ))}.</>
              : "No input raises the score noticeably."}
          </p>
          <p className="disclaimer" style={{ marginTop: 8 }}>
            SHAP splits the model's output among the inputs, starting from a base value of {result.explainability.base_value.toFixed(2)}; the
            contributions add up to the final raw score of {result.explainability.raw_margin.toFixed(2)} (log-odds).
          </p>
          <Callouts result={result} />
        </div>
      </div>
    </div>
  );
}
