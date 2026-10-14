import { useState } from "react";
import Car3D from "../components/Car3D.jsx";
import ComponentMap from "../components/ComponentMap.jsx";
import { Callouts } from "../components/ResultParts.jsx";
import { COMP_ICON, HealthRing, Icon, StatusBadge, StatusDot } from "../components/widgets.jsx";
import { BAND_COLOR, pct, signed } from "../lib.js";

export default function Home({ result, samples, activeSample, onSample, inputs, selected, onSelect, goto }) {
  const [mode, setMode] = useState("exterior");
  const [spin, setSpin] = useState(false);
  const sample = samples.find((s) => s.id === activeSample);
  const r = result;
  const color = r ? BAND_COLOR[r.risk_level] : "#5b6e8f";

  return (
    <div className="cols-home">
      <div className="stack">
        <div className="card health">
          <h3>Vehicle health score</h3>
          <HealthRing value={r ? r.vehicle_health_index : 0} color={color} sub="/100" label="Vehicle health index" />
          <div className="level" style={{ color }}>{r ? `${r.risk_level[0]}${r.risk_level.slice(1).toLowerCase()} risk` : "…"}</div>
          {r && (
            <>
              <div className="disclaimer">
                Risk index {Math.round(r.risk_index)}/100 · health = 100 − risk (display value)
              </div>
              <span className={`badge ${r.maintenance_priority}`} style={{ marginTop: 6 }}>
                Maintenance priority: {r.maintenance_priority}
              </span>
              <p className="disclaimer" style={{ marginTop: 8 }}>
                In this band the dataset's high-risk label appeared {pct(r.band_context.dataset_failure_rate)} of the time
                (overall 4.5%).
              </p>
            </>
          )}
        </div>

        <div className="card">
          <h3 style={{ marginBottom: 10 }}>Vehicle details</h3>
          <dl className="mini-kv">
            <dt>Mileage</dt><dd>{inputs ? Math.round(inputs.mileage_km).toLocaleString() : "–"} km</dd>
            <dt>Age</dt><dd>{inputs ? inputs.vehicle_age_years.toFixed(1) : "–"} years</dd>
            <dt>Engine temp.</dt><dd>{inputs ? inputs.engine_temp_c.toFixed(1) : "–"} °C</dd>
            <dt>Dataset label</dt>
            <dd>{sample ? (sample.dataset_label === 1 ? "high-risk" : "not high-risk") : "custom input"}</dd>
          </dl>
          <h3 style={{ margin: "14px 0 8px" }}>Demo vehicles</h3>
          <div className="demos">
            {samples.map((s) => (
              <button key={s.id} className="demo" aria-pressed={activeSample === s.id} title={s.description} onClick={() => onSample(s)}>
                {s.name}
              </button>
            ))}
          </div>
          <button className="btn ghost" style={{ width: "100%", marginTop: 12 }} onClick={() => goto("overview")}>
            Edit vehicle parameters
          </button>
        </div>
      </div>

      <div className="scene">
        <div className="scene-top">
          <div className="seg" role="group" aria-label="View mode">
            {[["exterior", "Exterior"], ["xray", "X-ray"], ["map", "Component map"]].map(([id, label]) => (
              <button key={id} aria-pressed={mode === id} onClick={() => setMode(id)}>{label}</button>
            ))}
          </div>
        </div>
        {mode === "map" ? (
          <div style={{ height: 640, display: "grid", alignItems: "center", padding: "56px 12px 12px" }}>
            <ComponentMap components={result?.components} selected={selected} onSelect={onSelect} />
          </div>
        ) : (
          <>
            <Car3D components={result?.components} selected={selected} onSelect={onSelect} mode={mode}
                   labels="all" height={640} autoRotate={spin} />
            <div className="scene-tools">
              <button className="tool" aria-pressed={spin} onClick={() => setSpin((v) => !v)}>Rotate</button>
              <button className="tool" onClick={() => goto("diagnostics")}>Inspect component</button>
            </div>
          </>
        )}
      </div>

      <div className="stack">
        <div className="card">
          <div className="card-head"><h2>Component status</h2></div>
          {result?.components.map((c) => (
            <button key={c.id} className="comp-row" aria-pressed={selected === c.id} onClick={() => onSelect(c.id)}>
              <span style={{ color: "var(--muted)" }}><Icon name={COMP_ICON[c.id]} size={17} /></span>
              <span>{c.label}</span>
              <span className="contrib" title="SHAP contribution of this subsystem's features (model log-odds)">
                {signed(c.ai_contribution.value)}
              </span>
              <StatusBadge status={c.status} />
            </button>
          )) || <p className="placeholder">Loading…</p>}
          <p className="axis-note">
            Status comes from SHAP contribution plus transparent rules. It is not a per-component failure probability.
          </p>
        </div>
        {result && <div className="card"><Callouts result={result} />
          <p className="disclaimer" style={{ marginTop: 10 }}>{result.recommendations[0]?.message}</p>
          <button className="btn" style={{ width: "100%", marginTop: 12 }} onClick={() => goto("service")}>
            View service recommendation
          </button>
        </div>}
      </div>
    </div>
  );
}
