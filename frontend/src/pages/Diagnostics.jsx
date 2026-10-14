import { useState } from "react";
import Car3D from "../components/Car3D.jsx";
import { COMP_ICON, Icon, RangeStrip, StatusBadge, StatusDot } from "../components/widgets.jsx";
import { Evidence } from "../components/ResultParts.jsx";
import { COMP_ORDER, STATUS_COLOR, STATUS_LABEL, signed } from "../lib.js";

const fmt = (v) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : Number(v.toFixed(3)).toString());

export default function Diagnostics({ result, metadata, selected, onSelect }) {
  const [tab, setTab] = useState("key");
  const comp = result?.components.find((c) => c.id === selected);
  const color = STATUS_COLOR[comp?.status] || STATUS_COLOR.idle;
  const feats = comp ? [...comp.features].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)) : [];
  const maxAbs = Math.max(...feats.map((f) => Math.abs(f.contribution)), 0.01);
  const advisories = result?.recommendations.filter((r) => comp?.advisories.includes(r.id)) || [];
  const meta = (name) => metadata?.features.find((f) => f.name === name);

  return (
    <div className="cols-diag">
      <div className="rail" role="group" aria-label="Components">
        {COMP_ORDER.map((id) => {
          const c = result?.components.find((x) => x.id === id);
          return (
            <button key={id} className="rail-btn" aria-pressed={selected === id} title={c?.label} aria-label={c?.label || id}
                    onClick={() => onSelect(id)}>
              <Icon name={COMP_ICON[id]} size={22} />
              <StatusDot status={c?.status || "idle"} />
            </button>
          );
        })}
      </div>

      <div className="stack">
        <div className="scene">
          <div className="scene-top"><span className="pill">{comp?.label || "Component"} · X-ray view</span></div>
          <Car3D components={result?.components} selected={selected} onSelect={onSelect} mode="xray" focus={selected}
                 labels="selected" height={400} />
        </div>
        <div className="card">
          <div className="card-head"><h2>Sensor positions</h2></div>
          {feats.map((f) => (
            <div className="sensor" key={f.feature}>
              <div className="sensor-top">
                <span>{f.label}</span>
                <b style={{ fontVariantNumeric: "tabular-nums" }}>{fmt(f.input_value)} {f.unit}</b>
              </div>
              {meta(f.feature) && <RangeStrip feature={meta(f.feature)} value={f.input_value} />}
              <div className="disclaimer" style={{ marginTop: 4 }}>
                Higher than {Math.round(f.input_percentile)}% of training vehicles
              </div>
            </div>
          ))}
          <p className="strip-legend">One reading per vehicle; the dataset has no time series, so there is no trend line.</p>
        </div>
      </div>

      <div className="stack">
        {comp ? (
          <>
            <div className="card">
              <div className="card-head">
                <h2>{comp.label}</h2>
                <StatusBadge status={comp.status} />
              </div>
              <div className="cond">
                <div className="cond-ring" style={{ borderColor: color, color, boxShadow: `0 0 18px ${color}44` }}>
                  {STATUS_LABEL[comp.status]}
                </div>
                <div>
                  <div className="disclaimer">AI contribution (SHAP, log-odds)</div>
                  <div style={{ fontSize: 24, fontWeight: 700 }}>{signed(comp.ai_contribution.value)}</div>
                  <div className="disclaimer">Impact: {comp.ai_contribution.level.toLowerCase()}</div>
                </div>
              </div>
              <div className="tabs" role="tablist">
                <button className="tab" role="tab" aria-selected={tab === "key"} onClick={() => setTab("key")}>Key indicators</button>
                <button className="tab" role="tab" aria-selected={tab === "shap"} onClick={() => setTab("shap")}>SHAP insights</button>
              </div>
              {tab === "key" ? (
                feats.map((f) => (
                  <div className="ind" key={f.feature}>
                    <span>{f.label}</span>
                    <span className="bar">
                      <i style={{
                        width: `${(Math.abs(f.contribution) / maxAbs) * 100}%`,
                        background: f.contribution >= 0 ? "linear-gradient(90deg,#ff7a45,#ff4d4f)" : "linear-gradient(90deg,#3b9bff,#22d3ee)",
                      }} />
                    </span>
                    <span style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{signed(f.contribution)}</span>
                  </div>
                ))
              ) : (
                <div style={{ fontSize: 13 }}>
                  {feats.map((f) => (
                    <p key={f.feature} style={{ marginBottom: 8 }}>
                      <b>{f.label}</b> ({fmt(f.input_value)} {f.unit}, higher than {Math.round(f.input_percentile)}% of vehicles){" "}
                      {f.contribution >= 0 ? "raises" : "lowers"} the model's risk score by {Math.abs(f.contribution).toFixed(2)} log-odds
                      {f.level !== "NONE" ? ` (${f.level.toLowerCase()} impact)` : ""}.
                    </p>
                  ))}
                  <p className="disclaimer">
                    Correlated wear features (mileage, age, brake pads, oil quality) share credit, so a single value is not a
                    precise measure of one part's condition.
                  </p>
                </div>
              )}
            </div>
            <div className="card">
              <div className="card-head"><h2>Recommended action</h2></div>
              {advisories.length ? (
                advisories.map((a) => (
                  <div className="rec" key={a.id}>
                    <div className="rec-top">
                      <span className={`badge ${a.severity === "high" ? "HIGH" : a.severity === "medium" ? "MEDIUM" : "neutral"}`}>{a.severity}</span>
                      <span className="src">rule: {a.condition}</span>
                    </div>
                    <div>{a.message}</div>
                    <Evidence evidence={a.evidence} />
                  </div>
                ))
              ) : comp.status === "ok" ? (
                <p className="disclaimer">No rule fired and the model does not single this subsystem out. Continue scheduled maintenance.</p>
              ) : (
                <p className="disclaimer">
                  No sensor rule fired, but the model gives this subsystem's readings a notable weight in the risk score. Include it in
                  the next routine inspection.
                </p>
              )}
            </div>
          </>
        ) : <div className="card"><p className="placeholder">Loading…</p></div>}
      </div>
    </div>
  );
}
