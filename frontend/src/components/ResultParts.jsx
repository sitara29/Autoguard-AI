import { pct } from "../lib.js";

export function ShapBars({ contributions, note = true }) {
  const max = Math.max(...contributions.map((c) => Math.abs(c.contribution)), 0.01);
  return (
    <div>
      {contributions.map((c) => {
        const width = (Math.abs(c.contribution) / max) * 50;
        const up = c.contribution >= 0;
        return (
          <div className="shap-row" key={c.feature} title={`${c.label} = ${c.input_value} ${c.unit}`}>
            <span className="lab">{c.label}</span>
            <span className="shap-track">
              <span className="zero" />
              <span className={`shap-bar ${up ? "up" : "down"}`} style={{ width: `${width}%` }} />
            </span>
            <span className="v">{up ? "+" : "−"}{Math.abs(c.contribution).toFixed(2)}</span>
          </div>
        );
      })}
      {note && (
        <div className="axis-note">
          Red pushes toward higher risk, blue toward lower. Units are model log-odds, not percentage points.
          Mileage, age, brake wear and oil quality are strongly correlated, so they share credit.
        </div>
      )}
    </div>
  );
}

export function Evidence({ evidence }) {
  if (!evidence || evidence.strength === "none" || evidence.lift == null) return null;
  return (
    <div className="evidence">
      In this dataset: high-risk label in {pct(evidence.failure_rate_when_triggered)} of vehicles when this fires
      (overall {pct(evidence.baseline_failure_rate)}, {evidence.lift.toFixed(1)}× lift, <b>{evidence.strength}</b> association).
    </div>
  );
}

export function RecList({ recommendations }) {
  return recommendations.map((rec) => (
    <div className="rec" key={rec.id}>
      <div className="rec-top">
        <span className={`badge ${rec.severity === "high" ? "HIGH" : rec.severity === "medium" ? "MEDIUM" : "neutral"}`}>
          {rec.severity}
        </span>
        <span className="src">
          {rec.source === "model_output" ? "from model" : rec.source === "rule_based_sensor_advisory" ? "rule-based" : rec.source}
        </span>
      </div>
      <div>{rec.message}</div>
      {rec.condition && <div className="evidence">Rule: {rec.condition}</div>}
      <Evidence evidence={rec.evidence} />
    </div>
  ));
}

export function Callouts({ result }) {
  const r = result;
  const ruleOverride = r.predicted_failure_risk === 0 && r.maintenance_priority === "HIGH";
  const fired = r.recommendations.filter((x) => x.source === "rule_based_sensor_advisory" && x.severity === "high");
  return (
    <>
      {ruleOverride && (
        <div className="callout">
          <b>Model and rules disagree.</b> The model alone rates this vehicle {r.risk_level}, but a high-severity
          sensor rule fired ({[...new Set(fired.map((x) => x.component))].join(", ")}), so the maintenance priority is
          raised to HIGH.
        </div>
      )}
      {r.input_warnings.map((w) => (
        <div className="callout warn" key={w.feature}>
          <b>Out-of-range input.</b> {w.message}
        </div>
      ))}
    </>
  );
}
