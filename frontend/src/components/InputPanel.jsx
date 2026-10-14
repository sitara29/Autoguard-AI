// Sliders are built from /metadata so ranges and units always match the training data.
function stepFor(feature) {
  const span = feature.max - feature.min;
  const raw = span / 400;
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / pow;
  return (n < 2 ? 1 : n < 5 ? 2 : 5) * pow;
}

function decimals(step) {
  return step >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(step)));
}

export default function InputPanel({ metadata, samples, inputs, activeSample, warnings, loading, onSample, onChange }) {
  if (!metadata || !inputs) {
    return (
      <div className="card">
        <h2>Vehicle parameters</h2>
        <p className="placeholder">Loading from the backend…</p>
      </div>
    );
  }
  const warned = new Set(warnings.map((w) => w.feature));

  return (
    <div className="card">
      <div className="card-head">
        <h2>Vehicle parameters</h2>
      </div>

      <h3 style={{ marginBottom: 6 }}>Demo vehicles</h3>
      <div className="demos">
        {samples.map((s) => (
          <button
            key={s.id}
            className="demo"
            aria-pressed={activeSample === s.id}
            title={s.description}
            onClick={() => onSample(s)}
          >
            {s.name}
          </button>
        ))}
      </div>

      {metadata.features.map((f) => {
        const step = stepFor(f);
        const dp = decimals(step);
        const value = inputs[f.name];
        const fmt = (n) => Number(n).toLocaleString(undefined, { maximumFractionDigits: dp });
        return (
          <div className="field" key={f.name}>
            <div className="field-top">
              <label htmlFor={f.name}>
                {f.label} <span className="unit">({f.unit})</span>
              </label>
              <input
                className={`num ${warned.has(f.name) ? "warn" : ""}`}
                type="number"
                aria-label={`${f.label} value`}
                step={step}
                value={Number.isFinite(value) ? Number(value.toFixed(dp + 1)) : ""}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (Number.isFinite(v)) onChange(f.name, v);
                }}
              />
            </div>
            <input
              id={f.name}
              type="range"
              min={f.min}
              max={f.max}
              step={step}
              value={Math.min(Math.max(value, f.min), f.max)}
              onChange={(e) => onChange(f.name, parseFloat(e.target.value))}
            />
            <div className="hint">
              Training range {fmt(f.min)}–{fmt(f.max)} · typical {fmt(f.p5)}–{fmt(f.p95)}
            </div>
          </div>
        );
      })}
      <div className="status-line" aria-live="polite">
        {loading ? "Analysing…" : "Updates automatically as you change a value."}
      </div>
    </div>
  );
}
