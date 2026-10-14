import InputPanel from "../components/InputPanel.jsx";
import { RangeStrip } from "../components/widgets.jsx";

export default function Overview({ metadata, samples, inputs, activeSample, result, loading, onSample, onChange }) {
  return (
    <div className="cols-over">
      <InputPanel
        metadata={metadata} samples={samples} inputs={inputs} activeSample={activeSample}
        warnings={result?.input_warnings || []} loading={loading} onSample={onSample} onChange={onChange}
      />
      <div className="card">
        <div className="card-head"><h2>Where this vehicle sits among the training vehicles</h2></div>
        <p className="disclaimer" style={{ marginBottom: 8 }}>
          Each bar spans the training range of a sensor. The pale band is the middle 90% of vehicles, the darker box the
          middle 50%, the thin line the median, and the white marker this vehicle.
        </p>
        {metadata && inputs ? (
          metadata.features.map((f) => {
            const v = inputs[f.name];
            const c = result?.explainability.contributions.find((x) => x.feature === f.name);
            return (
              <div className="sensor" key={f.name}>
                <div className="sensor-top">
                  <span><b>{f.label}</b> <span className="disclaimer">({f.unit})</span></span>
                  <span style={{ fontVariantNumeric: "tabular-nums" }}>
                    {Number(v).toLocaleString(undefined, { maximumFractionDigits: 3 })}
                    {c && <span className="disclaimer"> · higher than {Math.round(c.input_percentile)}% of vehicles</span>}
                  </span>
                </div>
                <RangeStrip feature={f} value={v} />
              </div>
            );
          })
        ) : <p className="placeholder">Loading…</p>}
        <p className="strip-legend">
          The dataset has one reading per vehicle (no time series), so this shows position within the population, not a trend.
        </p>
      </div>
    </div>
  );
}
