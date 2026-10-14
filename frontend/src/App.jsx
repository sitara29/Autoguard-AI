import { useEffect, useRef, useState } from "react";
import { getMetadata, getSamples, predict } from "./api.js";
import { Icon } from "./components/widgets.jsx";
import ModelInfo from "./components/ModelInfo.jsx";
import Home from "./pages/Home.jsx";
import Overview from "./pages/Overview.jsx";
import Diagnostics from "./pages/Diagnostics.jsx";
import Explain from "./pages/Explain.jsx";
import Service from "./pages/Service.jsx";
import { BRAND } from "./lib.js";

const NAV = [
  { id: "home", label: "Home", icon: "home", title: "Vehicle dashboard" },
  { id: "overview", label: "Vehicle overview", icon: "sliders", title: "Vehicle overview" },
  { id: "diagnostics", label: "Diagnostics", icon: "activity", title: "Component diagnostics" },
  { id: "explain", label: "Explainability", icon: "cpu", title: "Explainable AI" },
  { id: "service", label: "Service insights", icon: "tool", title: "Service recommendation" },
  { id: "model", label: "Model & evaluation", icon: "chart", title: "Model & evaluation" },
];

export default function App() {
  const [page, setPage] = useState("home");
  const [metadata, setMetadata] = useState(null);
  const [samples, setSamples] = useState([]);
  const [inputs, setInputs] = useState(null);
  const [activeSample, setActiveSample] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState("engine");

  useEffect(() => {
    (async () => {
      try {
        const [meta, samp] = await Promise.all([getMetadata(), getSamples()]);
        setMetadata(meta);
        setSamples(samp.samples);
        const first = samp.samples.find((s) => s.id === "high_risk") || samp.samples[0];
        setInputs(first.inputs);
        setActiveSample(first.id);
      } catch (e) {
        setError(e.message);
      }
    })();
  }, []);

  // live prediction: re-run shortly after the inputs stop changing
  const seq = useRef(0);
  useEffect(() => {
    if (!inputs) return;
    const controller = new AbortController();
    const mine = ++seq.current;
    setLoading(true);
    const timer = setTimeout(async () => {
      try {
        const body = await predict(inputs, controller.signal);
        if (mine === seq.current) {
          setResult(body);
          setError("");
        }
      } catch (e) {
        if (e.name !== "AbortError" && mine === seq.current) setError(e.message);
      } finally {
        if (mine === seq.current) setLoading(false);
      }
    }, 300);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [inputs]);

  const loadSample = (sample) => {
    setInputs(sample.inputs);
    setActiveSample(sample.id);
  };
  const editInput = (name, value) => {
    setInputs((prev) => ({ ...prev, [name]: value }));
    setActiveSample(null);
  };
  const select = (id) => setSelected(id);
  const current = NAV.find((n) => n.id === page);

  return (
    <div className="shell">
      <nav className="side" aria-label="Main">
        <div className="brand">
          <div className="brand-mark">V</div>
          <div><b>{BRAND}</b><span>Predictive maintenance</span></div>
        </div>
        {NAV.map((n) => (
          <button key={n.id} className="nav" aria-current={page === n.id ? "page" : undefined} onClick={() => setPage(n.id)}>
            <Icon name={n.icon} />
            <span className="nav-label">{n.label}</span>
          </button>
        ))}
        <div className="side-foot">Research prototype · synthetic training data</div>
      </nav>

      <div className="main">
        <header className="topbar">
          <div>
            <h1>{current.title}</h1>
            <div className="sub">AI failure-risk prediction · SHAP explanation · maintenance decision support</div>
          </div>
          <span className="spacer" />
          {loading && <span className="pill">Analysing…</span>}
          <span className="pill">Synthetic dataset: not real-fleet performance</span>
        </header>
        {error && <div className="banner err" role="alert">{error}</div>}

        <div className="page">
          {page === "home" && (
            <Home result={result} samples={samples} activeSample={activeSample} onSample={loadSample}
                  inputs={inputs} selected={selected} onSelect={select} goto={setPage} />
          )}
          {page === "overview" && (
            <Overview metadata={metadata} samples={samples} inputs={inputs} activeSample={activeSample} result={result}
                      loading={loading} onSample={loadSample} onChange={editInput} />
          )}
          {page === "diagnostics" && (
            <Diagnostics result={result} metadata={metadata} selected={selected} onSelect={select} />
          )}
          {page === "explain" && <Explain result={result} metadata={metadata} inputs={inputs} />}
          {page === "service" && <Service result={result} inputs={inputs} />}
          {page === "model" && <ModelInfo />}
        </div>
      </div>
    </div>
  );
}
