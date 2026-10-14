// Thin client for the FastAPI backend. Override the address with VITE_API_URL.
const BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BASE}${path}`, options);
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new Error(
      `Cannot reach the backend at ${BASE}. Start it with: uvicorn backend.app:app --port 8000`
    );
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (Array.isArray(body.detail)) {
        detail = body.detail.map((d) => `${d.loc?.slice(-1)[0]}: ${d.msg}`).join("; ");
      } else if (body.detail) {
        detail = String(body.detail);
      }
    } catch {
      /* keep the status text */
    }
    throw new Error(detail);
  }
  return response.json();
}

export const getMetadata = () => request("/metadata");
export const getSamples = () => request("/samples");
export const getModelInfo = () => request("/model-info");
export const predict = (inputs, signal) =>
  request("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(inputs),
    signal,
  });
