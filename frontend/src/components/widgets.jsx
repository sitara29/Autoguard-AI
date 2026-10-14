import { STATUS_COLOR } from "../lib.js";

const ICONS = {
  home: <path d="M3 11l9-8 9 8v10a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1z" />,
  car: <path d="M5 17h14M3 13l2-6h14l2 6v5h-3v-2H6v2H3zM7 13h.01M17 13h.01" />,
  activity: <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />,
  cpu: (
    <>
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <rect x="9" y="9" width="6" height="6" />
      <path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3" />
    </>
  ),
  tool: <path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.5 2.5-2.1-.6-.6-2.1z" />,
  chart: <path d="M12 20V10M18 20V4M6 20v-4" />,
  gear: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </>
  ),
  thermo: <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z" />,
  drop: <path d="M12 2.7l5.7 5.6a8 8 0 1 1-11.4 0z" />,
  disc: (
    <>
      <circle cx="12" cy="12" r="9" />
      <circle cx="12" cy="12" r="3" />
    </>
  ),
  battery: (
    <>
      <rect x="2" y="7" width="16" height="10" rx="2" />
      <path d="M22 11v2" />
    </>
  ),
  layers: (
    <>
      <polygon points="12 2 2 7 12 12 22 7 12 2" />
      <polyline points="2 17 12 22 22 17" />
      <polyline points="2 12 12 17 22 12" />
    </>
  ),
  speaker: (
    <>
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <path d="M15.5 8.5a5 5 0 0 1 0 7M19 5a10 10 0 0 1 0 14" />
    </>
  ),
  calendar: (
    <>
      <rect x="3" y="4" width="18" height="18" rx="2" />
      <path d="M16 2v4M8 2v4M3 10h18" />
    </>
  ),
  sliders: <path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6" />,
};

export const COMP_ICON = {
  engine: "gear",
  cooling: "thermo",
  lubrication: "drop",
  brakes: "disc",
  battery: "battery",
  general_wear: "layers",
};

export function Icon({ name, size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
         strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {ICONS[name]}
    </svg>
  );
}

export function HealthRing({ value, color, size = 150, label, sub }) {
  const r = 52, c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(100, value));
  return (
    <svg width={size} height={size} viewBox="0 0 140 140" role="img" aria-label={`${label}: ${Math.round(v)} out of 100`}>
      <circle cx="70" cy="70" r={r} stroke="#16294a" strokeWidth="11" fill="none" />
      <circle
        cx="70" cy="70" r={r} stroke={color} strokeWidth="11" fill="none" strokeLinecap="round"
        strokeDasharray={`${(v / 100) * c} ${c}`} transform="rotate(-90 70 70)"
        style={{ filter: `drop-shadow(0 0 6px ${color}88)`, transition: "stroke-dasharray .5s" }}
      />
      <text x="70" y="72" textAnchor="middle" fill="#fff" fontSize="34" fontWeight="700">{Math.round(v)}</text>
      <text x="70" y="92" textAnchor="middle" fill="#8ea3c4" fontSize="11">{sub}</text>
    </svg>
  );
}

// Where a value sits among the training vehicles: p5–p95 band, p25–p75 box, median tick, marker.
export function RangeStrip({ feature, value }) {
  const { min, max, p5, p25, p50, p75, p95 } = feature;
  const span = max - min;
  const at = (v) => `${Math.max(0, Math.min(100, ((v - min) / span) * 100))}%`;
  const out = value < min || value > max;
  return (
    <div className="strip" title={`Training range ${min}–${max}; median ${p50}`}>
      <div className="strip-track">
        <span className="strip-band" style={{ left: at(p5), width: `calc(${at(p95)} - ${at(p5)})` }} />
        <span className="strip-box" style={{ left: at(p25), width: `calc(${at(p75)} - ${at(p25)})` }} />
        <span className="strip-med" style={{ left: at(p50) }} />
        <span className={`strip-mark ${out ? "out" : ""}`} style={{ left: at(value) }} />
      </div>
    </div>
  );
}

export function StatusBadge({ status }) {
  const label = { ok: "Normal", watch: "Watch", critical: "Critical" }[status] || status;
  return <span className={`badge ${status}`}>{label}</span>;
}

export function StatusDot({ status }) {
  return <span className="dot" style={{ background: STATUS_COLOR[status] || STATUS_COLOR.idle }} />;
}
