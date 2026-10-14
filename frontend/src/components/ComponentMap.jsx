import { COMP_SHORT, STATUS_COLOR, STATUS_LABEL, signed } from "../lib.js";

// Top-down component map: the car outline in the middle, one tile per subsystem.
// Front of the car is on the right. Tile positions are layout only.
const TILES = {
  engine: { x: 478, y: 18, dot: [405, 150] },
  cooling: { x: 478, y: 128, dot: [452, 190] },
  battery: { x: 478, y: 238, dot: [395, 240] },
  lubrication: { x: 14, y: 18, dot: [340, 175] },
  general_wear: { x: 14, y: 128, dot: [200, 175] },
  brakes: { x: 14, y: 238, dot: [335, 262] },
};

export default function ComponentMap({ components, selected, onSelect }) {
  const byId = (id) => components?.find((c) => c.id === id);
  return (
    <svg viewBox="0 0 640 330" width="100%" role="img" aria-label="Component map of the vehicle" style={{ display: "block" }}>
      {/* car outline, top view */}
      <g stroke="#3b9bff" strokeWidth="2" fill="rgba(59,155,255,.06)">
        <rect x="150" y="70" width="320" height="200" rx="64" />
        <path d="M250 90 h110 q24 0 34 22 v96 q-10 22 -34 22 h-110 q-24 0 -34 -22 v-96 q10 -22 34 -22z" fill="rgba(59,155,255,.10)" />
      </g>
      {[[210, 78], [210, 262], [405, 78], [405, 262]].map(([x, y], i) => (
        <rect key={i} x={x - 24} y={y - 9} width="48" height="18" rx="6" fill="#0b0f16" stroke="#3b9bff" />
      ))}
      {Object.entries(TILES).map(([id, t]) => {
        const c = byId(id);
        const color = STATUS_COLOR[c?.status] || STATUS_COLOR.idle;
        const left = t.x < 300;
        const ax = left ? t.x + 138 : t.x;
        const ay = t.y + 38;
        return (
          <g key={id}>
            <path d={`M${ax} ${ay} L${t.dot[0]} ${t.dot[1]}`} stroke={color} strokeWidth="1.5" strokeDasharray="4 3" opacity=".8" fill="none" />
            <circle cx={t.dot[0]} cy={t.dot[1]} r="6" fill={color} style={{ filter: `drop-shadow(0 0 6px ${color})` }} />
            <g onClick={() => onSelect(id)} style={{ cursor: "pointer" }} role="button" tabIndex={0}
               onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onSelect(id)}>
              <rect x={t.x} y={t.y} width="138" height="76" rx="12" fill="rgba(8,16,32,.9)" stroke={color}
                    strokeWidth={selected === id ? 3 : 1.5} />
              <text x={t.x + 12} y={t.y + 24} fill="#e8eefb" fontSize="14" fontWeight="600">{COMP_SHORT[id]}</text>
              <text x={t.x + 12} y={t.y + 46} fill={color} fontSize="15" fontWeight="700">{STATUS_LABEL[c?.status || "idle"]}</text>
              <text x={t.x + 12} y={t.y + 64} fill="#8ea3c4" fontSize="11">
                SHAP {c ? signed(c.ai_contribution.value) : "…"}
              </text>
            </g>
          </g>
        );
      })}
    </svg>
  );
}
