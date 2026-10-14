export const BRAND = "Vehicle Risk Console";

export const STATUS_COLOR = { ok: "#22c55e", watch: "#ff9f1a", critical: "#ff4d4f", idle: "#5b6e8f" };
export const STATUS_LABEL = { ok: "Normal", watch: "Watch", critical: "Critical", idle: "No data" };
export const BAND_COLOR = { LOW: "#22c55e", MODERATE: "#ff9f1a", HIGH: "#ff4d4f" };

export const COMP_ORDER = ["engine", "cooling", "lubrication", "brakes", "battery", "general_wear"];
export const COMP_SHORT = {
  engine: "Engine",
  cooling: "Cooling",
  lubrication: "Oil",
  brakes: "Brakes",
  battery: "Battery",
  general_wear: "Chassis wear",
};

export const signed = (x, d = 2) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`;
export const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;

// Display mapping from the backend's maintenance priority to a plain-language urgency.
// It is a wording of the priority, not a predicted date.
export function urgency(priority) {
  if (priority === "HIGH") return "Arrange an inspection soon";
  if (priority === "MEDIUM") return "Schedule a routine inspection";
  return "No action needed: continue scheduled maintenance";
}

export function topDrivers(result, n = 3) {
  return result.explainability.contributions
    .filter((c) => c.contribution > 0)
    .slice(0, n);
}

// Plain-language summary assembled only from fields the API returned.
export function narrative(result) {
  const r = result;
  const parts = [
    `This vehicle is rated ${r.risk_level.toLowerCase()} risk, with a risk index of ${Math.round(r.risk_index)} out of 100.`,
  ];
  const drivers = topDrivers(r, 3);
  if (drivers.length) {
    parts.push(`The strongest factors raising the risk are ${drivers.map((d) => d.label.toLowerCase()).join(", ")}.`);
  } else {
    parts.push("No input is pushing the risk upward in a notable way.");
  }
  const flagged = r.components.filter((c) => c.status !== "ok");
  if (flagged.length) {
    parts.push(`Subsystems to look at: ${flagged.map((c) => c.label.toLowerCase()).join(", ")}.`);
  }
  parts.push(`Maintenance priority is ${r.maintenance_priority.toLowerCase()}. ${urgency(r.maintenance_priority)}.`);
  parts.push("This is decision support from a model trained on a synthetic dataset, not a diagnosis.");
  return parts.join(" ");
}
