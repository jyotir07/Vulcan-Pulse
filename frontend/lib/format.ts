export function pct(value: number, digits = 2): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function pp(value: number, digits = 2): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)} pp`;
}

export function rupees(value: number): string {
  const sign = value < 0 ? "-" : "";
  const v = Math.abs(value);
  if (v >= 1e7) return `${sign}₹${(v / 1e7).toFixed(2)} Cr`;
  if (v >= 1e5) return `${sign}₹${(v / 1e5).toFixed(2)} L`;
  return `${sign}₹${Math.round(v).toLocaleString("en-IN")}`;
}

export function count(value: number): string {
  return Math.round(value).toLocaleString("en-IN");
}

export function signedCount(value: number): string {
  return `${value >= 0 ? "+" : ""}${count(value)}`;
}

export function ms(value: number): string {
  return `${Math.round(value).toLocaleString("en-IN")} ms`;
}

export function clock(minute: number): string {
  const h = Math.floor(minute / 60) % 24;
  const m = minute % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}
