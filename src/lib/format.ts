// Number formatting shared by content components and interactives.

export function sig(x: number, digits = 3): string {
  if (!isFinite(x)) return '—';
  if (x === 0) return '0';
  const p = Math.floor(Math.log10(Math.abs(x)));
  const d = Math.max(0, digits - 1 - p);
  return x.toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: 0 });
}

const SI: [number, string][] = [
  [1e18, 'E'], [1e15, 'P'], [1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'k'],
];

export function si(x: number, unit = '', digits = 3): string {
  if (!isFinite(x)) return '—';
  for (const [v, p] of SI) if (Math.abs(x) >= v) return `${sig(x / v, digits)} ${p}${unit}`;
  return `${sig(x, digits)} ${unit}`.trim();
}

export function time(s: number): string {
  if (!isFinite(s)) return '—';
  if (s >= 1) return `${sig(s)} s`;
  if (s >= 1e-3) return `${sig(s * 1e3)} ms`;
  if (s >= 1e-6) return `${sig(s * 1e6)} µs`;
  return `${sig(s * 1e9)} ns`;
}

export function bytes(b: number): string {
  if (b >= 2 ** 30) return `${sig(b / 2 ** 30)} GiB`;
  if (b >= 2 ** 20) return `${sig(b / 2 ** 20)} MiB`;
  if (b >= 2 ** 10) return `${sig(b / 2 ** 10)} KiB`;
  return `${b} B`;
}

export function duration(ms: number): string {
  const m = Math.round(ms / 60000);
  if (m < 60) return `${m} min`;
  return `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, '0')} min`;
}

/** Parse a learner's numeric answer: accepts 1,200 / 1.2e3 / 1.2k / 12% */
export function parseNumber(raw: string): number | null {
  const s = raw.trim().replace(/,/g, '').replace(/\s+/g, '').replace(/×10\^?/i, 'e');
  const m = s.match(/^([-+]?\d*\.?\d+(?:e[-+]?\d+)?)([kmgtp%]?)$/i);
  if (!m) return null;
  const mult: Record<string, number> = { '': 1, k: 1e3, m: 1e6, g: 1e9, t: 1e12, p: 1e15, '%': 1 };
  return parseFloat(m[1]) * mult[m[2].toLowerCase()];
}
