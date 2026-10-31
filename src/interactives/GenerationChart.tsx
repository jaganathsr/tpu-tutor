// Growth of per-chip BF16 FLOP/s and HBM bandwidth across flagship TPU
// generations, both indexed to TPUv2 = 1 (one axis, no dual scale).
// Data: scripts/ch01_roofline.py GENERATIONS.
import { useState } from 'preact/hooks';

const GEN = [
  { name: 'TPUv2', year: 2017, tf: 46, bw: 700 },
  { name: 'TPUv3', year: 2018, tf: 123, bw: 900 },
  { name: 'TPU v4', year: 2020, tf: 275, bw: 1200 },
  { name: 'TPU v5p', year: 2023, tf: 459, bw: 2765 },
  { name: 'TPU7x', year: 2025, tf: 2307, bw: 7380 },
];

const W = 640, H = 300;
const M = { l: 48, r: 92, t: 18, b: 44 };
const Y0 = 0, Y1 = Math.log10(80);

export function GenerationChart() {
  const [hover, setHover] = useState<number | null>(null);
  const x = (i: number) => M.l + (i / (GEN.length - 1)) * (W - M.l - M.r);
  const y = (v: number) => H - M.b - ((Math.log10(v) - Y0) / (Y1 - Y0)) * (H - M.t - M.b);
  const fl = GEN.map((g) => g.tf / GEN[0].tf);
  const bw = GEN.map((g) => g.bw / GEN[0].bw);
  const path = (vs: number[]) => vs.map((v, i) => `${i ? 'L' : 'M'}${x(i)},${y(v)}`).join(' ');
  const ticks = [1, 2, 5, 10, 20, 50];

  return (
    <div class="ix">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="BF16 FLOP/s grew 50 times from TPUv2 to TPU7x; HBM bandwidth grew 10.5 times.">
        {ticks.map((t) => (
          <g>
            <line class="grid" x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} />
            <text x={M.l - 8} y={y(t) + 4} text-anchor="end">{t}×</text>
          </g>
        ))}
        {GEN.map((g, i) => (
          <text x={x(i)} y={H - M.b + 18} text-anchor="middle">
            {g.name}
          </text>
        ))}
        {GEN.map((g, i) => (
          <text x={x(i)} y={H - M.b + 32} text-anchor="middle" style="fill: var(--ink-3)">
            {g.year}
          </text>
        ))}
        <line class="axis" x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} />
        <path d={path(fl)} fill="none" stroke="var(--s1)" stroke-width="2" />
        <path d={path(bw)} fill="none" stroke="var(--s2)" stroke-width="2" />
        {fl.map((v, i) => (
          <circle cx={x(i)} cy={y(v)} r="4.5" fill="var(--s1)" stroke="var(--paper-2)" stroke-width="2" />
        ))}
        {bw.map((v, i) => (
          <circle cx={x(i)} cy={y(v)} r="4.5" fill="var(--s2)" stroke="var(--paper-2)" stroke-width="2" />
        ))}
        <text x={x(4) + 10} y={y(fl[4]) + 4} style="fill: var(--ink)">FLOP/s 50×</text>
        <text x={x(4) + 10} y={y(bw[4]) + 4} style="fill: var(--ink)">HBM BW 10.5×</text>
        {GEN.map((_, i) => (
          <rect
            x={x(i) - 40}
            y={M.t}
            width="80"
            height={H - M.t - M.b}
            fill="transparent"
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
        {hover != null ? (
          <g pointer-events="none">
            <line x1={x(hover)} x2={x(hover)} y1={M.t} y2={H - M.b} stroke="var(--ink-3)" />
            <text x={x(hover) + (hover > 2 ? -8 : 8)} y={M.t + 12} text-anchor={hover > 2 ? 'end' : 'start'} style="fill: var(--ink)">
              {GEN[hover].name}: {GEN[hover].tf.toLocaleString('en-US')} TFLOP/s, {GEN[hover].bw.toLocaleString('en-US')} GB/s, ridge{' '}
              {Math.round((GEN[hover].tf * 1000) / GEN[hover].bw)}
            </text>
          </g>
        ) : null}
      </svg>
      <div class="legend">
        <span><i style="border-color: var(--s1)" />Dense BF16 FLOP/s per chip</span>
        <span><i style="border-color: var(--s2)" />HBM bandwidth per chip</span>
      </div>
      <div class="table-wrap" style="margin-top:.8rem">
        <table class="spec-table">
          <thead>
            <tr><th>Chip</th><th class="r">BF16 TFLOP/s</th><th class="r">HBM GB/s</th><th class="r">Ridge, FLOP/B</th></tr>
          </thead>
          <tbody>
            {GEN.map((g) => (
              <tr>
                <td>{g.name} ({g.year})</td>
                <td class="r">{g.tf.toLocaleString('en-US')}</td>
                <td class="r">{g.bw.toLocaleString('en-US')}</td>
                <td class="r">{Math.round((g.tf * 1000) / g.bw)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
