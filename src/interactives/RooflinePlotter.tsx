// Roofline plotter: one matmul X[B,D] @ W[D,F] against a chip's HBM roofline.
// Spec numbers come from src/generated/chips.json (scripts/specs.py).
import { useMemo, useState } from 'preact/hooks';
import chipsJson from '../generated/chips.json';
import { si, sig, time } from '../lib/format';

type DType = 'bf16' | 'fp8' | 'int8' | 'int4' | 'fp4';
const BYTES: Record<DType, number> = { bf16: 2, fp8: 1, int8: 1, int4: 0.5, fp4: 0.5 };
const PROV_FIELD: Record<DType, string> = {
  bf16: 'flops_bf16', fp8: 'flops_fp8', int8: 'flops_int8', int4: 'flops_int4', fp4: 'flops_fp4',
};

type ChipJ = { name: string; flops: Record<DType, number | null>; hbm_bw: number; provisional: string[] };
const CHIPS = chipsJson as unknown as Record<string, ChipJ>;

const W = 640, H = 340;
const M = { l: 56, r: 16, t: 16, b: 40 };
const X0 = 0, X1 = 4.5; // log10 intensity range: 1 .. ~31,600 FLOP/B

export function RooflinePlotter({ chip: initialChip = '7x', b: initialB = 64 }: { chip?: string; b?: number }) {
  const [chip, setChip] = useState(initialChip);
  const [wt, setWt] = useState<DType>('bf16');
  const [ct, setCt] = useState<DType>('bf16');
  const [logB, setLogB] = useState(Math.log2(initialB));
  const [D, setD] = useState(8192);
  const [F, setF] = useState(28672);
  const [hoverX, setHoverX] = useState<number | null>(null);

  const c = CHIPS[chip];
  const C = c.flops[ct] ?? NaN;
  const B = Math.round(2 ** logB);
  const a = BYTES[ct];
  const w = BYTES[wt];

  const m = useMemo(() => {
    const flops = 2 * B * D * F;
    const bytes = a * B * D + w * D * F + a * B * F;
    const ai = flops / bytes;
    const crit = C / c.hbm_bw;
    const tMath = flops / C;
    const tMem = bytes / c.hbm_bw;
    // Solve 2BDF / (aBD + wDF + aBF) = C / BW for B.
    const denom = 2 * D * F * c.hbm_bw - C * a * (D + F);
    const bStar = denom > 0 ? (C * w * D * F) / denom : Infinity;
    return { flops, bytes, ai, crit, tMath, tMem, bStar, attained: Math.min(C, c.hbm_bw * ai) };
  }, [B, D, F, a, w, C, c.hbm_bw]);

  const Y1 = Math.log10(Math.max(...Object.values(CHIPS).flatMap((k) => Object.values(k.flops).filter(Boolean) as number[]))) + 0.15;
  const Y0 = 12; // 1 TFLOP/s
  const px = (lx: number) => M.l + ((lx - X0) / (X1 - X0)) * (W - M.l - M.r);
  const py = (ly: number) => H - M.b - ((ly - Y0) / (Y1 - Y0)) * (H - M.t - M.b);
  const roof = (peak: number) => {
    const ridge = Math.log10(peak / c.hbm_bw);
    const yAt = (lx: number) => Math.min(Math.log10(peak), Math.log10(c.hbm_bw) + lx);
    const pts = [X0, Math.max(X0, Math.min(X1, ridge)), X1].map((lx) => `${px(lx)},${py(Math.max(Y0, yAt(lx)))}`);
    return pts.join(' ');
  };

  const available = (Object.keys(BYTES) as DType[]).filter((d) => c.flops[d]);
  const bound = m.tMath >= m.tMem ? 'compute-bound' : 'HBM-bound';
  const prov = c.provisional.includes(PROV_FIELD[ct]);
  const decades = Array.from({ length: Math.floor(Y1) - Y0 + 1 }, (_, i) => Y0 + i);
  const xTicks = [0, 1, 2, 3, 4];

  const onMove = (e: MouseEvent) => {
    const svg = e.currentTarget as SVGSVGElement;
    const r = svg.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W;
    const lx = X0 + ((x - M.l) / (W - M.l - M.r)) * (X1 - X0);
    setHoverX(lx >= X0 && lx <= X1 ? lx : null);
  };

  const hoverAI = hoverX == null ? null : 10 ** hoverX;
  const hoverY = hoverAI == null ? null : Math.min(C, c.hbm_bw * hoverAI);

  return (
    <div class="ix">
      <div class="ix-controls">
        <label class="field">
          <span class="lbl">Chip</span>
          <select
            value={chip}
            onChange={(e) => {
              const k = (e.target as HTMLSelectElement).value;
              setChip(k);
              if (!CHIPS[k].flops[ct]) setCt('bf16');
            }}
          >
            {Object.entries(CHIPS).map(([k, v]) => (
              <option value={k}>{v.name}</option>
            ))}
          </select>
        </label>
        <label class="field">
          <span class="lbl">Weights</span>
          <select value={wt} onChange={(e) => setWt((e.target as HTMLSelectElement).value as DType)}>
            {(Object.keys(BYTES) as DType[]).map((d) => (
              <option value={d}>{d}</option>
            ))}
          </select>
        </label>
        <label class="field">
          <span class="lbl">Math and activations</span>
          <select value={ct} onChange={(e) => setCt((e.target as HTMLSelectElement).value as DType)}>
            {available.map((d) => (
              <option value={d}>{d}</option>
            ))}
          </select>
        </label>
        <label class="field">
          <span class="lbl">
            Token batch B <output>{B.toLocaleString('en-US')}</output>
          </span>
          <input type="range" min={0} max={14} step={0.25} value={logB} onInput={(e) => setLogB(+(e.target as HTMLInputElement).value)} />
        </label>
        <label class="field">
          <span class="lbl">D (contracting)</span>
          <select value={D} onChange={(e) => setD(+(e.target as HTMLSelectElement).value)}>
            {[2048, 4096, 8192, 16384].map((v) => (
              <option value={v}>{v.toLocaleString('en-US')}</option>
            ))}
          </select>
        </label>
        <label class="field">
          <span class="lbl">F (output)</span>
          <select value={F} onChange={(e) => setF(+(e.target as HTMLSelectElement).value)}>
            {[4096, 8192, 14336, 28672, 53248].map((v) => (
              <option value={v}>{v.toLocaleString('en-US')}</option>
            ))}
          </select>
        </label>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Roofline for ${c.name}: operating point at ${sig(m.ai)} FLOPs per byte, ${bound}.`}
        onMouseMove={onMove}
        onMouseLeave={() => setHoverX(null)}
      >
        {decades.map((d) => (
          <g>
            <line class="grid" x1={M.l} x2={W - M.r} y1={py(d)} y2={py(d)} />
            <text x={M.l - 8} y={py(d) + 4} text-anchor="end">
              {si(10 ** d, '', 1)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <g>
            <line class="grid" x1={px(t)} x2={px(t)} y1={M.t} y2={H - M.b} />
            <text x={px(t)} y={H - M.b + 16} text-anchor="middle">
              {(10 ** t).toLocaleString('en-US')}
            </text>
          </g>
        ))}
        <line class="axis" x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} />
        <text x={(M.l + W - M.r) / 2} y={H - 6} text-anchor="middle">
          Arithmetic intensity, FLOPs per byte of HBM traffic
        </text>
        <text x={14} y={(M.t + H - M.b) / 2} text-anchor="middle" transform={`rotate(-90 14 ${(M.t + H - M.b) / 2})`}>
          Attainable FLOP/s
        </text>

        {ct !== 'bf16' && c.flops.bf16 ? (
          <polyline points={roof(c.flops.bf16)} fill="none" stroke="var(--s2)" stroke-width="2" stroke-dasharray="5 4" />
        ) : null}
        {isFinite(C) ? <polyline points={roof(C)} fill="none" stroke="var(--s1)" stroke-width="2" stroke-linejoin="round" /> : null}

        {isFinite(m.crit) ? (
          <g>
            <line x1={px(Math.log10(m.crit))} x2={px(Math.log10(m.crit))} y1={M.t} y2={H - M.b} stroke="var(--rule-strong)" stroke-dasharray="2 3" />
            <text x={px(Math.log10(m.crit)) + 5} y={M.t + 12}>
              ridge {Math.round(m.crit)}
            </text>
          </g>
        ) : null}

        {hoverX != null && hoverY != null ? (
          <g pointer-events="none">
            <line x1={px(hoverX)} x2={px(hoverX)} y1={M.t} y2={H - M.b} stroke="var(--ink-3)" />
            <text x={px(hoverX) + (hoverX > 3.4 ? -6 : 6)} y={py(Math.log10(hoverY)) - 8} text-anchor={hoverX > 3.4 ? 'end' : 'start'} style="fill: var(--ink)">
              {sig(hoverAI!)} FLOP/B, {si(hoverY, 'FLOP/s')}
            </text>
          </g>
        ) : null}

        {isFinite(m.attained) ? (
          <g>
            <circle cx={px(Math.log10(m.ai))} cy={py(Math.log10(m.attained))} r="5.5" fill="var(--s1)" stroke="var(--paper-2)" stroke-width="2" />
            <text x={px(Math.log10(m.ai)) + 9} y={py(Math.log10(m.attained)) + 16} style="fill: var(--ink)">
              B = {B.toLocaleString('en-US')}
            </text>
          </g>
        ) : null}
      </svg>
      <div class="legend">
        <span>
          <i style="border-color: var(--s1)" />
          {c.name} {ct} roofline
        </span>
        {ct !== 'bf16' && c.flops.bf16 ? (
          <span>
            <i class="dash" style="border-color: var(--s2)" />
            bf16 roofline, for comparison
          </span>
        ) : null}
      </div>

      <div class="ix-readout">
        <div><span class="k">Arithmetic intensity</span><span class="v">{sig(m.ai)} FLOP/B</span></div>
        <div><span class="k">Ridge point (C / HBM BW)</span><span class="v">{sig(m.crit)} FLOP/B</span></div>
        <div><span class="k">T_math</span><span class="v">{time(m.tMath)}</span></div>
        <div><span class="k">T_HBM</span><span class="v">{time(m.tMem)}</span></div>
        <div><span class="k">Compute-bound above B ≈</span><span class="v">{isFinite(m.bStar) ? Math.ceil(m.bStar).toLocaleString('en-US') : 'never'}</span></div>
      </div>
      <p class="ix-verdict">
        <b>{bound[0].toUpperCase() + bound.slice(1)}.</b>{' '}
        {bound === 'HBM-bound'
          ? `Loading ${si(m.bytes, 'B')} takes ${time(m.tMem)}; the ${si(m.flops, 'FLOP')} of math would take ${time(m.tMath)}.`
          : `The ${si(m.flops, 'FLOP')} of math take ${time(m.tMath)}; loading ${si(m.bytes, 'B')} takes ${time(m.tMem)}.`}
      </p>
      {prov ? (
        <p class="ix-note">
          <span class="vtag">provisional</span> {c.name} {ct} peak comes from the JAX Pallas hardware table, not from Google's published spec.
        </p>
      ) : null}
    </div>
  );
}
