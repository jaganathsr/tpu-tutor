// Training mesh planner (Scaling Book ch. 5 model, forward pass of one layer
// In[B,D] W_in[D,F] W_out[F,D]):
//   T_math = 4BDF / (N C), T_FSDP = 4DF / (Y W M_X), T_TP = 4BD / (X W M_Y)
// with N = X * Y chips, X-way FSDP over M_X mesh axes and Y-way TP over M_Y axes.
// The two collectives run on different axes, so communication time is their max.
import { useState } from 'preact/hooks';
import chipsJson from '../generated/chips.json';
import { PRESETS } from '../lib/model';
import { time } from '../lib/format';

type ChipJ = { name: string; flops: Record<string, number | null>; W_axis: number | null; provisional: string[] };
const CHIPS = Object.fromEntries(Object.entries(chipsJson as unknown as Record<string, ChipJ>).filter(([, c]) => c.W_axis && c.flops.bf16)) as Record<string, ChipJ>;
const DENSE = PRESETS.filter((p) => p.E === 1);

const W = 640, H = 260;
const M = { l: 56, r: 16, t: 14, b: 40 };

export function MeshPlanner() {
  const [chip, setChip] = useState('7x');
  const [mi, setMi] = useState(DENSE.findIndex((p) => p.name === 'LLaMA-3-70B'));
  const [logB, setLogB] = useState(22); // 4M tokens
  const [logN, setLogN] = useState(11); // 2048 chips
  const [logX, setLogX] = useState(9);
  const [mx, setMx] = useState(2);
  const my = 3 - mx;

  const c = CHIPS[chip];
  const m = DENSE[mi];
  const C = c.flops.bf16!, Wi = c.W_axis!;
  const B = 2 ** logB, N = 2 ** logN;
  const X = 2 ** Math.min(logX, logN), Y = N / X;
  const tMath = (4 * B * m.D * m.F) / (N * C);
  const tF = (x: number) => (4 * m.D * m.F) / ((N / x) * Wi * mx);
  const tT = (x: number) => (4 * B * m.D) / (x * Wi * my);
  // FSDP and TP collectives use different torus axes and run concurrently (SB5),
  // so the slower of the two sets the communication time.
  const tComms = Math.max(tF(X), tT(X));
  const xOpt = Math.sqrt((B / m.F) * (mx / my) * N);
  const alpha = C / Wi;
  const thresh = (alpha * alpha) / (mx * my * m.F);

  const xs = Array.from({ length: logN + 1 }, (_, i) => 2 ** i);
  const all = xs.flatMap((x) => [tF(x), tT(x)]).concat([tMath]);
  const y0 = Math.log10(Math.min(...all)) - 0.2, y1 = Math.log10(Math.max(...all)) + 0.2;
  const px = (x: number) => M.l + (Math.log2(x) / logN) * (W - M.l - M.r);
  const py = (v: number) => H - M.b - ((Math.log10(v) - y0) / (y1 - y0)) * (H - M.t - M.b);
  const line = (f: (x: number) => number) => xs.map((x, i) => `${i ? 'L' : 'M'}${px(x)},${py(f(x))}`).join(' ');

  return (
    <div class="ix">
      <div class="ix-controls">
        <label class="field"><span class="lbl">Chip</span>
          <select value={chip} onChange={(e) => setChip((e.target as HTMLSelectElement).value)}>
            {Object.entries(CHIPS).map(([k, v]) => <option value={k}>{v.name}</option>)}
          </select></label>
        <label class="field"><span class="lbl">Model</span>
          <select value={mi} onChange={(e) => setMi(+(e.target as HTMLSelectElement).value)}>
            {DENSE.map((p, i) => <option value={i}>{p.name}</option>)}
          </select></label>
        <label class="field"><span class="lbl">Global batch <output>{(B / 2 ** 20).toLocaleString('en-US')}M tokens</output></span>
          <input type="range" min={18} max={26} value={logB} onInput={(e) => setLogB(+(e.target as HTMLInputElement).value)} /></label>
        <label class="field"><span class="lbl">Chips N <output>{N.toLocaleString('en-US')}</output></span>
          <input type="range" min={4} max={14} value={logN} onInput={(e) => setLogN(+(e.target as HTMLInputElement).value)} /></label>
        <label class="field"><span class="lbl">FSDP degree X <output>{X.toLocaleString('en-US')} (TP {Y})</output></span>
          <input type="range" min={0} max={logN} value={Math.min(logX, logN)} onInput={(e) => setLogX(+(e.target as HTMLInputElement).value)} /></label>
        <label class="field"><span class="lbl">Torus axes for FSDP / TP</span>
          <select value={mx} onChange={(e) => setMx(+(e.target as HTMLSelectElement).value)}>
            <option value={2}>2 / 1</option><option value={1}>1 / 2</option>
          </select></label>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Per-layer times against FSDP degree; optimum near ${Math.round(xOpt)}.`}>
        {xs.filter((_, i) => i % 2 === 0).map((x) => (
          <g>
            <line class="grid" x1={px(x)} x2={px(x)} y1={M.t} y2={H - M.b} />
            <text x={px(x)} y={H - M.b + 16} text-anchor="middle">{x.toLocaleString('en-US')}</text>
          </g>
        ))}
        <text x={(M.l + W - M.r) / 2} y={H - 6} text-anchor="middle">FSDP degree X (TP degree is N / X)</text>
        <line class="axis" x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} />
        <line x1={M.l} x2={W - M.r} y1={py(tMath)} y2={py(tMath)} stroke="var(--s3)" stroke-width="2" />
        <path d={line(tF)} fill="none" stroke="var(--s1)" stroke-width="2" />
        <path d={line(tT)} fill="none" stroke="var(--s2)" stroke-width="2" />
        <path d={line((x) => Math.max(tF(x), tT(x)))} fill="none" stroke="var(--ink-2)" stroke-width="1.5" stroke-dasharray="4 3" />
        {xOpt >= 1 && xOpt <= N ? <line x1={px(xOpt)} x2={px(xOpt)} y1={M.t} y2={H - M.b} stroke="var(--rule-strong)" stroke-dasharray="2 3" /> : null}
        <circle cx={px(X)} cy={py(tComms)} r="5" fill="var(--ink)" stroke="var(--paper-2)" stroke-width="2" />
        <text x={M.l - 6} y={py(tMath) + 4} text-anchor="end">{time(tMath)}</text>
      </svg>
      <div class="legend">
        <span><i style="border-color: var(--s3)" />math per layer</span>
        <span><i style="border-color: var(--s1)" />FSDP weight gathers</span>
        <span><i style="border-color: var(--s2)" />TP activation collectives</span>
        <span><i class="dash" style="border-color: var(--ink-2)" />communication: the slower of the two</span>
      </div>
      <div class="ix-readout">
        <div><span class="k">Tokens per chip</span><span class="v">{Math.round(B / N).toLocaleString('en-US')}</span></div>
        <div><span class="k">Needed for FSDP+TP, α²/(M_X M_Y F)</span><span class="v">{Math.round(thresh).toLocaleString('en-US')}</span></div>
        <div><span class="k">Best FSDP degree X_opt</span><span class="v">{Math.round(xOpt).toLocaleString('en-US')}</span></div>
        <div><span class="k">α = C / W_ici</span><span class="v">{Math.round(alpha).toLocaleString('en-US')}</span></div>
        <div><span class="k">At this X: math / comms</span><span class="v">{time(tMath)} / {time(tComms)}</span></div>
      </div>
      <p class="ix-verdict">
        <b>{tMath >= tComms ? 'Compute-bound' : 'Communication-bound'} at X = {X}, Y = {Y}.</b>{' '}
        {B / N >= thresh
          ? `${Math.round(B / N)} tokens per chip clears the ${Math.round(thresh)} needed, so a split near X = ${Math.round(xOpt)} keeps the matrix units busy.`
          : `${Math.round(B / N)} tokens per chip is below the ${Math.round(thresh)} needed: no FSDP/TP split is compute-bound. Use fewer chips, a larger batch, or accept lower MFU.`}
      </p>
      {c.provisional.includes('flops_bf16') ? <p class="ix-note"><span class="vtag">provisional</span> {c.name} BF16 peak and ICI are not from Google's published spec.</p> : null}
    </div>
  );
}
