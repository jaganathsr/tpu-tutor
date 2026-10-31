// Prefill/decode calculator for a model on N chips (weights and KV caches
// sharded evenly). Decode step (Scaling Book ch. 7, mirrored in
// scripts/ch07_inference.py):
//   T_step = B * KV / BW + max(2 * B * active / C, param_bytes / BW)
// plus a separate estimate of tensor-parallel collectives over one ICI axis.
import { useMemo, useState } from 'preact/hooks';
import chipsJson from '../generated/chips.json';
import { PRESETS, activeParams, attnFlopsPerToken, kvBytesPerToken, params } from '../lib/model';
import { si, time } from '../lib/format';

type ChipJ = { name: string; flops: Record<string, number | null>; hbm_bw: number; hbm_bytes: number; W_axis: number | null; provisional: string[] };
const CHIPS = chipsJson as unknown as Record<string, ChipJ>;
const BYTES: Record<string, number> = { bf16: 2, fp8: 1, int8: 1, int4: 0.5, fp4: 0.5 };

const W = 640, H = 230;
const M = { l: 60, r: 16, t: 14, b: 40 };

export function BreakevenCalc() {
  const [mi, setMi] = useState(PRESETS.findIndex((p) => p.name === 'LLaMA-3-70B'));
  const [chip, setChip] = useState('7x');
  const [logN, setLogN] = useState(3);
  const [wt, setWt] = useState('bf16');
  const [kvt, setKvt] = useState('bf16');
  const [mt, setMt] = useState('bf16');
  const [logS, setLogS] = useState(15);
  const [logB, setLogB] = useState(6);
  const [prompt, setPrompt] = useState(8192);

  const m = PRESETS[mi];
  const c = CHIPS[chip];
  const N = 2 ** logN, S = 2 ** logS, B = 2 ** logB;
  const C1 = c.flops[mt] ?? c.flops.bf16!;
  const C = C1 * N, BW = c.hbm_bw * N;
  const P = params(m), A = activeParams(m);
  const pBytes = P * BYTES[wt];
  const kvTok = kvBytesPerToken(m, BYTES[kvt]);

  const step = (b: number) => {
    const tKv = (b * S * kvTok) / BW;
    const tW = pBytes / BW;
    const tF = (2 * b * A) / C;
    return { t: tKv + Math.max(tW, tF), tKv, tW, tF, mem: (pBytes + b * S * kvTok) / N };
  };
  const r = step(B);
  const fits = r.mem <= c.hbm_bytes;
  const maxB = Math.max(0, Math.floor((c.hbm_bytes * N - pBytes) / (S * kvTok)));
  // MLP becomes compute-bound when 2bA/C = pBytes/BW.
  const bStar = (pBytes * C) / (2 * A * BW);
  // Tensor-parallel collectives: AllGather + ReduceScatter around attention and MLP per layer, one axis.
  const actBytes = BYTES[mt] >= 1 ? BYTES[mt] : 1;
  const tIciBw = N > 1 && c.W_axis ? (4 * m.L * B * m.D * actBytes) / c.W_axis : 0;
  const tIciLat = N > 1 ? 4 * m.L * (N / 2) * 1e-6 : 0;
  const tIci = Math.max(tIciBw, tIciLat);
  const ttft = (2 * A * prompt + attnFlopsPerToken(m, prompt) * prompt * 0.5) / (C * 0.4);

  const curve = useMemo(() => {
    const pts: { b: number; t: number; tput: number }[] = [];
    for (let lb = 0; lb <= 12; lb++) {
      const b = 2 ** lb;
      if (b > maxB) break;
      const s = step(b);
      pts.push({ b, t: s.t, tput: b / s.t / N });
    }
    return pts;
  }, [mi, chip, logN, wt, kvt, mt, logS, maxB]);

  const xs = curve.map((p) => p.t), ys = curve.map((p) => p.tput);
  const x0 = Math.log10(Math.min(...xs, r.t) * 0.8), x1 = Math.log10(Math.max(...xs, r.t) * 1.25);
  const y0 = Math.log10(Math.min(...ys, B / r.t / N) * 0.8), y1 = Math.log10(Math.max(...ys, B / r.t / N) * 1.25);
  const px = (v: number) => M.l + ((Math.log10(v) - x0) / (x1 - x0 || 1)) * (W - M.l - M.r);
  const py = (v: number) => H - M.b - ((Math.log10(v) - y0) / (y1 - y0 || 1)) * (H - M.t - M.b);

  const dtypeSel = (val: string, set: (v: string) => void, opts: string[]) => (
    <select value={val} onChange={(e) => set((e.target as HTMLSelectElement).value)}>
      {opts.map((o) => <option value={o}>{o}</option>)}
    </select>
  );
  const mathOpts = Object.keys(BYTES).filter((d) => c.flops[d]);

  return (
    <div class="ix">
      <div class="ix-controls">
        <label class="field"><span class="lbl">Model</span>
          <select value={mi} onChange={(e) => setMi(+(e.target as HTMLSelectElement).value)}>
            {PRESETS.map((p, i) => <option value={i}>{p.name}</option>)}
          </select></label>
        <label class="field"><span class="lbl">Chip</span>
          <select value={chip} onChange={(e) => { const k = (e.target as HTMLSelectElement).value; setChip(k); if (!CHIPS[k].flops[mt]) setMt('bf16'); }}>
            {Object.entries(CHIPS).map(([k, v]) => <option value={k}>{v.name}</option>)}
          </select></label>
        <label class="field"><span class="lbl">Chips (model-parallel) <output>{N}</output></span>
          <input type="range" min={0} max={6} value={logN} onInput={(e) => setLogN(+(e.target as HTMLInputElement).value)} /></label>
        <label class="field"><span class="lbl">Weights</span>{dtypeSel(wt, setWt, ['bf16', 'fp8', 'int8', 'int4'])}</label>
        <label class="field"><span class="lbl">KV cache</span>{dtypeSel(kvt, setKvt, ['bf16', 'fp8', 'int8'])}</label>
        <label class="field"><span class="lbl">Math</span>{dtypeSel(mt, setMt, mathOpts)}</label>
        <label class="field"><span class="lbl">Context S <output>{S.toLocaleString('en-US')}</output></span>
          <input type="range" min={10} max={17} value={logS} onInput={(e) => setLogS(+(e.target as HTMLInputElement).value)} /></label>
        <label class="field"><span class="lbl">Decode batch B <output>{B}</output></span>
          <input type="range" min={0} max={12} value={logB} onInput={(e) => setLogB(+(e.target as HTMLInputElement).value)} /></label>
        <label class="field"><span class="lbl">Prompt length for TTFT <output>{prompt.toLocaleString('en-US')}</output></span>
          <input type="range" min={7} max={17} value={Math.log2(prompt)} onInput={(e) => setPrompt(2 ** +(e.target as HTMLInputElement).value)} /></label>
      </div>

      <div class="ix-readout">
        <div><span class="k">Memory per chip</span><span class="v">{si(r.mem, 'B')} of {si(c.hbm_bytes, 'B')}</span></div>
        <div><span class="k">Largest batch that fits</span><span class="v">{maxB.toLocaleString('en-US')}</span></div>
        <div><span class="k">Step: KV read</span><span class="v">{time(r.tKv)}</span></div>
        <div><span class="k">Step: weight read</span><span class="v">{time(r.tW)}</span></div>
        <div><span class="k">Step: MLP FLOPs</span><span class="v">{time(r.tF)}</span></div>
        <div><span class="k">Step total (lower bound)</span><span class="v">{time(r.t)}</span></div>
        <div><span class="k">Tokens/s per chip</span><span class="v">{Math.round(B / r.t / N).toLocaleString('en-US')}</span></div>
        <div><span class="k">MLP compute-bound above B ≈</span><span class="v">{Math.round(bStar).toLocaleString('en-US')}</span></div>
        <div><span class="k">TP collectives, if not overlapped</span><span class="v">{N > 1 ? time(tIci) : 'none'}</span></div>
        <div><span class="k">Prefill TTFT at 40% MFU</span><span class="v">{time(ttft)}</span></div>
      </div>
      <p class={`ix-verdict ${fits ? '' : 'result wrong'}`}>
        {!fits ? (
          <><b>Does not fit.</b> Weights plus {B} sequences of {S.toLocaleString('en-US')} tokens need {si(r.mem, 'B')} per chip. Lower the batch, shorten the context, quantize, or add chips.</>
        ) : r.tKv > r.tW && r.tKv > r.tF ? (
          <><b>KV-bandwidth-bound.</b> Reading {B} caches takes {time(r.tKv)}, more than the weights ({time(r.tW)}). More batch adds KV time one-for-one; quantizing the KV cache or shorter context helps most.</>
        ) : r.tF > r.tW ? (
          <><b>MLP compute-bound, attention HBM-bound.</b> The batch is above {Math.round(bStar)}, so the matrix units are busy during the MLP.</>
        ) : (
          <><b>Weight-bandwidth-bound.</b> The batch of {B} is below the {Math.round(bStar)} needed to make the MLP compute-bound; a larger batch raises throughput almost for free.</>
        )}
        {N > 1 && tIci > 0.3 * r.t ? <> The tensor-parallel collectives ({tIciLat > tIciBw ? 'latency-bound' : 'bandwidth-bound'}) are a large share of the step unless overlapped.</> : null}
      </p>

      {curve.length > 1 ? (
        <>
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Throughput per chip against step latency for batch sizes that fit.">
            <line class="axis" x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} />
            <text x={(M.l + W - M.r) / 2} y={H - 6} text-anchor="middle">Step latency (per output token, log scale)</text>
            <text x={12} y={(H - M.b + M.t) / 2} text-anchor="middle" transform={`rotate(-90 12 ${(H - M.b + M.t) / 2})`}>tokens/s per chip</text>
            <path d={curve.map((p, i) => `${i ? 'L' : 'M'}${px(p.t)},${py(p.tput)}`).join(' ')} fill="none" stroke="var(--s1)" stroke-width="2" />
            {curve.map((p) => (
              <g>
                <circle cx={px(p.t)} cy={py(p.tput)} r={p.b === B ? 6 : 3.5} fill={p.b === B ? 'var(--s2)' : 'var(--s1)'} stroke="var(--paper-2)" stroke-width="2" />
                {Math.log2(p.b) % 2 === 0 ? <text x={px(p.t)} y={py(p.tput) - 9} text-anchor="middle">B={p.b}</text> : null}
              </g>
            ))}
            <text x={M.l - 6} y={py(Math.max(...ys)) + 4} text-anchor="end">{Math.round(Math.max(...ys)).toLocaleString('en-US')}</text>
            <text x={M.l - 6} y={py(Math.min(...ys)) + 4} text-anchor="end">{Math.round(Math.min(...ys)).toLocaleString('en-US')}</text>
            <text x={M.l} y={H - M.b + 16}>{time(Math.min(...xs))}</text>
            <text x={W - M.r} y={H - M.b + 16} text-anchor="end">{time(Math.max(...xs))}</text>
          </svg>
          <div class="legend"><span><i style="border-color: var(--s1)" />batches that fit in HBM, 1 to {curve[curve.length - 1].b}</span><span><i style="border-color: var(--s2)" />current batch</span></div>
        </>
      ) : null}
      <p class="ix-note">
        Lower bounds: no overheads, perfect overlap of FLOPs with reads, and attention FLOPs ignored in decode. TTFT counts
        weight FLOPs plus causal attention for one prompt. Spec values from scripts/specs.py
        {c.provisional.includes('flops_bf16') ? '; this chip\'s BF16 and FP8 peaks are provisional' : ''}.
      </p>
    </div>
  );
}
