// Side-by-side predictions for 2-4 chips on one workload. Formulas mirror
// scripts/ch11_compare.py (collectives) and scripts/ch07_inference.py (decode).
import { useState } from 'preact/hooks';
import chipsJson from '../generated/chips.json';
import { PRESETS, activeParams, kvBytesPerToken, params } from '../lib/model';
import { bytes as fmtBytes, time } from '../lib/format';

type Net = { kind?: 'torus' | 'switch'; axes?: number; domain?: number; link?: number; egress?: number; node?: number };
type ChipJ = { name: string; flops: Record<string, number | null>; hbm_bw: number; hbm_bytes: number; W_axis: number | null; provisional: string[]; net: Net };
const CHIPS = chipsJson as unknown as Record<string, ChipJ>;
const DEFAULT = ['7x', 'H100', 'GB200', 'Trn3'];

type Work = 'matmul' | 'decode' | 'allgather' | 'alltoall';
const WORK: Record<Work, string> = {
  matmul: 'bf16 matmul on one chip',
  decode: 'Decode step, model sharded over N chips',
  allgather: 'AllGather of V bytes over N chips',
  alltoall: 'AllToAll of V bytes over N chips (MoE dispatch)',
};

function collective(c: ChipJ, kind: 'allgather' | 'alltoall', V: number, N: number): { t: number; why: string } {
  const n = c.net;
  if (n.kind === 'torus' && c.W_axis) {
    const axes = N <= 16 ? 1 : n.axes ?? 1;
    if (N > (n.domain ?? Infinity)) return { t: NaN, why: `beyond one ICI domain (${n.domain} chips); crosses DCN` };
    if (kind === 'allgather') return { t: V / (c.W_axis * axes), why: `torus, ${axes} ${axes > 1 ? 'axes' : 'axis'} at ${fmtBytes(c.W_axis).replace('iB', 'B')}/s each` };
    const side = N ** (1 / axes);
    return { t: (V * side) / (4 * N * c.W_axis), why: `torus bisection, ${axes} ${axes > 1 ? 'axes' : 'axis'}` };
  }
  if (n.kind === 'switch' && n.link && n.domain) {
    if (N <= n.domain) {
      return kind === 'allgather'
        ? { t: (V * (N - 1)) / N / n.link, why: `one-hop switched domain of ${n.domain}` }
        : { t: V / (N * n.link), why: `one-hop switched domain of ${n.domain}` };
    }
    if (!n.egress || !n.node) return { t: NaN, why: `beyond the ${n.domain}-chip domain; scale-out not modelled` };
    if (kind === 'allgather') return { t: Math.max(V / n.link, V / n.egress), why: `crosses nodes: ${fmtBytes(n.egress).replace('iB', 'B')}/s per ${n.node}-GPU node` };
    return { t: V / ((N / n.node) * n.egress), why: `crosses ${Math.ceil(N / n.node)} nodes over scale-out` };
  }
  return { t: NaN, why: 'no network model' };
}

export function CompareExplorer() {
  const [sel, setSel] = useState<string[]>(DEFAULT);
  const [work, setWork] = useState<Work>('alltoall');
  const [logV, setLogV] = useState(30);
  const [logN, setLogN] = useState(6);
  const [logB, setLogB] = useState(6);
  const [mi, setMi] = useState(PRESETS.findIndex((p) => p.name === 'LLaMA-3-70B'));
  const [logS, setLogS] = useState(15);
  const V = 2 ** logV, N = 2 ** logN, B = 2 ** logB, S = 2 ** logS;
  const D = 8192, F = 28672;

  const rows = sel.map((k) => {
    const c = CHIPS[k];
    if (work === 'matmul') {
      const C = c.flops.bf16!;
      const flops = 2 * B * D * F, by = 2 * (B * D + D * F + B * F);
      const tm = flops / C, th = by / c.hbm_bw;
      return { k, c, t: Math.max(tm, th), why: tm > th ? 'compute-bound' : `HBM-bound (${Math.round(flops / by)} FLOP/B against a ridge of ${Math.round(C / c.hbm_bw)})` };
    }
    if (work === 'decode') {
      const m = PRESETS[mi];
      const C = (c.flops.fp8 ?? c.flops.bf16!) * N, bw = c.hbm_bw * N;
      const pB = params(m), kv = B * S * kvBytesPerToken(m, 1);
      if (pB + kv > c.hbm_bytes * N) return { k, c, t: NaN, why: `does not fit: needs ${fmtBytes((pB + kv) / N)} per chip` };
      const tKv = kv / bw, tW = pB / bw, tF = (2 * B * activeParams(m)) / C;
      return { k, c, t: tKv + Math.max(tW, tF), why: tKv > Math.max(tW, tF) ? 'KV-read-bound' : tF > tW ? 'MLP compute-bound' : 'weight-read-bound' };
    }
    const r = collective(c, work, V, N);
    return { k, c, ...r };
  });
  const finite = rows.filter((r) => isFinite(r.t));
  const tmax = Math.max(...finite.map((r) => r.t), 1e-12);

  const toggle = (k: string) => setSel(sel.includes(k) ? sel.filter((x) => x !== k) : sel.length < 4 ? [...sel, k] : sel);

  return (
    <div class="ix">
      <div class="ix-controls">
        <label class="field" style="grid-column: 1 / -1"><span class="lbl">Chips (up to 4)</span>
          <span style="display:flex;flex-wrap:wrap;gap:.4rem 1rem">
            {Object.entries(CHIPS).map(([k, c]) => (
              <label><input type="checkbox" checked={sel.includes(k)} onChange={() => toggle(k)} /> {c.name}</label>
            ))}
          </span></label>
        <label class="field"><span class="lbl">Workload</span>
          <select value={work} onChange={(e) => setWork((e.target as HTMLSelectElement).value as Work)}>
            {Object.entries(WORK).map(([k, v]) => <option value={k}>{v}</option>)}
          </select></label>
        {work === 'allgather' || work === 'alltoall' ? (
          <label class="field"><span class="lbl">V <output>{fmtBytes(V)}</output></span>
            <input type="range" min={20} max={34} value={logV} onInput={(e) => setLogV(+(e.target as HTMLInputElement).value)} /></label>
        ) : null}
        {work !== 'matmul' ? (
          <label class="field"><span class="lbl">Chips N <output>{N}</output></span>
            <input type="range" min={1} max={10} value={logN} onInput={(e) => setLogN(+(e.target as HTMLInputElement).value)} /></label>
        ) : null}
        {work === 'matmul' || work === 'decode' ? (
          <label class="field"><span class="lbl">Batch B <output>{B}</output></span>
            <input type="range" min={0} max={12} value={logB} onInput={(e) => setLogB(+(e.target as HTMLInputElement).value)} /></label>
        ) : null}
        {work === 'decode' ? (
          <>
            <label class="field"><span class="lbl">Model</span>
              <select value={mi} onChange={(e) => setMi(+(e.target as HTMLSelectElement).value)}>
                {PRESETS.map((p, i) => <option value={i}>{p.name}</option>)}
              </select></label>
            <label class="field"><span class="lbl">Context <output>{S.toLocaleString('en-US')}</output></span>
              <input type="range" min={10} max={17} value={logS} onInput={(e) => setLogS(+(e.target as HTMLInputElement).value)} /></label>
          </>
        ) : null}
      </div>
      <table class="spec-table compare-table">
        <thead><tr><th>Chip</th><th style="width:40%"></th><th class="r">Time</th><th>Why</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr>
              <td>{r.c.name}{r.c.provisional.length ? <span class="vtag" title={`Provisional: ${r.c.provisional.join(', ')}`}>p</span> : null}</td>
              <td>{isFinite(r.t) ? <div style={`height:10px;background:var(--s1);width:${Math.max(1, (100 * r.t) / tmax)}%`} /> : null}</td>
              <td class="r">{isFinite(r.t) ? time(r.t) : '—'}</td>
              <td style="white-space:normal">{r.why}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p class="ix-note">
        Idealized models from the Scaling Book (TPU ch. 3, GPU ch. 12): no latency floor, no software overhead. The book notes GPU
        AllReduce reaches about 370 of 450 GB/s only at about 10 GB messages, and multi-node AllToAll may not reach 50%. Matmul: D = 8,192, F = 28,672.
        Decode uses fp8 weights and KV, fp8 math where the chip has it. Trainium3's per-direction link rate and Rubin's figures are provisional.
      </p>
    </div>
  );
}
