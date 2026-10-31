// Transformer accounting calculator: parameters, FLOPs per token, KV cache and
// how many chips the weights and caches need.
import { useState } from 'preact/hooks';
import chipsJson from '../generated/chips.json';
import { PRESETS, activeParams, attnFlopsPerToken, attnParams, kvBytesPerToken, mlpParams, params, vocabParams, type ModelCfg } from '../lib/model';
import { si, sig } from '../lib/format';

type ChipJ = { name: string; hbm_bytes: number };
const CHIPS = chipsJson as unknown as Record<string, ChipJ>;
const FIELDS: [keyof ModelCfg, string][] = [
  ['L', 'Layers L'], ['D', 'd_model D'], ['F', 'd_ff F (per expert)'], ['N', 'Query heads N'],
  ['K', 'KV heads K'], ['H', 'Head dim H'], ['V', 'Vocab V'], ['E', 'Experts E'], ['k', 'Active experts k'],
];

export function ModelCalc() {
  const [m, setM] = useState<ModelCfg>(PRESETS[0]);
  const [wBytes, setWBytes] = useState(2);
  const [kvB, setKvB] = useState(2);
  const [S, setS] = useState(32768);
  const [chip, setChip] = useState('7x');

  const P = params(m), A = activeParams(m);
  const kvTok = kvBytesPerToken(m, kvB);
  const kvSeq = kvTok * S;
  const fwd = 2 * A + attnFlopsPerToken(m, S);
  const attnShare = attnFlopsPerToken(m, S) / fwd;
  const hbm = CHIPS[chip].hbm_bytes;
  const chipsForWeights = Math.ceil((P * wBytes) / hbm);

  const set = (f: keyof ModelCfg, v: number) => setM({ ...m, name: 'Custom', [f]: v });

  return (
    <div class="ix">
      <div class="ix-controls">
        <label class="field"><span class="lbl">Model</span>
          <select value={PRESETS.findIndex((p) => p.name === m.name)} onChange={(e) => {
            const i = +(e.target as HTMLSelectElement).value; if (i >= 0) setM(PRESETS[i]);
          }}>
            {PRESETS.map((p, i) => <option value={i}>{p.name}</option>)}
            {m.name === 'Custom' ? <option value={-1}>Custom</option> : null}
          </select></label>
        {FIELDS.map(([f, label]) => (
          <label class="field"><span class="lbl">{label}</span>
            <input type="number" min={1} value={m[f] as number} onInput={(e) => set(f, Math.max(1, +(e.target as HTMLInputElement).value || 1))} /></label>
        ))}
        <label class="field"><span class="lbl">Weight bytes</span>
          <select value={wBytes} onChange={(e) => setWBytes(+(e.target as HTMLSelectElement).value)}>
            <option value={2}>bf16 (2)</option><option value={1}>int8 / fp8 (1)</option><option value={0.5}>int4 / fp4 (0.5)</option>
          </select></label>
        <label class="field"><span class="lbl">KV bytes</span>
          <select value={kvB} onChange={(e) => setKvB(+(e.target as HTMLSelectElement).value)}>
            <option value={2}>bf16 (2)</option><option value={1}>int8 / fp8 (1)</option>
          </select></label>
        <label class="field"><span class="lbl">Context S <output>{S.toLocaleString('en-US')}</output></span>
          <input type="range" min={10} max={20} step={1} value={Math.log2(S)} onInput={(e) => setS(2 ** +(e.target as HTMLInputElement).value)} /></label>
        <label class="field"><span class="lbl">Chip</span>
          <select value={chip} onChange={(e) => setChip((e.target as HTMLSelectElement).value)}>
            {Object.entries(CHIPS).map(([k, v]) => <option value={k}>{v.name}</option>)}
          </select></label>
      </div>
      <div class="ix-readout">
        <div><span class="k">Total parameters</span><span class="v">{sig(P / 1e9, 4)} B</span></div>
        <div><span class="k">Active per token</span><span class="v">{sig(A / 1e9, 4)} B</span></div>
        <div><span class="k">MLP / attention / vocab</span><span class="v">{sig((m.L * mlpParams(m)) / 1e9)} / {sig((m.L * attnParams(m)) / 1e9)} / {sig(vocabParams(m) / 1e9)} B</span></div>
        <div><span class="k">Training FLOPs per token (6 × active)</span><span class="v">{si(6 * A, 'FLOP')}</span></div>
        <div><span class="k">Forward FLOPs per token at S</span><span class="v">{si(fwd, 'FLOP')}</span></div>
        <div><span class="k">Attention share of forward FLOPs at S</span><span class="v">{Math.round(attnShare * 100)}%</span></div>
        <div><span class="k">KV per token</span><span class="v">{si(kvTok, 'B')}</span></div>
        <div><span class="k">KV per sequence at S</span><span class="v">{si(kvSeq, 'B')}</span></div>
        <div><span class="k">Chips just to hold the weights</span><span class="v">{chipsForWeights}</span></div>
      </div>
      <p class="ix-verdict">
        <b>{si(P * wBytes, 'B')} of weights.</b> On {CHIPS[chip].name}, {chipsForWeights} {chipsForWeights === 1 ? 'chip holds' : 'chips hold'} them.
        Each {S.toLocaleString('en-US')}-token sequence needs {si(kvSeq, 'B')} of KV cache, so the HBM left on{' '}
        {chipsForWeights === 1 ? 'that chip' : `those ${chipsForWeights} chips`} holds about {Math.max(0, Math.floor((chipsForWeights * hbm - P * wBytes) / kvSeq))} sequences.
      </p>
      {m.E > 1 ? (
        <p class="ix-note">MoE: compute per token follows active parameters ({sig(A / 1e9, 3)} B); memory follows total ({sig(P / 1e9, 3)} B). The batch needed to be compute-bound grows by E/k = {sig(m.E / m.k)}.</p>
      ) : null}
    </div>
  );
}
