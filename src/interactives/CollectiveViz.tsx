// Ring collectives, step by step: which shards each device holds after each
// step, the busiest link's load, and the resulting time from
//   T = (busiest directional link load in chunks) * chunk / W_link + steps * 1 us.
import { useEffect, useMemo, useState } from 'preact/hooks';
import { bytes as fmtBytes, time } from '../lib/format';

type Kind = 'ag-uni' | 'ag-bi' | 'rs-bi' | 'a2a-bi';
const KINDS: Record<Kind, string> = {
  'ag-uni': 'AllGather, unidirectional ring',
  'ag-bi': 'AllGather, bidirectional ring',
  'rs-bi': 'ReduceScatter, bidirectional ring',
  'a2a-bi': 'AllToAll, bidirectional ring',
};
const LINK = { v5e: 45e9, '7x': 100e9 } as const;

function ringDist(a: number, b: number, n: number) {
  const f = (b - a + n) % n;
  return Math.min(f, n - f);
}

/** Busiest directional link (in chunks) and step count, per collective. */
function linkLoad(kind: Kind, n: number) {
  if (kind === 'ag-uni') return { load: n - 1, steps: n - 1, chunkDiv: n };
  if (kind === 'ag-bi' || kind === 'rs-bi') return { load: Math.ceil((n - 1) / 2), steps: Math.ceil((n - 1) / 2), chunkDiv: n };
  // AllToAll: route every (src, dst) pair the short way; split ties alternately.
  const load = new Map<string, number>();
  let tie = 0;
  for (let s = 0; s < n; s++) for (let d = 0; d < n; d++) {
    if (s === d) continue;
    const f = (d - s + n) % n, b = n - f;
    const dir = f < b ? 1 : f > b ? -1 : (tie++ % 2 ? 1 : -1);
    const hops = Math.min(f, b);
    for (let h = 0, at = s; h < hops; h++, at = (at + dir + n) % n) {
      const key = `${at}${dir > 0 ? '+' : '-'}`;
      load.set(key, (load.get(key) ?? 0) + 1);
    }
  }
  return { load: Math.max(...load.values()), steps: Math.floor(n / 2), chunkDiv: n * n };
}

export function CollectiveViz() {
  const [kind, setKind] = useState<Kind>('ag-bi');
  const [n, setN] = useState(8);
  const [t, setT] = useState(0);
  const [logV, setLogV] = useState(27); // 128 MiB
  const [chip, setChip] = useState<keyof typeof LINK>('7x');
  const [playing, setPlaying] = useState(false);

  const { load, steps, chunkDiv } = useMemo(() => linkLoad(kind, n), [kind, n]);
  useEffect(() => { setT(0); setPlaying(false); }, [kind, n]);
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setT((x) => (x >= steps ? x : x + 1)), 700);
    return () => clearInterval(id);
  }, [playing, steps]);
  useEffect(() => { if (t >= steps) setPlaying(false); }, [t, steps]);

  const V = 2 ** logV;
  const W1 = LINK[chip];
  const chunk = V / chunkDiv;
  const tBw = (load * chunk) / W1;
  const tLat = steps * 1e-6;
  const formula =
    kind === 'ag-uni' ? { text: '(N − 1)/N · V / W_link', v: ((n - 1) / n) * V / W1 }
    : kind === 'a2a-bi' ? { text: 'V / (4 W_ici)', v: V / (4 * 2 * W1) }
    : { text: 'V / W_ici', v: V / (2 * W1) };

  // What each device holds after step t.
  const holds = (d: number): { label: string; fresh: boolean }[] => {
    if (kind === 'ag-uni') return Array.from({ length: Math.min(t, n - 1) + 1 }, (_, s) => ({ label: String((d - s + n) % n), fresh: s === t && t > 0 }));
    if (kind === 'ag-bi') {
      const set = new Map<number, boolean>();
      for (let s = 0; s <= Math.min(t, steps); s++) { set.set((d - s + n) % n, s === t && t > 0); set.set((d + s) % n, s === t && t > 0); }
      return [...set].sort((a, b) => a[0] - b[0]).map(([k, f]) => ({ label: String(k), fresh: f }));
    }
    if (kind === 'rs-bi') {
      const got = Math.min(n, 1 + 2 * t);
      return [{ label: `Σ${got}/${n}`, fresh: t > 0 }];
    }
    // AllToAll: chunks that have arrived at d.
    return Array.from({ length: n }, (_, s) => s).filter((s) => ringDist(s, d, n) <= t).map((s) => ({ label: `${s}→${d}`, fresh: ringDist(s, d, n) === t && t > 0 }));
  };

  const R = 128, cx = 190, cy = 178;
  const pos = (i: number) => [cx + R * Math.sin((2 * Math.PI * i) / n), cy - R * Math.cos((2 * Math.PI * i) / n)];

  return (
    <div class="ix">
      <div class="ix-controls">
        <label class="field"><span class="lbl">Collective</span>
          <select value={kind} onChange={(e) => setKind((e.target as HTMLSelectElement).value as Kind)}>
            {Object.entries(KINDS).map(([k, v]) => <option value={k}>{v}</option>)}
          </select></label>
        <label class="field"><span class="lbl">Devices on the ring <output>{n}</output></span>
          <input type="range" min={4} max={8} value={n} onInput={(e) => setN(+(e.target as HTMLInputElement).value)} /></label>
        <label class="field"><span class="lbl">Total array V <output>{fmtBytes(V)}</output></span>
          <input type="range" min={14} max={32} value={logV} onInput={(e) => setLogV(+(e.target as HTMLInputElement).value)} /></label>
        <label class="field"><span class="lbl">Link</span>
          <select value={chip} onChange={(e) => setChip((e.target as HTMLSelectElement).value as any)}>
            <option value="v5e">v5e, 45 GB/s one way (book)</option>
            <option value="7x">Ironwood, 100 GB/s one way (docs)</option>
          </select></label>
      </div>
      <svg viewBox="0 0 580 350" role="img" aria-label={`${KINDS[kind]} on ${n} devices, step ${t} of ${steps}.`}>
        <circle cx={cx} cy={cy} r={R} fill="none" stroke="var(--rule-strong)" stroke-width="1.5" />
        {Array.from({ length: n }, (_, d) => {
          const [x, y] = pos(d);
          const h = holds(d);
          return (
            <g>
              <rect x={x - 34} y={y - 22} width="68" height="44" rx="3" fill="var(--paper)" stroke="var(--rule-strong)" />
              <text x={x} y={y - 26} text-anchor="middle" style="font-size:10px">device {d}</text>
              {kind === 'rs-bi' ? (
                <g>
                  <rect x={x - 28} y={y - 4} width="56" height="10" fill="var(--paper-2)" stroke="var(--rule)" />
                  <rect x={x - 28} y={y - 4} width={56 * Math.min(1, (1 + 2 * t) / n)} height="10" fill="var(--s1)" />
                  <text x={x} y={y + 18} text-anchor="middle" style="font-size:9px; fill: var(--ink)">block {d}: {h[0].label}</text>
                </g>
              ) : (
                h.slice(0, 8).map((c, i) => (
                  <g>
                    <rect x={x - 30 + (i % 4) * 15} y={y - 16 + Math.floor(i / 4) * 17} width="13" height="14" rx="1.5" fill={c.fresh ? 'var(--s2)' : 'var(--s1)'} />
                    <text x={x - 23.5 + (i % 4) * 15} y={y - 5.5 + Math.floor(i / 4) * 17} text-anchor="middle" style="font-size:7px; fill: var(--on-accent)">
                      {kind === 'a2a-bi' ? c.label.split('→')[0] : c.label}
                    </text>
                  </g>
                ))
              )}
            </g>
          );
        })}
        <g transform="translate(372, 40)">
          <text y="0" style="fill: var(--ink); font-weight: 600">Step {t} of {steps}</text>
          <text y="22">Chunk size: {fmtBytes(chunk)}</text>
          <text y="42">Busiest link carries {load} chunks</text>
          <text y="62">Bandwidth time: {time(tBw)}</text>
          <text y="82">Latency, {steps} hops: {time(tLat)}</text>
          <text y="102" style="fill: var(--ink)">Total: {time(tBw + tLat)}</text>
          <text y="130">Book formula: {formula.text}</text>
          <text y="148">W_ici = 2 × one link</text>
          <text y="166" style="fill: var(--ink)">= {time(formula.v)}</text>
          <text y="194">{kind === 'a2a-bi' ? 'Squares: origin of chunks that' : kind === 'rs-bi' ? 'Bar: partial sums folded into' : 'Squares: shards held'}</text>
          <text y="210">{kind === 'a2a-bi' ? 'have arrived here.' : kind === 'rs-bi' ? 'this device\'s block.' : '(orange: arrived this step).'}</text>
        </g>
      </svg>
      <div class="btn-row">
        <button class="btn primary" onClick={() => (t >= steps ? (setT(0), setPlaying(true)) : setPlaying(!playing))}>
          {playing ? 'Pause' : t >= steps ? 'Replay' : 'Play'}
        </button>
        <button class="btn" onClick={() => setT((x) => Math.min(steps, x + 1))} disabled={t >= steps}>Step</button>
        <button class="btn" onClick={() => { setPlaying(false); setT(0); }}>Reset</button>
      </div>
      <p class="ix-verdict">
        <b>{tLat > tBw ? 'Latency-bound' : 'Bandwidth-bound'}.</b>{' '}
        {kind === 'a2a-bi'
          ? 'An AllToAll moves each chunk only as far as its destination, so the busiest link carries about a quarter of what an AllGather sends.'
          : 'The bandwidth time does not depend on the number of devices: more devices means more steps, but smaller chunks.'}
      </p>
    </div>
  );
}
