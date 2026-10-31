// Point-to-point routing on a ring or 2D torus: hop counts, diameter and the
// latency + bandwidth time of a transfer, with and without wraparound links.
// Dimension-order routing (X first, then Y), the shortest direction per axis.
import type { JSX } from 'preact';
import { useState } from 'preact/hooks';
import { time, bytes as fmtBytes } from '../lib/format';

const LINKS = {
  v5e: { name: 'TPU v5e (45 GB/s per link, one way; book)', bw: 45e9 },
  '7x': { name: 'Ironwood (100 GB/s per link, one way; docs)', bw: 100e9 },
};

function axisSteps(a: number, b: number, n: number, wrap: boolean): number[] {
  // Returns the sequence of +1/-1 moves along one axis.
  const fwd = (b - a + n) % n, back = (a - b + n) % n;
  if (!wrap) return Array(Math.abs(b - a)).fill(Math.sign(b - a));
  return fwd <= back ? Array(fwd).fill(1) : Array(back).fill(-1);
}

export function TorusExplorer({ mode = '2d' }: { mode?: 'ring' | '2d' }) {
  const [kind, setKind] = useState<'ring' | '2d'>(mode);
  const [X, setX] = useState(kind === 'ring' ? 8 : 4);
  const [Y, setY] = useState(4);
  const [wrapX, setWrapX] = useState(true);
  const [wrapY, setWrapY] = useState(true);
  const [src, setSrc] = useState<[number, number]>([0, 0]);
  const [dst, setDst] = useState<[number, number]>([3, 3]);
  const [pick, setPick] = useState<'src' | 'dst'>('dst');
  const [logV, setLogV] = useState(24); // 16 MiB
  const [chip, setChip] = useState<keyof typeof LINKS>('v5e');

  const ny = kind === 'ring' ? 1 : Y;
  const clamp = (p: [number, number]): [number, number] => [Math.min(p[0], X - 1), Math.min(p[1], ny - 1)];
  const s = clamp(src), d = clamp(dst);
  const wy = kind === 'ring' ? false : wrapY;

  const sx = axisSteps(s[0], d[0], X, wrapX);
  const sy = axisSteps(s[1], d[1], ny, wy);
  const path: [number, number][] = [s];
  let cur: [number, number] = [...s];
  for (const st of sx) { cur = [(cur[0] + st + X) % X, cur[1]]; path.push(cur); }
  for (const st of sy) { cur = [cur[0], (cur[1] + st + ny) % ny]; path.push(cur); }
  const hops = path.length - 1;
  const diam = (wrapX ? Math.floor(X / 2) : X - 1) + (kind === 'ring' ? 0 : wy ? Math.floor(ny / 2) : ny - 1);

  const V = 2 ** logV;
  const bw = LINKS[chip].bw;
  const tBw = V / bw, tLat = hops * 1e-6;

  const cell = 56, pad = 34;
  const W = pad * 2 + X * cell, H = pad * 2 + ny * cell + (kind === 'ring' ? 10 : 0);
  const cx = (i: number) => pad + i * cell + cell / 2;
  const cy = (j: number) => pad + j * cell + cell / 2;
  const onPath = (i: number, j: number) => path.some((p) => p[0] === i && p[1] === j);

  const edges: JSX.Element[] = [];
  const pathEdge = (a: [number, number], b: [number, number]) =>
    path.some((p, k) => k > 0 && ((path[k - 1][0] === a[0] && path[k - 1][1] === a[1] && p[0] === b[0] && p[1] === b[1]) ||
      (path[k - 1][0] === b[0] && path[k - 1][1] === b[1] && p[0] === a[0] && p[1] === a[1])));
  for (let j = 0; j < ny; j++) for (let i = 0; i < X; i++) {
    if (i + 1 < X) edges.push(<line x1={cx(i)} y1={cy(j)} x2={cx(i + 1)} y2={cy(j)} stroke={pathEdge([i, j], [i + 1, j]) ? 'var(--s2)' : 'var(--rule-strong)'} stroke-width={pathEdge([i, j], [i + 1, j]) ? 4 : 1.5} />);
    if (j + 1 < ny) edges.push(<line x1={cx(i)} y1={cy(j)} x2={cx(i)} y2={cy(j + 1)} stroke={pathEdge([i, j], [i, j + 1]) ? 'var(--s2)' : 'var(--rule-strong)'} stroke-width={pathEdge([i, j], [i, j + 1]) ? 4 : 1.5} />);
  }
  // Wraparound links drawn as stubs leaving each edge.
  const wraps: JSX.Element[] = [];
  if (wrapX && X > 2) for (let j = 0; j < ny; j++) {
    const used = pathEdge([0, j], [X - 1, j]);
    const st = used ? 'var(--s2)' : 'var(--s1)';
    wraps.push(<line x1={cx(0)} y1={cy(j)} x2={cx(0) - 26} y2={cy(j)} stroke={st} stroke-width={used ? 4 : 1.5} stroke-dasharray="4 3" />);
    wraps.push(<line x1={cx(X - 1)} y1={cy(j)} x2={cx(X - 1) + 26} y2={cy(j)} stroke={st} stroke-width={used ? 4 : 1.5} stroke-dasharray="4 3" />);
  }
  if (kind === '2d' && wy && ny > 2) for (let i = 0; i < X; i++) {
    const used = pathEdge([i, 0], [i, ny - 1]);
    const st = used ? 'var(--s2)' : 'var(--s1)';
    wraps.push(<line x1={cx(i)} y1={cy(0)} x2={cx(i)} y2={cy(0) - 26} stroke={st} stroke-width={used ? 4 : 1.5} stroke-dasharray="4 3" />);
    wraps.push(<line x1={cx(i)} y1={cy(ny - 1)} x2={cx(i)} y2={cy(ny - 1) + 26} stroke={st} stroke-width={used ? 4 : 1.5} stroke-dasharray="4 3" />);
  }

  const click = (i: number, j: number) => {
    if (pick === 'src') { setSrc([i, j]); setPick('dst'); } else { setDst([i, j]); setPick('src'); }
  };

  return (
    <div class="ix">
      <div class="ix-controls">
        <label class="field"><span class="lbl">Topology</span>
          <select value={kind} onChange={(e) => setKind((e.target as HTMLSelectElement).value as any)}>
            <option value="ring">1D ring</option>
            <option value="2d">2D torus</option>
          </select></label>
        <label class="field"><span class="lbl">X <output>{X}</output></span>
          <input type="range" min={2} max={kind === 'ring' ? 16 : 8} value={X} onInput={(e) => setX(+(e.target as HTMLInputElement).value)} /></label>
        {kind === '2d' ? (
          <label class="field"><span class="lbl">Y <output>{Y}</output></span>
            <input type="range" min={2} max={8} value={Y} onInput={(e) => setY(+(e.target as HTMLInputElement).value)} /></label>
        ) : null}
        <label class="field"><span class="lbl">Wraparound</span>
          <span style="display:flex;gap:1rem">
            <label><input type="checkbox" checked={wrapX} onChange={() => setWrapX(!wrapX)} /> X</label>
            {kind === '2d' ? <label><input type="checkbox" checked={wrapY} onChange={() => setWrapY(!wrapY)} /> Y</label> : null}
          </span></label>
        <label class="field"><span class="lbl">Bytes to send <output>{fmtBytes(V)}</output></span>
          <input type="range" min={10} max={32} value={logV} onInput={(e) => setLogV(+(e.target as HTMLInputElement).value)} /></label>
        <label class="field"><span class="lbl">Link</span>
          <select value={chip} onChange={(e) => setChip((e.target as HTMLSelectElement).value as any)}>
            {Object.entries(LINKS).map(([k, v]) => <option value={k}>{v.name}</option>)}
          </select></label>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={`max-width:${W * 1.4}px`} role="img"
        aria-label={`${hops} hops from chip (${s[0]},${s[1]}) to (${d[0]},${d[1]}); diameter ${diam}.`}>
        {wraps}
        {edges}
        {Array.from({ length: ny }, (_, j) => Array.from({ length: X }, (_, i) => {
          const isS = s[0] === i && s[1] === j, isD = d[0] === i && d[1] === j;
          return (
            <g style="cursor:pointer" onClick={() => click(i, j)}>
              <rect x={cx(i) - 15} y={cy(j) - 15} width="30" height="30" rx="3"
                fill={isS ? 'var(--s1)' : isD ? 'var(--s2)' : onPath(i, j) ? 'var(--accent-tint)' : 'var(--paper)'}
                stroke={isS || isD ? 'none' : 'var(--rule-strong)'} />
              <text x={cx(i)} y={cy(j) + 4} text-anchor="middle" style={`fill:${isS || isD ? 'var(--on-accent)' : 'var(--ink-2)'};font-size:10px`}>
                {kind === 'ring' ? i : `${i},${j}`}
              </text>
            </g>
          );
        }))}
      </svg>
      <div class="legend">
        <span><i style="border-color: var(--s1)" />source</span>
        <span><i style="border-color: var(--s2)" />destination and route</span>
        <span><i class="dash" style="border-color: var(--s1)" />wraparound link</span>
        <span>Click a chip to set the {pick === 'src' ? 'source' : 'destination'}.</span>
      </div>
      <div class="ix-readout">
        <div><span class="k">Hops on this route</span><span class="v">{hops}</span></div>
        <div><span class="k">Diameter of this slice</span><span class="v">{diam}</span></div>
        <div><span class="k">Latency, 1 µs per hop</span><span class="v">{time(tLat)}</span></div>
        <div><span class="k">Bandwidth time, one link</span><span class="v">{time(tBw)}</span></div>
        <div><span class="k">Total</span><span class="v">{time(tLat + tBw)}</span></div>
      </div>
      <p class="ix-verdict">
        <b>{tLat > tBw ? 'Latency-bound.' : 'Bandwidth-bound.'}</b>{' '}
        {tLat > tBw
          ? `${hops} hops cost more than moving ${fmtBytes(V)}. Fewer hops matter more than more bandwidth.`
          : `Moving ${fmtBytes(V)} over one link dominates. Using several links in parallel matters more than hop count.`}
      </p>
    </div>
  );
}
