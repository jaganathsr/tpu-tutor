// Sharding sandbox: a mesh, a 2-D array and a PartitionSpec; optionally a second
// operand to see the matmul case, the collectives it needs and their cost.
import type { JSX } from 'preact';
import { useMemo, useState } from 'preact/hooks';
import chipsJson from '../generated/chips.json';
import { axisProd, localShape, parseMesh, parseSpec, planMatmul, replication, type Mesh, type Spec } from '../lib/sharding';
import { si, time } from '../lib/format';

type ChipJ = { name: string; flops: Record<string, number | null>; W_axis: number | null };
const CHIPS = Object.fromEntries(Object.entries(chipsJson as unknown as Record<string, ChipJ>).filter(([, c]) => c.W_axis)) as Record<string, ChipJ>;
const BLOCK_COLORS = ['var(--s1)', 'var(--s2)', 'var(--s3)'];

function Sub({ letter, axes }: { letter: string; axes: string[] }) {
  return <>{letter}{axes.length ? <sub>{axes.join('')}</sub> : null}</>;
}

function devices(mesh: Mesh) {
  const out: Record<string, number>[] = [];
  const rec = (i: number, cur: Record<string, number>) => {
    if (i === mesh.length) { out.push({ ...cur }); return; }
    for (let v = 0; v < mesh[i].size; v++) rec(i + 1, { ...cur, [mesh[i].name]: v });
  };
  rec(0, {});
  return out;
}

function blockIndex(axes: string[], coord: Record<string, number>, mesh: Mesh) {
  let idx = 0;
  for (const a of axes) idx = idx * mesh.find((m) => m.name === a)!.size + coord[a];
  return idx;
}

function BlockGrid({ shape, spec, mesh, label }: { shape: number[]; spec: Spec; mesh: Mesh; label: string }) {
  const rows = axisProd(spec[0], mesh), cols = axisProd(spec[1], mesh);
  const devs = devices(mesh);
  const holders: number[][][] = Array.from({ length: rows }, () => Array.from({ length: cols }, () => []));
  devs.forEach((d, i) => holders[blockIndex(spec[0], d, mesh)][blockIndex(spec[1], d, mesh)].push(i));
  const W = 300, H = 200;
  const bw = W / cols, bh = H / rows;
  const cells: JSX.Element[] = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const ids = holders[r][c];
    cells.push(
      <g>
        <rect x={c * bw} y={r * bh} width={bw} height={bh} fill={BLOCK_COLORS[(r + c) % 3]} fill-opacity="0.18" stroke="var(--rule-strong)" />
        {bw > 34 && bh > 16 ? (
          <text x={c * bw + bw / 2} y={r * bh + bh / 2 + 4} text-anchor="middle" style="font-size:10px; fill: var(--ink)">
            {ids.length <= 4 ? ids.map((i) => `d${i}`).join(' ') : `${ids.length} devices`}
          </text>
        ) : null}
      </g>,
    );
  }
  return (
    <figure style="margin:0">
      <svg viewBox={`-2 -2 ${W + 4} ${H + 4}`} style="max-width:320px" role="img" aria-label={`${label}: ${rows} by ${cols} blocks`}>{cells}</svg>
      <figcaption class="ix-note">{label}: {rows} × {cols} blocks of [{shape[0] / rows}, {shape[1] / cols}]. Each block lists the devices holding it.</figcaption>
    </figure>
  );
}

export function ShardingSandbox() {
  const [meshS, setMeshS] = useState('X:4, Y:2');
  const [shapeS, setShapeS] = useState('8192, 28672');
  const [specA, setSpecA] = useState("P('X', 'Y')");
  const [useB, setUseB] = useState(true);
  const [kS, setKS] = useState('8192');
  const [specB, setSpecB] = useState("P('Y', None)");
  const [chip, setChip] = useState('7x');
  const [bytesPer, setBytesPer] = useState(2);

  const r = useMemo(() => {
    const mesh = parseMesh(meshS);
    if (typeof mesh === 'string') return { err: mesh };
    const shape = shapeS.split(',').map((x) => +x.trim());
    if (shape.length !== 2 || shape.some((x) => !(x > 0))) return { err: 'Enter the array shape as two positive integers, e.g. 1024, 4096.' };
    const sA = parseSpec(specA, 2, mesh);
    if (typeof sA === 'string') return { err: `A: ${sA}` };
    const local = localShape(shape, sA, mesh);
    if (local.some((x) => !Number.isInteger(x))) return { err: `A's dimensions are not divisible by the mesh axes they are sharded over.` };
    let plan, sB: Spec | undefined, K = 0;
    if (useB) {
      K = +kS;
      const parsed = parseSpec(specB, 2, mesh);
      if (typeof parsed === 'string') return { err: `B: ${parsed}` };
      sB = parsed;
      const c = CHIPS[chip];
      const C = (bytesPer === 1 ? c.flops.fp8 ?? c.flops.int8 : c.flops.bf16) ?? c.flops.bf16!;
      plan = planMatmul(shape[0], shape[1], K, sA, sB, mesh, bytesPer, c.W_axis!, C);
      if (typeof plan === 'string') return { err: plan };
    }
    return { mesh, shape, sA, local, sB, K, plan };
  }, [meshS, shapeS, specA, useB, kS, specB, chip, bytesPer]);

  return (
    <div class="ix">
      <div class="ix-controls">
        <label class="field"><span class="lbl">Mesh</span>
          <input type="text" value={meshS} onInput={(e) => setMeshS((e.target as HTMLInputElement).value)} /></label>
        <label class="field"><span class="lbl">A shape [I, J]</span>
          <input type="text" value={shapeS} onInput={(e) => setShapeS((e.target as HTMLInputElement).value)} /></label>
        <label class="field"><span class="lbl">A PartitionSpec</span>
          <input type="text" value={specA} onInput={(e) => setSpecA((e.target as HTMLInputElement).value)} /></label>
        <label class="field"><span class="lbl">Multiply by B[J, K]</span>
          <span><input type="checkbox" checked={useB} onChange={() => setUseB(!useB)} /> on</span></label>
        {useB ? (
          <>
            <label class="field"><span class="lbl">K</span>
              <input type="text" value={kS} onInput={(e) => setKS((e.target as HTMLInputElement).value)} /></label>
            <label class="field"><span class="lbl">B PartitionSpec</span>
              <input type="text" value={specB} onInput={(e) => setSpecB((e.target as HTMLInputElement).value)} /></label>
            <label class="field"><span class="lbl">Chip</span>
              <select value={chip} onChange={(e) => setChip((e.target as HTMLSelectElement).value)}>
                {Object.entries(CHIPS).map(([k, v]) => <option value={k}>{v.name}</option>)}
              </select></label>
            <label class="field"><span class="lbl">Bytes per element</span>
              <select value={bytesPer} onChange={(e) => setBytesPer(+(e.target as HTMLSelectElement).value)}>
                <option value={2}>2 (bf16)</option><option value={1}>1 (fp8 / int8)</option>
              </select></label>
          </>
        ) : null}
      </div>

      {'err' in r ? (
        <p class="result wrong" role="status"><span class="verdict">Cannot shard this.</span> {r.err}</p>
      ) : (
        <>
          <div class="ix-readout">
            <div><span class="k">A in sharding notation</span><span class="v">A[<Sub letter="I" axes={r.sA[0]} />, <Sub letter="J" axes={r.sA[1]} />]</span></div>
            <div><span class="k">Shard per device</span><span class="v">[{r.local.join(', ')}]</span></div>
            <div><span class="k">Bytes per device</span><span class="v">{si(r.local[0] * r.local[1] * bytesPer, 'B')}</span></div>
            <div><span class="k">Copies of each shard</span><span class="v">{replication(r.sA, r.mesh)}</span></div>
            <div><span class="k">Devices</span><span class="v">{r.mesh.reduce((p, m) => p * m.size, 1)}</span></div>
          </div>
          <div style="display:flex;gap:2rem;flex-wrap:wrap;margin-top:1rem">
            <BlockGrid shape={r.shape} spec={r.sA} mesh={r.mesh} label="A" />
            {r.sB ? <BlockGrid shape={[r.shape[1], r.K]} spec={r.sB} mesh={r.mesh} label="B" /> : null}
          </div>
          {r.plan && typeof r.plan !== 'string' ? (
            <div style="margin-top:1rem">
              <p class="ix-verdict">
                <b>Case {r.plan.caseNo}.</b> {r.plan.caseName}
              </p>
              {r.plan.steps.length ? (
                <table class="spec-table">
                  <thead><tr><th>Collective</th><th class="r">Bytes per device</th><th class="r">Time</th></tr></thead>
                  <tbody>
                    {r.plan.steps.map((s) => (
                      <tr><td style="white-space:normal">{s.what}</td><td class="r">{si(s.bytes, 'B')}</td><td class="r">{time(s.time)}</td></tr>
                    ))}
                  </tbody>
                </table>
              ) : null}
              <div class="ix-readout">
                <div><span class="k">Result</span><span class="v">{r.plan.out.replace(/_\{(\w+)\}/g, '_$1')}</span></div>
                <div><span class="k">Local matmul</span><span class="v">{si(r.plan.flopsPerDevice, 'FLOP')}, {time(r.plan.tMath)}</span></div>
                <div><span class="k">Communication</span><span class="v">{time(r.plan.tComms)}</span></div>
              </div>
              <p class="ix-verdict">
                <b>{r.plan.tComms > r.plan.tMath ? 'Communication-bound' : 'Compute-bound'}</b>
                {r.plan.tComms ? ` if nothing overlaps: ${time(r.plan.tComms)} of collectives against ${time(r.plan.tMath)} of math.` : '.'}
              </p>
            </div>
          ) : null}
        </>
      )}
      <p class="ix-note">
        Accepts P('X', None), ('X','Y'), None and similar. Collective times use the Scaling Book's ring formulas with W_ici per axis from
        scripts/specs.py (Ironwood 200 GB/s), full rings with wraparound, and a 1 µs-per-hop latency floor. Overlap is not modelled.
      </p>
    </div>
  );
}
