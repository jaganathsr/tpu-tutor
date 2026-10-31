// An illustrative (not captured) TPU:0 timeline for two FSDP layers, shown with
// and without AllGather overlap. Ops are named the way the XLA Ops row names them.
import { useState } from 'preact/hooks';

type Op = { name: string; start: number; end: number; row: 0 | 1; kind: 'mxu' | 'vpu' | 'comm' | 'idle'; note: string };

function layers(overlap: boolean): Op[] {
  const ops: Op[] = [];
  let t = 0;
  for (let l = 0; l < 2; l++) {
    const ag = 180, mm1 = 140, act = 25, mm2 = 140, rs = 90;
    if (overlap) {
      // The next layer's weight gather is in flight while this layer computes.
      ops.push({ name: `all-gather-start.${l}`, start: t, end: t + 4, row: 0, kind: 'comm', note: 'Starts the async AllGather of the next layer\'s weights.' });
      ops.push({ name: `all-gather (in flight)`, start: t + 4, end: t + 4 + ag, row: 1, kind: 'comm', note: 'Data moving over ICI; the TensorCore is free to compute.' });
      ops.push({ name: `fusion.${10 * l + 1} (dot)`, start: t + 4, end: t + 4 + mm1, row: 0, kind: 'mxu', note: 'First matmul: MXU-bound.' });
      ops.push({ name: `fusion.${10 * l + 2} (gelu)`, start: t + 4 + mm1, end: t + 4 + mm1 + act, row: 0, kind: 'vpu', note: 'Elementwise: VPU.' });
      ops.push({ name: `all-gather-done.${l}`, start: t + 4 + ag, end: t + 8 + ag, row: 0, kind: 'comm', note: 'Done arrives while compute continues: the gather is hidden.' });
      const s = t + 8 + ag;
      ops.push({ name: `fusion.${10 * l + 3} (dot)`, start: s, end: s + mm2, row: 0, kind: 'mxu', note: 'Second matmul.' });
      ops.push({ name: `reduce-scatter (async)`, start: s + mm2, end: s + mm2 + rs, row: 1, kind: 'comm', note: 'Gradient ReduceScatter overlapping the next layer.' });
      t = s + mm2;
    } else {
      ops.push({ name: `all-gather.${l}`, start: t, end: t + ag, row: 0, kind: 'comm', note: 'Synchronous AllGather: the TensorCore waits for the weights.' });
      ops.push({ name: `fusion.${10 * l + 1} (dot)`, start: t + ag, end: t + ag + mm1, row: 0, kind: 'mxu', note: 'First matmul.' });
      ops.push({ name: `fusion.${10 * l + 2} (gelu)`, start: t + ag + mm1, end: t + ag + mm1 + act, row: 0, kind: 'vpu', note: 'Elementwise: VPU.' });
      const s = t + ag + mm1 + act;
      ops.push({ name: `fusion.${10 * l + 3} (dot)`, start: s, end: s + mm2, row: 0, kind: 'mxu', note: 'Second matmul.' });
      ops.push({ name: `reduce-scatter.${l}`, start: s + mm2, end: s + mm2 + rs, row: 0, kind: 'comm', note: 'Synchronous ReduceScatter.' });
      t = s + mm2 + rs;
    }
  }
  return ops;
}

const FILL: Record<Op['kind'], string> = { mxu: 'var(--s1)', vpu: 'var(--s3)', comm: 'var(--s2)', idle: 'var(--rule)' };

export function TraceReader() {
  const [overlap, setOverlap] = useState(false);
  const [sel, setSel] = useState<Op | null>(null);
  const ops = layers(overlap);
  const end = Math.max(...ops.map((o) => o.end));
  const W = 640, x = (v: number) => 120 + (v / 1300) * (W - 140);
  const busy = ops.filter((o) => o.row === 0 && o.kind !== 'comm').reduce((s, o) => s + o.end - o.start, 0);
  const blocking = ops.filter((o) => o.row === 0 && o.kind === 'comm').reduce((s, o) => s + o.end - o.start, 0);

  return (
    <div class="ix">
      <div class="btn-row" style="margin-top:0">
        <button class={`btn ${!overlap ? 'primary' : ''}`} onClick={() => { setOverlap(false); setSel(null); }}>Without overlap</button>
        <button class={`btn ${overlap ? 'primary' : ''}`} onClick={() => { setOverlap(true); setSel(null); }}>With async collectives</button>
      </div>
      <svg viewBox={`0 0 ${W} 130`} role="img" aria-label={`Illustrative trace, ${overlap ? 'with' : 'without'} overlap: step ${end} microseconds.`}>
        <text x="8" y="44">XLA Ops</text>
        <text x="8" y="58" style="font-size:9px">(TPU:0)</text>
        <text x="8" y="92">Async (ICI)</text>
        <line class="axis" x1="120" x2={W - 20} y1="112" y2="112" />
        {[0, 250, 500, 750, 1000, 1250].map((v) => <text x={x(v)} y="126" text-anchor="middle">{v} µs</text>)}
        {ops.map((o) => (
          <rect x={x(o.start)} y={o.row === 0 ? 30 : 78} width={Math.max(2, x(o.end) - x(o.start))} height="26" rx="2"
            fill={FILL[o.kind]} fill-opacity={o.row === 1 ? 0.55 : 1} stroke={sel === o ? 'var(--ink)' : 'var(--paper-2)'} stroke-width={sel === o ? 2 : 1}
            style="cursor:pointer" onClick={() => setSel(o)} />
        ))}
      </svg>
      <div class="legend">
        <span><i style="border-color: var(--s1)" />MXU fusion</span>
        <span><i style="border-color: var(--s3)" />VPU fusion</span>
        <span><i style="border-color: var(--s2)" />collective</span>
        <span>Click an op for details.</span>
      </div>
      <div class="ix-readout">
        <div><span class="k">Step (two layers)</span><span class="v">{end} µs</span></div>
        <div><span class="k">TensorCore computing</span><span class="v">{busy} µs ({Math.round((100 * busy) / end)}%)</span></div>
        <div><span class="k">TensorCore waiting on collectives</span><span class="v">{blocking} µs</span></div>
      </div>
      <p class="ix-verdict">
        {sel ? <><b><code>{sel.name}</code></b> ({sel.end - sel.start} µs): {sel.note}</> : overlap
          ? <><b>Overlapped.</b> Only the short -start and -done markers sit on the TensorCore's row; the transfers run underneath compute.</>
          : <><b>Not overlapped.</b> The all-gather and reduce-scatter bars sit on the XLA Ops row with nothing beside them: the TensorCore is idle for {blocking} µs.</>}
      </p>
      <p class="ix-note">Illustrative timings, not a captured profile. Real traces show the same signature: synchronous collectives as long bars on the XLA Ops row, async ones as -start/-done pairs with compute between them.</p>
    </div>
  );
}
