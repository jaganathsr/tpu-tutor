// Decodes one HLO instruction line as printed in TPU profiles and HLO dumps.
// Grammar follows the Scaling Book ch. 9 "How to read an XLA op" and the
// OpenXLA tiled-layout docs: name = dtype[dims]{minor_to_major:T(tiles)S(space)} op(operands), attrs
import { useMemo, useState } from 'preact/hooks';
import { si } from '../lib/format';

const EXAMPLES: { label: string; line: string }[] = [
  {
    label: 'Scaling Book ch. 9: a fusion that calls a reduce-scatter',
    line: '%fusion.3 = bf16[32,32,4096]{2,1,0:T(8,128)(2,1)S(1)} fusion(bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32), kind=kCustom, calls=%all-reduce-scatter.3',
  },
  {
    label: 'Scaling Book ch. 9, problem 1: a loop fusion',
    line: '%fusion.1 = bf16[4096]{0:T(1024)(128)(2,1)} fusion(bf16[4096,8192]{1,0:T(8,128)(2,1)} %param.1, bf16[8192]{0:T(1024)(128)(2,1)} %reduce.6), kind=kLoop, calls=%fused_computation.1',
  },
  {
    label: 'Constructed example: an all-reduce with replica groups',
    line: '%all-reduce.5 = bf16[8,32768]{1,0:T(8,128)(2,1)} all-reduce(bf16[8,32768]{1,0:T(8,128)(2,1)} %fusion.2), channel_id=3, replica_groups={{0,16,32,48,64,80,96,112},{1,17,33,49,65,81,97,113}}, to_apply=%add',
  },
  {
    label: 'Chapter 9 dump (CPU): the inserted all-reduce',
    line: '%all-reduce.1 = f32[4,512]{1,0} all-reduce(%dot.1), channel_id=1, replica_groups=[2,2]<=[4], use_global_device_ids=true, to_apply=%add.clone_promoted',
  },
];

const DTYPE_BYTES: Record<string, number> = { f64: 8, f32: 4, s32: 4, u32: 4, bf16: 2, f16: 2, s16: 2, s8: 1, u8: 1, f8e4m3fn: 1, f8e5m2: 1, pred: 1, s4: 0.5, u4: 0.5 };
const KINDS: Record<string, string> = {
  kLoop: 'loop fusion: elementwise work fused into one pass over the output (VPU work)',
  kInput: 'input fusion: a reduction fused with the elementwise ops that feed it',
  kOutput: 'output fusion: an op (often a matmul) fused with elementwise ops on its output',
  kCustom: 'custom fusion: wraps a special computation, here a collective or a kernel',
};

function parse(line: string) {
  const m = line.trim().match(/^(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*(\w+)\[([\d,]*)\](?:\{([^}]*)\})?\s+([\w\-]+)\((.*)$/);
  if (!m) return null;
  const [, name, dtype, dimsS, layoutS, op, rest] = m;
  const dims = dimsS ? dimsS.split(',').map(Number) : [];
  let minorToMajor: number[] = [], tiles: number[][] = [], space: string | null = null;
  if (layoutS) {
    const [order, extra = ''] = layoutS.split(':');
    minorToMajor = order.split(',').filter(Boolean).map(Number);
    tiles = [...extra.matchAll(/T\(([\d,]+)\)/g)].map((t) => t[1].split(',').map(Number));
    // Tiles are written as one group of parenthesised sizes after T: T(8,128)(2,1).
    const tgroup = extra.match(/T((?:\([\d,]+\))+)/);
    if (tgroup) tiles = [...tgroup[1].matchAll(/\(([\d,]+)\)/g)].map((t) => t[1].split(',').map(Number));
    const sp = extra.match(/S\((\d+)\)/);
    space = sp ? sp[1] : null;
  }
  const attrs: Record<string, string> = {};
  for (const a of rest.matchAll(/(\w+)=(\{\{[^}]*\}(?:,\{[^}]*\})*\}|\[[^\]]*\]<=\[[^\]]*\](?:T\([\d,]+\))?|[^,\s)]+)/g)) attrs[a[1]] = a[2];
  const operands = [...rest.matchAll(/(\w+)\[([\d,]*)\]\{[^}]*\}\s+%([\w.\-]+)/g)].map((o) => ({ dtype: o[1], dims: o[2], name: o[3] }));
  const bareOperands = operands.length ? [] : [...(rest.split(')')[0] ?? '').matchAll(/%([\w.\-]+)/g)].map((o) => o[1]);
  return { name, dtype, dims, minorToMajor, tiles, space, op, attrs, operands, bareOperands };
}

function paddedBytes(dims: number[], m2m: number[], tile: number[] | undefined, bytes: number) {
  if (!tile || !dims.length) return dims.reduce((p, d) => p * d, 1) * bytes;
  // The tile applies to the most-minor dimensions, in major-to-minor order.
  const order = [...m2m].reverse(); // major to minor
  const padded = dims.slice();
  const t = tile.slice(-order.length);
  for (let i = 0; i < t.length; i++) {
    const dim = order[order.length - t.length + i];
    padded[dim] = Math.ceil(dims[dim] / t[i]) * t[i];
  }
  return padded.reduce((p, d) => p * d, 1) * bytes;
}

function groupInfo(rg: string | undefined) {
  if (!rg) return null;
  if (rg.startsWith('{{')) {
    const first = rg.slice(2).split('}')[0].split(',').filter(Boolean);
    const groups = rg.split('},{').length;
    return `${groups} group${groups > 1 ? 's' : ''} shown; first has ${first.length} devices (${first.slice(0, 4).join(', ')}${first.length > 4 ? ', …' : ''}). A ${first.length}-way collective; stride ${first.length > 1 ? +first[1] - +first[0] : 0} between members.`;
  }
  const m = rg.match(/^\[([\d,]+)\]<=\[([\d,]+)\]/);
  if (m) {
    const [g, size] = m[1].split(',').map(Number);
    return `Iota form: ${g} groups of ${size} devices drawn from the device list [${m[2]}]. A ${size}-way collective.`;
  }
  return rg;
}

export function HloDecoder() {
  const [line, setLine] = useState(EXAMPLES[0].line);
  const p = useMemo(() => parse(line), [line]);
  const bytes = p ? DTYPE_BYTES[p.dtype] ?? null : null;
  const logical = p && bytes ? p.dims.reduce((a, d) => a * d, 1) * bytes : null;
  const padded = p && bytes ? paddedBytes(p.dims, p.minorToMajor, p.tiles[0], bytes) : null;
  const calls = p?.attrs.calls ?? '';

  return (
    <div class="ix">
      <div class="ix-controls" style="grid-template-columns: 1fr">
        <label class="field"><span class="lbl">Example</span>
          <select onChange={(e) => setLine(EXAMPLES[+(e.target as HTMLSelectElement).value].line)}>
            {EXAMPLES.map((x, i) => <option value={i}>{x.label}</option>)}
          </select></label>
        <label class="field"><span class="lbl">HLO line (paste your own)</span>
          <textarea style="min-height:4.5rem;font-family:var(--font-mono);font-size:var(--fs-xsmall)" value={line} onInput={(e) => setLine((e.target as HTMLTextAreaElement).value)} /></label>
      </div>
      {!p ? (
        <p class="result wrong"><span class="verdict">Could not parse this line.</span> Expected: %name = dtype[dims]{'{'}layout{'}'} op(operands), attributes.</p>
      ) : (
        <table class="decoder-table">
          <tbody>
            <tr><th>Name</th><td><code>%{p.name}</code></td></tr>
            <tr><th>Op</th><td><code>{p.op}</code>{p.op === 'fusion' ? ': several HLO ops compiled as one kernel; on TPU a fusion holds at most one matmul plus the elementwise work around it.' : ''}</td></tr>
            <tr><th>Output</th><td><code>{p.dtype}[{p.dims.join(',')}]</code>{logical ? `: ${si(logical, 'B')} logical` : ''}</td></tr>
            {p.minorToMajor.length ? (
              <tr><th>Layout</th><td>Minor-to-major <code>{'{'}{p.minorToMajor.join(',')}{'}'}</code>: dimension {p.minorToMajor[0]} varies fastest in memory{p.minorToMajor.length > 1 ? `, then ${p.minorToMajor.slice(1).join(', then ')}` : ''}.</td></tr>
            ) : null}
            {p.tiles.length ? (
              <tr><th>Tiling</th><td>
                {p.tiles.map((t) => `(${t.join(',')})`).join(' then ')}.{' '}
                {p.tiles.length > 1 && p.tiles[p.tiles.length - 1][0] > 1 ? `The inner (${p.tiles[p.tiles.length - 1].join(',')}) packs ${p.tiles[p.tiles.length - 1][0]} values into each 32-bit word. ` : ''}
                {padded && logical && padded > logical ? `Padding to the outer tile makes the buffer ${si(padded, 'B')}, ${(padded / logical).toFixed(2)}× the logical size.` : 'No padding: the minor dimensions are already multiples of the tile.'}
              </td></tr>
            ) : null}
            <tr><th>Memory space</th><td>{p.space === '1' ? 'S(1): VMEM, on chip.' : p.space && p.space !== '0' ? `S(${p.space}): another memory space; the Scaling Book does not say which.` : 'No S(n), or S(0): HBM.'}</td></tr>
            {p.attrs.kind ? <tr><th>Fusion kind</th><td><code>{p.attrs.kind}</code>: {KINDS[p.attrs.kind] ?? 'unrecognized kind'}</td></tr> : null}
            {calls ? (
              <tr><th>Calls</th><td><code>{calls}</code>{/all-reduce-scatter|reduce-scatter/.test(calls) ? ': a ReduceScatter collective.' : /all-gather/.test(calls) ? ': an AllGather collective.' : /all-reduce/.test(calls) ? ': an AllReduce.' : ': the fused computation body; open it in the graph viewer.'}</td></tr>
            ) : null}
            {p.attrs.replica_groups ? <tr><th>Replica groups</th><td>{groupInfo(p.attrs.replica_groups)}</td></tr> : null}
            {p.operands.length ? (
              <tr><th>Operands</th><td>{p.operands.map((o) => <div><code>%{o.name}</code>: {o.dtype}[{o.dims}]</div>)}</td></tr>
            ) : p.bareOperands.length ? (
              <tr><th>Operands</th><td>{p.bareOperands.map((o) => <code>%{o} </code>)}</td></tr>
            ) : null}
          </tbody>
        </table>
      )}
    </div>
  );
}
