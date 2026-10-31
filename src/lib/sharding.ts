// Sharding arithmetic for the sandbox: parse a mesh and PartitionSpecs, compute
// local shards, and classify A[I,J] @ B[J,K] into the Scaling Book's four cases
// (SB3 #computation-with-sharded-arrays) with collective costs (SB3 cost summary):
//   AllGather / ReduceScatter: V / (W_ici * axes), AllReduce: 2V / (W_ici * axes),
//   latency floor: 1 us per hop, sum(|X_i|)/2 hops. Full rings (wraparound) assumed.

export type Mesh = { name: string; size: number }[];
export type Spec = string[][]; // per array dimension, the mesh axes it is sharded over

export function parseMesh(s: string): Mesh | string {
  const parts = s.split(',').map((p) => p.trim()).filter(Boolean);
  const mesh: Mesh = [];
  for (const p of parts) {
    const m = p.match(/^([A-Za-z_]\w*)\s*[:=]\s*(\d+)$/);
    if (!m) return `Could not read "${p}". Write axes as name:size, e.g. X:4, Y:2.`;
    if (mesh.some((a) => a.name === m[1])) return `Axis ${m[1]} appears twice.`;
    mesh.push({ name: m[1], size: +m[2] });
  }
  return mesh.length ? mesh : 'Enter at least one axis, e.g. X:4.';
}

/** Parses P('X', None) style or plain "X, None" / "(X,Y), None". */
export function parseSpec(s: string, ndim: number, mesh: Mesh): Spec | string {
  let t = s.trim();
  const wrapped = t.match(/^(?:jax\.)?P\s*\(([\s\S]*)\)$/);
  if (wrapped) t = wrapped[1];
  const dims: string[] = [];
  let depth = 0, cur = '';
  for (const ch of t) {
    if (ch === '(') depth++;
    if (ch === ')') depth--;
    if (ch === ',' && depth === 0) { dims.push(cur); cur = ''; } else cur += ch;
  }
  if (cur.trim() || dims.length) dims.push(cur);
  while (dims.length < ndim) dims.push('None');
  if (dims.length > ndim) return `The spec has ${dims.length} entries for a ${ndim}-D array.`;
  const spec: Spec = dims.map((d) =>
    d.replace(/[()'"\s]/g, '').split(',').filter((a) => a && a !== 'None' && a !== '_'),
  );
  const used = spec.flat();
  for (const a of used) if (!mesh.some((m) => m.name === a)) return `Axis ${a} is not in the mesh.`;
  if (new Set(used).size !== used.length) return 'A mesh axis can shard only one dimension of an array.';
  return spec;
}

export const axisProd = (axes: string[], mesh: Mesh) => axes.reduce((p, a) => p * (mesh.find((m) => m.name === a)?.size ?? 1), 1);

export function localShape(shape: number[], spec: Spec, mesh: Mesh) {
  return shape.map((n, i) => n / axisProd(spec[i], mesh));
}

export function replication(spec: Spec, mesh: Mesh) {
  const used = new Set(spec.flat());
  return mesh.filter((m) => !used.has(m.name)).reduce((p, m) => p * m.size, 1);
}

export const fmtSpec = (letters: string[], spec: Spec) =>
  letters.map((l, i) => (spec[i].length ? `${l}_{${spec[i].join('')}}` : l)).join(', ');

export interface Step { what: string; axes: string[]; bytes: number; time: number }

export interface MatmulPlan {
  caseNo: 1 | 2 | 3 | 4;
  caseName: string;
  steps: Step[];
  out: string;
  flopsPerDevice: number;
  tMath: number;
  tComms: number;
}

const diff = (a: string[], b: string[]) => a.filter((x) => !b.includes(x));
const inter = (a: string[], b: string[]) => a.filter((x) => b.includes(x));

function collectiveTime(kind: 'AG' | 'RS' | 'AR', V: number, axes: string[], mesh: Mesh, W: number) {
  if (!axes.length) return 0;
  const hops = axes.reduce((s, a) => s + (mesh.find((m) => m.name === a)!.size) / 2, 0);
  const bw = ((kind === 'AR' ? 2 : 1) * V) / (W * axes.length);
  return Math.max(bw, hops * 1e-6);
}

/** A[I,J] @ B[J,K] with given specs. bytesPer: element size. W: per-axis W_ici. C: FLOP/s. */
export function planMatmul(
  I: number, J: number, K: number, sA: Spec, sB: Spec, mesh: Mesh, bytesPer: number, W: number, C: number,
): MatmulPlan | string {
  let [aI, aJ] = sA.map((x) => [...x]);
  let [bJ, bK] = sB.map((x) => [...x]);
  const steps: Step[] = [];
  let caseNo: MatmulPlan['caseNo'] = 1;
  let caseName = 'Neither operand is sharded on the contracting dimension. No communication.';

  // Case 4: a non-contracting dimension of each operand sharded on the same axis.
  const clash = inter(aI, bK);
  if (clash.length) {
    caseNo = 4;
    caseName = `I and K are both sharded on ${clash.join(', ')}, so only diagonal blocks are computable. AllGather B's K dimension first.`;
    const bytes = (J / axisProd(bJ, mesh)) * (K / axisProd(diff(bK, clash), mesh)) * bytesPer;
    steps.push({ what: `AllGather B over ${clash.join(', ')} (K dimension)`, axes: clash, bytes, time: collectiveTime('AG', bytes, clash, mesh, W) });
    bK = diff(bK, clash);
  }
  // An axis used for I in A and for J in B (or K in B and J in A) also forces a gather.
  const aIinBJ = inter(aI, bJ);
  if (aIinBJ.length) {
    const bytes = (J / axisProd(diff(bJ, aIinBJ), mesh)) * (K / axisProd(bK, mesh)) * bytesPer;
    steps.push({ what: `AllGather B over ${aIinBJ.join(', ')} (J dimension), because A uses that axis for I`, axes: aIinBJ, bytes, time: collectiveTime('AG', bytes, aIinBJ, mesh, W) });
    bJ = diff(bJ, aIinBJ);
    if (caseNo === 1) { caseNo = 2; caseName = 'One operand is sharded on the contracting dimension. AllGather it.'; }
  }
  const bKinAJ = inter(bK, aJ);
  if (bKinAJ.length) {
    const bytes = (I / axisProd(aI, mesh)) * (J / axisProd(diff(aJ, bKinAJ), mesh)) * bytesPer;
    steps.push({ what: `AllGather A over ${bKinAJ.join(', ')} (J dimension), because B uses that axis for K`, axes: bKinAJ, bytes, time: collectiveTime('AG', bytes, bKinAJ, mesh, W) });
    aJ = diff(aJ, bKinAJ);
    if (caseNo === 1) { caseNo = 2; caseName = 'One operand is sharded on the contracting dimension. AllGather it.'; }
  }
  // Contracting dimension: shard both the same way or gather the mismatch.
  const onlyA = diff(aJ, bJ), onlyB = diff(bJ, aJ);
  if (onlyA.length) {
    const bytes = (I / axisProd(aI, mesh)) * (J / axisProd(diff(aJ, onlyA), mesh)) * bytesPer;
    steps.push({ what: `AllGather A over ${onlyA.join(', ')} (J dimension)`, axes: onlyA, bytes, time: collectiveTime('AG', bytes, onlyA, mesh, W) });
    aJ = diff(aJ, onlyA);
    if (caseNo === 1) { caseNo = 2; caseName = 'One operand is sharded on the contracting dimension. AllGather it.'; }
  }
  if (onlyB.length) {
    const bytes = (J / axisProd(diff(bJ, onlyB), mesh)) * (K / axisProd(bK, mesh)) * bytesPer;
    steps.push({ what: `AllGather B over ${onlyB.join(', ')} (J dimension)`, axes: onlyB, bytes, time: collectiveTime('AG', bytes, onlyB, mesh, W) });
    bJ = diff(bJ, onlyB);
    if (caseNo === 1) { caseNo = 2; caseName = 'One operand is sharded on the contracting dimension. AllGather it.'; }
  }
  const shared = inter(aJ, bJ);
  const localI = I / axisProd(aI, mesh), localJ = J / axisProd(shared, mesh), localK = K / axisProd(bK, mesh);
  const flops = 2 * localI * localJ * localK;
  let out = `C[${fmtSpec(['I', 'K'], [aI, bK])}]`;
  if (shared.length) {
    if (caseNo !== 4) {
      caseNo = 3;
      caseName = `Both operands are sharded on the contracting dimension over ${shared.join(', ')}. Multiply locally, then reduce the partial sums.`;
    }
    const bytes = localI * localK * bytesPer;
    steps.push({ what: `AllReduce C over ${shared.join(', ')} (or ReduceScatter at half the cost, leaving C sharded)`, axes: shared, bytes, time: collectiveTime('AR', bytes, shared, mesh, W) });
    out = `C[${fmtSpec(['I', 'K'], [aI, bK])}] after the AllReduce; unreduced {U_${shared.join('')}} before it`;
  }
  const tComms = steps.reduce((s, x) => s + x.time, 0);
  return { caseNo, caseName, steps, out, flopsPerDevice: flops, tMath: flops / C, tComms };
}
