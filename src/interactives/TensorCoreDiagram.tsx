// Static block diagram of one TensorCore and its memory path.
// Sizes and widths are Ironwood (TPU7x) figures from J-hw and SB2.
export function TensorCoreDiagram() {
  const box = (x: number, y: number, w: number, h: number, title: string, sub: string, strong = false) => (
    <g>
      <rect x={x} y={y} width={w} height={h} rx="3" fill={strong ? 'var(--accent-tint)' : 'var(--paper)'} stroke={strong ? 'var(--s1)' : 'var(--rule-strong)'} stroke-width={strong ? 2 : 1} />
      <text x={x + 10} y={y + 20} style="fill: var(--ink); font-weight: 600">{title}</text>
      <text x={x + 10} y={y + 37}>{sub}</text>
    </g>
  );
  const arrow = (x1: number, y1: number, x2: number, y2: number, label?: string, lx = 0, ly = 0) => (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--s2)" stroke-width="2" marker-end="url(#ah)" />
      {label ? <text x={(x1 + x2) / 2 + lx} y={(y1 + y2) / 2 + ly} style="fill: var(--ink)">{label}</text> : null}
    </g>
  );
  return (
    <div class="ix">
      <svg viewBox="0 0 640 404" role="img" aria-label="TensorCore block diagram: HBM feeds VMEM by DMA; VMEM feeds vector registers; vector registers feed the MXUs and the VPU; the scalar unit issues instructions and DMAs.">
        <defs>
          <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--s2)" />
          </marker>
        </defs>
        {box(20, 20, 600, 46, 'HBM', '96 GiB per TensorCore, 3.7 TB/s per TensorCore', false)}
        <rect x="20" y="96" width="600" height="300" rx="4" fill="none" stroke="var(--s1)" stroke-dasharray="5 4" />
        <text x="30" y="114" style="fill: var(--s1); font-weight: 600">One TensorCore</text>
        {box(40, 126, 260, 50, 'Scalar unit + SMEM', 'issues VLIW bundles, loops, DMAs')}
        {box(330, 126, 270, 50, 'VMEM', '64 MiB scratchpad, compiler-managed', true)}
        {box(330, 206, 270, 50, 'Vector registers (VREGs)', '64 × 8×128 × 32-bit, about 256 KiB')}
        {box(40, 296, 170, 70, 'MXU × 2', '256×256 systolic arrays', true)}
        {box(230, 296, 170, 70, 'VPU', '8 sublanes × 128 lanes')}
        {box(420, 296, 180, 70, 'XLU × 2', 'transpose, permute, reduce')}
        {arrow(465, 66, 465, 124, 'DMA', 8, 4)}
        {arrow(300, 151, 328, 151)}
        {arrow(465, 176, 465, 204, 'load / store', 8, 4)}
        {arrow(380, 256, 140, 294)}
        {arrow(440, 256, 315, 294)}
        {arrow(520, 256, 510, 294)}
        <text x="40" y="381">The scalar unit drives every unit; there are no warps.</text>
      </svg>
      <p class="ix-note">
        Ironwood figures, per TensorCore: two TensorCores per chip, each on its own chiplet. HBM and VMEM sizes from the JAX hardware table; VREG and VPU shapes from the
        Scaling Book. Earlier chips differ in counts (v5p: 4 MXUs of 128×128 per core), not in structure.
      </p>
    </div>
  );
}
