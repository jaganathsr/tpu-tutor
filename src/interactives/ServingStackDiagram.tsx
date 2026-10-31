// The three paths from a vLLM model definition to a TPU, as of October 2026.
// Sources: R-tpuinf, R-torchax-guide, B-vllmtpu, R-vllm-torchtpu, T-engines.
export function ServingStackDiagram() {
  const colX = [20, 230, 440];
  const colW = 190;
  const box = (x: number, y: number, h: number, title: string, sub: string, strong = false) => (
    <g>
      <rect x={x} y={y} width={colW} height={h} rx="3" fill={strong ? 'var(--accent-tint)' : 'var(--paper)'} stroke={strong ? 'var(--s1)' : 'var(--rule-strong)'} stroke-width={strong ? 2 : 1} />
      <text x={x + 10} y={y + 18} style="fill: var(--ink); font-weight: 600; font-size: 11.5px">{title}</text>
      {sub.split('\n').map((l, i) => <text x={x + 10} y={y + 34 + i * 14} style="font-size: 10.5px">{l}</text>)}
    </g>
  );
  const wide = (y: number, h: number, title: string, sub: string) => (
    <g>
      <rect x={20} y={y} width={610} height={h} rx="3" fill="var(--paper)" stroke="var(--rule-strong)" />
      <text x={30} y={y + 18} style="fill: var(--ink); font-weight: 600; font-size: 11.5px">{title}</text>
      <text x={30} y={y + 34} style="font-size: 10.5px">{sub}</text>
    </g>
  );
  const down = (x: number, y1: number, y2: number) => <line x1={x} y1={y1} x2={x} y2={y2} stroke="var(--s2)" stroke-width="2" marker-end="url(#ah2)" />;
  return (
    <div class="ix">
      <svg viewBox="0 0 650 470" role="img" aria-label="Upstream vLLM scheduler feeds three TPU paths: JAX-native models in vLLM JAX, PyTorch models through torchax in vLLM JAX, and PyTorch models on TorchTPU in vllm-torchtpu. All share Pallas kernels, XLA and libtpu.">
        <defs>
          <marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--s2)" />
          </marker>
        </defs>
        {wide(10, 46, 'Upstream vLLM v1', 'API server, scheduler (continuous batching), KV block manager, prefix cache. Unchanged on TPU.')}
        {down(115, 56, 78)}{down(325, 56, 78)}{down(535, 56, 78)}
        <text x={colX[0]} y={92} style="fill: var(--s1); font-weight: 600; font-size: 11px">vLLM JAX (tpu-inference)</text>
        <text x={colX[1]} y={92} style="fill: var(--s1); font-weight: 600; font-size: 11px">vLLM JAX, torchax path</text>
        <text x={colX[2]} y={92} style="fill: var(--s1); font-weight: 600; font-size: 11px">vllm-torchtpu</text>
        {box(colX[0], 100, 74, 'JAX model (Flax NNX)', 'MODEL_IMPL_TYPE=flax_nnx\nhand-written for TPU;\ndefault when one exists', true)}
        {box(colX[1], 100, 74, 'PyTorch model via torchax', 'MODEL_IMPL_TYPE=vllm\nupstream vLLM model code;\nATen ops mapped to jnp')}
        {box(colX[2], 100, 74, 'PyTorch model on TorchTPU', 'torch.compile(backend="tpu")\nupstream model code;\n"recommended path" (Oct 2026)', true)}
        {down(115, 174, 196)}{down(325, 174, 196)}{down(535, 174, 196)}
        {box(colX[0], 198, 60, 'jax.jit', 'model_fn(weights, kv_caches,\ninputs) -> (kv_caches, ...)')}
        {box(colX[1], 198, 60, 'jax.jit of a pure function', 'torch.func.functional_call;\nKV cache passed in, returned')}
        {box(colX[2], 198, 60, 'Dynamo FX graph', 'one StableHLO program per\ngraph; no TorchInductor')}
        {down(115, 258, 280)}{down(325, 258, 280)}{down(535, 258, 280)}
        {wide(282, 46, 'Pallas kernels (Mosaic TPU)', 'Ragged Paged Attention v3, MLA, fused MoE (megablox), quantized matmul, SparseCore gather: custom calls in XLA')}
        {down(325, 328, 350)}
        {wide(352, 46, 'StableHLO → XLA → libtpu (PjRT)', 'whole-step compilation per shape bucket; SPMD partitioning over the mesh; collectives as HLO ops')}
        {down(325, 398, 420)}
        {wide(422, 40, 'TPU slice', 'v5e, v6e, Ironwood (7x recommended for vllm-torchtpu); multi-host through Pathways or multi-controller')}
      </svg>
      <p class="ix-note">
        As of 2026-10-23. tpu-inference is being renamed vLLM JAX and focuses on JAX models; its torchax path still exists. New PyTorch-native TPU
        serving is in vllm-torchtpu. SGLang has the same split (sglang-jax, sglang-torchtpu).
      </p>
    </div>
  );
}
