# TPU Deep Dive: Syllabus

5 days × 2 blocks of ~55 min = 10 chapters. Audience: an experienced GPU engineer learning TPUs well enough to teach them. Source IDs refer to `research/sources.md`. Numbers marked † are recomputed by `scripts/specs.py` from those sources. They do not appear in any single source.

## Course-wide conventions

- **Reference chips.** The current-generation reference is **TPU7x (Ironwood)**: it is GA (2026-03-31, [D-rn]) and fully documented. Book problems keep their original chip (v5e, v5p) so answers can be checked against the book. Each chapter then re-runs the key example on Ironwood. **TPU 8t and 8i** appear wherever Google has published a number. Their BF16 and FP8 peaks exist only in the JAX hardware table [J-hw] and carry a "provisional" tag.
- **Conflicting numbers** follow the precedence rule in `sources.md`. Cloud docs come first and the Scaling Book fifth. Where the book's number differs, the chapter footnotes both.
- **Bandwidth convention.** $W_\text{ici}$ is the bandwidth one chip pushes into one torus axis during a bidirectional ring collective. That equals two links times the one-way link rate, which is the book's convention [SB3]. Cloud docs quote per-chip totals summed over all links and both directions. Chapter 3 shows the conversion once, explicitly.
- **Every chapter** has inline worked examples, "GPU analogy" and "where the analogy breaks" callouts, and a source sidebar. It ends with a quiz (8–12 questions, at least one adapted from the book's problems), 15–25 flashcards, and a teach-back checklist.

### Headline numbers the course will use (†, from `scripts/specs.py`)

| Chip | BF16 HBM intensity (FLOPs/B) | FP8 HBM intensity | $B_\text{crit}$, int8/fp8 weights + bf16 math | $C/W_\text{ici}$ per axis, bf16 |
|---|---|---|---|---|
| v5e | 241 (book: 240) | — | 120 | 1,970 |
| v5p | 166 | — | 83 | 2,295 (book: 2,550, using 1.8e11) |
| v6e | 560 | — | 280 | 4,590 |
| 7x Ironwood | 313 | 625 | 156 | 11,535 |
| 8t (provisional) | 153 | 916 | 76 | ~2,490 (ICI "2× Ironwood", derived) |
| 8i (provisional) | 128 | 1,024 | 64 | not computable (Boardfly per-link split unpublished) |
| H100 | 295 | 591 | 148 | — |
| B200 | 281 | 562 | 141 | — |

Two observations from this table drive several chapters. Ironwood's per-chip ICI (1,200 GB/s [D-7x]) is the same as v5p's, while its BF16 FLOPs are 5× higher. That makes its ICI roofline about 5× stricter than v5p's (Ch 5–6). The provisional TPU 8 figures show a split: 8t's BF16 intensity is lower than Ironwood's, and 8i's FP8 intensity is about 1,000. At that level decode is HBM-bound at any realistic batch, which is why 8i spends its die on 384 MB of SRAM and HBM bandwidth (Ch 2, Ch 7).

---

## Day 1. Rooflines and the chip

### Chapter 1: TPU lineage and roofline thinking (55 min)

**Time split:** history 15 min · roofline 30 min · quiz/cards 10 min.

**Learning objectives**
- I can explain each TPU generation as a response to a named bottleneck: v1 (inference, int8, DDR3-bound), v2 (training: HBM, bf16, ICI torus), v3 (2× MXU, liquid cooling, 1,024-chip pods), v4 (OCS, 3D torus, SparseCore, CMEM), v5e/v5p (cost vs scale split), v6e (256×256 MXU), Ironwood (chiplets, FP8, 9,216-chip pod), and 8t/8i (training and inference split, Boardfly, CAE).
- I can compute $T_\text{math}$, $T_\text{comms}$ and the critical arithmetic intensity for any chip from its spec sheet. I can state which regime a kernel is in.
- I can derive that a bf16 matmul's intensity is about the token batch $B$ when $B \ll D,F$, and give the compute-bound batch for v5e, Ironwood, H100 and 8i.
- I can show how int8/fp8 weights with bf16 math halve $B_\text{crit}$, and how fp8 math doubles it again.
- I can apply the same roofline to an ICI link and derive the 2-chip contracting-dimension threshold ($D > 8{,}755$ on v5e).

**Reading**
- [SB1] all sections, especially `#matrix-multiplication` and `#network-communication-rooflines`.
- [P-v1] §2 and §4: the 1,350 ops/byte ridge and why most v1 workloads were memory-bound.
- [P-v23] §2–3 and [P-Norrie] "Derivation from TPUv1": what changed for training.
- [P-v4] §2: OCS and cubes, as preview only.
- [G-7x], [G-8dd]: the current two generations.
- [HK] "Background" and "Design philosophy" as a narrative primer. Check its numbers against the generation table.

**Content notes**
- Two "slides" of generational progress:
  - A generation table, v1 → 8i, with per-chip BF16 (or int8 for v1), HBM capacity and bandwidth, ICI, max pod and MXU geometry.
  - A log-scale chart of FLOPs and HBM bandwidth by generation, showing FLOPs growing faster than bandwidth. P-Micro26 reports "100× node perf, 10× HBM" from v2 to Ironwood.
- Context, mentioned once: TPUs now reach third parties. The Anthropic agreement is primary-sourced; the Meta and Fluidstack items are tagged unverified.

**Interactive:** roofline plotter. Chip selector (v5e, v5p, v6e, 7x, 8t, 8i, H100, B200), dtype (bf16, int8-w, fp8), and sliders for B, D and F. Plots the operating point against the HBM and VMEM rooflines.

**Book problems adapted:** SB1 Q2 (int8 weights, $B > 120$), SB1 Q4 (per-example weights: intensity about 2), SB1 Q5 (H100: 295).

**Teach-back prompts**
1. "Your H100 intuition says a decode step at batch 64 is memory-bound. Show on a whiteboard whether that holds on Ironwood, and at what batch it stops holding."
2. "Why did Google build an inference-only chip first (v1), then a training chip, then split again in 2026? Name the bottleneck each time."
3. "Why does int8 weights with bf16 math lower the compute-bound batch, while int8 math does not?"

### Chapter 2: Anatomy of a TensorCore (55 min)

**Learning objectives**
- I can draw a TensorCore and name each part: the scalar unit issuing VLIW bundles, a VPU of 8 sublanes × 128 lanes, the MXU systolic arrays (128×128 before v6e, 256×256 on v6e and later), the XLU, VMEM, SMEM, VREGs, and the DMA engines. I can say what each one is for.
- I can explain weight-stationary systolic dataflow. I can explain why MXU utilization depends on padding K and N to multiples of 128 or 256, and why the contracting dimension sets the fill and drain cost.
- I can explain why VMEM is a compiler- or programmer-managed scratchpad and not a cache. I can compute the effective roofline when weights are already resident in VMEM.
- I can explain (8,128) tiling, the bf16 (2,1) and int8 (4,1) packing, and what padding does to memory and FLOPs.
- I can explain megacore (v4, v5p), single-core chips (v5e, v6e), Ironwood's two chiplets presenting as 2 JAX devices, and 8t (1 TC) vs 8i (2 TC) per [J-hw].
- I can say what SparseCore is for (embeddings, gather/scatter, collective offload on 7x), and what 8i's Collectives Acceleration Engine replaces.
- I can map SM, warp scheduler, CUDA core, SMEM, Tensor Core and HBM onto TPU equivalents, and state three places the mapping breaks. The three: one thread of control, no hardware warp switching to hide latency, and VMEM at about 2× the size of GPU L2 [SB12].

**Reading**
- [SB2] `#what-is-a-tpu`, `#appendix-a-more-on-tpu-internals`, `#appendix-b-how-does-a-systolic-array-work`.
- [P-Norrie] scalar, vector and MXU sections.
- [P-v4] §2.2 (chip) and §3 (SparseCore).
- [D-arch].
- [X-tiled] TPU section.
- [J-hw] table.
- [SB12] `#what-is-a-gpu`, `#gpus-vs-tpus-at-the-chip-level`.
- [A-trn2-nki] NeuronCore engines, read for the comparison callout.

**Content notes**
- **Trainium callout:** 8 NeuronCores per chip. Each has one 128×128 Tensor Engine plus separate Vector, Scalar and GpSimd engines, a 28–32 MiB SBUF with 128 partitions, and a separate 2 MiB PSUM accumulation buffer [A-trn2-nki, A-trn3-nki]. Trainium sits between the TPU (few large cores, systolic) and the GPU (many independent cores).
- **8i design callout:** 384 MB of on-chip SRAM is 3× Ironwood, sized to keep the KV cache on chip [G-8dd]. This is the decode bandwidth problem answered in silicon.

**Interactive:** systolic array animation.
- Choose a 4×4 to 8×8 array and M, K, N.
- Watch weights preload, activations stream in diagonally, partial sums flow down and results drain.
- Counters show cycles, MAC utilization and the fill/drain overhead as K and N vary.
- A toggle shows the padding waste for N = 100 on a 128-wide array.

**Book problems adapted:** SB2 Q1 (200B model on 32 v6e: 7.8 ms lower bound), SB2 Q4 (int8 matmul on v5e: $B > 267$ from HBM, $B > 11$ from VMEM), SB2 Appendix A pop quiz (VPU 9e12 f32 FLOPs/s per core, 64× below the MXU).

**Teach-back prompts**
1. "A GPU hides memory latency with 64 resident warps per SM. What hides it on a TPU, and what does that imply for who has to be clever?"
2. "Why did Google move from 128×128 to 256×256 MXUs in v6e, and what does that cost a model with head_dim 64?"
3. "What is VMEM, how big is it on Ironwood and 8i, and why does 8i make it 3× larger?"

---

## Day 2. The network and the model

### Chapter 3: ICI, OCS, slices and pods (55 min)

**Learning objectives**
- I can describe the topology at each level: a 2D torus (v5e/v6e, 16×16, wraps only on 16-long axes), 4×4×4 cubes joined by OCS into a 3D torus (v4, v5p, 7x), and twisted tori. I can explain why slices smaller than a cube lose wraparound and what that does to collective time (about 2×).
- I can convert Cloud-doc ICI figures (per chip, bidirectional, summed over links) to the per-axis $W_\text{ici}$ used in cost formulas.
- I can explain what the OCS buys: reconfiguration in milliseconds, routing around failed racks, non-contiguous slices, and twisted shapes. I can state its cost: under 5% of system cost [P-v4], under 6% by another accounting [P-Lightwave].
- I can rank the bandwidth ladder HBM > ICI > PCIe > DCN with numbers for Ironwood. I can explain multislice over DCN, Pathways vs multi-controller JAX, and Virgo on 8t.
- I can explain Boardfly on 8i: 4-chip boards, 8-board groups, 36 groups over OCS, at most 7 hops vs 16 for an 8×8×16 torus [G-8dd]. I can explain why decode-heavy MoE serving wants low diameter more than bisection.
- I can contrast all three scale-up designs:
  - NVLink switched domains (8 or 72 GPUs, one hop, in-switch SHARP reduction), then an InfiniBand fat tree at 400 GB/s per node [SB12].
  - Trainium: a 4×4 torus on Trn2, replaced by the NeuronSwitch all-to-all fabric on Trn3 [A-trn2-sys, A-trn3-sys].
  - TPU: direct-connect torus, no switch, about 9,000 chips.

**Reading**
- [SB2] `#tpu-networking`, `#tpu-specs`.
- [P-v4] §2.1–2.8 (OCS, availability, twisted tori).
- [P-Lightwave] §1–3.
- [P-Resil] (how OCS is used to survive failures).
- [D-arch], [D-7x], [D-v5p].
- [G-8dd] "Boardfly vs. torus math".
- [P-Pathways] §2 and §4.
- [SB12] `#networking`, `#beyond-the-node-level`.
- [A-trn3-sys].

**Interactive:** a hop-count view inside the torus visualizer. Pick a slice shape, source and destination, and see the hop count and the path with and without wraparound. This reuses the Chapter 5 component.

**Book problems adapted:** SB2 Q5 (v5e 4×4 point-to-point: 6 µs for the first byte, 194 µs in total), SB2 Q2 (pod accounting for v5e and v5p, re-run for Ironwood†), SB2 Q6 (host-offloaded int8 matrix, about 200 ms, bound by ICI).

**Teach-back prompts**
1. "NVLink gives every GPU one hop to 71 others. A TPU torus gives each chip 6 neighbours. Argue for each design, with numbers."
2. "Why do you get roughly half the collective bandwidth on a 2×2×2 slice compared with a 4×4×4?"
3. "What did Google change in 8i's network and why, given what decode traffic looks like?"

### Chapter 4: Transformer accounting on TPU (55 min)

**Learning objectives**
- I can count parameters and FLOPs for a dense GQA transformer and for an MoE. I can derive $6 \cdot \text{params} \cdot \text{tokens}$ from 2× forward plus 4× backward, and $\approx 8ND$ with block remat.
- I can derive when attention FLOPs dominate ($T > 8D$) and when self-attention equals the QKVO projections ($T = 2D$).
- I can compute KV cache bytes per token ($2 \cdot L \cdot K \cdot H \cdot \text{bytes}$) for LLaMA-3-70B and a current open-weight MoE. I can explain how GQA, MLA-style compression and int8/fp8 KV change it.
- I can compute attention arithmetic intensity: about $T/2$ in prefill and about 1 per KV head group in decode. I can explain why decode attention never becomes compute-bound.
- I can compute training memory (10 bytes per parameter with Adam) and activation memory under different remat policies. I can state why large clusters are FLOPs-motivated, not memory-motivated.
- I can state the TPU-specific consequences: dimensions padded to 256 on v6e and later, head_dim 64 using half an MXU dimension, and MoE needing $B > 120 \cdot E/k$ (int8 weights on v5e) to be compute-bound.

**Reading**
- [SB4] all sections, including Appendix A on flash attention mapped to VMEM.
- [SB6] `#counting-parameters-and-flops`.
- [SB7] `#what-about-attention`.

**Book problems adapted:** SB4 Q1 (16B params, 512 KiB/token int8 KV), SB4 Q4 (GQA attention intensity), SB4 Q5 ($T = 2D$), SB4 Q7 (DeepSeek-V3 MFU of 21.7%), SB4 Q8 (MoE critical batch of 3,840, re-run for Ironwood fp8†), SB5 Q1 (LLaMA-2-13B parameter count).

**Teach-back prompts**
1. "Estimate on a whiteboard, in 2 minutes, the FLOPs and the Ironwood-days to train a 70B dense model on 15T tokens."
2. "Why does MoE make decode harder to make compute-bound, and by what factor?"
3. "Your model has head_dim 64 and you are on v6e. What happens in the MXU?"

---

## Day 3. Sharding and training

### Chapter 5: Sharded matmuls and collectives on a torus (55 min)

**Learning objectives**
- I can read and write sharding notation ($A[I_X, J_Y]$, unreduced $\{U_X\}$) and the equivalent `PartitionSpec`. I can say what each chip holds for a given mesh and spec.
- I can classify any sharded matmul into the 4 cases and name the collective each one needs: none, AllGather, AllReduce or ReduceScatter, or reshard first.
- I can derive the bidirectional-ring AllGather time $V/W_\text{ici}$, independent of shard count. I can extend it to multiple axes, to AllReduce at 2×, and to AllToAll at ¼ of an AllGather in 1D. I can state when a collective is latency-bound (below about 45 kB on v5e at 1 µs per hop).
- I can explain collective matmul: the AllGather is decomposed into collective-permutes overlapped with partial matmuls. I can explain why it is a compiler transform on TPU and a library or kernel feature on GPU (async-TP with SymmetricMemory, Flux, TE userbuffers).
- I can relate the book's formulas to NCCL's algbw and busbw factors. I can explain why TPUs reach peak collective bandwidth at about 10 MB per collective, while 8×H100 AllReduce needs messages near 10 GB per device to reach about 370 GB/s [SB3, SB12].

**Reading**
- [SB3] all sections.
- [SB12] `#how-do-collectives-work-on-gpus`.
- [P-CollMM] (via [SB10] `#manual-sharding-mode-via-shard-map` and [J-shmap] matmul examples).
- [N-nccltests].
- [PT-asynctp].
- [P-Megatron] §3 and [P-SeqPar] §4, for the GPU origin of the AllGather/ReduceScatter-around-matmul pattern.

**Interactives**
- **Sharding-spec sandbox.** Enter a mesh (e.g. `{X:4, Y:2}`), an array shape, and a `PartitionSpec`. See the block each device holds, bytes per device, and the replication factor. Optionally enter a second operand to get the matmul case, the collective required and its cost on the selected chip.
- **Torus collective visualizer.** AllGather, ReduceScatter and AllToAll on a 1D ring (unidirectional or bidirectional) vs a 2D torus. Chunks move per step, with a running count of hops × bytes and time on the chosen chip. A wraparound toggle is included.

**Book problems adapted:**
- SB3 pop quiz: AllGather over a 4-long v5e axis without wraparound takes 560 µs (680 µs measured).
- SB3 Q2: v4 4×4×4 AllGather and AllReduce at 23, 46 and 11.6 µs, and the free reshard to $[B_{XZ}, D_Y]$.
- SB3 Q4: AllGather-first vs matmul-then-AllReduce.
- SB3 Q10: derivation of AllToAll's ¼.
- Re-run on Ironwood: the threshold above which collective time dominates a 2-way contracting-dim matmul†.

**Teach-back prompts**
1. "Why doesn't AllGather time on a ring depend on how many chips are in the ring?"
2. "Explain to a Megatron user what XLA's collective matmul does, and why they would have to write it by hand on GPU."
3. "Given `A: [B, D] @ P('X', 'Y')` and `W: [D, F] @ P('Y', None)`, what happens, and what does it cost on a v5p 4×4×4?"

### Chapter 6: Choosing a mesh for training: rooflines and a case study (55 min)

**Learning objectives**
- I can derive the compute-bound condition for each scheme:
  - DP and FSDP: per-chip batch $> C/W_\text{ici}$ per axis used.
  - TP: $F > Y \cdot C / W_\text{ici}$.
  - FSDP+TP: per-chip batch $> \alpha^2 / (M_X M_Y F)$, about 100 on v5p.
  - Cross-pod DP over DCN: per-slice batch $>$ about 73k on v5p.
- I can compute $X_\text{opt} = \sqrt{(B/F)(M_X/M_Y)N}$ and pick a mesh for a given model, batch and slice.
- I can explain why pipelining is "somewhat less essential on TPUs" [SB5]. I can explain why GPU recipes lean on PP and EP across nodes instead (DeepSeek-V3: 64 EP × 16 PP × 2 DP; LLaMA-3: 8 TP × 16 PP × 128 DP) [SB12].
- I can redo the LLaMA-3-70B plan on a v5p pod (2,048 FSDP × 4 TP, about 44 days at 40% MFU). I can then redo it for an Ironwood pod and explain how the stricter ICI roofline† changes the mesh.
- I can map these choices to MaxText config: the `ici_*` vs `dcn_*` parallelism axes, `logical_axis_rules`, and why `dcn_tensor_parallelism` is "never recommended".

**Reading**
- [SB5] all sections.
- [SB6] all sections.
- [SB12] `#rooflines-for-llm-scaling-on-gpus`.
- [R-maxtext] `base.yml` mesh section and `docs/guides/optimization/sharding.md`.
- [P-GSPMD] §3.3 (pipelining as sharding).

**Book problems adapted:**
- SB5 Q3: LLaMA-2-13B on 4,096 v5p gives 1,024 DP × 4 TP at about 300 ms per step.
- SB6 worked Q&A (70B on a v5p pod).
- SB6 end-of-chapter Q1 (4 pods) and Q2 (405B on 8 pods). The book gives no answer for these; they are derived in `scripts/` and labelled.
- SB12 Quiz 5 Q2: 70B on 4,096 H100s, where only 8-way PP is compute-bound.

**Teach-back prompts**
1. "Give the one-line rule for when FSDP stops being free, and the number on v5p and on Ironwood."
2. "Why do GPU shops use 16-way pipeline parallelism for LLaMA-3 and TPU shops mostly don't?"
3. "Walk through choosing a mesh for a 70B dense model on a 2,048-chip Ironwood slice with a 4M-token batch."

---

## Day 4. Inference

### Chapter 7: Inference rooflines: prefill, decode, KV and quantization (55 min)

**Learning objectives**
- I can state the decode step-time lower bound, (B · KV bytes + param bytes) / HBM BW, and the general step-time formula with the MLP term $\max(2B \cdot \text{params}/C,\ \text{params}/W_\text{hbm})$.
- I can compute the batch-size breakeven for decode, $B_\text{crit} = \beta \cdot C/W_\text{hbm}$, for each weight/compute dtype pair on v5e, Ironwood and 8i. I can explain why KV growth makes the breakeven unreachable at long context.
- I can explain why prefill is compute-bound and decode is not. I can explain what that implies for disaggregated serving and for the prefill:decode server ratio (about 3:1 in the book's example).
- I can choose decode sharding:
  - Model-parallel up to $Y \approx F/(B\beta)$ with $\beta = W_\text{hbm}/W_\text{ici}$.
  - KV sharded by heads, then by batch, with 2 AllToAlls per layer.
  - Never FSDP at decode time.
- I can identify latency-bound collectives at small decode batch: under about 360 kB for 8-way on v5e.
- I can explain int8, fp8 and fp4 weights, KV and activations on TPU. I can say which generations have native fp8 MXU paths (Ironwood, 8t/8i; v5p lists fp8 = bf16 [D-v5p]). I can describe how Qwix intercepts `dot_general`, `einsum` and `ragged_dot` to apply PTQ rules.
- I can explain speculative decoding as spending idle FLOPs at a memory-bound step.

**Reading**
- [SB7] all sections, including Appendices A–D.
- [SB8] `#whats-the-llama-serving-story`, `#visualizing-the-latency-throughput-tradeoff`.
- [P-ESTI] §2–3.
- [R-qwix] README and `_src/interception.py`.
- [D-arch] supported dtypes per generation.

**Interactive:** prefill/decode breakeven calculator.
- Inputs: model (presets LLaMA-3-8B, 70B, 405B, a 2026 MoE, or custom L/D/F/N/K/H/E/k), chip and slice size, weight, KV and compute dtypes, context length, batch.
- Outputs:
  - Memory fit.
  - Step-time breakdown: weight load, KV load, FLOPs, ICI.
  - Tokens/s per chip, and $B_\text{crit}$.
  - The batch where decode crosses to compute-bound, or "unreachable at this context".
- The roofline plotter from Chapter 1 is reused in "decode mode".

**Book problems adapted:**
- SB7 pop quiz: 30B on v5e 4×4 at 2.5 ms (batch 4) and about 21 ms (batch 256).
- SB7 Q1–Q3: 18.4B params, 262 kB/token, max batch 7 at 128k, 1.4 ms.
- SB7 Q5: MoE needs 1,920 tokens.
- SB7 Q4 (no book answer) is derived. Every number is re-run on Ironwood and 8i†.

**Teach-back prompts**
1. "Your PM wants 2× decode throughput on the same chips. List what moves the needle and by how much: batch, int8 weights, fp8 KV, more TP, speculative decoding."
2. "Why is FSDP an order-of-magnitude regression for decode, when it is free for training?"
3. "At 32k context, why does 8i's FP8 peak barely matter for decode, and what does matter?"

### Chapter 8: The TPU serving stack: vLLM, tpu-inference, torchax (55 min)

**Learning objectives**
- I can describe the 2025–26 history and say which path to use today for a new port:
  - JetStream (archived 2026-02-01).
  - PyTorch/XLA vLLM backend.
  - tpu-inference, a unified JAX lowering path for JAX models and PyTorch-via-torchax (2025-10).
  - The October 2026 split: vLLM JAX (tpu-inference renamed) for JAX/Flax NNX models, and vllm-torchtpu as "the recommended path" for PyTorch model definitions [R-tpuinf, R-vllm-torchtpu, T-engines].
- I can explain torchax: a `torch.Tensor` subclass holding a `jax.Array`, with `__torch_dispatch__` mapping ATen ops to jnp/lax. I can explain the "sandwich" pattern in tpu-inference (JAX KV cache, sampler and RPA kernel around torch model code), and why the model must be a pure function that returns its KV cache.
- I can explain how continuous batching works on a static-shape compiler:
  - The upstream vLLM v1 scheduler.
  - Padded shape buckets precompiled by `compilation_manager.py`.
  - Ragged Paged Attention v3 with decode, prefill and mixed sub-kernels and a fused KV update [P-RPA].
- I can read the tpu-inference KV cache layout `(num_blocks, block_size, ceil(2·kv_heads/packing), packing, head_dim)` and explain the packing dimension from (8,128)(2,1) tiling.
- I can explain the tpu-inference mesh axes (`data`, `model`, `expert`, `attn_dp`, `dcp`, `pcp`) and map them to Chapter 7's sharding rules.
- I can apply the book's LLaMA-3-70B serving analysis to pick a slice and quantization for a latency target. I can explain disaggregated prefill/decode on TPU (TPU Sync, KV offload to host DRAM).

**Reading**
- [SB8] all sections.
- [B-vllmtpu].
- [R-tpuinf] README, `docs/developers_guide/torchax_model_development.md`, `runner/kv_cache.py`, `layers/common/sharding.py`.
- [R-torchax] README.
- [P-RPA].
- [T-engines].
- [R-vllm-torchtpu] README.

**Content notes:** This chapter directly affects your current torchax port, and says so plainly. The torchax path in tpu-inference (`MODEL_IMPL_TYPE=vllm`) still works. As of 2026-10-20, though, new PyTorch-native TPU serving work lives in vllm-torchtpu on TorchTPU. Chapter 9 explains what that changes underneath.

**Book problems adapted:**
- SB8 Q2–Q3: LLaMA-3-70B with batch 32 at 8k fits a 4×2, at 17 ms/step and 235 tok/s/chip.
- SB8 Q9: bf16 on a 4×8 is HBM-bound at batch 64 (11, 18 and 4 µs).
- SB8 Q12: prefill:decode servers ≈ 3:1.
- SB8 Q3 (405B under a 15 ms/token limit; no book answer) is derived. Re-run on Ironwood†.

**Teach-back prompts**
1. "Explain to a vLLM-on-CUDA engineer how continuous batching survives a compiler that wants static shapes."
2. "What does torchax actually do to a `torch.nn.Linear` call, and where does the code end up as HLO?"
3. "Why does the KV cache tensor have a 'packing' dimension on TPU?"

---

## Day 5. Software stack, kernels, profiling

### Chapter 9: From Python to the MXU: JAX, XLA and TorchTPU (55 min)

**Time split:** compiler pipeline 20 min · JAX sharding modes 10 min · TorchTPU 20 min · Trainium software contrast 5 min.

**Learning objectives**
- I can trace a function down the stack: jaxpr → StableHLO → HLO → XLA passes (fusion, layout assignment, memory-space assignment between HBM and VMEM, latency-hiding scheduler, SPMD partitioning) → LLO → TPU instructions. I can dump each stage (`make_jaxpr`, `.lower().as_text()`, `.compile().as_text()`, `--xla_dump_to`).
- I can explain what I control and what I don't:
  - Controllable: shardings and constraints, layouts (`AUTO`), remat policy, donation, XLA flags via `LIBTPU_INIT_ARGS`, Pallas kernels.
  - Not controllable: TPU codegen, fusion decisions beyond flags, VLIW scheduling. The XLA TPU backend is closed source.
- I can explain Shardy vs GSPMD: Shardy has been the default since JAX 0.7.0, with named axes and open/closed dimensions. I can compare the three JAX sharding modes (auto, explicit "sharding in types", manual `jax.shard_map`) and say when to drop to each.
- I can compare XLA whole-program compilation with CUDA Graphs plus torch.compile/Inductor. A CUDA graph replays a fixed DAG of already-compiled kernels; XLA chooses fusion, layout, memory assignment and collective schedule together.
- I can explain TorchTPU in detail:
  - An out-of-tree PrivateUse1 backend (`torch.device("tpu")`) with plain tensors and no subclass.
  - ATen ops become deferred StableHLO builders.
  - Three eager modes (strict, debug, defer-and-fuse). `torch.compile(backend="tpu")` turns the whole FX graph into one StableHLO program for XLA, with no Inductor.
  - A 3-tier compile cache and PjRT/libtpu runtime.
  - The `tpu_dist` process group with DDP, FSDP2 and DTensor. The SPMD handshake reconciles PyTorch's per-rank MPMD code with XLA's SPMD collectives.
  - Pallas kernels through `torch.tpu.pallas.jax_op`.
- I can contrast TorchTPU with JAX (global-view SPMD vs per-rank PyTorch), with torchax (Tensor subclass inside JAX tracing), and with PyTorch/XLA (lazy tensors and `mark_step`; deprecated, archive date 2026-11-15).
- I can contrast the Trainium path: neuronx-cc and NEFF, the move to TorchNeuron (also PrivateUse1), and JAX via a PjRT plugin. Both vendors moved PyTorch off torch-xla in 2025–26; Google kept XLA as the compiler [A-torch, T-xla].

**Reading**
- [SB9] `#a-thousand-foot-view-of-the-tpu-software-stack`.
- [SB10] all sections.
- [J-shard], [J-aot], [X-arch], [X-shardy-jax], [X-flags], [X-async].
- [P-GSPMD] §3–4.
- [T-how], [T-exec], [T-dist], [T-kern], [T-pt].
- [D-pytorch].
- [N-graphs] §4.2.
- [A-torch].

**Book problems adapted:**
- SB10 Q1: per-shard mean and roll with jit vs shard_map.
- SB10 collective matmul result: 311 → 244 µs, against a 224 µs baseline.
- A derived exercise: given a jaxpr, predict the HLO collectives Shardy will insert.

**Teach-back prompts**
1. "A CUDA engineer says 'XLA is just CUDA Graphs.' Correct them in three sentences."
2. "Draw how `y = model(x)` reaches the TPU under JAX jit, torchax, TorchTPU strict eager, and TorchTPU `torch.compile`."
3. "Why did TorchTPU need an 'SPMD handshake' that JAX never needed?"

### Chapter 10: Pallas kernels, XProf, and a design review (55 min)

**Time split:** Pallas 15 min · XProf 25 min · capstone 15 min.

**Learning objectives**
- I can write and explain a Pallas TPU kernel skeleton, and explain how it lowers through Mosaic to a custom call inside XLA. The skeleton covers:
  - The `grid`, which runs sequentially, unlike a CUDA grid.
  - `BlockSpec` index maps, with the last two dimensions divisible by (8,128).
  - Memory spaces (VMEM, SMEM, `pl.ANY`) and implicit double-buffered HBM↔VMEM pipelining.
  - `dimension_semantics` for megacore.
  - Scalar prefetch for paged attention.
- I can say when a Pallas kernel is worth writing on TPU (ragged or paged attention, fused MoE, quantized matmul, collective matmul via remote DMA) and when XLA fusion already does the job.
- I can capture a profile (`jax.profiler.trace`; tpu-inference `PHASED_PROFILING_DIR` and the profiler server). I can read the Trace Viewer: the XLA Ops row, step markers, and `-start`/`-done` async pairs whose gap is the overlap window.
- I can decode an HLO op line from a profile (`bf16[32,32,4096]{2,1,0:T(8,128)(2,1)S(1)}`): shape, minor-to-major layout, tiling and memory space (S(1) = VMEM).
- I can find a communication/compute overlap failure and a sharding mistake from a trace. Example: a giant AllGather taking 80% of the step, fixed with `with_sharding_constraint` [SB10].
- I can use HLO Op Profile and the Roofline Model tool to place each op against HBM and VMEM rooflines and decide which ops to attack.
- In the capstone, I can run a design review: given a model, a chip, an SLO and a profile summary, produce the mesh, quantization, batch and expected step time, with the bottleneck stated in one sentence.

**Reading**
- [PL-tpu], [PL-pipe], [PL-grid], [PL-sparse], [PL-dist] (ring AllGather example), [PL-mm].
- [SB9] all sections.
- [XP-trace], [XP-opprof], [XP-roof], [XP-mem].
- [R-tpuinf-prof].
- [PL-gpu], for the Mosaic GPU contrast (warpgroups and TMA vs VMEM and DMA semaphores).
- [A-nki] for the NKI contrast: explicit DMA and engine choice, lower-level than Pallas.

**Book problems adapted:**
- SB9 Q1: diagnose from `replica_groups` that the model is 8-way model parallel and recover the true shapes.
- SB9 worked profile: 95.6 ms predicted vs 96 ms measured for the FFW fusion; 1.1 ms vs 1.13 ms for the ReduceScatter.
- Capstone: LLaMA-3-70B (or your current port) on an Ironwood slice under a stated TTFT and TPOT SLO.

**Teach-back prompts**
1. "Show me in this trace where communication failed to overlap, and what you'd change."
2. "What is the Pallas equivalent of a CUDA thread block, and why is the answer 'there isn't one'?"
3. "Read this HLO line aloud and tell me which memory the output lives in."

---

## Reinforcement beyond chapters

- **Day-end mini-exams** (Days 1–5): 10 questions drawn from that day's two chapters plus 2 from earlier days.
- **Final exam:** 30 questions. At least 2 per chapter, at least 10 numeric (checked within tolerance), at least 4 "spot the bottleneck", and 2 free-response "explain to a GPU engineer" questions with model answers.
- **Flashcards:** 15–25 per chapter, about 200 in total, scheduled with SM-2.
- **Comparison reference page** (not a chapter): one consolidated TPU vs GPU vs Trainium table covering compute unit, matrix unit, on-chip memory, scale-up topology, collectives implementation, kernel language, compiler path and partitioning model, linked from Chapters 2, 3, 5 and 9.
- **Glossary:** TensorCore vs Tensor Core, slice, cube, pod, superpod, megacore, chiplet, OCS, ICI, DCN, MXU, VPU, XLU, VMEM, SMEM, CMEM, SparseCore, CAE, Boardfly, HLO, LLO, Mosaic, Shardy and the rest.

## Changes from the suggested arc, and why

1. **History opens Chapter 1** instead of being a separate block, as asked. It takes 15 minutes and frames each generation as a response to a roofline, so it leads straight into the roofline material.
2. **Interconnect gets its own chapter (3).** The brief paired it with transformer accounting on Day 2. The topology material grew with three additions in the last 12 months: Boardfly on 8i, Virgo on 8t, and Trainium's move from torus to switch. That justifies a full block. Transformer accounting keeps its own block (4).
3. **TorchTPU sits in Chapter 9 (Day 5), as asked, but is introduced in Chapter 8.** Research showed TorchTPU is now the recommended PyTorch serving path through vllm-torchtpu. Chapter 8 covers that at the system level, and Chapter 9 covers its internals after XLA has been explained.
4. **Trainium is woven in as callouts**: chip in Ch 2, topology in Ch 3, kernels in Ch 10, software in Ch 9. A reference page consolidates them, rather than spending a whole block on it.
5. **Added items not in the brief:**
   - Numerics by generation: fp8 native only from Ironwood; fp4 on 8t/8i.
   - Static shapes, shape buckets and compile caches as a first-class inference constraint.
   - Ironwood's chiplets showing up as 2 JAX devices per chip.
   - SparseCore and the CAE.
   - OCS-based resilience.
   - The derived ICI-roofline gap on Ironwood.
   - The current status of every repo (PyTorch/XLA deprecated; JetStream archived; tpu-inference renamed).

## Open items before content is written

1. **Local resources.** The path in the brief was the literal placeholder `[path/to/my/resources]`. Send a real path and I'll index it and fold it into the reading lists.
2. **TPU 8 BF16 and FP8 peaks** come only from the JAX hardware table, which is generated from libtpu metadata and not from a Google spec sheet. The plan is to use them with a "provisional" tag. The alternative is to show only FP4 and memory figures for TPU 8.
3. **Ironwood ICI roofline (C/W ≈ 11.5k per axis†).** This follows from docs-published per-link bandwidth and differs from the book's TPU7x ICI table (1.8e11 per link) by about 10%, and by 2× from the book's prose figure of 4e11 "per axis", which sums both links. It will be stated as derived, with the arithmetic shown in Chapter 3.
