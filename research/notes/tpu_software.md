# TPU Software Stack: Source Notes

Course: TPU Deep Dive. Audience: GPU/ML-infra engineer porting open-weight LLMs to TPU inference (tpu-inference / torchax, Qwix, XProf).
All URLs accessed and checked (HTTP 200 or content fetched) on **2026-10-23** unless marked otherwise.
Labels: **VERIFIED** = page fetched and content read. **URL-OK** = HTTP 200, content not read in detail. **UNVERIFIED** = from memory or secondary sources only.

---

## 0. Big picture: what changed in 2025-2026 (read first)

1. **TorchTPU is real and replaces PyTorch/XLA.** It was announced 2026-04-07 (Google Developers Blog) and open-sourced 2026-10-20 at PyTorch Conference 2026. Repo: github.com/google-pytorch/torch_tpu. PyPI `torch_tpu` 0.2.0 was uploaded 2026-10-19. The PyTorch/XLA README says the project is "deprecated and replaced by TorchTPU" and that the repo "will be archived on November 15, 2026." The last torch_xla on PyPI is 2.9.0 (2025-11-17).
2. **vLLM on TPU is now two projects.**
   - `vllm-project/tpu-inference` is being renamed to **vLLM JAX** (`vllm-project/vllm-jax`). It now focuses on JAX (Flax NNX) models. PyPI tpu-inference 0.32.0, 2026-10-22.
   - **vllm-torchtpu** (`vllm-project/vllm-torchtpu`, PyPI 0.29.0, 2026-10-19) is "the recommended path for serving PyTorch model definitions on TPU." It runs on TorchTPU, not torchax.
   - The torchax path (`MODEL_IMPL_TYPE=vllm`) still exists in tpu-inference source, but new PyTorch-native work goes to vllm-torchtpu. **This matters for the learner's current torchax port.**
3. **SGLang has the same split:** `sgl-project/sglang-jax` and `sgl-project/sglang-torchtpu`.
4. **torchax moved** from pytorch/xla/torchax to **github.com/google/torchax** on 2025-10-06. PyPI 0.0.13, 2026-06-17.
5. **JAX partitioning moved to Shardy.** JAX 0.7.0 (2025-07-22) made Shardy the default. In the Unreleased section after 0.12.0, `jax_use_shardy_partitioner` "is deprecated and is now a no-op." `jax.shard_map` became the public API in JAX 0.8.0 (2025-10-15). `jax.set_mesh` arrived in 0.7.1. pmap is now implemented via jit + shard_map, and the C++ pmap was removed in 0.10.0 (2026-04-16). The latest JAX is 0.12.0.1 (2026-10-21/23).
6. **Pallas changes:** `pl.kernel` / `core_map` per-core programming, SparseCore kernels (`plsc`), Mosaic GPU as the default GPU backend (0.9.0), and Triton deprecated (0.11.0). `core_map` was deprecated in favor of `pl.kernel` in 0.11.1 (2026-08-17).
7. **JetStream** was archived on 2026-02-01. Its functionality moved to tpu-inference / vLLM TPU.
8. **MaxText** migrated from Linen to Flax NNX (2026-06-29). It has a `src/maxtext` layout and uses Qwix for quantization; AQT is deprecated.

---

## 1. Papers

### 1.1 Pathways: Asynchronous Distributed Dataflow for ML (VERIFIED)
- Barham, Chowdhery, Dean, Ghemawat, Hand, Hurt, Isard, Lim, Pang, Roy, Saeta, Schuh, Sepassi, El Shafey, Thekkath, Wu. MLSys 2022. arXiv 2203.12533, submitted 2022-03-23.
- https://arxiv.org/abs/2203.12533 (PDF: https://arxiv.org/pdf/2203.12533)
- **Key ideas:**
  - A **single-controller** runtime, as opposed to the multi-controller SPMD model where each host runs the same program (JAX's classic multi-host model).
  - Programs form a sharded dataflow graph of asynchronous operators that pass futures.
  - Gang-scheduled dynamic dispatch over accelerator "islands" connected by ICI, with DCN between islands.
  - Parallel asynchronous dispatch hides the control-plane latency of the single controller.
  - Results: about 100% accelerator utilization on SPMD workloads at 2048 TPUs. Comparable throughput when pipelining a Transformer across 16 stages or across two islands over DCN.
- **Section map** (taken from the PDF text):
  - §2 Design Motivation: single- vs multi-controller (Fig. 2).
  - §3 Pathways Programming Model.
  - §4 System Architecture: 4.1 Resource Manager, 4.2 Client, 4.3 Coordination implementation (PLAQUE), 4.4 Gang-scheduled dynamic dispatch, 4.5 Parallel asynchronous dispatch, 4.6 Data management.
  - §5 Evaluation: 5.1 Single-controller dispatch overheads, 5.2 Multi-tenancy, 5.3 Large-scale model performance.
  - §6 Discussion: 6.1 Pathways design vs. alternatives, 6.2 Resource management, 6.3 Data-dependent vectorized control flow.
  - §7 Related Work.
- **Course relevance:**
  - Pathways on Cloud is how MaxText and vLLM do multi-host inference and weight transfer: https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro (linked from MaxText docs; URL-OK via MaxText docs, not fetched).
  - tpu-inference has `models/common/pathways_dummy_loader.py`.

### 1.2 GSPMD: General and Scalable Parallelization for ML Computation Graphs (VERIFIED)
- Xu, Lee, Chen, Hechtman, Huang, Joshi, Krikun, Lepikhin, Ly, Maggioni, Pang, Shazeer, Wang, Wang, Wu, Chen. arXiv 2105.04663. v1 2021-05-10, v2 2021-12-23.
- https://arxiv.org/abs/2105.04663
- **Key ideas:**
  - Write a single-device program, add a few sharding annotations, and the compiler propagates ("completes") shardings and partitions the program into one SPMD program per device.
  - The sharding representation is a device-assignment tensor that describes replicated, tiled, partially tiled, and manual shardings.
  - Pipelining is reduced to sharding along a stage axis.
  - Reported 50-62% compute utilization on up to 2048 TPUv3 cores, for models up to 1T parameters.
- **Section map:**
  - §2 Background: 2.1 parallelism patterns, 2.2 XLA and TensorFlow.
  - **§3 Tensor Sharding and Auto Completion:** 3.1 Representation (Replicated / Tiled / Partially Tiled), 3.2 in-operator parallelism examples, 3.3 Pipeline parallelism reduced to tensor sharding, 3.4 Manually partitioned subgraphs and pipelining (the ancestor of shard_map), 3.5 Intuitive sharding completion (propagation), 3.6 API integration.
  - **§4 The SPMD Partitioner:** 4.1 Static constraints, 4.2 Communication primitives, 4.3 Halo exchange with dynamic bounds, 4.4 Grouping and recursive partitioning, 4.5 Resharding, 4.6 Compiler optimizations for data formatting.
  - §5 Case studies: 5.2 pipelining + in-layer sharding, 5.3 Conformer, 5.4 Sparse Transformer, 5.5 Hybrid sparse/dense, 5.6 Image spatial partitioning.
- **Note:** In JAX, GSPMD *propagation* is now replaced by Shardy (see §2.5). The XLA SPMD partitioner (§4 ideas) still does the per-device lowering. The exact division of labor in current XLA is UNVERIFIED in detail.

### 1.3 Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models (partially VERIFIED)
- Shibo Wang, Jinliang Wei, Amit Sabne, Andy Davis, Berkin Ilbeyi, Blake Hechtman, Dehao Chen, Karthik Srinivasa Murthy, Marcello Maggioni, Qiao Zhang, Sameer Kumar, Tongfei Guo, Yuanzhong Xu, Zongwei Zhou. ASPLOS 2023 (vol. 1), pp. 93-106.
- The DBLP listing (title, venue, pages, Wang/Wei/Sabne/Xu) is VERIFIED via search. The full author list is from memory (UNVERIFIED).
- DOI: https://doi.org/10.1145/3567955.3567959. The DOI is from memory; doi.org/ACM returned 403 to the bot, so it is UNVERIFIED but likely correct.
- **Key ideas:**
  - Decompose an AllGather or ReduceScatter that feeds or consumes an einsum into a sequence of collective-permutes (ring steps).
  - Each step's partial einsum overlaps with the next step's transfer. This is "collective matmul" / windowed einsum in XLA.
  - Bidirectional variants use both ICI directions.
  - Implemented in the XLA SPMD partitioner, with evaluation on TPU. The paper reports end-to-end speedups for large models; exact numbers UNVERIFIED.
- **Where it shows up today:**
  - XLA flag `xla_jf_spmd_threshold_for_windowed_einsum_mib` (see §2.6).
  - tpu-inference kernel `tpu_inference/kernels/collectives/all_gather_matmul.py`, a Pallas collective matmul.
  - Pallas TPU distributed docs (remote DMA ring patterns).
  - Scaling book ch. 3 (sharding) discusses collective matmul.

### 1.4 Ragged Paged Attention (VERIFIED, new 2026)
- Jevin Jiang, Ying Chen, Blake A. Hechtman, Fenghui Zhang, Yarong Mu. arXiv 2604.15464, submitted 2026-04-16. https://arxiv.org/abs/2604.15464
- **Techniques:**
  - Fine-grained tiling for dynamic slicing over ragged memory.
  - A custom software pipeline that **fuses the KV-cache update with attention**.
  - **Distribution-aware compilation:** specialized decode / prefill / mixed kernels.
- **Results:** up to 86% MBU in decode and 73% MFU in prefill on Llama-3-8B on TPU7x.
- It is the primary TPU attention backend in vLLM and SGLang. Must-read for the inference module.

---

## 2. XLA / OpenXLA

TPU XLA backend source is **closed** (it ships in libtpu). The public docs describe HLO, the generic passes, the SPMD/Shardy layers, and TPU flags, but not TPU codegen internals.

### 2.1 Architecture and HLO
- XLA architecture: https://openxla.org/xla/architecture (URL-OK). Pipeline: StableHLO → HLO → target-independent passes → backend.
- **Operation semantics** (canonical HLO op reference: dot_general, all-gather, all-reduce, collective-permute, reduce-scatter, all-to-all, dynamic-slice, while, etc.): https://openxla.org/xla/operation_semantics (URL-OK).
- Shapes and layout (minor-to-major): https://openxla.org/xla/shapes (URL-OK).
- Terminology: https://openxla.org/xla/terminology (URL-OK).
- HLO passes overview: https://openxla.org/xla/hlo_passes (URL-OK).
- HLO dumps (`XLA_FLAGS=--xla_dump_to=...`, `--xla_dump_hlo_as_text`, pass-by-pass dumps): https://openxla.org/xla/hlo_dumps (URL-OK).
- Tools (hlo-opt, run_hlo_module, etc.): https://openxla.org/xla/tools (URL-OK).
- Async ops (VERIFIED): https://openxla.org/xla/async_ops
  - Generic `async-start` / `async-update` / `async-done` wrap any HLO op.
  - The parser desugars `-start` / `-done` suffixes.
  - `copy-start`/`copy-done` and `collective-permute-start`/`done` keep first-class forms.
  - In a trace you will see e.g. `all-gather-start` ... `all-gather-done`; the compute between the pair is the overlap window.
- LHS cost model (VERIFIED, **GPU-oriented**): https://openxla.org/xla/lhs_cost_model. Covers performance tables for collectives and GEMMs plus an analytical DCN S-curve. The page does not cover TPU.
- Fusion: public fusion docs are GPU-centric (https://openxla.org/xla/gpu_architecture, https://openxla.org/xla/emitters; URL-OK). TPU fusion decisions are internal. For TPU, teach fusion via HLO dumps and XProf (look at `fusion.N` ops, `kLoop`/`kInput`/`kOutput` fusion kinds, and convolution fusions for matmuls). TPU fusion-kind naming is UNVERIFIED for current libtpu.

### 2.2 StableHLO
- Home: https://openxla.org/stablehlo (URL-OK).
- Spec: https://openxla.org/stablehlo/spec, source at https://github.com/openxla/stablehlo/blob/main/docs/spec.md (URL-OK).
- Compatibility guarantees (versioned portable artifacts, used by jax.export): https://openxla.org/stablehlo/compatibility (URL-OK).
- StableHLO is the frontend IR shared by JAX, TorchTPU, and PyTorch/XLA. TorchTPU builds StableHLO per ATen op (see §7).

### 2.3 Layouts and tiling on TPU (VERIFIED): https://openxla.org/xla/tiled_layout
- Notation `F32[3,5]{1,0:T(2,2)}`, where `{1,0}` is minor-to-major and `T(...)` is the tile.
- Multi-level tiling, e.g. `(8,128)(2,1)`.
- Pad → reshape (a bitcast) → transpose view of tiling.
- TPU specifics:
  - **8x128** tiles for 32-bit types match the vreg (8 sublanes x 128 lanes).
  - Small second-minor dims use 2x128 or 4x128.
  - **bf16 uses `(8,128)(2,1)`**, packing two rows into one 32-bit word. **int8/fp8 use `(8,128)(4,1)`.**
  - PRED uses one byte per value.
- This explains the tpu-inference KV cache "packing" dimension (§6.4) and the Pallas block-shape rule (last two dims divisible by 8 and 128).
- JAX-side layout control: https://docs.jax.dev/en/latest/notebooks/layout.html ("Device-local array layout control", URL-OK).

### 2.4 SPMD partitioner (GSPMD)
- There is no standalone OpenXLA doc page for the SPMD partitioner (`openxla.org/xla/sharding` is a 404). Use the GSPMD paper §3-4 plus the Shardy docs.
- The partitioner source lives in the XLA repo under `xla/service/spmd/` (UNVERIFIED path; check before citing).

### 2.5 Shardy (VERIFIED)
- Home: https://openxla.org/shardy. An "MLIR-based tensor partitioning system for all dialects," built jointly by the GSPMD and PartIR teams. Dialect `sdy`.
- Subpages (all listed on the Shardy home page):
  - Overview: https://openxla.org/shardy/overview
  - Use Shardy from JAX: https://openxla.org/shardy/getting_started_jax
  - Sharding representation: https://openxla.org/shardy/sharding_representation
  - Propagation: https://openxla.org/shardy/propagation
  - Compiler API: https://openxla.org/shardy/compiler_api
  - SDY dialect: https://openxla.org/shardy/sdy_dialect
  - Import / export / propagation passes; MPMD dialect: https://openxla.org/shardy/mpmd/mpmd_dialect
- **Representation:**
  - A mesh has named axes.
  - Per-dimension shardings list axes major→minor.
  - **Open vs closed** dims: open dims can be sharded further by propagation.
  - **Priorities:** 0 is highest; lower numbers propagate first.
  - Explicitly vs implicitly replicated axes.
  - **Sub-axes** `"x":(m)k`.
- **Shardy vs GSPMD** (from getting_started_jax):
  - Named axes instead of `{devices=[4,1,2]<=[8]}`.
  - `sdy.sharding_constraint` is an op, not a custom call.
  - `{?}` marks open dims.
  - shard_map becomes `sdy.manual_computation`.
  - `custom_partitioning` uses an einsum-like `sharding_rule` string.
  - The post-propagation dump is `sdy_module_after_sdy_export.mlir` (under `XLA_DUMP_TO`).
  - `GSPMDSharding`, `PositionalSharding`, and `PmapSharding` are not supported.
- **JAX status:** default since **JAX 0.7.0 (2025-07-22)**. The flag `jax_use_shardy_partitioner` is now a no-op (JAX changelog, Unreleased after 0.12.0).
- There is also an **MPMD dialect** (`mpmd`) for pipeline-style MPMD partitioning. It is under-documented; describe it only as "exists."

### 2.6 TPU-relevant XLA flags (VERIFIED): https://openxla.org/xla/flags_guidance
On TPU, pass these via `LIBTPU_INIT_ARGS="--flag=value ..."`. The env var is standard practice but **not documented on this page**; it is documented in MaxText / tpu-recipes (UNVERIFIED citation).
- **Scheduler:**
  - `xla_tpu_enable_latency_hiding_scheduler` enables LHS, which allows async collectives. Disabling it saves memory but loses overlap.
  - `xla_latency_hiding_scheduler_rerun` reruns LHS with a lower memory limit each time.
- **Async collectives:**
  - `xla_enable_async_all_gather`, `xla_tpu_enable_async_collective_fusion`, `..._fuse_all_gather`, and `..._fuse_all_reduce` (v5e examples).
  - `xla_tpu_enable_async_all_to_all`.
  - `xla_enable_async_collective_permute`.
- **Pipelining ICI all-gathers:** `xla_tpu_enable_ici_ag_pipelining` (together with the loop-variant/invariant chain flags). `xla_tpu_enable_ag_backward_pipelining` pipelines all-gathers backwards through scan loops.
- **Collective matmul:** `xla_jf_spmd_threshold_for_windowed_einsum_mib` sets the minimum dot size that triggers windowed einsum.
- **Latency-bound (inference):** `xla_{all_gather,all_reduce,collective_permute,all_to_all}_latency_bound_threshold_in_bytes`. The default is -1 (disabled); the page says these are "typically used for inference."
- **Memory and VMEM:**
  - `xla_tpu_scoped_vmem_limit_kib` sets per-op scoped VMEM.
  - `xla_tpu_async_copy_bandwidth_scaling_factor` and `xla_tpu_msa_inefficient_use_to_copy_ratio` tune MSA (memory-space assignment: HBM↔VMEM prefetch).
  - `xla_tpu_rwb_fusion`.
- **Compute:** `xla_tpu_enable_dot_strength_reduction`, `xla_tpu_dot_dot_fusion`, `xla_jf_enable_multi_output_fusion`.
- **Data parallel and DCN:** `xla_tpu_enable_data_parallel_all_reduce_opt`, `xla_tpu_data_parallel_opt_different_sized_ops`.
- **Megacore:** `xla_tpu_megacore_fusion_allow_ags`.
- **Correctness:** `xla_mosaic_on_device_checks` adds bounds checks for Mosaic (Pallas) kernels.

---

## 3. JAX docs

- `jax.jit` API (in_shardings / out_shardings, donate_argnums, static args): https://docs.jax.dev/en/latest/_autosummary/jax.jit.html (URL-OK).
- `jax.sharding` API (Mesh, AbstractMesh, NamedSharding, PartitionSpec/`jax.P`, AxisType): https://docs.jax.dev/en/latest/jax.sharding.html (URL-OK).
- **Sharding guide, current:** https://docs.jax.dev/en/latest/201/sharding.html. This is the maintained location; the older `notebooks/Distributed_arrays_and_automatic_parallelization.html` / `parallel.html` say so.
  - Sections: Mesh; Sharding; **Explicit sharding mode ("sharding in types")**; `@auto_axes`; Auto sharding mode; Manual mode.
  - (VERIFIED via https://docs.jax.dev/en/latest/parallel.html; `notebooks/explicit-sharding.html` redirects there.)
  - Key concepts:
    - `AxisType.{Auto, Explicit, Manual}`.
    - `jax.typeof(x)` shows e.g. `f32[8@X, 4@Y]`.
    - `jax.reshard`.
    - `jax.set_mesh`, which replaced `use_mesh` in 0.7.1.
    - `auto_axes` / `explicit_axes`.
    - **unreduced** axes: `jax.P('X', None, unreduced={'Y'})`.
  - The page states "By default, all mesh axis types are `AxisType.Explicit`." The changelog (0.8.1) announced that `jax.make_mesh` would default to Explicit in 0.9.0; whether that landed is UNVERIFIED in the changelog text read.
- Older automatic-parallelization tutorial: https://docs.jax.dev/en/latest/sharded-computation.html (URL-OK).
- **shard_map:** API https://docs.jax.dev/en/latest/_autosummary/jax.shard_map.html; tutorial https://docs.jax.dev/en/latest/notebooks/shard_map.html (URL-OK).
  - `jax.experimental.shard_map.shard_map` was deprecated in JAX 0.8.0 (2025-10-15) in favor of `jax.shard_map`.
  - Since JAX 0.9.1, Explicit mode errors on PartitionSpec mismatch with `in_specs`.
- **jaxpr:** https://docs.jax.dev/en/latest/jaxpr.html (URL-OK). Use `jax.make_jaxpr` and `jit(f).lower(...).as_text()` for StableHLO, then `.compile().as_text()` for optimized HLO. AOT docs: https://docs.jax.dev/en/latest/aot.html (URL-OK).
- Changelog (VERIFIED): https://docs.jax.dev/en/latest/changelog.html. Key dates are in §0.
  - Also: `with mesh:` deprecated in 0.10.1 (2026-05-20).
  - `jax.sharding.no_collectives()` added (0.12.0 / Unreleased).
- Training cookbook: https://docs.jax.dev/en/latest/the-training-cookbook.html (URL-OK).
- **Scaling book** ("How to Scale Your Model", Austin et al., published 2025-02-04; VERIFIED): https://jax-ml.github.io/scaling-book/
  - Chapters: roofline, tpus, sharding, transformers, training, applied-training, inference, applied-inference, **profiling**, **jax-stuff** (Programming TPUs in JAX), conclusion, gpus.
  - Covered separately in `scaling_book_part1.md` and `scaling_book_part2.md`.

---

## 4. Pallas and Mosaic

### 4.1 Docs map
- Pallas index: https://docs.jax.dev/en/latest/pallas/index.html (VERIFIED). GPU and TPU backends "build on shared ideas (`Ref`s, `BlockSpec`s, and pipelining) but provide distinct, hardware-specific APIs."
- Quickstart: https://docs.jax.dev/en/latest/pallas/quickstart.html
- Grid and BlockSpec: https://docs.jax.dev/en/latest/pallas/grid_blockspec.html
- Generic pipelining: https://docs.jax.dev/en/latest/pallas/pipelining.html
- TPU index: https://docs.jax.dev/en/latest/pallas/tpu/index.html
- **TPU pipelining** (VERIFIED): https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html
- **Writing TPU kernels / details** (VERIFIED): https://docs.jax.dev/en/latest/pallas/tpu/details.html
- TPU matmul tutorial: https://docs.jax.dev/en/latest/pallas/tpu/matmul.html
- **Scalar prefetch and block-sparse** (the basis of paged attention): https://docs.jax.dev/en/latest/pallas/tpu/sparse.html
- **Distributed / remote DMA** (VERIFIED): https://docs.jax.dev/en/latest/pallas/tpu/distributed.html
- **Core-specific programming (`core_map` / `pl.kernel`)** (VERIFIED): https://docs.jax.dev/en/latest/pallas/tpu/core_map.html
- **SparseCore kernel writing** (VERIFIED): https://docs.jax.dev/en/latest/pallas/tpu/sparsecore.html
- Async ops design note: https://docs.jax.dev/en/latest/pallas/design/async_note.html
- GPU (Mosaic GPU): https://docs.jax.dev/en/latest/pallas/gpu/index.html, reference https://docs.jax.dev/en/latest/pallas/gpu/reference.html
- **Pallas changelog** (VERIFIED): https://docs.jax.dev/en/latest/pallas/CHANGELOG.html

### 4.2 Core TPU model (from the details and pipelining pages)
- **Memory spaces:**
  - HBM: `pallas_call` args are assumed to live in HBM.
  - **VMEM** ("16MB+") holds kernel-body refs by default.
  - **SMEM** handles scalars, control flow, and data-dependent indexing; 32-bit only.
  - **`pl.ANY`** means "compiler decides, usually HBM." It cannot be indexed directly; DMA it into VMEM/SMEM manually.
  - `pltpu.ANY` was removed in favor of `pl.ANY` in JAX 0.10.0. `pltpu.HOST` was deprecated in favor of `pl.HOST` in 0.10.2 and removed in 0.11.2.
- **BlockSpec rule:** the last two block dims must be divisible by 8 and 128, or equal the full dims. This follows directly from (8,128) tiling.
- **Grid:** runs **sequentially in lexicographic order** (unlike a CUDA grid). HBM transfers are skipped when consecutive iterations use the same block. Put the reduction axis last and accumulate in the output ref.
- **Pipelining:** the default is double buffering. `pipeline_mode=pl.Buffered(n)` sets the depth. `pltpu.emit_pipeline` builds nested pipelines inside a kernel and supports lookahead prefetch and dynamic block shapes (`pl.BoundedSlice`).
- **Megacore:**
  - Mark grid axes `dimension_semantics=("parallel","arbitrary")` via `pltpu.CompilerParams`. `TPUCompilerParams` was renamed in 0.7, with the old name removed in 0.8.1.
  - The pipelining page says the annotation is effective only on chips that present 2 TensorCores as one device (**v4, v5p**) and is a no-op elsewhere.
  - How v7x's dual chiplets are exposed is not stated on that page: UNVERIFIED (the SparseCore page says TPU7x has "2 SparseCores per chip (4 physical cores)").
- **Ops:**
  - The MXU accumulates in f32, and inputs are rounded to bf16 unless f32 precision is requested.
  - Elementwise ops are mostly 32-bit.
  - Transposes are limited to the last two dims.
  - Loops are fully unrolled.

### 4.3 2025-2026 features
- **`pl.kernel` / `core_map`:**
  - Per-physical-core programming: `pltpu.create_tensorcore_mesh('core')`, `pl.run_scoped` for VMEM and semaphores, barriers via `pl.semaphore_signal/wait`, and inter-core `pltpu.make_async_remote_copy(device_id={'core':...})`.
  - `pltpu.emit_pipeline` splits PARALLEL axes across cores.
  - `core_map` was deprecated in favor of `pl.kernel` in 0.11.1 (2026-08-17). `pl.kernel` takes `out_type` / `scratch_types` since 0.10.0.
- **SparseCore:**
  - `plsc.VectorSubcoreMesh`, `plsc.ScalarSubcoreMesh`; `pltpu.get_tpu_info().sparse_core`.
  - Gather/scatter by indexing a Ref with an index Ref. DMAs are 32-bit only, so pack bf16 pairs.
  - Hardware: v5p 4 SC/chip; v6e 2 SC; 7x 2 SC (4 physical), SIMD 16 f32 / 32 bf16; 16 vector subcores each.
  - The docs example measured a gather at 4.05 ms on SparseCore vs 18.1 ms with `jnp.take` on the TensorCore.
  - SparseCore via `pallas_call` was removed in 0.10.1; use `pl.kernel`.
  - tpu-inference uses this in `kernels/sparse_core/` (ragged gather/scatter for MoE).
- **Remote DMA collectives:**
  - Push-only RDMA via `make_async_remote_copy(src, dst, send_sem, recv_sem, device_id)`.
  - Barrier semaphores need a `collective_id`; automatic collective-ID assignment arrived in 0.10.2.
  - Examples on the distributed page: ppermute, ring all-gather, all-reduce with double-buffering, bidirectional reduce-scatter, and nested remote+local pipelines.
  - Megacore guidance: issue DMAs from one core.
- **Other changelog items:**
  - `jax.debug.print` works in kernels (0.7).
  - `tpu.get_tpu_info()` (0.8.1).
  - `enable_poison_buffers` (0.10.2).
  - `tpu.annotate()` and `get_barrier_semaphore(tag=...)` (0.11.2).
  - `ref.memory_space_cast` and `tpu.has_memory_space()` (0.12.0, 2026-10-19).
  - `DeviceIdType` deprecated (0.12.0).

### 4.4 Mosaic TPU vs Mosaic GPU
- **Mosaic TPU** is the MLIR-based compiler that Pallas TPU lowers into. It emits a custom call that XLA embeds in the program. The TorchTPU docs call "StableHLO and (even more so) Mosaic" roughly analogous to PTX.
- **Mosaic GPU** is a separate MLIR backend for Hopper and Blackwell. It became the default Pallas GPU lowering in JAX 0.9.0; the Triton backend was deprecated in 0.11.0 and the `pallas.ops.gpu` Triton ops in 0.11.1, with Tokamax suggested as the replacement.
- The two backends share Pallas concepts (Refs, BlockSpecs, pipelining) but have hardware-specific APIs: warpgroups, wgmma, TMA, and SMEM barriers on GPU vs VMEM, SMEM, DMA semaphores, and the MXU on TPU.

### 4.5 Attention kernels
- **Splash attention** (block-sparse flash attention for training/prefill, with mask types and segment ids):
  - In JAX at https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/tpu/splash_attention (URL-OK).
  - Also packaged in **Tokamax**: https://github.com/openxla/tokamax (URL-OK).
  - MaxText has `kernels/tokamax_splash_attention` and the `use_tokamax_splash` flag.
- **Ragged paged attention:**
  - Upstream JAX copy at https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/tpu/ragged_paged_attention (URL-OK).
  - The production copy lives in tpu-inference: `tpu_inference/kernels/ragged_paged_attention/{v2,v3}`. v3 has `kernel.py`, `kernel_hd64.py`, and `tuned_block_sizes*.py`.
  - The v3 kernel signature (from source, VERIFIED): queries `[max_num_tokens, num_q_heads, head_dim]`, keys/values `[max_num_tokens, num_kv_heads, head_dim]`, and `kv_cache [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]`. Metadata: `kv_lens i32[max_num_seqs]`, `page_indices i32[max_num_seqs*pages_per_seq]`, `cu_q_lens i32[max_num_seqs+1]`, and `distribution i32[3]`, which gives the cumulative counts of decode / prefill / mixed sequences. `RpaCase` = DECODE, PREFILL, MIXED.
  - Paper: §1.4.
- Other tpu-inference kernels (`tpu_inference/kernels/`, from source tree): `flash_attention`, `mla`, `fused_moe`, `megablox` (gmm), `quantized_matmul`, `gdn` (Gated DeltaNet), `causal_conv1d`, `collectives/all_gather_matmul.py`, `sparse_core/*`, `structured_sparse_matmul`.

---

## 5. MaxText, JetStream, MaxDiffusion

### 5.1 MaxText (VERIFIED): https://github.com/AI-Hypercomputer/maxtext, docs https://maxtext.readthedocs.io
- A pure Python/JAX reference LLM library that "relies on the XLA compiler" and stays largely optimization-free.
- Models: Gemma 1-4, Qwen 2.5/3/3.5/3.8, DeepSeek V2-V4/R1, Kimi K2, Llama 2-4, GPT-OSS, and Mixtral.
- Post-training via **Tunix** (SFT, GRPO, GSPO), with vLLM used for RL sampling.
- **News:**
  - `src` layout (2025-09-05), PyPI (2025-09).
  - Restructure (2026-02-27).
  - **Linen → Flax NNX (2026-06-29)**.
  - Streaming DiLoCo (2026-08-19).
  - DeepSeek V4 Pro / Qwen 3.8 (2026-10).
- **Layout** (from the repo tree): `src/maxtext/{configs,layers,kernels,models,inference,trainers,...}`. `configs/base.yml` holds all defaults.
- **Mapping to scaling-book ideas** (from `base.yml` and the docs):
  - The mesh axes `mesh_axes: ['diloco','data','stage','fsdp','fsdp_transpose','context','context_usp_ulysses','context_autoregressive','tensor','tensor_sequence','expert','autoregressive']` map to scaling-book ch. 5 (DP, FSDP, TP, PP, plus EP and CP).
  - The **`ici_*_parallelism`** and **`dcn_*_parallelism`** pairs form a hierarchical mesh: ICI within a slice, DCN across slices. The recommended defaults are `ici_fsdp_parallelism: -1` and `dcn_data_parallelism: -1`. `dcn_tensor_parallelism` is commented "never recommended," which matches the scaling book's ICI vs DCN bandwidth reasoning.
  - **`logical_axis_rules`** map logical names (embed, mlp, heads, ...) to mesh axes, the Flax-style logical sharding pattern.
  - **`shard_mode: auto | explicit`** selects JAX sharding-in-types. `docs/guides/optimization/shard_mode_performance.md` is a measured explicit-vs-auto study (2026-08/09, v5p, JAX 0.11.1).
  - The sharding guide `docs/guides/optimization/sharding.md` is explicitly built on arithmetic intensity and links scaling-book ch. 3. It covers DP, FSDP, CP, TP, TP-transpose, EP, and PP (gPipe, circular) with per-strategy arithmetic-intensity derivations, and notes that an ICI all-to-all costs about 1/4 of a full all-gather.
  - Other guides: `pallas_kernels_performance.md`, `fsdp_prefetch.md`, `custom_mesh_and_rule.md`, `benchmark_and_performance.md`.
  - Other relevant knobs: `remat_policy`, `attention: autoselected|dot_product|flash|...`, `sa_block_*` (splash block sizes), `megablox` / `sparse_matmul` / `use_tokamax_gmm` for MoE.
  - Quantization: `use_qwix_quantization`. The base.yml comment reads "[DEPRECATED: AQT will be removed ...] strongly recommended to set use_qwix_quantization to true."
  - Reference doc `docs/reference/core_concepts/jax_xla_and_pallas.md` exists (not read).

### 5.2 JetStream (VERIFIED): https://github.com/AI-Hypercomputer/JetStream
- The README warning says: "we will be archiving Jetstream on February 1st 2026." The core functionality moved to tpu-inference / tpu.vllm.ai.
- Historical engines: MaxText (JAX) and jetstream-pytorch. Teach it as history; the continuous-batching / prefill-insert-generate design was absorbed into vLLM TPU.

### 5.3 MaxDiffusion (VERIFIED): https://github.com/AI-Hypercomputer/maxdiffusion
- Pure JAX latent diffusion reference: SD 2.x, SDXL, Flux dev/schnell, Flux.2-Klein, LTX-Video/LTX-2, Wan2.1/2.2.
- Latest updates (Jul-Sep 2026): Wan2.2 dual-expert training, Flux.2-Klein editing with KV caching, attention tile-size search.

---

## 6. tpu-inference (vLLM JAX), torchax, Qwix

### 6.1 Repo status (VERIFIED): https://github.com/vllm-project/tpu-inference
- The README says the repo "is moving to" `vllm-project/vllm-jax` and that "PyTorch-native TPU serving has moved to its own project" (vllm-torchtpu).
- It still describes a backend "unifying JAX and PyTorch under a single lowering path."
- Docs: https://docs.vllm.ai/projects/tpu/en/latest/ (VERIFIED). Recipes: AI-Hypercomputer/tpu-recipes.
- Support matrix (dashboard 2026-08-27):
  - Release-validated models include Gemma 3/4, Llama 3.1-8B / 3.3-70B, Qwen3 / Qwen3.5-397B, DeepSeek-R1, Kimi-K2.6, and gpt-oss-120b.
  - Parallelism (PP/DP/EP/TP/CP/SP) is tabulated separately for **Flax** and **Torchax** columns.
  - Recommended TPUs: v7x, v5e, v6e. Experimental: v3, v4, v5p.
- PyPI tpu-inference **0.32.0** (2026-10-22).

### 6.2 Architecture (blog + source, VERIFIED)
- **Blog:** "vLLM TPU: A New Unified Backend Supporting PyTorch and JAX on TPU," 2025-10-16, https://vllm.ai/blog/vllm-tpu (also blog.vllm.ai/2025/10/16/vllm-tpu.html).
  - "vLLM TPU now uses JAX as the lowering path for all vLLM models."
  - It defaults to the TPU-optimized JAX model in tpu-inference if one exists, and otherwise runs upstream vLLM PyTorch model code via torchax.
  - Torchax-lowered and naive JAX reimplementations performed about the same.
  - RPA v3 fuses the KV-cache update ("completely hides scatter latency") and compiles to 3 sub-kernels (prefill-only / decode-only / mixed).
  - SPMD is native to XLA.
  - Numbers: 3.6x (Llama-3.1-8B, v6e-1) and 2.1x (70B, v6e-8) over the PyTorch/XLA backend; about 20% from JAX lowering; about 10% from RPA v3 over v2; "nearly 5x" over the Feb 2025 prototype.
- **Model selection** (source, `tpu_inference/envs.py`, `models/common/model_loader.py`):
  - `MODEL_IMPL_TYPE` ∈ {`auto`, `vllm`, `flax_nnx`, `jetpack`}; default `auto`.
  - `auto` resolves to `flax_nnx` unless the architecture is in `_VLLM_PREFERRED_ARCHITECTURES` or JAX doesn't support it. Fallback is `vllm`, i.e. the torchax path through `models/vllm/vllm_model_wrapper.py`.
  - `jetpack` is not explained in source comments read (UNVERIFIED meaning).
  - Native JAX models live in `tpu_inference/models/jax/` (llama3, llama4, qwen2, qwen3, qwen2_5_vl, deepseek_v3, gemma4, gpt_oss, eagle3/dflash).
- **Package layout:** `core`, `distributed`, `executors`, `kernels`, `layers/{common,jax,vllm}`, `lora`, `models/{common,jax,vllm}`, `offload`, `platforms`, `rl`, `runner`, `spec_decode`, `worker`.
- **Runner** (`tpu_inference/runner/`):
  - `tpu_runner.py`, `kv_cache.py`, `kv_cache_manager.py`, `block_table.py`, `input_batch.py`, `persistent_batch_manager.py`, `compilation_manager.py` (precompiles shape buckets so nothing compiles during serving), `decode_loop.py`, `speculative_decoding_manager.py`, `structured_decoding_manager.py`.
  - Continuous batching and scheduling come from **upstream vLLM v1** (scheduler, KV block manager). TPU code supplies the worker/runner and padded shape buckets (inference from structure plus the torchax guide).
- **torchax model guide** (VERIFIED): `docs/developers_guide/torchax_model_development.md` (Siyuan Liu, Hongmin Fan, Han Qi; updated 2025-09-26).
  - The JAX worker calls a **pure function** `model_fn(weights, kv_caches, inputs)`.
  - PyTorch modules are made pure via `torch.func.functional_call`.
  - The KV cache is passed in and **returned**, functional update style.
  - "Mental model: `torchax` is `JAX`." A "sandwich": torch model code in the middle, with JAX inputs/outputs, JAX sampler, JAX KV cache, and the Pallas RPA kernel around it. `torch_view` / `jax_view` convert at the boundaries.
- **JAX model guide:** `docs/developers_guide/jax_model_development.md`.
  - Flax NNX `JaxModule` + `LoadableWithIterator`.
  - Forward signature `__call__(kv_caches, input_ids, attention_metadata) -> (kv_caches, hidden, aux)`; `compute_logits`.
  - Reusable `layers/jax` blocks (JaxLinear/JaxEinsum/JaxMoE), with weight loading reused from vLLM.
- Profiling guide: `docs/developers_guide/profiling.md` (see §8).

### 6.3 Sharding (source: `tpu_inference/layers/common/sharding.py`)
- Mesh axes: `("data","attn_dp","attn_dp_expert","expert","model","dcp","pcp")`. The default 2D mesh is `('data','model')`.
- Logical→mesh mapping classes: `ShardingAxisName2D` (default) and `ShardingAxisNameBase` (used when `USE_2D_TP` or `NEW_MODEL_DESIGN` is set).
  - In 2D: ATTN_HEAD = KV_HEAD = MLP_TENSOR = EXPERT = `'model'`, BATCH = `'data'`, VOCAB = `('data','model')`.
  - DCP/PCP are decode / prefill context parallelism.
- Other env toggles: `NUM_SLICES` (multi-slice), MoE kernel toggles, `ENABLE_QUANTIZED_MATMUL_KERNEL`.

### 6.4 KV cache layout (source: `tpu_inference/runner/kv_cache.py`, VERIFIED)
- Per layer: **`(num_blocks, block_size, cdiv(num_kv_heads*2, packing), packing, head_dim)`**, where `packing = 32 // dtype_bits`.
- K and V are interleaved on the head axis (x2) and packed into 32-bit sublanes. This mirrors the `(8,128)(2,1)` / `(4,1)` tiling in §2.3.
- Block size scales by the context-parallel axes. KV heads are sharded on the KV_HEAD axes.
- MLA has its own shape (`mla.get_kv_cache_shape`).

### 6.5 Quantization in tpu-inference
- Checkpoint formats (README matrix, loading-compat only): mxfp4, compressed-tensors FP8/INT8, AWQ INT4, modelopt NVFP4.
- **Qwix for online quantization of JAX models** (jax_model_development.md):
  `--additional_config='{"quantization": {"qwix": {"rules": [{"module_path": ".*", "weight_qtype": "float8_e4m3fn", "act_qtype": "float8_e4m3fn"}]}}}'`
- YAML configs: `tpu_inference/models/jax/utils/quantization/configs`.
- Code: `models/jax/utils/qwix/qwix_utils.py`.

### 6.6 torchax (VERIFIED): https://github.com/google/torchax
- Moved from https://github.com/pytorch/xla/tree/master/torchax on **2025-10-06**. PyPI 0.0.13 (2026-06-17).
- **How torch ops lower to JAX:**
  - `torchax.tensor.Tensor` is a `torch.Tensor` subclass holding a `jax.Array`.
  - It overrides **`__torch_dispatch__`**. Each ATen op is swapped for its JAX implementation (op registry in `docs/ops_registry.md`): unwrap → run the jnp/lax implementation → re-wrap.
  - `torchax.enable_globally()` turns this on.
  - `torchax.interop.jax_jit` / `JittableModule` let `jax.jit` trace torch code. Use a functional model via `torch.func.functional_call`.
- So a PyTorch model under torchax becomes a jaxpr → StableHLO → XLA program. JAX sharding (NamedSharding), Pallas kernels, and Qwix-style JAX tooling all apply.
- Torchax's long-term status relative to TorchTPU is **not stated** in its README. A SemiAnalysis glossary calls TorchTPU its replacement (secondary source; https://inferencex.semianalysis.com/glossary/torchax, UNVERIFIED as official).

### 6.7 Qwix (VERIFIED): https://github.com/google/qwix
- A JAX quantization library: QAT (fake quant), PTQ (serving on XLA TPU/GPU), ODML (LiteRT), LoRA/QLoRA.
- Schemes: weight-only, dynamic-range, static-range. Native int4/int8/fp8; emulated int1-7 and nf4. Calibration: absmax, minmax, rms, fixed.
- Works with Flax Linen and NNX "via a single function call." Regex `QuantizationRule(module_path=..., weight_qtype=..., act_qtype=...)`.
  ```python
  rules = [qwix.QuantizationRule(module_path='.*', weight_qtype='int8', act_qtype='int8')]
  ptq_model = qwix.quantize_model(model, qwix.PtqProvider(rules))
  ptq_params = qwix.quantize_params(fp_params, abs_ptq_params)
  ```
- **Mechanism** (from source, `qwix/_src/interception.py`, "Intercepts Python functions by patching"):
  - Providers intercept `jax.lax.dot_general`, `jax.numpy.einsum`, `jax.lax.conv_general_dilated`, and `jax.lax.ragged_dot` while the model runs.
  - Interception is disabled inside `pallas_call`.
- **Pallas support:** `qwix.pallas` (`qpl.quantize`, `qpl.pallas_call` with auto-adjusted BlockSpecs for QArrays, `qpl.dot_general/einsum` inside kernels).
- `qwix/contrib/kernels/` contains quantized-matmul kernels, including fused Hadamard ones.
- Version at HEAD and on PyPI: 0.1.8 (PyPI 2026-06-22). The README says it is not on PyPI, which is outdated.

---

## 7. TorchTPU and PyTorch/XLA

### 7.1 Primary sources (all VERIFIED)
- **Announcement / design blog:** "TorchTPU: Running PyTorch Natively on TPUs at Google Scale," Google Developers Blog, **2026-04-07**. Authors: Claudio Basile, Kat Ko, Ben Wilson, Lee Howes, Bill Jia, Joe Pamer, Michael Voznesensky, Robert Hundt. https://developers.googleblog.com/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/
- **Open-source launch:** "TorchTPU is now open source," Google Cloud Blog, **2026-10-20** (PyTorch Conference 2026). https://cloud.google.com/blog/topics/ai-infrastructure/torchtpu-is-now-open-source-run-native-pytorch-on-tpus
- **Repo:** https://github.com/google-pytorch/torch_tpu (Apache-2.0; Bazel build; Python 3.11+; `pip install torch_tpu==0.2.0 --extra-index-url https://download.pytorch.org/whl/cpu`).
- **Docs:** https://torchtpu.readthedocs.io (0.2.0).
  - How TorchTPU works: https://torchtpu.readthedocs.io/en/0.2.0/how_torchtpu_works/
  - Execution model: https://torchtpu.readthedocs.io/en/0.2.0/torchtpu_execution_model/
  - Memory model: `/torchtpu_memory_model/` (not read)
  - Distributed: https://torchtpu.readthedocs.io/en/0.2.0/distributed_support/
  - Custom kernels: https://torchtpu.readthedocs.io/en/0.2.0/custom_kernels/
  - Dynamic shapes: `/dynamic_shapes/` (not read)
  - XProf profiling: `/performance_profiling_with_xprof/` (not read)
  - Feature set and roadmap: `/feature_set/` (not read)
- **Google Cloud doc:** "PyTorch on TPU": https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu (updated 2026-10-22). "TorchTPU replaces PyTorch/XLA." The old approach required lazy-tensor graph recording and marking compilable regions; TorchTPU runs eagerly by default.
- **Inference engines blog:** "Announcing New vLLM and SGLang Inference Engines on TPU," 2026-10-20 (Qi Zhou, Chris Chan): https://developers.googleblog.com/announcing-new-vllm-and-sglang-inference-engines-on-tpu/
- Related repos (URL-OK):
  - https://github.com/vllm-project/vllm-torchtpu
  - https://github.com/sgl-project/sglang-torchtpu
  - https://github.com/google/tpu-sync (KV transfer library)
  - https://github.com/google-pytorch/tpu-agent-suite
- Forum post: https://dev-discuss.pytorch.org/t/now-open-source-torchtpu-and-the-tpu-agent-suite-for-native-pytorch-on-tpus/3467 (not fetched).

### 7.2 Architecture (from "How TorchTPU works" + execution model + blog)
- **Out-of-tree PyTorch backend on the `PrivateUse1` dispatch key**, renamed to `tpu`.
  - It autoloads on import, enabling `torch.device("tpu")` and `torch.accelerator`.
  - "No subclasses, no wrappers; just ordinary ... PyTorch Tensors on a TPU." Contrast torchax's Tensor subclass and PyTorch/XLA's `xla` device lazy tensors.
- **Op tracing at ATen level.**
  - Each dispatched op creates a **Deferred Op** holding a **StableHLO builder** (e.g. `aten::mul` → `stablehlo.multiply`). The returned `at::Tensor`'s storage points to it.
  - View semantics and in-place mutation are handled by view decomposition/inversion, since StableHLO is SSA.
- **Execution modes** (execution model doc):
  - **Strict eager** (default): every op flags its stream for execution, compiled singly at `-O1`. Still asynchronous.
  - **Debug eager:** `TORCH_TPU_LAUNCH_BLOCKING=1`; the host blocks after every op.
  - **Optimized / "fused" eager:** `TORCH_TPU_DEFER_AND_FUSE=1`. Ops stay deferred until a sync point (D2H copy, explicit sync, data-dependent shape) or until a repeated op loop is detected. The deferred region is compiled at `-O2` as one graph. The blog says "50% to 100+%" over strict eager; the docs say "up to 50%".
  - **`torch.compile(model, backend="tpu")`:** a TorchDynamo backend. The FX graph becomes a single StableHLO program, then XLA. **"TorchTPU does not use TorchInductor."**
- **Runtime:**
  - Async dispatch via a background thread → **PjRT** → libtpu executables.
  - Errors may surface late.
  - **3-tier compile cache:** process-local, machine-local shared memory, remote persistent. Keys cover graph, shapes, compiler options, and hardware/libtpu/PjRT versions.
  - First startup compile can take minutes (vllm-torchtpu suggests `VLLM_XLA_CACHE_PATH`).
- **Distributed:**
  - A `tpu_dist` c10d ProcessGroup. Collectives lower to StableHLO collectives compiled by XLA ("No separate library like NCCL").
  - Works with DDP, **FSDPv2**, **DTensor**, and SimpleFSDP.
  - `torch.tpu.distributed.topology_aware_mesh()` builds a DeviceMesh aligned with ICI rings.
  - **SPMD vs MPMD:** XLA wants identical programs per rank, but PyTorch code can be MPMD (rank-conditional code). TorchTPU isolates collectives by default ("skinny" collectives). `@spmd_safe` and `TORCH_TPU_SPMD_HANDSHAKE_STAGE` (`OFF` | `COMPILE_STAGE` | `DISPATCH_STAGE`) allow fusing and overlap. PyTorch/XLA supported only pure SPMD.
- **Custom kernels:**
  - Pallas via JAX, registered with **`torch.tpu.pallas.jax_op("ns::name", ...)`** (docs). The blog instead names `@torch_tpu.pallas.custom_jax_kernel`; the API name may have changed between April and October, so treat the docs as current.
  - **Tokamax** kernels (MoE, attention, TopK) are also usable.
  - **Helion** support is experimental.
- **Dynamic shapes:** bounded dynamism in XLA is the stated compiler focus (blog roadmap).
- The docs mention "TPU 8i" (288 GB HBM at 8.6 TB/s) in the memory section. Hardware generation naming is outside this note's scope (UNVERIFIED here).
- **Claims:** 52% MFU training Qwen3-8B on up to 8,192 v7x chips via TorchTitan (secondary reports of Google figures). "Thousands" of HF models validated. HF Transformers v5 supports `device_map="tpu"`.

### 7.3 How TorchTPU differs from JAX (synthesis for the course)
| Aspect | JAX | TorchTPU |
|---|---|---|
| Frontend | Trace pure functions to jaxpr | Eager ATen dispatch (PrivateUse1); Dynamo for compile |
| Default execution | Op-by-op dispatch (each primitive compiled), `jit` for real perf | Strict eager (per-op `-O1` executables), fused-eager, `torch.compile` |
| IR to XLA | jaxpr → StableHLO | Deferred Op → StableHLO (per op, per fused region, or per FX graph) |
| Parallelism model | Global-view SPMD: NamedSharding + Shardy propagation; `shard_map` for manual | Per-rank PyTorch (MPMD-capable): torch.distributed, DTensor, FSDP2; collectives as StableHLO ops; SPMD handshake to fuse |
| Runtime | PjRT + libtpu; multi-controller (or Pathways single-controller) | PjRT + libtpu; one process per TPU device (`tpu_dist`) |
| Kernels | Pallas native | Pallas via `torch.tpu.pallas.jax_op`, Tokamax, Helion (exp.) |
| Mutation | Functional (donation for in-place) | Real in-place / views emulated over SSA |
The "Default execution" and "Runtime" rows are partly inference from the docs. The "one process per device" claim comes from "Each worker is bound to one TPU device" in the distributed doc.

### 7.4 vLLM / SGLang on TorchTPU (VERIFIED)
- **vllm-torchtpu:**
  - An out-of-tree vLLM platform plugin running upstream vLLM PyTorch model definitions on TorchTPU via `torch.compile`.
  - Pallas kernels called as custom ops in the compiled graph: RPA, dense/sparse MLA, Gated DeltaNet, fused MoE with SparseCore offload.
  - Quantized matmul: FP8, INT8, MXFP4.
  - Parallelism: TP, DP, PP, EP, prefill/decode CP; P/D disaggregation.
  - Python 3.12. v7x is recommended and CI-validated; v6e is supported; v5e/v5p are experimental.
- **Inference blog details:**
  - Upstream scheduler, KV cache manager, and API server are preserved.
  - Prefix caching includes hybrid models. Host-DRAM / Managed Lustre KV offload, FP8 KV.
  - Disaggregated serving uses **TPU Sync**, which reads device buffers behind torch tensors directly.
  - A Pallas **megakernel** runs a whole decode step for Kimi K3.
  - The JAX engines "will continue to be maintained."
- **Implication for the learner:** for a PyTorch model port started in 2026, the default recommendation is now vllm-torchtpu. tpu-inference's torchax path still works but is no longer where PyTorch-native development happens. Qwix applies to the JAX path; vllm-torchtpu quantization is described only as FP8/INT8/MXFP4 matmuls, and its Qwix relation is UNVERIFIED.

### 7.5 PyTorch/XLA status (VERIFIED): https://github.com/pytorch/xla
- README: "This project is deprecated and replaced by TorchTPU." "Please start new PyTorch TPU workloads with TorchTPU and migrate existing PyTorch/XLA workloads." "The PyTorch/XLA repo will be archived on November 15, 2026."
- Last PyPI release: torch_xla 2.9.0 (2025-11-17). The README's stable install still shows 2.8.0.
- Architecture (for contrast): a **lazy tensor** `xla` device that records an IR graph and compiles/executes on `mark_step` / sync. It offered an SPMD mode and a Dynamo bridge. Docs: https://docs.pytorch.org/xla/master/ (URL-OK).
- AWS Neuron also dropped PyTorch/XLA for Trainium training as of Neuron 2.30 (2026) in favor of "TorchNeuron" (secondary context; https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/announcements/neuron2.x/announce-no-support-pytorch-xla-training.html, not fetched).
- Historical: "PyTorch 2.0 & XLA: the path forward" https://pytorch.org/blog/pytorch-2.0-xla-path-forward/ (not fetched).

---

## 8. XProf / profiling

- **XProf home** (VERIFIED): https://openxla.org/xprof. Repo: https://github.com/openxla/xprof.
  - Overhead is under 1% on TPU. XProf runs standalone (`xprof --logdir ...`) or as the TensorBoard "Profile" plugin; TensorBoard is now optional.
- **Tool pages:**
  - Overview page: https://openxla.org/xprof/overview_page
  - **Trace Viewer:** https://openxla.org/xprof/trace_viewer (device timelines per TensorCore; XLA ops / XLA modules / steps rows; host threads)
  - **Graph Viewer:** https://openxla.org/xprof/graph_viewer (HLO graph)
  - **HLO Op Profile:** https://openxla.org/xprof/hlo_op_profile (per-op time and FLOP/bandwidth utilization by category)
  - **HLO Op Stats:** https://openxla.org/xprof/hlo_op_stats
  - Framework Op Stats: https://openxla.org/xprof/framework_op_stats
  - **Memory Viewer:** https://openxla.org/xprof/memory_viewer (static HBM allocation over program order, peak usage)
  - Memory Profile: https://openxla.org/xprof/memory_profile
  - **Roofline Model** (VERIFIED): https://openxla.org/xprof/roofline_model. Not `roofline_analysis`, which is a 404.
    - Plots OI (FLOPs/byte) vs achieved FLOP/s, with **HBM, VMEM, and CMEM (v4 only)** rooflines on TPU.
    - Has a program-level view and a per-op view of the top 1,000 ops, with filters by bottleneck resource.
    - Slanted region means memory-bound; flat means compute-bound.
  - Megascale Viewer (multi-slice DCN): https://openxla.org/xprof/megascale_viewer
  - Utilization Viewer: https://openxla.org/xprof/utilization_viewer (URL-OK)
  - Perf Counters: https://openxla.org/xprof/perf_counters (URL-OK)
  - GPU Kernel Stats: https://openxla.org/xprof/gpu_kernel_stats
- **Capture docs:**
  - General: https://openxla.org/xprof/capturing_profiles
  - **JAX:** https://openxla.org/xprof/jax_profiling (VERIFIED)
  - Advanced options: https://openxla.org/xprof/advanced_profiler_options
  - Custom-call (Pallas) profiling: https://openxla.org/xprof/custom_call_profiling
  - Kernel profiling guide: https://openxla.org/xprof/kernel-profiling
  - Memory leaks: https://openxla.org/xprof/memory_leaks
  - PyTorch/XLA: https://openxla.org/xprof/pytorch_xla_profiling
  - These URLs are from the XProf nav listing; only jax_profiling, roofline_model, and the core tool pages were HTTP-checked.
  - JAX profiling page: https://docs.jax.dev/en/latest/profiling.html (sections "General options" and "Advanced configuration options"). API: https://docs.jax.dev/en/latest/jax.profiler.html.
- **Capture recipes:**
  - `jax.profiler.start_trace(dir)` ... `y.block_until_ready()` ... `stop_trace()`, or `with jax.profiler.trace(dir):`.
  - `jax.profiler.start_server(9999)` plus "CAPTURE PROFILE" in the UI at `localhost:9999`.
  - Annotations: `jax.profiler.TraceAnnotation`, `annotate_function`. `StepTraceAnnotation` exists in the API but isn't on that page.
  - `ProfileOptions`: `host_tracer_level` 0-3 (default 2), `device_tracer_level`, `python_tracer_level`, `advanced_configuration` (e.g. `{"tpu_trace_mode": "TRACE_ONLY_HOST"}`).
- **tpu-inference profiling** (VERIFIED, `docs/developers_guide/profiling.md`):
  - `examples/tpu_profiling.py --model ... --input-len --output-len --batch-size`. Profile prefill with batch 1 and a long input; profile decode with input/output len 1 and batch 256.
  - **`PHASED_PROFILING_DIR`** auto-captures prefill-heavy (≥0.9 prefill share), decode-heavy (≤0.2), and mixed (0.4-0.6) batches.
  - **`USE_JAX_PROFILER_SERVER=True`** with `JAX_PROFILER_SERVER_PORT` (default 9999) enables on-demand capture from TensorBoard.
  - Profiler options go through `profile_prefix` as `key:value;...`; unknown keys are forwarded to `advanced_configuration`, e.g. `tpu_perf_counters`.
- TorchTPU has its own XProf guide (`/performance_profiling_with_xprof/`, not read).
- Scaling book ch. 9 "How to Profile TPU Code": https://jax-ml.github.io/scaling-book/profiling/

---

## 9. Open items / UNVERIFIED list
- ASPLOS'23 collective-matmul paper: full author list, DOI, and exact speedups (ACM blocked).
- Exact meaning of `MODEL_IMPL_TYPE=jetpack` in tpu-inference.
- Whether `jax.make_mesh` now defaults to `AxisType.Explicit` (the guide says axes default to Explicit; the changelog confirmation was not read).
- Megacore / dual-chiplet exposure on v7x in Pallas: the pipelining doc says megacore applies only to v4/v5p.
- Whether vllm-torchtpu uses Qwix.
- TorchTPU custom-kernel API naming: blog `@torch_tpu.pallas.custom_jax_kernel` vs docs `torch.tpu.pallas.jax_op`.
- The `LIBTPU_INIT_ARGS` convention is not on the OpenXLA flags page.
- The XLA SPMD partitioner source path, and the current split between Shardy and the SPMD partitioner inside XLA.
