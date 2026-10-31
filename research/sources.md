# Sources

Every source the course cites, with an ID used for inline citations in chapter content (e.g. `[SB2 §tpu-specs]`, `[D-7x]`).
**Accessed 2026-10-23** for every entry unless a different date is given. Raw reading notes, with page-level detail and every problem-set answer, are in `research/notes/`:

| Notes file | Covers |
|---|---|
| `notes/scaling_book_part1.md` | Scaling Book intro + ch. 1–6, all problems and answers |
| `notes/scaling_book_part2.md` | Scaling Book ch. 7–12, all problems and answers |
| `notes/tpu_hardware.md` | TPU papers, Cloud docs, launch blogs, generation table, conflicts register |
| `notes/tpu_software.md` | XLA, JAX, Pallas, MaxText, tpu-inference, torchax, Qwix, TorchTPU, XProf |
| `notes/comparisons.md` | NVIDIA, Trainium, Megatron/NCCL literature, dated recent-changes list |

Status labels: **read** = fetched and read in full or in the cited sections. **live** = URL returned HTTP 200, content skimmed or not needed beyond the title. **blocked** = publisher blocks automated fetches (ACM returns 403); the DOI resolves and an open copy is given where one exists. **UNVERIFIED** = the claim or URL comes only from secondary reporting or search summaries.

Link liveness for every URL in this file was re-checked by `scripts/check_urls.py` on 2026-10-24: 146 of 152 return HTTP 200, the 2 IEEE pages return 202, and the 4 ACM DOIs return 403 to scripts. Results are in `research/url_check.txt`.

---

## Precedence rule for conflicting numbers

When sources disagree, the course uses the first available in this order and footnotes the others:

1. Google Cloud TPU docs (`docs.cloud.google.com/tpu/docs/...`)
2. Google-authored papers (ISCA, IEEE Micro, NSDI, SIGCOMM)
3. Google launch blogs (Cloud blog, The Keyword)
4. JAX docs, including the Pallas hardware table (generated from libtpu metadata)
5. The Scaling Book
6. Secondary press, marked UNVERIFIED

A number that appears only at levels 4–6 is shown with its source tag; a number at level 6 only is shown with an "unverified" tag.

---

## 1. Spine: the Scaling Book

Austin, Douglas, Frostig, Levskaya, Chen, Vikram, Lebron, Choy, Ramasesh, Webson, Pope. "How to Scale Your Model." Google DeepMind, 2025. Parts 1–11 dated 2025-02-04, with later in-place revisions (the TPU chapter now uses TPU7x as its reference chip). Part 12 dated 2025-08-18 (Austin, Patil, Paszke, Pope). Status: **read**, all chapters, all answer blocks.

| ID | Chapter | URL |
|---|---|---|
| SB0 | Introduction | https://jax-ml.github.io/scaling-book/ |
| SB1 | All About Rooflines | https://jax-ml.github.io/scaling-book/roofline/ |
| SB2 | How to Think About TPUs | https://jax-ml.github.io/scaling-book/tpus/ |
| SB3 | Sharded Matrices and How to Multiply Them | https://jax-ml.github.io/scaling-book/sharding/ |
| SB4 | All the Transformer Math You Need to Know | https://jax-ml.github.io/scaling-book/transformers/ |
| SB5 | How to Parallelize a Transformer for Training | https://jax-ml.github.io/scaling-book/training/ |
| SB6 | Training LLaMA 3 on TPUs | https://jax-ml.github.io/scaling-book/applied-training/ |
| SB7 | All About Transformer Inference | https://jax-ml.github.io/scaling-book/inference/ |
| SB8 | Serving LLaMA 3-70B on TPUs | https://jax-ml.github.io/scaling-book/applied-inference/ |
| SB9 | How to Profile TPU Programs | https://jax-ml.github.io/scaling-book/profiling/ |
| SB10 | Programming TPUs in JAX | https://jax-ml.github.io/scaling-book/jax-stuff/ |
| SB11 | Conclusions and Further Reading | https://jax-ml.github.io/scaling-book/conclusion/ |
| SB12 | How to Think About GPUs | https://jax-ml.github.io/scaling-book/gpus/ |

Section anchors for each chapter are listed in the notes files and are used directly in citations (e.g. `[SB3 #case-2-one-multiplicand-has-a-sharded-contracting-dimension]`).

Known internal inconsistencies (flagged where they affect course content):
- v6e bf16 peak: 9.1e14 (SB1) vs 9.20e14 (SB2 table).
- TPU7x ICI: 1.8e11 B/s bidirectional per link (SB2 table) vs 1.2e12 B/s per chip and 4e11 per axis (SB2 prose). Cloud docs give 1,200 GB/s bidirectional per chip over 6 links of 200 GB/s [D-7x].
- "290 trillion multiply-adds/s" in SB2 key takeaway is left over from an older revision.
- VMEM bandwidth relative to HBM: 22× (SB2 Q4, v5e) vs ~7× (SB2 prose, TPU7x).
- H100 critical intensity: ~280 (SB7) vs 295 (SB1 Q5, SB12).
- B200 HBM bandwidth: 9 TB/s (SB12 prose) vs 8.0e12 (SB12 table).
- TPU VMEM: "120MB" vs 128 MB (SB12).
- LLaMA 3-70B int8 KV per token: 160 kB vs 162 kB (SB8).
- Problems with no published answer: SB3 Q6, Q8 (Q5, Q7 partial); SB6 Q1, Q2; SB7 Q4, Q6; SB8 Q1–3; SB9 Q2; SB10 Q3, Q4 (Q2 partial). Any of these used in a quiz is derived by `scripts/` and labelled "derived; no book answer".

---

## 2. TPU hardware

### 2.1 Papers

| ID | Citation | URL | Status |
|---|---|---|---|
| P-v1 | Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit," ISCA 2017 | https://arxiv.org/abs/1704.04760 · DOI https://doi.org/10.1145/3079856.3080246 | read (arXiv); DOI blocked |
| P-v23 | Jouppi et al., "A Domain-Specific Supercomputer for Training Deep Neural Networks," CACM 63(7), 2020 | https://doi.org/10.1145/3360307 · copy https://www.gwern.net/doc/ai/scaling/hardware/2020-jouppi.pdf | read (copy); DOI blocked |
| P-Norrie | Norrie et al., "The Design Process for Google's Training Chips: TPUv2 and TPUv3," IEEE Micro 41(2), 2021 | https://ieeexplore.ieee.org/document/9351692/ · copy https://www.gwern.net/doc/ai/scaling/hardware/2021-norrie.pdf | read (copy) |
| P-v4i | Jouppi et al., "Ten Lessons From Three Generations Shaped Google's TPUv4i," ISCA 2021 | https://www.cs.cmu.edu/~18742/papers/Jouppi2021.pdf | read |
| P-v4 | Jouppi et al., "TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings," ISCA 2023 | https://arxiv.org/abs/2304.01433 | read |
| P-Micro26 | Jouppi, Patterson, Lakshmanamurthy, Young, "Google's Training Supercomputers from TPU v2 to Ironwood…," IEEE Micro 2026 | https://research.google/pubs/googles-training-supercomputers-from-tpu-v2-to-ironwood-architectural-stability-scale-resilience-power-efficiency-and-sustainability-across-five-generation/ · https://ieeexplore.ieee.org/document/11550052 | abstract only |
| P-Lightwave | Liu et al., "Lightwave Fabrics: At-Scale Optical Circuit Switching for Datacenter and ML Systems," SIGCOMM 2023 | https://doi.org/10.1145/3603269.3604836 · copy https://cse.hkust.edu.hk/~kaichen/courses/spring24/comp7215/papers/lightwave-sigcomm23.pdf | read (copy) |
| P-Apollo | Poutievski et al., "Mission Apollo: Landing Optical Circuit Switching at Datacenter Scale," arXiv 2208.10041 | https://arxiv.org/abs/2208.10041 | live |
| P-Resil | Zu et al., "Resiliency at Scale: Managing Google's TPUv4 Machine Learning Supercomputer," NSDI 2024 | https://www.usenix.org/conference/nsdi24/presentation/zu | read |

### 2.2 Hot Chips

| ID | Talk | URL | Status |
|---|---|---|---|
| HC-v23 | Norrie & Patil, "Google's Training Chips Revealed: TPUv2 and TPUv3," Hot Chips 32 (2020) | https://hc32.hotchips.org/assets/program/conference/day2/HotChips2020_ML_Training_Google_Norrie_Patil.v01.pdf | read |
| HC-v4 | Jouppi & Swing, TPU v4, Hot Chips 2023 | https://hc2023.hotchips.org/assets/program/conference/day2/ML%20training/HC2023.Session5.ML_Training.Google.Norm_Jouppi.Andy_Swing.Final_2023-08-25.pdf | read |
| HC-7x | Ironwood at Hot Chips 2025, as reported by ServeTheHome (2025-08-26); no official slide PDF found | https://www.servethehome.com/google-ironwood-tpu-swings-for-reasoning-model-leadership-at-hot-chips-2025/ | read; secondary |

### 2.3 Google Cloud TPU documentation

All pages showed "Last updated 2026-10-22 UTC". The old `cloud.google.com/tpu/docs/...` paths 301-redirect here.

| ID | Page | URL |
|---|---|---|
| D-arch | System architecture (TensorCore, MXU size per gen, slices, cubes, OCS, twisted torus) | https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm |
| D-v2 | TPU v2 | https://docs.cloud.google.com/tpu/docs/v2 |
| D-v3 | TPU v3 | https://docs.cloud.google.com/tpu/docs/v3 |
| D-v4 | TPU v4 (incl. twisted-torus bisection gains) | https://docs.cloud.google.com/tpu/docs/v4 |
| D-v5e | TPU v5e | https://docs.cloud.google.com/tpu/docs/v5e |
| D-v5p | TPU v5p | https://docs.cloud.google.com/tpu/docs/v5p |
| D-v6e | TPU v6e (Trillium) | https://docs.cloud.google.com/tpu/docs/v6e |
| D-7x | TPU7x (Ironwood): 2 chiplets, JAX sees 2 devices/chip, 6 × 200 GB/s ICI | https://docs.cloud.google.com/tpu/docs/tpu7x |
| D-rn | Release notes (v6e GA 2024-12-16; TPU7x Preview 2025-11-24, GA 2026-03-31; no TPU 8 entries as of access) | https://docs.cloud.google.com/tpu/docs/release-notes |
| D-pytorch | PyTorch on TPU ("TorchTPU replaces PyTorch/XLA") | https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu |
| D-gke7x | GKE Ironwood concepts | https://docs.cloud.google.com/kubernetes-engine/docs/concepts/tpu-ironwood |

No docs page exists yet for TPU 8t or 8i (`/tpu/docs/tpu8t` and `/tpu8i` return 404 as of access).

### 2.4 Google launch posts

| ID | Post | Date | URL |
|---|---|---|---|
| G-v5e | Announcing Cloud TPU v5e | 2023-08 | https://cloud.google.com/blog/products/compute/announcing-cloud-tpu-v5e-and-a3-gpus-in-ga |
| G-v5p | Introducing Cloud TPU v5p and AI Hypercomputer | 2023-12-06 | https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer |
| G-v6e | Introducing Trillium, 6th-gen TPUs | 2024-05-14 | https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus |
| G-v6e-ga | Trillium is GA | 2024-12 | https://cloud.google.com/blog/products/compute/trillium-tpu-is-ga |
| G-7x | Ironwood: the first Google TPU for the age of inference | 2025-04-09 | https://blog.google/products/google-cloud/ironwood-tpu-age-of-inference/ |
| G-7x-ga | Ironwood TPUs and new Axion-based VMs | 2025-11-06 | https://cloud.google.com/blog/products/compute/ironwood-tpus-and-new-axion-based-vms-for-your-ai-workloads |
| G-8 | Vahdat, "Our eighth generation TPUs: two chips for the agentic era" | 2026-04-22 | https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/ |
| G-8dd | Gupta & Mugazambi, "TPU 8t and TPU 8i technical deep dive" (only official TPU 8 spec table) | 2026-04-22 | https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive |

### 2.5 JAX/Pallas hardware table

| ID | Page | URL | Status |
|---|---|---|---|
| J-hw | Pallas "TPU Hardware Reference." Per-TensorCore VMEM, CMEM, SMEM, HBM, HBM BW, BF16/FP8/INT8/INT4 peaks and SparseCore config, generated in-page from `pltpu.get_tpu_info_for_chip`. The only public source with BF16 and FP8 peaks for TPU 8t and 8i | https://docs.jax.dev/en/latest/pallas/tpu/hardware.html | read |

Per-TensorCore values from J-hw (chip value = per-TC × TCs/chip):

| Chip | TC/chip | VMEM/TC | CMEM/TC | HBM/TC | HBM BW/TC | BF16/TC | FP8/TC | INT8/TC | SparseCore/TC |
|---|---|---|---|---|---|---|---|---|---|
| v4 | 2 | 16 MiB | ~64 MiB | 17 GB | 615 GB/s | 138 TF | — | — | — |
| v5e | 1 | 128 MiB | — | 17 GB | 820 GB/s | 197 TF | — | 394 TOPs | — |
| v5p | 2 | 64 MiB | — | 51 GB | 1,230 GB/s | 230 TF | — | 459 TOPs | 4 SC |
| v6e | 1 | 128 MiB | — | 34 GB | 1,640 GB/s | 920 TF | 920 TF | 1,840 TOPs | 2 SC |
| 7x | 2 | 64 MiB | — | 103 GB | 3,700 GB/s | 1,155 TF | 2,300 TF | — | 2 SC |
| 8i | 2 | 192 MiB | — | 154 GB | 4,300 GB/s | 550 TF | 4,404 TF | — | 1 SC (4 subcores) |
| 8t | 1 | 128 MiB | — | 231 GB | 6,400 GB/s | 996 TF | 5,977 TF | 996 TOPs | 2 SC |

How the course uses J-hw: VMEM sizes for v5e onward come only from here (and SB2 for 7x). The TPU 8 BF16 and FP8 figures are shown with the tag "provisional: JAX hardware table, not in Google's published spec" because they are not in [G-8dd]. Two conflicts are flagged in content: (a) J-hw gives 8i one SparseCore per TensorCore, while G-8dd says the CAE replaces SparseCores on 8i; (b) J-hw 8t HBM BW is 6,400 GB/s vs 6,528 GB/s in G-8dd. Unit note: J-hw HBM capacities, summed per chip and converted, match the official figures read as GiB (7x 2 × 103 GB ≈ 192 GiB; 8t 231 GB ≈ 216 GiB; 8i 2 × 154 GB ≈ 288 GiB). So Google's "216 GB" and "288 GB" are most likely GiB. This is an inference and is labelled as one.

### 2.6 Secondary hardware material

| ID | Source | URL | Notes |
|---|---|---|---|
| HK | Henry Ko, "TPU Deep Dive," 2025-06-18 | https://henryhmko.github.io/posts/tpu/tpu.html | read. Good narrative. Known errors: calls Trillium "v6p"; v4 VMEM 32 MiB/TC (paper: 16 MiB/TC); "SMEM 10 MiB" is v4 spMEM; PaLM "6144 v4 = 2 pods" (1.5 pods); v5p "500 TFLOPS" (docs: 459). |
| S-TC | TechCrunch on TPU 8 | https://techcrunch.com/2026/04/22/google-cloud-next-new-tpu-ai-chips-compete-with-nvidia/ | secondary |
| S-TNW | TheNextWeb on TPU 8 (2 nm, codenames, "late 2027") | https://thenextweb.com/news/google-ironwood-tpu-inference-cloud-next | secondary; UNVERIFIED claims |
| S-TH | Tom's Hardware on TPU 8 (N3) | https://www.tomshardware.com/tech-industry/semiconductors/google-splits-its-tpu-into-two-chips-for-the-first-time-with-training-and-inference-variants | search snippet only; UNVERIFIED |

---

## 3. TPU software stack

### 3.1 Papers

| ID | Citation | URL | Status |
|---|---|---|---|
| P-Pathways | Barham et al., "Pathways: Asynchronous Distributed Dataflow for ML," MLSys 2022 | https://arxiv.org/abs/2203.12533 | read |
| P-GSPMD | Xu et al., "GSPMD: General and Scalable Parallelization for ML Computation Graphs," 2021 | https://arxiv.org/abs/2105.04663 | read |
| P-CollMM | Wang et al., "Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models," ASPLOS 2023, pp. 93–106 | DOI https://doi.org/10.1145/3567955.3567959 (UNVERIFIED DOI; bibliographic data verified via DBLP) | blocked |
| P-RPA | Jiang, Chen, Hechtman, Zhang, Mu, "Ragged Paged Attention," 2026-04-16 | https://arxiv.org/abs/2604.15464 | read |
| P-ESTI | Pope et al., "Efficiently Scaling Transformer Inference," 2022 (basis of SB7) | https://arxiv.org/abs/2211.05102 | live |

### 3.2 XLA / OpenXLA

| ID | Page | URL |
|---|---|---|
| X-arch | XLA architecture | https://openxla.org/xla/architecture |
| X-ops | Operation semantics (HLO op reference) | https://openxla.org/xla/operation_semantics |
| X-shapes | Shapes and layout | https://openxla.org/xla/shapes |
| X-tiled | Tiled layout (TPU (8,128) tiling, bf16 (2,1), int8 (4,1)) | https://openxla.org/xla/tiled_layout |
| X-passes | HLO passes | https://openxla.org/xla/hlo_passes |
| X-dumps | HLO dumps | https://openxla.org/xla/hlo_dumps |
| X-async | Async HLO ops (`-start`/`-done`) | https://openxla.org/xla/async_ops |
| X-flags | Flags guidance (TPU scheduler, async collectives, windowed einsum, VMEM) | https://openxla.org/xla/flags_guidance |
| X-shardy | Shardy (default partitioner since JAX 0.7.0) | https://openxla.org/shardy |
| X-shardy-jax | Using Shardy from JAX | https://openxla.org/shardy/getting_started_jax |
| X-shardy-prop | Shardy propagation | https://openxla.org/shardy/propagation |
| X-stablehlo | StableHLO spec | https://openxla.org/stablehlo/spec |
| X-gpu | XLA:GPU architecture (for the fusion comparison) | https://openxla.org/xla/gpu_architecture |

The TPU backend of XLA is closed source (ships in libtpu). Claims about TPU-specific passes beyond what these pages and SB9 state are marked UNVERIFIED in content.

### 3.3 JAX

| ID | Page | URL |
|---|---|---|
| J-shard | Sharding guide (auto / explicit / manual) | https://docs.jax.dev/en/latest/parallel.html |
| J-sharding-api | `jax.sharding` API | https://docs.jax.dev/en/latest/jax.sharding.html |
| J-shmap | `shard_map` tutorial | https://docs.jax.dev/en/latest/notebooks/shard_map.html |
| J-jaxpr | jaxpr language | https://docs.jax.dev/en/latest/jaxpr.html |
| J-aot | Ahead-of-time lowering | https://docs.jax.dev/en/latest/aot.html |
| J-prof | JAX profiling | https://docs.jax.dev/en/latest/profiling.html |
| J-log | JAX changelog (Shardy default 0.7.0; `jax.shard_map` 0.8.0; C++ pmap removed 0.10.0; 0.12.0 on 2026-10-19) | https://docs.jax.dev/en/latest/changelog.html |

### 3.4 Pallas / Mosaic

| ID | Page | URL |
|---|---|---|
| PL-index | Pallas overview | https://docs.jax.dev/en/latest/pallas/index.html |
| PL-grid | Grids and BlockSpecs | https://docs.jax.dev/en/latest/pallas/grid_blockspec.html |
| PL-tpu | Writing TPU kernels (memory spaces, (8,128) rule, sequential grid) | https://docs.jax.dev/en/latest/pallas/tpu/details.html |
| PL-pipe | TPU pipelining (double buffering, megacore `dimension_semantics`) | https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html |
| PL-mm | TPU matmul tutorial | https://docs.jax.dev/en/latest/pallas/tpu/matmul.html |
| PL-sparse | Scalar prefetch and block-sparse (basis of paged attention) | https://docs.jax.dev/en/latest/pallas/tpu/sparse.html |
| PL-dist | Distributed Pallas (remote DMA collectives) | https://docs.jax.dev/en/latest/pallas/tpu/distributed.html |
| PL-core | Core-specific programming (`pl.kernel`) | https://docs.jax.dev/en/latest/pallas/tpu/core_map.html |
| PL-sc | SparseCore kernels | https://docs.jax.dev/en/latest/pallas/tpu/sparsecore.html |
| PL-log | Pallas changelog | https://docs.jax.dev/en/latest/pallas/CHANGELOG.html |
| PL-gpu | Pallas Mosaic GPU (for comparison) | https://docs.jax.dev/en/latest/pallas/gpu/index.html |
| K-tokamax | Tokamax kernel library | https://github.com/openxla/tokamax |

### 3.5 Reference implementations and serving

| ID | Source | URL | Status / notes |
|---|---|---|---|
| R-maxtext | MaxText repo | https://github.com/AI-Hypercomputer/maxtext | read. Mesh axes and `ici_*`/`dcn_*` parallelism in `src/maxtext/configs/base.yml`; sharding guide `docs/guides/optimization/sharding.md`; Flax NNX since 2026-06-29; Qwix replaces AQT |
| R-jetstream | JetStream | https://github.com/AI-Hypercomputer/JetStream | read. Archived 2026-02-01; history only |
| R-tpuinf | tpu-inference (being renamed vLLM JAX) | https://github.com/vllm-project/tpu-inference | read. PyPI 0.32.0 (2026-10-22). Source files cited: `tpu_inference/runner/kv_cache.py`, `layers/common/sharding.py`, `envs.py`, `models/common/model_loader.py`, `kernels/ragged_paged_attention/v3/` |
| R-tpuinf-docs | vLLM TPU docs | https://docs.vllm.ai/projects/tpu/en/latest/ | read |
| R-torchax-guide | tpu-inference torchax model development guide | https://github.com/vllm-project/tpu-inference/blob/main/docs/developers_guide/torchax_model_development.md | read |
| R-tpuinf-prof | tpu-inference profiling guide | https://github.com/vllm-project/tpu-inference/blob/main/docs/developers_guide/profiling.md | read |
| B-vllmtpu | "vLLM TPU: A New Unified Backend Supporting PyTorch and JAX on TPU," 2025-10-16 | https://vllm.ai/blog/vllm-tpu | read |
| R-vllm-torchtpu | vllm-torchtpu ("recommended path for serving PyTorch model definitions on TPU") | https://github.com/vllm-project/vllm-torchtpu | read |
| R-torchax | torchax (moved here 2025-10-06) | https://github.com/google/torchax | read |
| R-qwix | Qwix | https://github.com/google/qwix | read |
| R-sglang-jax | SGLang JAX | https://github.com/sgl-project/sglang-jax | live |

### 3.6 TorchTPU and PyTorch/XLA

| ID | Source | Date | URL | Status |
|---|---|---|---|---|
| T-blog1 | Google Developers Blog, "TorchTPU: Running PyTorch Natively on TPUs at Google Scale" | 2026-04-07 | https://developers.googleblog.com/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/ | read |
| T-oss | Google Cloud Blog, "TorchTPU is now open source" | 2026-10-20 | https://cloud.google.com/blog/topics/ai-infrastructure/torchtpu-is-now-open-source-run-native-pytorch-on-tpus | read |
| T-pt | PyTorch blog, "Introducing TorchTPU" | 2026-10-21 | https://pytorch.org/blog/introducing-torchtpu/ | read |
| T-repo | torch_tpu repo (0.2.0) | 2026-10 | https://github.com/google-pytorch/torch_tpu | read |
| T-how | Docs: How TorchTPU works | 2026-10 | https://torchtpu.readthedocs.io/en/0.2.0/how_torchtpu_works/ | read |
| T-exec | Docs: Execution model (strict / debug / defer-and-fuse eager; `torch.compile(backend="tpu")`) | 2026-10 | https://torchtpu.readthedocs.io/en/0.2.0/torchtpu_execution_model/ | read |
| T-dist | Docs: Distributed support (`tpu_dist`, SPMD handshake) | 2026-10 | https://torchtpu.readthedocs.io/en/0.2.0/distributed_support/ | read |
| T-kern | Docs: Custom kernels (`torch.tpu.pallas.jax_op`) | 2026-10 | https://torchtpu.readthedocs.io/en/0.2.0/custom_kernels/ | read |
| T-engines | "Announcing New vLLM and SGLang Inference Engines on TPU" | 2026-10-20 | https://developers.googleblog.com/announcing-new-vllm-and-sglang-inference-engines-on-tpu/ | read |
| T-xla | PyTorch/XLA repo ("deprecated and replaced by TorchTPU"; archived 2026-11-15) | 2026-10 | https://github.com/pytorch/xla | read |

TorchTPU performance claims: [T-pt] reports Qwen3-MoE-235B on 8,192 TPU v7 chips at 27.1% MFU, and a 30B model at 59% prefill MFU on one v7 chip. A "52% MFU on 8,192 v7x chips (Qwen3-8B)" figure appears only in secondary reporting and is tagged unverified in content.

### 3.7 XProf

| ID | Page | URL |
|---|---|---|
| XP-home | XProf | https://openxla.org/xprof |
| XP-trace | Trace Viewer | https://openxla.org/xprof/trace_viewer |
| XP-opprof | HLO Op Profile | https://openxla.org/xprof/hlo_op_profile |
| XP-opstats | HLO Op Stats | https://openxla.org/xprof/hlo_op_stats |
| XP-mem | Memory Viewer | https://openxla.org/xprof/memory_viewer |
| XP-roof | Roofline Model (HBM, VMEM, CMEM rooflines) | https://openxla.org/xprof/roofline_model |
| XP-graph | Graph Viewer | https://openxla.org/xprof/graph_viewer |
| XP-mega | Megascale Viewer (multislice DCN) | https://openxla.org/xprof/megascale_viewer |
| XP-jax | Profiling JAX programs | https://openxla.org/xprof/jax_profiling |

---

## 4. Comparisons: NVIDIA and AWS Trainium

### 4.1 NVIDIA

| ID | Source | URL | Notes |
|---|---|---|---|
| N-h100 | H100 product page | https://www.nvidia.com/en-us/data-center/h100/ | Tensor Core figures are sparse; dense = ½ |
| N-h200 | H200 product page | https://www.nvidia.com/en-us/data-center/h200/ | |
| N-b200 | DGX B200 | https://www.nvidia.com/en-us/data-center/dgx-b200/ | per-GPU values derived from 8-GPU totals |
| N-hgx | HGX (B200, B300) | https://www.nvidia.com/en-us/data-center/hgx/ | |
| N-gb200 | GB200 NVL72 | https://www.nvidia.com/en-us/data-center/gb200-nvl72/ | per-GPU derived from rack totals |
| N-gb300 | GB300 NVL72 | https://www.nvidia.com/en-us/data-center/gb300-nvl72/ | |
| N-rubin | Vera Rubin NVL72 | https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/ | product page says 19.2 TB/s HBM4, 3 TB/s NVLink per GPU; CES 2026 coverage said 22 and 3.6 (conflict noted) |
| N-wp | H100 architecture whitepaper (mirror; official copy is registration-gated) | https://www.ece.lsu.edu/koppel/gp/refs/gtc22-whitepaper-hopper.pdf | read |
| N-tune | Hopper tuning guide (228 KB SMEM/SM, 50 MB L2) | https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html | read |
| N-graphs | CUDA Programming Guide §4.2, CUDA Graphs | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html | read |

### 4.2 GPU parallelism and collectives literature

| ID | Citation | URL |
|---|---|---|
| P-Megatron | Shoeybi et al., "Megatron-LM," 2019 | https://arxiv.org/abs/1909.08053 |
| P-SeqPar | Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models," 2022 | https://arxiv.org/abs/2205.05198 |
| P-Flux | Chang et al., "FLUX: Fast Software-based Communication Overlap on GPUs," 2024 | https://arxiv.org/abs/2406.06858 |
| P-Llama3 | Llama Team, Meta, "The Llama 3 Herd of Models," 2024. Table 3 (8B/70B/405B shapes), Table 4 (16,384-GPU parallelism: TP 8, PP 16, DP 128, 41% MFU), §3.4 (batch 4M→8M→16M tokens; 405B on 15.6T tokens, 3.8e25 FLOPs). Read 2026-10-23 | https://arxiv.org/abs/2407.21783 |
| N-nccltests | NCCL-tests PERFORMANCE.md (algbw vs busbw) | https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md |
| N-ncclenv | NCCL env docs (NCCL_ALGO: Ring, Tree, NVLS, PAT…) | https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html |
| N-nccl24 | "Massively Scale Your Deep Learning Training with NCCL 2.4" (double binary trees) | https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4 |
| PT-asynctp | "Introducing Async Tensor Parallelism in PyTorch," 2024-09-12 | https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487 |
| N-teoverlap | NeMo TP communication overlap | https://docs.nvidia.com/nemo-framework/user-guide/26.02/nemotoolkit/features/optimizations/communication_overlap.html |

### 4.3 AWS Trainium

| ID | Source | URL |
|---|---|---|
| A-trn2 | Trainium2 chip architecture | https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/_sources/about-neuron/arch/neuron-hardware/trainium2.rst |
| A-trn2-nki | Trainium2 architecture guide for NKI (NeuronCore-v3 engines, SBUF/PSUM) | https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/_sources/nki/guides/architecture/trainium2_arch.rst |
| A-trn2-sys | Trn2 instance and UltraServer architecture (4×4 2D torus) | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/about-neuron/arch/neuron-hardware/trn2-arch.rst |
| A-trn3 | Trainium3 chip architecture | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trainium3.html |
| A-trn3-nki | Trainium3 architecture guide for NKI | https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.1/nki/guides/architecture/trainium3_arch.html |
| A-trn3-sys | Trn3 UltraServer architecture (NeuronSwitch, replaces 2D torus) | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn3-arch.html |
| A-trn3-prod | EC2 Trn3 product page | https://aws.amazon.com/ec2/instance-types/trn3/ |
| A-trn3-ga | Trn3 UltraServers GA, 2025-12-02 | https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-ec2-trn3-ultraservers/ |
| A-nki | NKI programming model | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/programming_model.html |
| A-torch | PyTorch on Neuron / TorchNeuron (beta; torch-neuronx legacy) | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/frameworks/torch/about/index.rst |
| A-jax | JAX on Neuron (PJRT plugin) | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/jax/index.html |
| A-trn4 | Trainium4 + NVLink Fusion preview (The Register, 2025-12-02) | https://www.theregister.com/2025/12/02/amazon_nvidia_trainium/ |

AWS pages disagree with each other on CC-Core count (16 vs 20), Trn3 HBM bandwidth (4.7 vs 4.9 TB/s), and NeuronLink-v4 bandwidth (2.0 vs 2.56 TB/s). The course gives both where they matter.

---

## 5. Recent changes, Oct 2025 – Oct 2026

Dated headlines that change the picture relative to the Scaling Book's Feb 2025 baseline. Detail and URLs are in `notes/comparisons.md §4` and `notes/tpu_software.md §0`.

| Date | Change | Source ID |
|---|---|---|
| 2025-10-06 | torchax moves to github.com/google/torchax | R-torchax |
| 2025-10-15 | JAX 0.8.0: `jax.shard_map` becomes the public API | J-log |
| 2025-10-16 | vLLM TPU rebuilt on tpu-inference: one JAX lowering path for JAX and PyTorch (torchax) models; RPA v3 | B-vllmtpu |
| 2025-11-06 | Ironwood GA announced (docs: GA 2026-03-31) | G-7x-ga, D-rn |
| 2025-12-02 | Trainium3 UltraServers GA; Trainium4 + NVLink Fusion previewed | A-trn3-ga, A-trn4 |
| 2025-12 | AWS TorchNeuron (native PyTorch, PrivateUse1) in Neuron 2.27 | A-torch |
| 2026-02-01 | JetStream archived | R-jetstream |
| 2026-04-07 | TorchTPU described publicly | T-blog1 |
| 2026-04-16 | Ragged Paged Attention paper; JAX 0.10.0 removes C++ pmap | P-RPA, J-log |
| 2026-04-22 | TPU 8t and TPU 8i announced; first split-generation TPU; Boardfly topology and CAE on 8i | G-8, G-8dd |
| 2026-06-29 | MaxText moves to Flax NNX | R-maxtext |
| 2026-10-19 | JAX 0.12.0 | J-log |
| 2026-10-20 | TorchTPU open-sourced; vllm-torchtpu and sglang-torchtpu announced; tpu-inference refocused as vLLM JAX | T-oss, T-engines, R-tpuinf |
| 2026-11-15 (scheduled) | PyTorch/XLA repo archived | T-xla |

Business news (Anthropic's up-to-1M-TPU agreement of 2025-10-23 and the multi-GW Broadcom deal of 2026-04-06; press reports of Meta renting TPUs and Fluidstack hosting TPUs in third-party data centres) is in `notes/comparisons.md §4`. The course mentions it once, in Chapter 1, as context only. The Meta and Fluidstack items are tagged unverified.

---

## 6. Local resources

Not indexed. The brief gives the path as the literal placeholder `[path/to/my/resources]`. To be folded in once a real path is provided.
