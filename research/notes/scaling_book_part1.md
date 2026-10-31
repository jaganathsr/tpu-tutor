# JAX Scaling Book ("How to Scale Your Model"), Part 1 notes: Intro + Chapters 1-6

Source: Austin et al., "How to Scale Your Model", Google DeepMind, 2025. https://jax-ml.github.io/scaling-book/
Accessed: 2026-10-23 (every page fetched live with curl, HTTP 200).
Pages list "Published Feb. 4, 2025", but the content has been revised since then. The TPU chapter now uses **TPU7x** (Ironwood) as its reference chip, and the intro links a new "Section 12 on NVIDIA GPUs". Some numbers from older revisions are still in the text, and these notes flag where they clash.

Authors: Jacob Austin, Sholto Douglas, Roy Frostig, Anselm Levskaya, Charlie Chen, Sharad Vikram, Federico Lebron, Peter Choy, Vinay Ramasesh, Albert Webson, Reiner Pope (*"Work done at Google DeepMind, now at MatX").

Citation string given by the book: `Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.`

## Verified chapter URLs

The TOC links `/scaling-book/<slug>`, which redirects to `/scaling-book/<slug>/`. Both resolve.

| Ch | Title (as on page) | URL |
|---|---|---|
| 0 | How to Scale Your Model (intro) | https://jax-ml.github.io/scaling-book/ |
| 1 | All About Rooflines | https://jax-ml.github.io/scaling-book/roofline/ |
| 2 | How to Think About TPUs | https://jax-ml.github.io/scaling-book/tpus/ |
| 3 | Sharded Matrices and How to Multiply Them | https://jax-ml.github.io/scaling-book/sharding/ |
| 4 | All the Transformer Math You Need to Know | https://jax-ml.github.io/scaling-book/transformers/ |
| 5 | How to Parallelize a Transformer for Training | https://jax-ml.github.io/scaling-book/training/ |
| 6 | Training LLaMA 3 on TPUs | https://jax-ml.github.io/scaling-book/applied-training/ |
| (7-12, out of scope) | inference, applied-inference, profiling, jax-stuff, conclusion, gpus | same pattern |

---

## Chapter 0: Intro / landing page

URL: https://jax-ml.github.io/scaling-book/ (accessed 2026-10-23)

Section headings: `#why-should-you-care` Why should you care? · `#high-level-outline` High-Level Outline · `#links-to-sections` Links to Sections

### Core concepts
- The book's aim is to "demystify the science of scaling language models": how TPUs (and GPUs) work and communicate, how LLMs run on real hardware, and how to parallelize for training and inference.
- **Strong scaling** means adding chips and getting a proportional, linear increase in throughput. You become "communication bound" when communication takes longer than computation.
- Single-chip bottlenecks: "Your shiny new TPU or GPU may be rated to perform 500 trillion operations per second, but if you aren't careful it can just as easily do a tenth of that if it's bogged down moving parameters around in memory."
- Hardware/algorithm co-design: chip designers must bet on what algorithms will look like "2 to 3 years down the road".
- The book uses FLOPs to mean a count of operations and FLOPs/s for a rate.
- Four primary parallelism techniques: **data, tensor, pipeline, expert**. Memory-reduction techniques: **rematerialization, optimizer/model sharding (ZeRO), host offload, gradient accumulation**.
- Expected background: basic LLM/Transformer knowledge and some JAX.
- Part structure: Part 1 Preliminaries (Ch 1-3), Part 2 Transformers (Ch 4-8), Part 3 Practical Tutorials (Ch 9 profiling, Ch 10 JAX), Part 4 Conclusions and Bonus (Ch 11 conclusions, Ch 12 GPUs).

### TPU-vs-GPU statements
- "Matrix multiplication ... uses far more FLOPs per byte of memory than almost any other (N FLOPs per byte), and early TPUs and their systolic array architecture achieved far better perf / $ than GPUs did at the time they were built."
- "TPUs were designed for ML workloads, and GPUs with their Tensor Cores are rapidly changing to fill this niche as well." TPUs are "inherently less flexible than GPUs".

No problems on this page.

---

## Chapter 1: All About Rooflines

URL: https://jax-ml.github.io/scaling-book/roofline/ (accessed 2026-10-23)

Section headings:
- `#where-does-the-time-go` Where Does the Time Go?
  - `#visualizing-rooflines` Visualizing rooflines
  - `#matrix-multiplication` Matrix multiplication
  - `#network-communication-rooflines` Network communication rooflines
- `#a-few-problems-to-work` A Few Problems to Work

### Core concepts and formulas
- Three bounds on any computation: compute (OPs/s), bandwidth (bytes/s), and total memory (bytes).
- Compute time: $T_\text{math} = \frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs/s}}$. Time spent doing arithmetic.
- Comms time, within a chip (HBM to cores) or between chips (ICI, DCN, PCIe): $T_\text{comms} = \frac{\text{Communication Bytes}}{\text{Network/Memory Bandwidth Bytes/s}}$. Time spent moving bytes.
- Bounds: $T_\text{lower} = \max(T_\text{math}, T_\text{comms})$ assumes perfect overlap, and $T_\text{upper} = T_\text{math} + T_\text{comms}$ assumes none. They differ by at most 2x. The book optimizes against the max.
- Compute-bound means $T_\text{math} > T_\text{comms}$. Comms-bound is the reverse. The book uses "communication-bound", "comms-bound", "memory-bound" and "bandwidth-bound" interchangeably.
- **Arithmetic intensity**: $\text{Intensity} = \frac{\text{Computation FLOPs}}{\text{Communication Bytes}}$. FLOPs done per byte moved.
- Compute-bound iff $\text{Intensity(Computation)} > \text{Intensity(Accelerator)} = \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}}$. The algorithm must do more FLOPs per byte than the hardware's ratio.
- **TPU v5e MXU critical intensity is about 240 FLOPs/byte**, from 1.97e14 FLOPs/s / 8.2e11 B/s HBM. The book notes this applies only to HBM-loaded weights running on the MXU. VMEM has higher bandwidth, and the VPU has a different peak.
- Dot product example (bf16[N]·bf16[N]): $\frac{2N-1}{4N+2} \to \frac12$, so it is always comms-bound. The book adds a caveat that the dot product actually runs on the VPU. "The TPU v5p VPU can do roughly 7e12 FLOPs / second per core, so its critical intensity is around 3."
- Roofline plot: log-log plot of achievable FLOPs/s against intensity. Throughput rises linearly until the critical intensity, then stays flat at peak. You improve either by raising intensity or by raising bandwidth (BW1 to BW2).
- **Matmul** bf16 X[B,D]·Y[D,F]: load $2BD+2DF$ bytes, do $2BDF$ FLOPs, write $2BF$ bytes.
  $\text{Intensity(matmul)} = \frac{BDF}{BD+DF+BF} \approx B$ when $B \ll D,F$. Intensity is roughly the token batch size.
  Compute-bound when $B > 1.97e14/8.20e11 = 240$.
- **Takeaway (verbatim)**: "for a bfloat16 matmul to be compute-bound on most TPUs, we need our batch size to be greater than 240 (tokens)." This counts tokens, not sequences, and applies per replica, because model sharding scales compute and bandwidth together. Example from the book: 512 seqs × 4096 tokens on 2048 GPUs gives a 2M-token global batch and a 1k local batch.
- "For GPUs, this number is slightly higher (closer to 300)."
- Tiling: an (m,k)·(k,n) matmul with tiles bm, bk, bn has intensity of roughly $\frac{bm \cdot bn}{bm+bn}$. Tiles live in "VMEM/SMEM/TMEM".
- **Network roofline example**: X[B,D], Y[D,F] sharded on D across 2 chips. Each chip computes half, then exchanges partial sums. $T_\text{math} = \frac{BDF}{1.97e14}$ and $T_\text{comms} = \frac{2BF}{4.5e10}$. Compute-bound when $\frac{D}{2} > \frac{1.97e14}{4.5e10} = 4377$, i.e. **D > 8755**. The threshold now depends on D, not B.

### Hardware numbers stated (Ch 1)
- NVIDIA H100: ~9.89e14 bf16 FLOPs/s; HBM ~3.35 TB/s. "H100s and B200s can usually only achieve around 80-85% of the claimed peak FLOPs, while TPUs can get closer to 95% in normal use."
- TPU v6e: 9.1e14 FLOPs/s (the Ch 2 table says 9.20e14); HBM ~1.6 TB/s.
- TPU v5e: 1.97e14 bf16 FLOPs/s; HBM 8.2e11 B/s; int8 3.94e14 OPs/s (Q1).
- TPU v5p VPU: ~7e12 FLOPs/s per core.
- Example: 1e12 FLOPs takes 1.01 ms on H100 and 1.1 ms on v6e. The book notes this ignores price.
- Interchip example link: 4.5e10 B/s each direction.

### TPU-vs-GPU comparisons (Ch 1)
- Achievable fraction of peak: GPUs 80-85%, TPUs close to 95%.
- Critical batch size: about 240 on TPU v5e versus about 295 on H100 ("rather similar").
- NVIDIA calls HBM bandwidth "memory bandwidth".

### Problems (Ch 1)
1. **int8 matmul** X[B,D]·Y[D,F] in int8, with HBM 8.2e11 and int8 3.94e14. Asks for bytes loaded and written, OPs, intensity, and $T$ bounds.
   *Answer*: load $BD+DF$, write $BF$. OPs $2BDF$. Intensity $2BDF/(BD+DF+BF) \approx 2B$. The int8 HBM intensity is 3.94e14/8.2e11 = 480, so the rule becomes $B > 480/2 = 240$, "basically unchanged". $T_\text{math}=2BDF/3.94e14$, $T_\text{comms}=(BD+DF+BF)/8.2e11$. Lower bound is the max, upper bound the sum.
2. **int8 weights + bf16 activations/compute** (bf16[B,D]·int8[D,F] to bf16[B,F]), with 1.97e14 bf16. Asks at what B it becomes compute-bound.
   *Answer*: $2BDF$ FLOPs but only $DF$ weight bytes, so compute-bound when $2B>240$, i.e. **B > 120**. "A meaningful win in efficiency (although int8 OPs would be better)."
3. **Roofline plot** of peak FLOPs/s against B for the Q2 setup, with D=F=4096 and D=F=1024, using exact bytes.
   *Answer*: a plot plus code. Bytes are `2*B*D + D*F + 2*B*F` and the v5e numbers are 1.97e14 and 8.2e11. Both curves reach peak, the larger D/F sooner. "D=F=1024 almost doubles the critical batch size."
4. **Per-example weight matrices**: int8[B,D]·int8[B,D,F] to int8[B,F]. Asks for the intensity.
   *Answer*: FLOPs are still $2BDF$, but bytes are $BD+BDF+BF$, so intensity is about **2** regardless of B. "We'll basically always be comms bound."
5. **H100 memory roofline**: using the H100 SXM spec sheet (the Tensor Core figure is 2x because of sparsity), find the batch size at which a bf16 matmul becomes compute-bound.
   *Answer*: 1.979e15 sparse halves to 9.89e14 dense. HBM is 3.35e12 B/s. $B_\text{crit}$ = 9.89e14/3.35e12 = **295**, "rather similar to the TPU".

---

## Chapter 2: How to Think About TPUs

URL: https://jax-ml.github.io/scaling-book/tpus/ (accessed 2026-10-23)

Section headings:
- `#what-is-a-tpu` What Is a TPU?
- `#tpu-networking` TPU Networking
- `#key-takeaways` Key Takeaways
  - `#tpu-specs` TPU specs
- `#worked-problems` Worked Problems
- `#appendix` Appendix
  - `#appendix-a-more-on-tpu-internals` Appendix A: More on TPU internals (`#vpu` VPU, `#scalar-core` Scalar Core)
  - `#appendix-b-how-does-a-systolic-array-work` Appendix B: How does a systolic array work?

### Core concepts
- TPU = **TensorCore** (a matmul-specialized compute core) + **HBM**. A TensorCore has three key units:
  - **MXU** (Matrix Multiply Unit), a systolic array. On TPU7x it does one bf16[8,256] @ bf16[256,256] -> f32[8,256] every 4 cycles.
  - **VPU** (Vector Processing Unit): elementwise ops (ReLU, add, multiply) and reductions.
  - **VMEM**: on-chip, programmer-controlled scratchpad. "Somewhat like an L1/L2 cache on CPUs but is much larger and programmer-controlled." Data in HBM must be copied into VMEM before compute.
  - Also SMEM and a scalar unit, used for control flow (Appendix A).
- Data path: HBM, then VMEM, then VREGs, then VPU/MXU, then back. Everything is pipelined and overlapped. A matmul streams 8x256 (X) and 256x256 (A) chunks.
- "Key takeaway: TPUs are very simple. They load weights from HBM into VMEM, then from VMEM into a systolic array which can perform around 290 trillion multiply-adds per second." (This figure looks left over from an earlier revision. It does not match TPU7x's ~1.15e15 FLOPs/s per core.)
- **VMEM and intensity**: VMEM bandwidth is about 7x HBM on TPU7x, so ops reading from VMEM need about 7x lower intensity to reach peak. VMEM prefetching example: load FFW weights into VMEM during attention.
- **Megacore**: v4, v5 and v6 chips have 2 cores sharing memory that act as one accelerator. "TPU v7 removes megacore and instead has a high-bandwidth link between the two cores." TPU v3 and older expose 2 separate accelerators. Inference chips (v5e) have 1 core per chip.
- **Trays and hosts**: 4 chips per tray, connected to a CPU host over PCIe. Inference chips (v6e) have 2 trays per host, giving 8 chips = 8 cores. On Cloud TPU VMs each tray is a separate VM.
- **PCIe** is slow. v4 has 16 GB/s each way, "close to 100x slower than HBM".
- **ICI networking**: v6e (Trillium) links each chip to 4 nearest neighbours in a 2D torus. TPU7x links to 6 neighbours in a 3D torus. These are direct chip-to-chip links that do not go through hosts. A torus cuts the max distance from N to N/2. A "twisted torus" (Möbius-like) cuts the average distance further.
- **TPU7x superpod**: 9216 chips made of reconfigurable 4x4x4 cubes joined by an **optical circuit switch (OCS)**. The cubes can form any 4n×4m×4k topology. The OCS "is simply a reconfigurable connection with the same ICI bandwidth."
- **Wraparound caveat**: smaller topologies such as 2x2x1, 2x2x2 and 2x2x4 have no wraparound, which "typically doubles the time of most communication". Any multiple of a full cube has wraparound. A v6e 8x16 *does* wrap on its 16-long axis, because v5e and v6e have no OCS and wrap on any axis of size 16.
- **v5e/v6e pods**: a single 16x16 2D torus that cannot grow larger. Pods talk to each other over DCN.
- **Slice**: a set of ICI-connected TPUs. **Multi-slice** means slices joined over DCN. The DCN path runs TPU, PCIe, host NIC, network, target host, PCIe, HBM.
- Bandwidth ladder (fastest first): HBM, then ICI, then PCIe, then DCN. Communication on each channel should be proportional to its speed.
- **MXU padding**: weight matrices should be padded to multiples of 256 (128 before v6e) in both dims.
- **Low precision**: v6e runs int8/int4 2x/4x faster than bf16. TPU7x runs fp8 at 2x bf16 FLOPs. VPU ops run in fp32 or bf16, and pre-v6e chips support fp32 VPU ops only.

### Hardware numbers (Ch 2, copy exactly)

TPU7x figures in the prose:
- MXU: bf16[8,256]@bf16[256,256]->f32[8,256] every 4 cycles. ~5.8e14 bf16 FLOPs/s per MXU at 2.2 GHz.
- 2 TensorCores per chip, 2 MXUs per TensorCore. ~2.3e15 bf16 FLOPs/s per chip, or 1.15e15 per core.
- fp8: 4.6e15 OPs/s per chip.
- VMEM: **128 MiB per chip**. HBM: **192 GiB**.
- HBM BW: "around 7TB/sec" in prose, and **7.4e12 bytes/s (7380 GB/s)** in the ICI comparison list.
- ICI: **1.2e12 B/s (1200 GB/s) bidirectional per chip, or 4e11 B/s (400 GB/s) per axis**. "TPU ICI links have slightly different bandwidths depending on the operation being performed."
- DCN: **1.25e10 B/s (12.5 GB/s) egress per chip**.
- A pod of 9216 chips delivers 21 bf16 exaFLOPs/s.
- VREGs: 64 32-bit VREGs per core (32 on v4), about 256 kB per core. Per cycle: 3 VREG loads and 2 stores, max 4 total.
- VPU: shape (8,128), with 128 lanes × 8 sublanes and 4 independent ALUs per (lane, sublane). Most ops take 1 cycle with 2-cycle latency. A single scalar core controls the VPU (4096 ALUs), 2 MXUs, 2 XLUs and multiple DMA engines. One DMA request per cycle per core.
- Systolic array: 256x256 (65,536 ALUs) on v6e and later, 128x128 before. Pre-v6e the 8x(128) multiply took 8 cycles.
- MXU count: 2 on v3, v6e and TPU7x; 4 on v4/v5. Tiling dims should exceed 256 (or 128) × the number of MXUs.

**Spec table** (`#tpu-specs`):

| Model | Pod size | Host size | HBM capacity/chip | HBM BW/chip (bytes/s) | FLOPs/s/chip (bf16) | FLOPs/s/chip (int8) |
|---|---|---|---|---|---|---|
| TPU v3 | 32x32 | 4x2 | 32GiB | 9.0e11 | 1.4e14 | 1.4e14 |
| TPU v4p | 16x16x16 | 2x2x1 | 32GiB | 1.2e12 | 2.75e14 | 2.75e14 |
| TPU v5p | 16x20x28 | 2x2x1 | 96GiB | 2.8e12 | 4.59e14 | 9.18e14 |
| TPU v5e | 16x16 | 4x2 | 16GiB | 8.2e11 | 1.97e14 | 3.94e14 |
| TPU v6e | 16x16 | 4x2 | 32GiB | 1.6e12 | 9.20e14 | 1.84e15 |
| TPU7x | 4x4x576 | 2x2x1 | 192GiB | 7.4e12 | 2.31e15 | 4.61e15 (fp8) |

**ICI table**:

| Model | ICI BW/link (one-way, bytes/s) | ICI BW/link (bidi, bytes/s) |
|---|---|---|
| TPU v3 | 1.0e11 | 2.0e11 |
| TPU v4p | 4.5e10 | 9.0e10 |
| TPU v5p | 9.0e10 | 1.8e11 |
| TPU v5e | 4.5e10 | 9.0e10 |
| TPU v6e | 9.0e10 | 1.8e11 |
| TPU7x | 9.0e10 | 1.8e11 |

- "Bidi" means total bytes over one link in both directions. It applies when a ring exists, i.e. with wraparound: a full 16-long axis on inference chips, or an axis that is a multiple of 4 on v*p training chips.
- PCIe: ~1.6e10 B/s per TPU (3.2e10 on v6e).
- DCN: ~6.25e9 B/s per TPU (12.5e9 on v6e and TPU7x, 3.125e9 on v5e).
- PCIe for TPU v4: 16 GB/s each way.
- Per-hop ICI latency: ~1 µs (used in the problems, e.g. v5e).
- Inconsistencies to watch: (a) the TPU7x ICI table gives 1.8e11 bidi per link, while the prose gives 4e11 per axis and 1.2e12 per chip. (b) The TPU7x HBM BW is "around 7TB/s" in prose and 7.4e12 in the table. (c) The "290 trillion multiply-adds/s" figure is stale. (d) Q4 uses VMEM at 22x HBM BW (v5e), while the prose says ~7x for TPU7x.

### TPU-vs-GPU comparisons (Ch 2)
- **Topology**: "This nearest-neighbor connectivity is a key difference between TPUs and GPUs. GPUs are connected with a hierarchy of switches that approximate a point-to-point connection between every GPU." GPUs within a node are directly connected: "8 GPUs for H100 or as many as 72 for B200 NVL72". Larger GPU topologies need O(log N) hops. TPUs are "dramatically cheaper (since NVLink switches are expensive), simpler to wire together, and can scale to much larger topologies because the number of links per device and the bandwidth per device is constant."
- **VPU and CUDA cores**: "each ALU in the VPU is analogous to a CUDA core, and a single VPU lane is analogous to a 'Warp Scheduler', i.e. the set of usually 32 CUDA Cores that perform SIMD arithmetic." Crossing lanes means going through VMEM, the XLU or SMEM, which is slow.
- VMEM is described relative to CPU L1/L2. (Ch 1 also pairs VMEM with GPU SMEM/TMEM.)

### Problems (Ch 2)
1. **LLM latency bound**: a 200B bf16 model on 32 v6e chips. How long to load all params from HBM?
   *Answer*: 400e9 B / 32 = 12.5e9 B per chip, at 1.6e12 B/s gives **~7.8 ms**. This is a lower bound on per-step sampling latency and is close to achievable at small batch.
2. **Pod accounting**: hosts, TensorCores, FLOPs/s and HBM for a full v5e pod and a full v5p pod.
   *Answer*: v5e: 16·16/8 = **32 hosts**, **256 cores** (1 per chip), 16·16·2e14 = **5.1e16** bf16 FLOPs/s, 256·16 = **4 TiB** HBM. v5p: 16·20·28 = 8960 chips, 8960/4 = **2,240 hosts**, **17,920 cores**, 8960·4.59e14 = **4.1e18** FLOPs/s, 8960·96 = **840 TiB**.
3. **PCIe intensity**: A bf16[D,F] and x bf16[B,D] sit in host DRAM. One v6e (9.2e14), PCIe 1.6e10, B≪D, F=4D. Find the minimum B to stay FLOPs-bound.
   *Answer*: $2BDF/9.2e14 > 2(BD+DF+BF)/1.6e10$, which reduces to $8BD^2/9.2e14 > 8D^2/1.6e10$, so **B > ~57,500**.
4. **General matmul latency**: int8[16384,4096] weights × int8[B,4096] on 1 v5e. (a) Time as a function of B? (b) From VMEM?
   *Answer*: (a) FLOPs = 2·4096·16384·B = 1.3e8·B, so $T_\text{math}$ = 1.3e8·B / 3.94e14. Bytes = 6.7e7 + 2e4·B, so $T_\text{comms}$ = (6.7e7 + 2e4·B)/8.2e11. FLOPs-bound when **B > 267**, a bit above 240 because the full D and F are counted. (b) Taking VMEM to MXU BW as 22x HBM (1.80e13) gives **B > 11**, and "in practice ... closer to 20".
5. **ICI bandwidth**: v5e 4x4 slice, send bf16[8,128,8192] from TPU{0,0} to TPU{3,3}, with 1 µs per hop. (a) When does the first byte arrive? (b) Total time?
   *Answer*: a 4x4 has no wraparound. 2·8·128·8192 = 1.7e7 bytes sent over 2 ports at 2·4.5e10 = 9e10 B/s takes **~188 µs**. 6 hops at 1 µs means the first byte arrives at **~6 µs**, and the total is **~194 µs**. Latency and bandwidth terms add.
6. **Pulling it together (hard)**: A int8[128·1024, 128·1024] (~16 GB) is sharded across a v5e 4x4 but offloaded to host DRAM. Copy it to TPU{0,0} and multiply by bf16[8, 128·1024].
   *Answer*: a 4x4 spans 2 hosts. Best path: PCIe-load each shard onto its own TPU, ICI-gather, then matmul. DCN would be worse. PCIe: 16 GB over 16 links at 1.6e10 takes **~63 ms**. ICI: TPU{0,0} must receive 15 GB over 2 links at 4.5e10 each, a lower bound of 15e9/(4.5e10·2) = **~167 ms** (comparable to a full AllGather, 16e9/(4.5e10·2)). HBM to MXU: 16e9/8.2e11 = **20 ms**. FLOPs: 2·8·128·1024·128·1024 = 2.7e11 at 1.97e14 takes **1.4 ms**. With pipelining the slowest piece dominates: **at least 167 ms, likely ~200 ms**.
7. **Pop quiz (Appendix A), VPU throughput** on TPU7x at 2.2 GHz.
   *Answer*: 4 vector instructions per cycle on 8·128 ALUs gives 8·128·4·2.2e9 = **9e12 f32 FLOPs/s per core** (**1.8e13 bf16**, packed). That is about 64x below the MXU's ~1.2e15 per core.

---

## Chapter 3: Sharded Matrices and How to Multiply Them

URL: https://jax-ml.github.io/scaling-book/sharding/ (accessed 2026-10-23)

Section headings:
- `#partitioning-notation-and-collective-operations` Partitioning Notation and Collective Operations
  - `#a-unified-notation-for-sharding` A unified notation for sharding
  - `#how-do-we-describe-this-in-code` How do we describe this in code?
- `#computation-with-sharded-arrays` Computation With Sharded Arrays
  - `#case-1-neither-multiplicand-has-a-sharded-contracting-dimension` Case 1
  - `#case-2-one-multiplicand-has-a-sharded-contracting-dimension` Case 2
  - `#case-3-both-multiplicands-have-sharded-contracting-dimensions` Case 3
  - `#case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis` Case 4
- `#a-deeper-dive-into-tpu-communication-primitives` A Deeper Dive into TPU Communication Primitives
  - `#our-final-communication-primitive-the-alltoall` Our final communication primitive: the AllToAll
  - `#more-about-the-reducescatter` More about the ReduceScatter
  - `#how-to-overlap-matmul-communication-with-compute` How to overlap matmul communication with compute
- `#what-have-we-learned` What Have We Learned?
- `#some-problems-to-work` Some Problems to Work

### Core concepts
- **Mesh**: a 2D/3D device grid with named axes, e.g. `Mesh(devices=((0,1),(2,3)), axis_names=('X','Y'))`. A **sharding** maps array dims to mesh axes.
- Notation: $A[I_X, J_Y]$ shards I over X and J over Y. A missing subscript means the dim is replicated. $A[I_{XY}, J]$ shards I over the flattened X·Y, and subscript order sets the traversal order. $A[I_X, J_X]$ is **forbidden**, because a mesh axis is "spent" once used.
- Global (logical) shape versus device-local shape.
- JAX code: `jax.make_mesh(axis_sizes=(4,2), axis_names=('X','Y'), axis_types=(Auto, Auto))`, `jax.NamedSharding(mesh, PartitionSpec(...))`, `jax.jit(..., out_shardings=...)`. `B.shape` is global, and `B.addressable_shards` shows local shards.
- Elementwise ops on sharded arrays cost nothing extra. Communication appears only on contractions.
- **The 4 sharded-matmul cases**:
  1. Neither input sharded on the contracting dim: no comms. E.g. $A[I_X,J]\cdot B[J,K_Y] \to C[I_X,K_Y]$.
  2. One input sharded on the contracting dim: **AllGather** it first. E.g. $\text{AllGather}_X A[I,J_X] \to A[I,J]$, then a local matmul. Alternative: a local partial matmul plus an AllReduce (Q4).
  3. Both sharded on the contracting dim along the same axis: a local matmul gives a result **unreduced** $C[I,K]\{U_X\}$. Then **AllReduce** (or **ReduceScatter** to keep the result sharded: $C[I,K_X]$).
  4. Both have a non-contracting dim sharded on the same axis ($A[I_X,J]\cdot B[J,K_X]$): invalid, since only diagonal blocks are computable. AllGather one input first.
- $\{U_X\}$ means "unreduced along X". $\cdot_\text{LOCAL}$ means a local sum without reduction.
- Block-matrix view: a sharded matmul is moving blocks, multiplying locally, and summing.
- **Collectives**:
  - **AllGather**: removes a subscript, $[A_X,B]\to[A,B]$.
  - **ReduceScatter**: sums an unreduced array and shards it, $[A,B]\{U_X\}\to[A_X,B]$.
  - **AllReduce** = ReduceScatter + AllGather, $[A_X,B]\{U_Y\}\to[A_X,B]$.
  - **AllToAll**: moves a subscript between dims, $[A,B_X]\to[A_X,B]$. Arises in MoE.
- Ring AllGather: unidirectional takes N−1 hops of bytes/N. Bidirectional takes ⌊N/2⌋ hops of 2·bytes/N.
- **AllGather cost (bidirectional ring)**: $T_{hop} = \frac{2V}{|X| W_\text{ici}}$ over $|X|/2$ hops gives $T_\text{total} = \frac{V}{W_\text{ici}}$. Here V is the total array bytes and $W_\text{ici}$ is the bidirectional BW. **Independent of the number of shards.** "Even though our TPUs are only locally connected, the locality of the connections doesn't matter."
- With latency: $T_{total} = \max\left[\frac{T_{min}\cdot X}{2}, \frac{V}{W_\text{ici}}\right]$ with $T_{min}$ ≈ 1 µs per hop. On v5e (4.5e10 unidirectional), any buffer under 4.5e10·1e-6 = **45 kB** is latency-bound.
- Multi-axis: $T_{total} = \max\left[\frac{T_{min}\sum_i|X_i|}{2}, \frac{V}{W_\text{ici}\cdot N_\text{axes}}\right]$. More axes give proportionally more bandwidth.
- Empirical result on a v5e 8x16 (gathered over the 16-axis): reaches ~95% of the 4.5e10 peak, and hits peak at about 10 MB total (~625 kB per device). "(aside: this is much better than GPUs)".
- **Cost summary** (bidirectional W, full ring):
  - $T_\text{AllGather} = T_\text{ReduceScatter} = \frac{V}{W_\text{ici}}$. Per mesh axis used, divide by num_axes.
  - $T_\text{AllReduce} = 2\frac{V}{W_\text{ici}}$.
  - $T_\text{AllToAll} = \frac{V\cdot\max(A,B,C,\ldots)}{4 N W_\text{ici}}$ on an A×B×C mesh. In 1D this is $V/(4W_\text{ici})$, **¼ of an AllGather**. In 2D or more it scales down with the smallest axis. Derivation: average distance N/4 on a ring, or bisection bandwidth.
  - Unidirectional form: $\frac{\text{Data volume}}{\text{bandwidth}}\cdot\frac{\text{Axis}-1}{\text{Axis}}$.
- **ReduceScatter is the derivative (transpose) of AllGather**, and vice versa. Broadcast and reduce are transposes, and AG and RS are their Kronecker products with $I_p$.
- Deferring the AllGather: use a ReduceScatter instead of an AllReduce to keep the output sharded.
- **Collective matmul** (Wang et al.): overlap the matmul on one chunk with the ring reduction of earlier chunks. Implemented in Part 10. The Mosaic docs have a GPU example.

### Hardware numbers used (Ch 3)
- H100 memory BW is taken as 3.4e12 in the pop quiz.
- v5e: 4.5e10 B/s unidirectional ICI per axis, ~1 µs per hop, latency threshold 45 kB.
- v4p: 9e10 B/s bidirectional, 4.5e10 unidirectional.
- $C/W_\text{ici} \approx 2550$ (Q4; this is the v5p value 4.59e14/1.8e11).

### TPU-vs-GPU comparisons (Ch 3)
- "A GPU AllGather can also work like this, where you create a ring out of the GPUs in a node and pass the chunks around in that (arbitrary) order."
- AllGather reaching peak BW at ~10 MB "is much better than GPUs".

### Problems (Ch 3)
- **Pop quiz 1**: fp32[1024,4096] as $A[I_{XY},J]$ on mesh {X:8, Y:2}. Bytes per device, and load time on H100 (3.4e12)?
  *Answer*: local fp32[64,4096] = 4·64·4096 = **1 MiB**, load time ≈ 1e6/3.4e12 = **294 ns**, more in practice because of overheads.
- **Pop quiz 2**: int8[128,2048], $A[I_{XY},J]$ on mesh {X:2, Y:8, Z:2} (32 devices). Per-device and total memory?
  *Answer*: int8[8,2048] = **16,384 B per device**. Replicated over Z, so 2 copies, **512 KiB total** (= 32 × 16,384).
- **Pop quiz (AllGather time)**: $\text{AllGather}_Y([E_Y,F])$ on v5e with mesh {X:8, Y:4}. Case (a) E=2048, F=8192 bf16. Case (b) E=F=256.
  *Answer*: (a) per-device 512·8192·2 = 8.4 MB, total 34 MB. The naive estimate is 34e6/9e10 = 377 µs. But the 4-long axis has no wraparound on v5e (that needs 16), so 3·8.4e6/4.5e10 = **560 µs**. The Colab profile measured **680 µs**. (b) shard 64·256·2 = 32 kB, 32e3/4.5e10 = 0.7 µs, so it is latency-bound: 3 hops ≈ **3 µs**, measured **~8 µs**.
1. **Replicated sharding**: $A[I_X,J,K,\ldots]$ on {X:4, Y:8, Z:2}. Ratio of total bytes across chips to one copy?
   *Answer*: each shard is size/4. It is replicated across Y·Z, so total = Y·Z·sizeof(A), giving a ratio of **16**.
2. **AllGather latency** on a v4p 4x4x4 with {X:4, Y:4, Z:4}, B=1024, D=4096 bf16. (a) $\text{AllGather}_X([B_X,D_Y])$, (b) $\text{AllGather}_{XY}$, (c) $\text{AllReduce}_Z([B_X,D_Y]\{U_Z\})$.
   *Answer*: a full cube has wraparound, so W = 9e10 bidirectional. (a) 2BD/(W·Y) = 2·1024·4096/(9e10·4) = **23 µs**. (b) 2BD/(2W) = **46 µs**, well above the ~4 µs latency bound. (c) 4BD/(X·Y·W) = 4·1024·4096/(16·9e10) ≈ **11.6 µs**. Fun fact from the book: (a) and (b) are not optimal. Reshard for free to $[B_{XZ},D_Y]$ to use the idle Z links, which cuts (a) to **11.5 µs** and (b) to **31 µs**. This is one reason to "shard arrays as finely as possible".
3. **Latency-bound AllGather**: $\text{AllGather}_X([B_X])$ with B=128 bf16 on a v4p 4x4x4.
   *Answer*: 256 B total, 64 B per device, so bandwidth time ≈ 0. With wraparound on a 4-axis the gather takes 2 hops, about **2 µs**.
4. **Matmul strategies**: $X[B,D]\cdot_D Y[D_X,F]$. Strategy 1 AllGathers Y and then multiplies. Strategy 2 multiplies locally ($X[B,D_X]$) and then AllReduces Z.
   *Answer*: S1: $T=\max(2BDF/C,\ 2DF/W)$. S2: $T=\max(2BDF/(XC),\ 4BF/W)$. With $C/W\approx2550$, S2 is compute-bound only if $X < D/(2\cdot2550)$, roughly X<2 for D≈8k, so it is basically always comms-bound. S1 is comms-bound when B<2550. If B<2550, S2 has lower comms when **D > 2B**. If B>2550, S2 beats S1 when **D > 5100**, which is "usually true for large models". It is rarely used in practice because one input is usually already sharded on that axis (e.g. FSDP).
5. **Minimum latency**: $A[I,J]\cdot B[J,K]$ on a v4p 4x4x4 with a fully replicated output. *Partial answer*: four options are (1) $A[I_{XYZ},J]$ + AG, (2) $B[J,K_{XYZ}]$ + AG, (3) $J_{XYZ}$ on both + AR, (4) fully replicated. FLOPs per TPU are equal except for (4). **(1) and (2) are equally good** (TLDR).
6. *(No answer given.)* On a v5e 4x4: comms vs compute for $A[I_X,J_Y]\cdot B[J_Y,K]\to C[I_X,K]$. Then $A[I_X,J]\cdot B[J_X,K_Y]\to C[I_X,K_Y]$ ("most standard setting for training ... data, tensor, and ZeRO"). Then $A[I_X,J]\cdot B[J,K_Y]\to C[I_X,K_Y]$ ("standard for inference ... pure tensor parallelism (+data)").
7. **Transformer block under a memory cap**: In[B,D]·W_in[D,F]·W_out[F,D] with D=8192, F=32768, B=128 bf16 on a v5e 2x2 with 300 MB free per TPU. *Partial answer*: each weight is 2·8192·32768 = **536 MB** and activations are 2 MB, so sharding is required. Options: FSDP $In[B_X,D]\cdot W_{in}[D_{XY},F]\cdot W_{out}[F,D_{XY}]$, which is "pretty bad" because it gathers the big weights. Or TP $In[B,D_{XY}]\cdot W_{in}[D,F_{XY}]\cdot W_{out}[F_{XY},D]\to Out[B,D_{XY}]$, which does an AllGather at the start and a ReduceScatter at the end. The rest is left as an exercise.
8. *(Challenge, no answer given.)* Benchmark AllGather, AllReduce, ReduceScatter and AllToAll with `jax.lax.all_gather`, `psum`, `psum_scatter` and `all_to_all` via pmap/shard_map.
9. **Matmul-then-AllReduce for one sharded input**: $A[I,J_X]\cdot B[J,K]$. (a) Write it with indices. (b) Change it for a sharded output. (c) Compare its comms with AllGather-first.
   *Answer*: (a) each device computes partial outer-product sums over its local J indices into O[N,K], then AllReduces. (b) Use a ReduceScatter over N or K instead. (c) AG-first comms scale as NM. Matmul-then-RS comms scale as NK. **Ratio M/K.**
10. **Where AllToAll's factor of 4 comes from** (N×N matrix, D devices on a ring). Unidirectional AG/RS: each link carries $N^2(1-1/D) \approx N^2$ scalars. Unidirectional AllToAll: $\frac{N^2}{2}(1-1/D) \approx N^2/2$, a factor of **½** from 1+2+…+n = n(n+1)/2. Bidirectional halves AG time (**2x win**) but cuts AllToAll by **4x** (sum ≈ D²/8 against D²/2). Net result: in a bidirectional ring AllToAll costs **¼** of an AllGather.

---

## Chapter 4: All the Transformer Math You Need to Know

URL: https://jax-ml.github.io/scaling-book/transformers/ (accessed 2026-10-23)

Section headings:
- `#counting-dots` Counting Dots
  - `#forward-and-reverse-flops` Forward and reverse FLOPs
- `#transformer-accounting` Transformer Accounting
- `#global-flops-and-params-calculation` Global FLOPs and Params Calculation
  - `#mlps` MLPs · `#attention` Attention · `#other-operations` Other Operations
  - `#general-rule-of-thumb-for-transformer-flops` General rule of thumb for Transformer FLOPs
  - `#fractional-cost-of-attention-with-context-length` Fractional cost of attention with context length
- `#miscellaneous-math` Miscellaneous Math
  - `#sparsity-and-mixture-of-experts` Sparsity and Mixture-of-Experts
  - `#gradient-checkpointing` Gradient checkpointing
  - `#key-value-kv-caching` Key-Value (KV) caching
- `#what-should-you-take-away-from-this-section` What Should You Take Away from this Section?
- `#a-few-problems-to-work` A Few Problems to Work
- `#appendix` · `#appendix-a-how-does-flash-attention-work` Appendix A: How does Flash Attention work?

### Core concepts and formulas
- FLOP counting: dot product = $2P$, matvec = $2NP$, matmul $AB$ = $2NPM$. General einsum FLOPs = 2 × the product of all dims, with batch and contracting dims counted once. A dim is "batching" only if it appears in both inputs and in the output. Contracting dims appear in both inputs but not the output. No factor of 2 for pure elementwise products.
- Matmul compute is $O(N^3)$ while data is $O(N^2)$, "why we use architectures dominated by matrix multiplication".
- **Backward = 2x forward**: $\partial L/\partial B = A^T(\partial L/\partial C)$ and $\partial L/\partial A = (\partial L/\partial C)B^T$, each $2NPM$. Training total is **6NPM** (2 forward + 4 backward). Hence **6 × params × tokens**.
- Shape letters: B batch, T query seq, S KV seq, D d_model, F d_ff, N query heads, K KV heads, H head dim, G = N/K (query groups), L layers, V vocab.
- Notes: gated MLP ("gating einsum") gives 3 matrices and 3DF params (LLaMA, DeepSeek), while non-gated gives 2DF. MHA: N=K. MQA: K=1. GQA ("GMQA"): K divides N. Pre-norm $x + \text{attn}(\text{norm}(x))$ (LLaMA-3) versus post-norm.
- **MLP (gated)**: 3 matmuls at $6BTDF$ each, $\approx 18BTDF$ train FLOPs, **3DF params**.
- **Attention projections (GQA)**: $12BTD(N+K)H$ train FLOPs, $2D(N+K)H$ params.
- **Dot-product attention**: $QK^T$ and $SV$ each $6BTSNH$, total $\approx 12BTSNH = 12BT^2NH$ for self-attention. A causal mask halves the useful FLOPs, but only with an attention kernel.
- Other: 2 layernorms at $O(BTD)$ with 2D params. Unembedding $6BTDV$ with DV params.
- **Rule of thumb**: $(18BTDF + 12BTD(N+K)H)L = 6 \cdot BT \cdot (3DF + 2D(N+K)H)L = 6 \cdot \text{tokens} \cdot \text{params}$.
- **Attention share** (F=4D, D=NH, N=K): $\frac{\text{attn FLOPs}}{\text{matmul FLOPs}} = \frac{12BT^2NH}{18BTDF + 24BTDNH} = \frac{T}{8D}$. Attention dominates only when **T > 8D**: ~64K tokens for D≈8k, and ~37k for Gemma-27B (D=4608).
- **MoE**: E experts, k active per token. Sparsity E/k is "usually between 8 and 64" (DeepSeek v3: k=8, E=256). Params grow by O(E), and active params scale with k. Adds 2 AllToAlls per MoE block (each ¼ of an AllGather on a bidirectional ring).
- **Gradient checkpointing (remat)**: saving everything costs ~$2\cdot20\cdot BTDL$ bytes. Example: BT=4M, L=64, D=8192 gives **84 TB** of bf16 activations ("20" ≈ intermediate nodes per layer). **Block remat** saves only layer inputs: 4.2 TB, and FLOPs rise from 6ND to **~8ND**. **"Big matmuls only"** cuts ~20 saves per layer to ~7. Controlled with `jax.remat` / `jax.checkpoint`.
- **KV cache**: shape [2, S, L, K, H], so **$2SLKH$ bytes in int8** (×2 for bf16). Example: 8k ctx, 64 layers, KH = NH = D = 8192 gives 2·8192·64·8192 = **8 GiB**. This motivates GQA with K≪N.
- **Summary table (MHA)**:

| Component | Params per layer | Training FLOPs per layer |
|---|---|---|
| MLP | 3DF | 18BTDF |
| Attention | 4DNH | 24BTDNH + 12BT²NH |
| Other | 2D | BTD |
| Vocab | DV (total, not per-layer) | 12BTDV |

  (The vocab row says 12BTDV, while the "Other Operations" section says 6BTDV for the unembedding. The 12 probably counts input and output embeddings.)
- **Flash attention (Appendix A)**: process K/V in chunks, carrying a running max M, running output O and running denominator L. Merge chunks with $L^\text{comb} = e^{M^1-\max(M^1,M^2)}L^1 + e^{M^2-\max(M^1,M^2)}L^2$. The Q chunk and running stats stay in **VMEM** (the paper's "on-chip SRAM"), which raises intensity. Backward trick: $S_{ij}\cdot_j dS_{ij} = dO_{id}\cdot_d O_{id}$ replaces a contraction over sequence length with one over head dim. This makes the block-local VJP possible and enables ring attention.

### Hardware numbers used (Ch 4)
- H800 FP8: 3,026 TFLOPs/s sparse, so 1.513e15 dense (Q7).
- TPU v5e: ~240 FLOPs/byte (Q8).

### TPU-vs-GPU comparisons (Ch 4)
- Flash attention's "on-chip SRAM" maps to TPU VMEM. Otherwise this chapter is hardware-neutral.

### Problems (Ch 4)
1. **Params**: D=4096, F=4D, V=32,000, L=64, NH=D, MHA, int8 KV. Total params, attention fraction, KV per token?
   *Answer*: $L(3DF + 4DNH + 2D) + 2DV$ ≈ **16B**. Attention fraction = $4D^2/(4D^2+12D^2)$ = **¼**. KV per token = $2\cdot L\cdot N\cdot H$ = 2·64·4096 B = **512 KiB/token** (int8).
2. **Sharded FLOPs**: $A[B_X,D_Y]\cdot_D W[D_Y,F]$ on {X:4, Y:8, Z:4}.
   *Answer*: the theoretical total is $2BDF$. It is not sharded over Z, so **$2BDF\cdot Z$** total FLOPs are actually executed. Per device ≈ **$2BDF/(X\cdot Y)$**.
3. **FLOPs of** $A[I,J,K,L]\cdot B[I,J,M,N,O]\to C[K,L,M,N,O]$.
   *Answer*: I and J contract, no batch dims, so **$2\cdot IJKLMNO$**.
4. **GQA attention intensity** (ignoring projections), as a function of T, S, G.
   *Answer*: bytes = $4BTNH + 4BSKH = 4BHK(TG+S)$. FLOPs = $4BTSNH$ (forward). Intensity = $\frac{4BTSKGH}{4BHK(TG+S)}$. Prefill (S=T) gives **$TG/(G+1) = O(T)$**, so attention is compute-bound around **S=240** (no sequence sharding). Generation (T=1) gives **$SG/(G+S)\to G$**, so it is never compute-bound because G is small. Larger G helps.
5. **When does self-attention FLOPs = QKVO projection FLOPs?**
   *Answer*: $24BTDNH = 12BT^2NH$, so **T = 2D**. For D=4096 that is **8192**.
6. **Remat cost when only the 7 matmul outputs are saved** (Q, K, V, O, W₁, W₂, W₃).
   *Answer*: recompute $QK^T$ and $\text{softmax}(QK^T)V$, adding **$4BT^2NH$** FLOPs, plus $O(BTD)$ and $O(BTF)$ elementwise recomputation.
7. **DeepSeek v3 MFU**: 2.79M H800-hours, 14.8T tokens, 37B active params, FP8 dense.
   *Answer*: 1.513e15 × 2.79e6 × 3600 = 1.52e25 available FLOPs. Needed: 6·37e9·14.8e12 = 3.3e24. **~21.7% utilization.**
8. **MoE critical batch** with int8 weights on v5e, and for DeepSeek (E=256, k=8).
   *Answer*: weight bytes per matrix = $EDF$, FLOPs = $2kBDF$. Need $2kB/E > 240$, i.e. $B > 120E/k$. For DeepSeek, **B > 3840** tokens, "remarkably large ... at generation time".

---

## Chapter 5: How to Parallelize a Transformer for Training

URL: https://jax-ml.github.io/scaling-book/training/ (accessed 2026-10-23)

Section headings:
- `#what-do-we-mean-by-scaling` What Do We Mean By Scaling?
  - `#data-parallelism` Data Parallelism
  - `#fully-sharded-data-parallelism-fsdp` Fully-Sharded Data Parallelism (FSDP)
  - `#tensor-parallelism` Tensor Parallelism
  - `#combining-fsdp-and-tensor-parallelism` Combining FSDP and Tensor Parallelism
  - `#pipelining` Pipelining
  - `#scaling-across-pods` Scaling Across Pods
- `#takeaways-from-llm-training-on-tpus` Takeaways from LLM Training on TPUs
- `#some-problems-to-work` Some Problems to Work
- `#appendix` · `#appendix-a-deriving-the-backward-pass-comms` Appendix A: Deriving the backward pass comms

### Core concepts
- Notation: D, F, B (total tokens, not per device), T, L; C = FLOPs/s per chip; W = bidirectional network BW ($W_\text{ici}$, $W_\text{dcn}$); X, Y, Z = chips per mesh axis. $M_X$ = number of hardware mesh axes a strategy spans, and collective time scales as $1/M_X$.
- Simplified model: each layer is In[B,D]·W_in[D,F]·W_out[F,D]. Attention and gating are ignored. Comms bounds are the focus, since memory rarely binds with remat at pretraining scale. Expert parallelism is not covered.
- **The four schemes (sharding formulas)**:
  - DP: $\text{In}[B_X,D]\cdot W_\text{in}[D,F]\cdot W_\text{out}[F,D]\to\text{Out}[B_X,D]$
  - FSDP: $\text{In}[B_X,D]\cdot W_\text{in}[D_X,F]\cdot W_\text{out}[F,D_X]\to\text{Out}[B_X,D]$
  - TP: $\text{In}[B,D_Y]\cdot W_\text{in}[D,F_Y]\cdot W_\text{out}[F_Y,D]\to\text{Out}[B,D_Y]$
  - FSDP+TP: $\text{In}[B_X,D_Y]\cdot W_\text{in}[D_X,F_Y]\cdot W_\text{out}[F_Y,D_X]\to\text{Out}[B_X,D_Y]$
  - Pipeline: weights sharded on L, i.e. $W_\text{in}[L_Z, D_X, F_Y]$.
- **Data parallelism**: no forward comms. The backward does 2 weight-gradient AllReduces per layer, which are **off the critical path** (async). Memory: bf16 params + fp32 Adam = **10 bytes/param**, so the max model is HBM/10. On v5p (96 GB) that is **~9B params**, an absolute lower bound at batch 1 without checkpoints.
  - $T_\text{comms} = \frac{2\cdot2\cdot2\cdot DF}{W_\text{ici}}$, $T_\text{math} = \frac{8BDF}{XC}$, so compute-bound iff **$\frac{B}{X} > \frac{C}{W_\text{ici}}$**. The per-device batch must exceed the ICI intensity, the counterpart of the single-chip B>240 rule.
  - v5p: C=4.6e14 and W=2·9e10 give **2,550** tokens per chip on 1 axis. With 3 axes: **850** per chip, i.e. **7.6M tokens per batch per pod (8960 chips)**.
- **Context/sequence parallelism**: to the MLP, "tokens are tokens", so you can do DP over sequence too. Attention needs ring attention.
- **FSDP / ZeRO-3**: shard weights, grads and optimizer on the data axis. AllGather weights just in time in the forward pass, ReduceScatter grads in the backward. AR = AG + RS, so comms are the same as DP: **compute-bound iff B/X > C/W_ici** (4.59e14/1.8e11 = **2550** on v5p). ZeRO-1/2/3 shard optimizer, gradients and weights respectively. All cost the same, "so we can basically always do ZeRO-3".
  - Takeaway: DP and FSDP become bandwidth-bound on TPUv5 when the per-device batch is **< 2550/M_X**.
  - DeepSeek-V2's ~40M-token batch would allow ~**47,000 chips** (~5 v5p pods).
  - LLaMA-3 70B, 6.3e24 FLOPs, 16M batch: 16e6/(2550/3) = **18,823 chips** (~2 pods) at 50% MFU would take **~17 days**.
  - Critical batch size: for a fixed chip count, a smaller global batch means more comms-bound. Bigger batches give diminishing returns or instability, so the real game is fitting a fixed scaling-law batch onto many chips.
- **Tensor parallelism (Megatron)**: AllGather activations before W_in and ReduceScatter after W_out, both **on the critical path**. $T_\text{math} = \frac{4BDF}{YC}$, $T_\text{comms} = \frac{4BD}{W_\text{ici}}$, so compute-bound iff **$F > Y\cdot\frac{C}{W_\text{ici}}$**, i.e. **$Y < M_Y\cdot F/2550$**. Independent of batch and of precision (int8 doubles C/W to 5100 but halves bytes). "For most models this is between 8 and 16-way."
  - LLaMA-3 70B (F≈30,000): 8-way is fine, 16-way is comms-bound, and 8-way needs F≥20k. Gemma 7B (F≈50k): comms-bound at 19-way, so 16-way is OK.
- **FSDP + TP**: "FSDP moves weights and tensor parallelism moves activations."
  - $T_\text{FSDP} = \frac{4DF}{Y W_\text{ici} M_X}$, $T_\text{TP} = \frac{4BD}{X W_\text{ici} M_Y}$, $T_\text{math} = \frac{4BDF}{NC}$.
  - **$X_{opt} = \sqrt{\frac{B}{F}\frac{M_X}{M_Y}N}$**: the optimal FSDP degree, balancing the two comms terms. Example: N=64 (4x4x4), B=48,000, F=32768 gives X≈13.9, so pick X=16, Y=4.
  - Compute-bound iff **$\frac{B}{N} > \frac{\alpha^2}{M_X M_Y F}$** with α = C/W_ici. With F=32,768, α=2550 and M_X·M_Y=2: **B/N > ~99**, about 8x better than FSDP alone (~850 on a 3D mesh). Takeaway: B/N as low as $2550^2/2F \approx 100$.
  - $T_\text{math}/T_\text{comms} = \frac{\sqrt{BF}\sqrt{M_XM_Y}}{\alpha\sqrt N}$, which grows as √B (FSDP grows as B, TP is flat).
  - Mixed FSDP+TP is required for B/N between ~100 and ~850.
  - v5p 16x16x16 with D=8192, F=32768: all strategies are comms-bound below ~6e5 batch, against a predicted 4096·2550²/(2·8192·4) = **4e5**. The best split is **FSDP=512, TP=8**, compute-bound above ~4e5.
- **Pipelining**: "a dominant strategy for GPU parallelism that is somewhat less essential on TPUs". Low inter-stage comms, but suffers a **pipeline bubble**. Mitigations: microbatching, and overlapping the forward, dx and dW matmuls (DeepSeek v3's "bubble-free" schedule, which prioritizes dL/dx).
- **Across pods (DCN)**: the largest v5p slice is **8960 chips / 2240 hosts**. Each host has ~**200 Gbps (25 GB/s)** full-duplex DCN, about **6.25 GB/s per TPU**. Do model parallelism or FSDP within ICI and pure DP across pods.
  - $T_\text{math} = \frac{8BDF}{NC}$, $T_\text{comms} = \frac{8DF}{M W_\text{dcn}}$ (M = TPUs per slice), so you need **B per slice > C/W_dcn** = 4.59e14/6.25e9 = **73,440** for v5p. (The summary says "about 71,000".)
  - LLaMA-3 70B at BS=2M: TP up to Y≈11·M_Y. FSDP alone (3 axes) reaches ≈2400 chips. FSDP+TP is comms-bound below B/N = 2550²/(2·30000) = **108**, which would allow ~18k chips, but a pod is 8k, so use DCN. Recipe: BS=1M uses X(FSDP)=1024 and Y(TP)=8. BS=2M uses 2 pods, 1M per pod, 111 per TPU.
  - Takeaway: cross-pod DP is easy if the per-pod batch is **≥ 73k tokens**.
- **Comms per layer table** (bytes, forward + backward):

| Strategy | Compute per layer | Comms per layer |
|---|---|---|
| DP | 4BDF/X + 8BDF/X | 0 + 8DF |
| FSDP | 4BDF/X + 8BDF/X | 4DF + 8DF |
| TP | 4BDF/Y + 8BDF/Y | 4BD + 4BD |
| FSDP+TP | 4BDF/(XY) + 8BDF/(XY) | (4BD/X + 4DF/Y) + (8BD/X + 8DF/Y) |

- Up to ~32k context, modelling the Transformer as an MLP stack is fine.
- Appendix A: derive backward comms by applying the sharded-matmul rules to dW_out = Tmp·_B dOut, dTmp = dOut·_D W_out, dW_in = In·_B dTmp and dIn = dTmp·_F W_in. dOut is sharded like Out.

### Hardware numbers used (Ch 5)
- v5p: C = 4.59e14 (sometimes written 4.6e14), W_ici = 1.8e11 bidirectional (2·9e10), C/W_ici = 2550, int8 C/W = 5100, HBM 96 GB, pod 8960 chips / 2240 hosts, DCN 25 GB/s per host and 6.25 GB/s per TPU, C/W_dcn = 73,440.
- v5p 16x16x16 slice: 393 TB HBM (Q3).

### TPU-vs-GPU comparisons (Ch 5)
- Pipelining "is a dominant strategy for GPU parallelism that is somewhat less essential on TPUs". It is "often very useful on GPUs since they are not densely connected by ICI in the way TPUs are". The book skips depth here "because it is less critical for TPUs (which have larger interconnected pods)".

### Problems (Ch 5), using LLaMA-2 13B (L=40, D=5,120, F=13824, N=40, K=40, H=128, V=32,000, separate embed/unembed, gated MLP)
1. **Param count**.
   *Answer*: FFW 3LDF = **8.5e9**. Attention 4DNHL = **4.2e9**. Vocab 2VD = **0.33e9**. Total **13.0e9**.
2. **Memory** at BS=16M tokens with Adam, bf16 params, fp32 optimizer, checkpoints after the 3 FFW matmuls.
   *Answer*: params + optimizer = (2+4+4)·13e9 ≈ **130 GB**. Activations = $2\cdot L\cdot B(D+2F)$ = 2·40·16e6·5120·(1+2·2.7) ≈ **4.2e13 = 42 TB**.
3. **32k seq, 3M-token batch, v5p 16x16x16** (4096 chips, 393 TB HBM; 96 sequences).
   *Answer*: (a) **Pure DP: no**. 130 GB of params+optimizer exceeds 96 GB per chip. (b) **Pure FSDP**: memory is ~7.86e12 activations + 1.3e11 ≈ **8 TB**, which fits in 393 TB. But the minimum batch is 850·4096 = **3.48M > 3M**, so it is **comms-bound**. (c) **FSDP+TP: yes**. The threshold is 2550²/(2·13824) = **235** per chip. $X_{opt}$ = √(3e6·2·4096/13824) = **1333**, so use roughly **1024-way DP × 4-way TP**. Step time = 6·3e6·13e9/(4096·4.6e14·0.4) = **~300 ms** at 40% MFU.

---

## Chapter 6: Training LLaMA 3 on TPUs

URL: https://jax-ml.github.io/scaling-book/applied-training/ (accessed 2026-10-23)

Section headings:
- `#what-does-llama-3-look-like` What does LLaMA 3 look like?
- `#counting-parameters-and-flops` Counting parameters and FLOPs
- `#how-to-shard-llama-3-70b-for-training` How to shard LLaMA 3-70B for training
- `#worked-problems` Worked Problems

The chapter is mostly worked Q&A, with answers hidden in collapsible boxes in the original. Target hardware is a TPU v5p pod.

### LLaMA 3-70B config (from HF)
L=80, D=8,192, F=28,672, N=64, K=8, H=128, V=128,256.

### Core concepts and worked Q&A
- **Param count**:
  - FFW: D·F·3·L = 8,192·8,192·3.5·3·80 = **56.3e9**.
  - Vocab: 2·V·D = **2.1e9**.
  - Attention: L·(2·D·N·H + 2·D·K·H) = **12e9**.
  - Total **70.4e9**.
  - Takeaway: the 3 MLP matrices hold 56B of the 70B.
- **FLOPs per token**: 6·70e9 = **4.2e11** (about half a TFLOP). At 4.59e14 that is ~**1 ms** on one v5p.
- **Total FLOPs over 15T tokens**: 4.2e11·15e12 = **6.3e24** ("6.3 yottaFLOPs"). On one TPU, **435 years**.
- **Full v5p pod** (16x20x28 = 8960 chips) at 40% MFU: 6.3e24/(8960·4.59e14·0.4) = **3.8e6 s ≈ 44 days**.
- **Minimum chips for a 4M-token batch** (bf16 params, fp32 optimizer, 2 checkpoints per layer):
  - Params 2·70 GB = ~140 GB. Optimizer 8·70 GB = ~560 GB. Activations 2·8192·4e6·2·80 = ~10.5 TB. Total ~**11.2 TB**.
  - 11.2e12/96e9 = **117 TPUs**, but training would take 44·8960/117 = **3369 days** (~10 years).
  - "We're using these large clusters not because we're bound by memory but rather because we need the extra FLOPs." Sparse checkpointing means closer to **8ND** than 6ND.
- **Per-chip memory on 8960 chips**: **~1.3 GB per chip**. With 12 checkpoints per layer, still only ~8 GB. "Nowhere near being memory bound."
- Takeaway: total FLOPs plus an assumed MFU plus a topology gives a ballpark training time.
- **Sharding** (4M batch = 1024 seqs × 4096, 8960 chips):
  - **FSDP alone, no sequence parallelism**: **no**. Only 1024 sequences, so at most 1024-way DP/FSDP.
  - **FSDP over batch + sequence**: 4·1024·1024/8960 = **468 tokens per chip** < 2550/3 = **850**, so **comms-bound**. No.
  - **FSDP + TP**: the threshold is $2550^2/2F$ = **113**, and 468 is above it, so it works. $X_{opt} = \sqrt{2BN/F} = \sqrt{2\cdot4.19e6\cdot8960/28672}$ = **1618**, so use **~2048-way FSDP × 4-way TP**.
  - Takeaway: on a full v5p pod, "a mixture of data parallelism (1024-way), sequence parallelism (2-way), and tensor parallelism (4-way)" avoids being comms-bound. Pure FSDP and FSDP+seq are comms-bound.

### Worked Problems (end of chapter; **no answers provided on the page**)
1. **Scale LLaMA 3-70B to 4 pods** at the same batch: which scheme, compute- or comms-bound, how long? "Make sure to use the correct roofline bound" (a hint pointing at the DCN bound from Ch 5).
2. **LLaMA 3-405B**: (a) build the hyperparameter table from the (gated) HF config, then compute total params, FLOPs per step, and FLOPs over 15T tokens. (b) On 8 v5p pods: which parallelism scheme, how long, compute- or comms-bound?
   (Course authors would need to derive these. Inputs from Ch 5: C/W_dcn = 73,440, 2550²/2F, $X_{opt}$.)

### Hardware numbers used (Ch 6)
- TPU v5p: 4.59e14 bf16 FLOPs/s, 96 GB HBM, pod 16x20x28 = 8960 chips.

### TPU-vs-GPU comparisons (Ch 6)
- None explicit.

---

## Cross-chapter cheat sheet (for quiz writers)

| Quantity | Value in book | Where |
|---|---|---|
| v5e HBM critical intensity | ~240 FLOPs/byte (1.97e14 / 8.2e11) | Ch1 |
| H100 critical intensity | ~295 (9.89e14 / 3.35e12); "closer to 300" | Ch1 Q5 |
| int8-weight / bf16-compute critical B (v5e) | 120 | Ch1 Q2 |
| v5e 2-chip D-sharded matmul threshold | D > 8755 | Ch1 |
| v5p ICI intensity C/W_ici | 2550 (1 axis), 850 (3 axes) | Ch5 |
| v5p DCN intensity C/W_dcn | 73,440 (summary says ~71,000) | Ch5 |
| TP limit | Y < M_Y·F/2550 (8-16 way typical) | Ch5 |
| FSDP+TP min per-chip batch | 2550²/(2F) ≈ 100 | Ch5 |
| X_opt | √((B/F)(M_X/M_Y)N) | Ch5 |
| AllGather / ReduceScatter | V / W_ici (bidi), / num_axes | Ch3 |
| AllReduce | 2V / W_ici | Ch3 |
| AllToAll | V·max(axis)/(4·N·W_ici); 1D: ¼ AllGather | Ch3 |
| ICI hop latency | ~1 µs; latency-bound below 45 kB (v5e) | Ch2/3 |
| Training FLOPs | 6·params·tokens (≈8 with block remat) | Ch4 |
| Attention dominance | T > 8D | Ch4 |
| KV cache | 2·S·L·K·H bytes (int8) | Ch4 |
| Adam memory | 10 bytes/param (bf16 + fp32 m,v) | Ch5 |
| MoE critical B (int8 w, v5e) | 120·E/k (DeepSeek: 3840) | Ch4 Q8 |

## Gaps / caveats
- Every page loaded. Equations were MathJax source in the HTML and were read as LaTeX. Images, animations and the Ch 5 interactive slider were not viewable. Their captions are summarized above.
- Problems with **no answer on the page**: Ch3 Q6 and Q8 (Q5 and Q7 are partial); Ch6 Worked Problems Q1 and Q2.
- The book was revised after its "Feb. 4, 2025" date (TPU7x content, GPU chapter). Some text still carries older numbers: "290 trillion multiply-adds/s"; VMEM 22x HBM in Ch2 Q4 against ~7x for TPU7x in prose; v6e at 9.1e14 in Ch1 against 9.20e14 in the Ch2 table; the TPU7x ICI per-link table (1.8e11 bidi) against prose (1.2e12 per chip, 4e11 per axis). Cite the table numbers and flag the discrepancies to learners.
- Neither the VMEM size for generations other than TPU7x (128 MiB) nor the clock speed for generations other than TPU7x (2.2 GHz) appears in this half of the book.
