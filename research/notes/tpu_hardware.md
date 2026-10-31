# TPU Hardware: History and Current Specs (source notes)

All sources were accessed on **2026-10-23** unless noted otherwise. "Docs" means the Cloud TPU documentation, which now lives at `docs.cloud.google.com/tpu/docs/...`. The old `cloud.google.com/tpu/docs/...` URLs return 301 redirects to it. Every docs page used here showed "Last updated 2026-10-22 UTC".

Conventions:
- **[P]** marks a primary source: a Google paper, Google docs, or a Google blog post.
- **[S]** marks a secondary source: press or third-party analysis.
- **UNVERIFIED** means no primary source could be found.
- **CONFLICT** marks a place where sources disagree.
- **(derived)** marks a number I computed from other numbers. No source states it.

---

## 1. Generation table

### 1a. Chip-level specs

| Gen (deployed / GA) | Process | Clock | TC/chip | MXUs per TC × size | Peak dense compute / chip | On-chip SRAM | HBM capacity / BW | TDP / measured power |
|---|---|---|---|---|---|---|---|---|
| **TPUv1** (2015, inference) | 28 nm [A1] | 700 MHz [A1 Tbl 2] | 1 | 1 × 256×256, int8 [A1 §2] | 92 TOPS int8 [A1] | 28 MiB: 24 MiB Unified Buffer + 4 MiB accumulators [A1 §2] | 8 GiB DDR3 at 34 GB/s (off-chip, not HBM) [A1 Tbl 2] | TDP 75 W; measured 28 W idle / 40 W busy [A1 Tbl 2] |
| **TPUv2** (2017) | 16 nm [A3, A5]. **CONFLICT:** A2 says ">12nm" | 700 MHz [A2 Tbl 3] | 2 [A2] | 1 × 128×128 [A2, A3] | 46 TFLOPS bf16 (3 TFLOPS fp32) [A2 Tbl 3] | 32 MiB [A2]. Figure 2 shows 8 MiB Vmem per core | 16 GiB at 700 GB/s [A2, A3]. Each core sees half of the HBM [A3] | TDP 280 W [A2] |
| **TPUv3** (2018) | 16 nm [A3, A4, A5]. **CONFLICT:** A2 says ">12nm" | 940 MHz [A2, A3, A4] | 2 [D-v3] | 2 × 128×128 [A2, D-v3] | 123 TFLOPS bf16 [A2, A4, D-v3]. **CONFLICT:** scaling book gives 1.4e14 [B1] | 32 MiB VMEM + 5 MiB spMEM [A4 Tbl 4] | 32 GiB HBM2 at 900 GB/s [A2, A4, D-v3] | TDP 450 W [A2, A5]; measured 123 idle, 175/220/262 W min/mean/max [A4, D-v3] |
| **TPUv4i** (2020, inference, internal only) | 7 nm [A5] | 1050 MHz [A5] | 1 | 4 × 128×128 [A5] | 138 TFLOPS bf16 or int8 [A5] | 144 MB total, including 128 MiB CMEM [A5 Tbl 1] | 8 GB at 614 GB/s [A5] | TDP 175 W [A5] |
| **TPU v4** (2020) | 7 nm, <600 mm², 22 B transistors [A4 Tbl 4] | 1050 MHz [A4] | 2 [A4, D-v4] | 4 × 128×128 [A4 §2.2, D-v4] | 275 TFLOPS bf16 or int8 [A4, D-v4] | 16 MiB VMEM per TC (32 MiB per chip) + 128 MiB CMEM + 10 MiB spMEM [A4 §2.2, Tbl 4] | 32 GiB HBM2 at 1200 GB/s [A4, D-v4] | Measured 90 idle, 121/170/192 W [A4, D-v4]. Hot Chips 2023 says "typical ~200W" [C2] |
| **TPU v5e** (Aug 2023) | not public | not public | 1 [D-v5e] | 4 × 128×128 [D-v5e, D-arch] | 197 TFLOPS bf16; 393 TOPS int8 [D-v5e] | not public | 16 GB at 800 GiBps [D-v5e]. Note GiBps vs GBps units | not public |
| **TPU v5p** (Dec 2023) | not public | not public | 2 [D-v5p] | 4 × 128×128 [D-v5p, D-arch] | 459 TFLOPS bf16 [D-v5p]. Docs list FP8 as 459, the same as bf16 [D-v5p, D-7x]. 918 TOPS int8 per [B1]; **UNVERIFIED in Google docs** | not public | 95 GiB at 2765 GBps [D-v5p] | not public |
| **TPU v6e "Trillium"** (announced May 2024; GA Dec 2024) | not public | not public | 1 [D-v6e] | 2 × 256×256 [D-v6e, D-arch] | 918 TFLOPS bf16; 1836 TOPS int8 [D-v6e]. The blog claims 4.7× v5e [G3] | not public | 32 GB at 1638 GBps [D-v6e] (2× v5e [G3]) | not public. Blog claims >67% more energy-efficient than v5e [G3] |
| **TPU7x "Ironwood"** (announced Apr 2025; Preview 24 Nov 2025; GA 31 Mar 2026 [D-rn]) | not public | ~2.2 GHz [B1]. **UNVERIFIED** by Google docs | 2: one per chiplet; 2 chiplets per chip [D-7x] | 2 × 256×256 [B1, D-arch] | 2307 TFLOPS bf16; 4614 TFLOPS FP8 [D-7x, G4] | VMEM 128 MiB per chip [B1]. Docs give no size | 192 GiB [D-7x], or "192 GB" [G4], at 7380 GBps [D-7x] / 7.37 TB/s [G4]. Eight stacks of HBM3e [S3] | not public. About 10 MW per 9,216-chip pod [G4] (≈1.1 kW/chip, derived and including overhead). 2× perf/W vs Trillium [G4] |
| **TPU 8t** (announced 22 Apr 2026; GA "later this year", not yet GA) | **UNVERIFIED.** TSMC 2nm per [S1, S5]; "N3 family" per Tom's Hardware search snippet [S6] | not public | not public | not public | **12.6 PFLOPS FP4** [G6]. Native FP4 "doubling MXU throughput" [G6]. No FP8 or BF16 figure published | **128 MB** VMEM [G6] | **216 GB at 6,528 GB/s** [G6] | not public. Up to 2× perf/W vs Ironwood [G5, G6] |
| **TPU 8i** (announced 22 Apr 2026; not yet GA) | **UNVERIFIED.** TSMC 2nm per [S1, S5] | not public | 2 TensorCores "on core dies" plus one CAE on a chiplet die [G6] | not public | **10.1 PFLOPS FP4** [G6]. Third-party reports of "11.6 FP8 EF/pod" are **UNVERIFIED** | **384 MB** VMEM, 3× Ironwood [G5, G6] | **288 GB at 8,601 GB/s** [G5, G6] | not public. Up to 2× perf/W vs Ironwood [G5, G6] |

### 1b. System-level specs

| Gen | Chips per pod / max slice | Topology | ICI per chip | SparseCores per chip | DCN | Chips per host |
|---|---|---|---|---|---|---|
| **TPUv1** | n/a (PCIe co-processor) | none | none; PCIe Gen3 x16 [A1] | 0 | n/a | 4 [A1] |
| **TPUv2** | 256 chips (16×16) [A2, D-v2] | 2D torus [A2] | 4 links × 496 Gb/s [A2, A5]. A3 says "500 Gbits/second" and "4x 60 GByte/second". Bisection 15.9 Tb/s for 256 chips [A2] | Present ("part of TPUs since TPU v2" [A4]); count **UNVERIFIED** | n/a | 4 [A2, A5] |
| **TPUv3** | 1024 chips [A2, D-v3]; 126 PF bf16 per pod [D-v3] | 2D torus [D-v3] | 4 × 656 Gb/s [A2, A5]. A3 says "650 Gbps". A4 Tbl 4 says "4 links @ 70 GB/s" (**CONFLICT**: 656 Gb/s is about 82 GB/s) | 2 [A4 Tbl 4] | n/a | **CONFLICT:** 8 per [A2, A4]; 4 per [A5] |
| **TPU v4** | 4096 chips = 64 cubes of 4×4×4 [A4, D-v4]; 1.1 EF per pod [D-v4] | 3D torus via OCS; twisted torus optional [A4 §2, D-v4] | 6 links @ 50 GB/s [A4 Tbl 4]. Bisection 24 TB/s per pod [D-v4] | 4 [A4 Tbl 4] | n/a | 4 [A4] |
| **TPU v5e** | 256 chips (16×16) [D-v5e] | 2D torus [D-v5e] | 400 GBps bidirectional, 4 ports [D-v5e] | none listed | 6.4 Tbps per pod; 2×100 Gbps NIC per host [D-v5e] | 8 per host; VMs of 1, 4, or 8 chips [D-v5e] |
| **TPU v5p** | 8960 per pod; max single slice 6144 (16×16×24); Multislice up to 18,432 [D-v5p] | 3D torus; twisted for some shapes [D-v5p] | 1200 GBps bidirectional [D-v5p]. The blog says "4,800 Gbps/chip" [G2] (= 600 GB/s, likely one direction) | 4 [D-v5p, D-arch]. The blog calls them "second-generation SparseCores" [G2] | 50 Gbps per chip [D-v5p, D-7x] | 4 [D-v5p] |
| **TPU v6e** | 256 chips (16×16) [D-v6e, G3] | 2D torus [D-v6e] | 800 GBps bidirectional, 4 ports [D-v6e] | 2 [D-arch, D-7x]. The blog says "third-generation SparseCore" [G3] | 100 Gbps per chip [D-7x]; 25.6 Tbps per pod [D-v6e] | 8 [D-v6e] |
| **TPU7x Ironwood** | 9216 per pod (144 cubes) [D-7x, G4]. Slices >2048 chips (8×16×16) need the account team [D-7x]. 1.77 PB HBM per superpod [G7] | 3D torus of 4×4×4 cubes via OCS [D-7x] | 1200 GBps bidirectional: 6 links × 200 GBps [D-7x]. "9.6 Tb/s" [G7] (= 1.2 TB/s). **CONFLICT:** scaling book gives 1.8e11 B/s bidirectional per link [B1] vs 200 GBps in docs | 4: 2 per chiplet [D-7x]. "Enhanced" [G4]; "4th-gen" per Hot Chips coverage [S3] | 100 Gbps per chip [D-7x] | 4 [D-7x] |
| **TPU 8t** | 9,600 per superpod; 2 PB HBM; 121 EF (FP4) per superpod [G5, G6] | 3D torus [G6] | 2× Ironwood [G5, G6]. That gives ~2400 GBps bidirectional (derived; no absolute figure published for 8t) | Yes, count not stated. "SparseCore (Embeddings) & LLM Decoder Engine" [G6] | Virgo network, up to 4× Ironwood DCN [G6]. Above 134,000 chips in one fabric, 47 Pb/s bisection [G6]. >1M chips per cluster with JAX/Pathways [G5, G6] | Arm Axion host [G6] |
| **TPU 8i** | 1,152 physical / up to 1,024 active chips per pod [G6]. See conflict note C-8 | **Boardfly** (new). Four-chip building block, 8 boards per group, 36 groups per pod via OCS; ≤7 hops chip to chip [G6] | **19.2 Tb/s** (2× Ironwood) [G5] | **0.** The CAE replaces Ironwood's 4 SparseCores [G6] | not stated | Arm Axion; 2× the physical CPU hosts per server [G5] |

TC = TensorCore. MXU = matrix-multiply unit (systolic array). CMEM = Common Memory. spMEM/Spmem = SparseCore memory. CAE = Collectives Acceleration Engine. OCS = Optical Circuit Switch.

---

## 2. TPU 8: the eighth generation (two variants, confirmed)

**Confirmed by primary sources.** Google announced the eighth-generation TPU at Google Cloud Next '26 on **2026-04-22** as two separate chips:

- **TPU 8t** ("t" for training). It targets large-scale pre-training and embedding-heavy workloads.
- **TPU 8i** ("i" for inference). It targets sampling, serving, reasoning, post-training RL and low-latency MoE serving.

Primary sources:
- **[G5]** Amin Vahdat, "Our eighth generation TPUs: two chips for the agentic era," The Keyword (blog.google). https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/
- **[G6]** Diwakar Gupta & Sabastian Mugazambi, "TPU 8t and TPU 8i technical deep dive," Google Cloud Blog, 2026-04-22. https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive. This post contains the only official spec table.
- Google's own X post: "dual chip approach: TPU 8t, optimized for training, and TPU 8i, optimized for inference" https://x.com/Google/status/2046993420841865508. Seen in search results only; I did not fetch it.

Official spec table [G6]:

| | TPU 8t | TPU 8i |
|---|---|---|
| Workload | Large-scale pre-training | Sampling, serving, reasoning |
| Topology | 3D torus | Boardfly |
| Special units | SparseCore (embeddings) + "LLM Decoder Engine" | CAE (Collectives Acceleration Engine) |
| HBM | 216 GB | 288 GB |
| On-chip SRAM (VMEM) | 128 MB | 384 MB |
| Peak FP4 | 12.6 PFLOPS | 10.1 PFLOPS |
| HBM BW | 6,528 GB/s | 8,601 GB/s |
| Host CPU | Arm Axion | Arm Axion |

Other primary claims:
- **8t** [G5, G6]:
  - 9,600-chip superpod with 2 PB shared HBM and 121 EF.
  - About 3× compute per pod vs Ironwood.
  - 2× ICI and up to 4× DCN vs Ironwood.
  - Virgo network: flat two-layer non-blocking fabric built from high-radix switches, multi-planar.
  - TPUDirect RDMA and TPUDirect Storage, with storage access 10× faster than Ironwood (Managed Lustre).
  - Target goodput above 97%.
  - Up to 2.7× training perf/$ vs Ironwood.
  - The VPU is scaled up so quantization, softmax and layernorm can overlap with MXU work.
- **8i** [G5, G6]:
  - 3× SRAM, so the KV cache can stay on chip.
  - The CAE cuts on-chip collective latency by up to 5×.
  - Boardfly cuts network diameter by more than 50%. Example: a 1,024-chip 3D torus of 8×8×16 has a 16-hop diameter; Boardfly has 7 hops.
  - 80% better perf/$ vs Ironwood.
- **Both** [G5, G6]:
  - Up to 2× perf/W vs Ironwood.
  - 4th-generation liquid cooling.
  - Software: JAX, MaxText, PyTorch native (preview, including eager mode), vLLM and SGLang. Pallas/Mosaic kernels for SparseCore and CAE. Bare-metal access.

**Availability:**
- Google [G5] says GA is "later this year" (2026), as part of AI Hypercomputer.
- As of 2026-10-23, the Cloud TPU release notes have **no TPU 8t/8i entries** [D-rn]. The latest entry is dated 2026-06-01.
- No `docs.cloud.google.com/tpu/docs/tpu8t` or `tpu8i` page exists; both return HTTP 404 (checked 2026-10-23).
- The cloud.google.com/tpu product page reportedly lists both chips as "Coming soon" (search snippets). The page is JS-rendered and I could not fetch it.
- **Status: announced, not GA. No public docs pages.**

**UNVERIFIED or secondary-only claims:**
- **Process node.** TheNextWeb [S5] and NAND Research report TSMC 2nm. A Tom's Hardware snippet [S6] says the TSMC N3 family. **CONFLICT, UNVERIFIED.**
- **Design partners.** Broadcom for 8t, MediaTek for 8i [S1 DCD, S5]. Google does not name them. **UNVERIFIED.**
- **Codenames.** "Sunfish" (8t) and "Zebrafish" (8i) [S5]. **UNVERIFIED.**
- **Packaging.** 8t: 2 compute dies, 1 I/O chiplet, 8× 12-high HBM3e. 8i: "one compute die", 1 I/O die, 6 HBM3e stacks [S5]. **UNVERIFIED.** This **CONFLICTS** with [G6], which says 8i's TensorCores sit on "core dies" (plural).
- **GA timing.** TheNextWeb [S5] and NAND Research say "late 2027". **CONFLICT** with Google's "later this year" [G5].
- **8i pod FP8 figure.** About 11.6 FP8 EF per pod (third-party aggregation). **UNVERIFIED.**

Secondary URLs, all 2026-04-22/23:
- [S1] DCD https://www.datacenterdynamics.com/en/news/google-unveils-eighth-generation-tpus-two-dedicated-training-and-inference-chips/. Returns 403 to automated fetches; title seen in search.
- [S4] TechCrunch https://techcrunch.com/2026/04/22/google-cloud-next-new-tpu-ai-chips-compete-with-nvidia/
- [S5] TheNextWeb https://thenextweb.com/news/google-ironwood-tpu-inference-cloud-next
- [S6] Tom's Hardware https://www.tomshardware.com/tech-industry/semiconductors/google-splits-its-tpu-into-two-chips-for-the-first-time-with-training-and-inference-variants. Body not fetchable; specs seen only in the search snippet.

---

## 3. Per-source summaries

### Papers

**[A1] Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit," ISCA 2017.**
- arXiv: https://arxiv.org/abs/1704.04760
- ACM DOI: https://doi.org/10.1145/3079856.3080246. ACM returns 403 to curl; the DOI resolves.

Key points:
- **§2 Architecture:**
  - A PCIe Gen3 x16 coprocessor using CISC instructions: Read_Host_Memory, Read_Weights, MatrixMultiply/Convolve, Activate, Write_Host_Memory.
  - 256×256 8-bit MAC systolic array (65,536 MACs).
  - 4 MiB of 32-bit accumulators (4096 × 256-element).
  - 24 MiB Unified Buffer and a Weight FIFO 4 tiles deep, fed from 8 GiB of DDR3 Weight Memory.
  - Die floor plan: 37% buffers, 30% compute, 10% I/O.
- **Table 2:** 28 nm, 700 MHz, 75 W TDP, 92 TOPS, 34 GB/s.
- **§4 Rooflines:** the ridge point is 1350 ops per weight byte. Most applications are memory-bound.
- **Results:** 15–30× faster than K80/Haswell, with 30–80× better TOPS/W.
- **§7:** swapping in GDDR5 would roughly triple performance.

**[A2] Jouppi et al., "A Domain-Specific Supercomputer for Training Deep Neural Networks," CACM 63(7), July 2020.**
- https://doi.org/10.1145/3360307. Redirects to dl.acm.org, which returns 403 to bots.
- Article page: https://cacm.acm.org/research/a-domain-specific-supercomputer-for-training-deep-neural-networks (403 to bots).
- Copy: https://www.gwern.net/doc/ai/scaling/hardware/2020-jouppi.pdf

Key points:
- Covers TPUv2/v3 supercomputers.
- 2D torus with no separate switches: ICI routers on chip.
- v2 has 4 × 496 Gb/s links; a 16×16 torus gives 15.9 Tb/s bisection.
- VLIW core with a 322-bit bundle.
- bf16 multiplies with fp32 accumulation.
- XLA compiler.
- **Table 3** compares v1, v2, v3 and Volta (numbers in the table above).
- v3 doubles the MXUs per core and is liquid cooled, allowing 1.6× power. v3 pods reach 1024 chips.
- Claims 50× faster training than general-purpose supercomputers on the stated benchmarks.

**[A3] Norrie et al., "The Design Process for Google's Training Chips: TPUv2 and TPUv3," IEEE Micro 41(2), Mar/Apr 2021.**
- https://doi.org/10.1109/MM.2021.3058217 → https://ieeexplore.ieee.org/document/9351692/
- Accepted-manuscript copy: https://www.gwern.net/doc/ai/scaling/hardware/2021-norrie.pdf

Key points:
- **Derivation from v1:** merges the buffers into a single vector memory and adds a programmable vector unit. The MXU becomes a coprocessor of the vector unit. DDR3 is replaced with HBM (20× bandwidth). ICI is added.
- **Scalar unit:** fetches 322-bit VLIW bundles with 2 scalar slots, 4 vector slots (2 for loads and stores), 2 matrix push slots, 1 matrix pop, 1 misc and 6 immediates.
- **Vector unit:** 128 lanes × 8 sublanes. Each sublane has a dual-issue 32-bit ALU and a 32-deep register file.
- **MXU:** 128×128, giving 32,768 ops per cycle. The RHS is preloaded (weight-stationary) and can be transposed on load. Results go to the Result FIFO.
- A transpose/reduce/permute unit is included.
- **Memory:** HBM is split between the 2 cores. ICI DMAs are push-only.
- PCIe is 16 GB/s per chip, vs 700 GB/s HBM and 4× 60 GB/s ICI.
- **v3 changes:** 2× MXUs; 700 → 940 MHz; +30% HBM bandwidth; 2× HBM capacity; ICI 650 Gbps per link; 1024-chip pods. Both v2 and v3 are 16 nm.

**[A4] Jouppi et al., "TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings," ISCA 2023 (Industry Track).**
- arXiv: https://arxiv.org/abs/2304.01433
- DOI: https://doi.org/10.1145/3579371.3589350

Key points:
- **§2.1–2.2 OCS:**
  - "Palomar" 3D-MEMS OCS that switches in milliseconds.
  - 136×136 ports (128 usable + 8 spares).
  - Circulators carry light both ways on one fiber, halving ports and cables.
  - Building block is a 4×4×4 cube = 64 chips = 1 rack with 16 hosts.
  - Each of a cube's 6 faces has 16 optical links (96 per block); opposing faces share an OCS, giving 48 OCSes per block.
  - 48 OCSes × 64 cubes = 4096 chips.
  - OCS is <5% of system cost and <3% of system power.
- **§2.2 chip:**
  - 2 TCs, each with 4 × 128×128 MXUs.
  - VPU with 128 lanes × 16 ALUs.
  - 16 MiB VMEM per TC; the TCs share a 128 MiB CMEM.
  - 4 chips per tray (2×2 mesh on the PCB) with 16 external ICI links.
- **§2.4–2.8:** availability and goodput, scheduling, modularity and security, and twisted tori (Fig. 5). Twisted shapes include 4×4×8 and 4×8×8. On Fig. 4, about 28% of large slices use twisted tori.
- **§3 SparseCore:**
  - Embeddings and their all-to-all traffic.
  - Bisection bandwidth scales as N^½ on a 2D torus vs N^⅔ on a 3D torus.
  - Each SC has 16 tiles. Each tile has an HBM channel, a Fetch Unit, an 8-wide SIMD scVPU and a Flush Unit, plus a 2.5 MiB slice of Spmem. There are 5 Cross-Channel Units.
  - SC is about 5% of die area and power.
  - Embedding-heavy models run 5–7× faster than with embeddings in CPU memory.
  - DLRM0 on v4 is 3.1× v3 and 30.1× CPUs.
- **§4:** ML-driven topology and partitioning search. Table 3 shows an LLM speedup of 2.3× from picking a better topology.
- **§5:** v4 vs v3 is 2.1× faster with 2.7× perf/W. CMEM contributes 1.2× overall and 2× on RNN1.
- **§6:** MLPerf vs A100 (1.2–1.7× faster, 1.3–1.9× less power) and vs the Graphcore IPU.
- **Table 4** is the v3 vs v4 feature table.

**[A5] Jouppi et al., "Ten Lessons From Three Generations Shaped Google's TPUv4i," ISCA 2021.**
- Copy: https://www.cs.cmu.edu/~18742/papers/Jouppi2021.pdf
- **Table 1** compares v1, v2, v3, v4i and T4: dates, clocks, TDP, die size, transistor count, memory and MXU configuration.
- **Table 2:** energy per operation at 45 nm vs 7 nm. Useful for teaching memory vs compute energy.
- **TPUv4i:** single TC with 4 MXUs, 128 MiB CMEM, 8 GB HBM, air cooled, 175 W.
- The ten lessons include: compiler compatibility over binary compatibility, TCO over CapEx, multi-tenancy, DNN demand growing about 1.5× per year, and latency mattering more than batch size.

**[A6] Jouppi, Patterson, Lakshmanamurthy, Young, "Google's Training Supercomputers from TPU v2 to Ironwood: Architectural Stability, Scale, Resilience, Power Efficiency, and Sustainability Across Five Generations," IEEE Micro 2026.**
- https://research.google/pubs/googles-training-supercomputers-from-tpu-v2-to-ironwood-architectural-stability-scale-resilience-power-efficiency-and-sustainability-across-five-generation/
- IEEE: https://ieeexplore.ieee.org/document/11550052
- Abstract only; I did not read the full text.
- Over eight years: 10× HBM capacity and bandwidth per node, 100× peak node performance, 3600× supercomputer performance and 30× perf/W.
- Also covers OCS plus built-in self-test for resilience, and embodied carbon.
- **Recommended:** obtain the full text. It is likely the best single primary source spanning v2 to Ironwood.

**[A7] Liu et al., "Lightwave Fabrics: At-Scale Optical Circuit Switching for Datacenter and Machine Learning Systems," SIGCOMM 2023, pp. 499–515.**
- DOI: https://doi.org/10.1145/3603269.3604836 (dl.acm.org, 403 to bots)
- Copy: https://cse.hkust.edu.hk/~kaichen/courses/spring24/comp7215/papers/lightwave-sigcomm23.pdf
- In-house OCS and transceivers, WDM plus circulators for bidirectional single-fiber links.
- For the 4096-chip TPU v4 superpod: up to 3× availability and up to 3.3× model-dependent performance vs a static fabric, at <6% of system cost.
- **CONFLICT (mild):** A4 says <5% of cost. Different accounting.

**[A8] Poutievski et al. (Google), "Mission Apollo: Landing Optical Circuit Switching at Datacenter Scale," arXiv 2208.10041.**
- https://arxiv.org/abs/2208.10041
- Background on the Palomar OCS hardware and its use in Jupiter DCN.

**[A9] Zu et al., "Resiliency at Scale: Managing Google's TPUv4 Machine Learning Supercomputer," NSDI 2024, pp. 761–774.**
- https://www.usenix.org/conference/nsdi24/presentation/zu
- PDF: https://www.usenix.org/system/files/nsdi24-zu.pdf
- Covers the software side: failure detection, OCS reconfiguration around failed hosts, chips and links, and repair workflows.
- Reports 99.98% system availability; about 1% of jobs see hardware outages.

### Hot Chips talks

**[C1] Norrie & Patil, "Google's Training Chips Revealed: TPUv2 and TPUv3," Hot Chips 32 (2020).**
- https://hc32.hotchips.org/assets/program/conference/day2/HotChips2020_ML_Training_Google_Norrie_Patil.v01.pdf
- Slide form of A3: VLIW, the 322-bit bundle, "Why 128x128?" (256 vs 128 vs 64 utilization trade-off), and the v3 vs v2 MXU memory-bound analysis.

**[C2] Jouppi & Swing, "A Machine Learning Supercomputer With An Optically Reconfigurable Interconnect and Embeddings Support" (TPU v4), Hot Chips 2023.**
- https://hc2023.hotchips.org/assets/program/conference/day2/ML%20training/HC2023.Session5.ML_Training.Google.Norm_Jouppi.Andy_Swing.Final_2023-08-25.pdf
- Calls the SparseCore "3rd-generation".
- "Typical power ~200W".
- 4 chips per board with parallel water flow (liquid cooled).
- "Distributed shared memory across up to 8K TensorCores and 16K SparseCores".
- Perf/W gains from CMEM.

**[S3] Ironwood at Hot Chips 2025 (ServeTheHome, Ryan Smith, 2025-08-26).** No official slide PDF was found.
- https://www.servethehome.com/google-ironwood-tpu-swings-for-reasoning-model-leadership-at-hot-chips-2025/
- First multi-compute-die TPU (2 dies).
- 8 HBM3e stacks: 192 GB at 7.3 TB/s.
- 4th-gen SparseCore, which also accelerates collectives.
- Trays of 4 chips; racks of 16 trays (64 TPUs).
- Copper inside a rack, OCS between racks.
- 9,216 chips per superpod, chosen to leave spare racks (vs 8,192).
- 3rd-gen liquid cooling.
- RAS features: root of trust, BIST, silent-data-corruption detection, in-flight arithmetic checking.
- Power smoothing.
- About 6× perf/W vs TPU v4.

### Google Cloud docs (all [P], "last updated 2026-10-22")

- **[D-arch] System architecture:** https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm
  - Definitions: TensorCore (MXUs + vector unit + scalar unit), slice, pod, topology AxB / AxBxC, cube (4×4×4, from v4), ICI, DCN, OCS, twisted torus.
  - MXU is **256×256 on v6e and TPU7x, 128×128 on earlier versions**, with fp32 accumulation.
  - SparseCores: v5p 4, v6e 2, TPU7x 4.
  - int8 on v5e/v6e; FP8 on v5p/TPU7x. Note that v5p's FP8 figure equals its BF16 figure, so v5p FP8 support is likely emulated or not faster.
  - ICI resiliency is on by default for v4, v5p and 7x.
  - Does not cover VMEM or CMEM.
- **[D-7x] TPU7x:** https://docs.cloud.google.com/tpu/docs/tpu7x
  - Comparison table of v5p, v6e and 7x.
  - 2 chiplets per chip, each with 1 TC, 2 SC and 96 GiB HBM.
  - The die-to-die link is 6× faster than a 1D ICI link.
  - **JAX sees 2 devices per chip.**
  - Topologies from 2×2×1 to 8×16×16; any multiple of 4 up to 9216 chips.
  - VM: 4 chips, 224 vCPUs, 960 GB RAM.
- **[D-v6e]** https://docs.cloud.google.com/tpu/docs/v6e. Values in table 1. Pod bf16 234.9 PF; all-reduce bandwidth 102.4 TB/s; bisection 3.2 TB/s.
- **[D-v5p]** https://docs.cloud.google.com/tpu/docs/v5p
- **[D-v5e]** https://docs.cloud.google.com/tpu/docs/v5e
- **[D-v4]** https://docs.cloud.google.com/tpu/docs/v4. Includes twisted-torus shapes: about +70% bisection for 4×4×8 and +40% for 4×8×8.
- **[D-v3]** https://docs.cloud.google.com/tpu/docs/v3
- **[D-v2]** https://docs.cloud.google.com/tpu/docs/v2. Points to A2 for architecture.
- **[D-rn] Release notes:** https://docs.cloud.google.com/tpu/docs/release-notes. v6e GA 2024-12-16; TPU7x Preview 2025-11-24; TPU7x GA 2026-03-31; nothing on TPU 8.
- GKE Ironwood concept page: https://docs.cloud.google.com/kubernetes-engine/docs/concepts/tpu-ironwood

### Google launch blogs ([P])

- **[G1] TPU v5e:**
  - Launch: https://cloud.google.com/blog/products/compute/announcing-cloud-tpu-v5e-and-a3-gpus-in-ga (Aug 2023)
  - GA: https://cloud.google.com/blog/products/compute/announcing-cloud-tpu-v5e-in-ga
  - Claims up to 2× training and 2.5× inference perf/$ vs v4.
- **[G2] TPU v5p + AI Hypercomputer**, 2023-12-06 (Vahdat & Lohmeyer). https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer
  - 8,960 chips; ICI 4,800 Gbps per chip; 3D torus.
  - More than 2× FLOPS and 3× HBM vs v4.
  - 2.8× faster LLM training.
  - "Second-generation SparseCores", with 1.9× faster embedding-dense training.
- **[G3] Trillium announcement**, 2024-05-14 (Vahdat). https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus
  - 4.7× peak per chip vs v5e, from a larger MXU and a faster clock.
  - 2× HBM and 2× ICI.
  - Third-generation SparseCore.
  - 256-chip pods.
  - >67% better energy efficiency.
  - GA post: https://cloud.google.com/blog/products/compute/trillium-tpu-is-ga
- **[G4] Ironwood announcement**, 2025-04-09, updated 2025-04-23 (Vahdat). https://blog.google/products/google-cloud/ironwood-tpu-age-of-inference/
  - "First TPU for the age of inference".
  - 4,614 TFLOPS per chip (FP8, per figure captions).
  - 192 GB HBM (6× Trillium) at 7.37 TB/s (4.5×).
  - ICI 1.2 TBps bidirectional (1.5×).
  - Pods of 256 or 9,216 chips; 42.5 EF per 9,216 chips; about 10 MW.
  - 2× perf/W vs Trillium.
  - Enhanced SparseCore; liquid cooled.
- **[G7] Ironwood + Axion GA announcement**, 2025-11-06 (Vahdat & Lohmeyer). https://cloud.google.com/blog/products/compute/ironwood-tpus-and-new-axion-based-vms-for-your-ai-workloads
  - "GA in coming weeks".
  - 10× peak vs v5p; >4× per chip vs v6e.
  - 9,216-chip superpod over 9.6 Tb/s ICI with 1.77 PB shared HBM.
  - Jupiter DCN to hundreds of thousands of TPUs.
  - OCS for routing around failures.
- **[G5], [G6]:** TPU 8 posts (see §2).

### Other learning material

- **[B1] Austin et al. (Google DeepMind), "How to Scale Your Model," Part 2: "How to Think About TPUs," published 2025-02-04.** https://jax-ml.github.io/scaling-book/tpus/
  - Spec and ICI tables for v3 through TPU7x.
  - TPU7x: 2 TC × 2 MXU (256×256), about 2.2 GHz, VMEM 128 MiB per chip, VMEM bandwidth about 7× HBM.
  - VPU is 8×128 ALUs × 4 issue.
  - One scalar core per TC, issuing one DMA per cycle.
  - Systolic array appendix.
  - Some numbers differ from docs (see conflicts). Treat it as semi-primary: written by Google authors but not official specs.
- **[H1] Henry Ko, "TPU Deep Dive," 2025-06-18.** https://henryhmko.github.io/posts/tpu/tpu.html. Blog home: https://henryhmko.github.io
  - Structure: Background (2006 ASIC evaluation, 2013 voice-search projection) → single chip (v4 focus) → design philosophy (systolic arrays and pipelining; AOT compilation with scratchpads instead of caches, XLA and JAX jit recompilation on shape change; energy, citing Horowitz 45 nm and v4i tables) → tray (4 chips) → rack/cube (4×4×4) → pod (4096 for v4, 9216 for Ironwood; slice shapes such as 8×8×8 vs 4×4×32 for all-to-all vs pipeline) → OCS benefits (wraparound, non-contiguous "bag of nodes" slices, twisted tori) → twisted-torus scenarios for decoder training → multislice over DCN (PaLM on 6144 v4 chips, hierarchical collectives) → photos.
  - Errors to watch:
    - Calls Trillium "TPUv6p". It is v6e.
    - Says "VMEM 32MiB" per TC. A4 gives 16 MiB per TC, 32 MiB per chip.
    - Lists "SMEM 10MiB". That number matches v4 spMEM total (A4 Tbl 4), not scalar memory.
    - Says "6144 TPUv4s (2 pods)". 6144 is 1.5 v4 pods; PaLM used 2 pods of 3072 each.
    - Cites v5p at "500 TFLOPS". Docs give 459.
  - A good narrative primer, but check its numbers against the sources above.

---

## 4. Conflicts register

- **C-1. TPUv2/v3 process node.**
  - A2 says ">12nm"; A3 and A5 say 16 nm.
  - Die size: A2 gives v2 <611 and v3 <648 mm²; A5 gives v2 <625 and v3 <700; A4 gives v3 <700 mm².
- **C-2. TPUv3 power.**
  - TDP is 450 W [A2, A5].
  - Measured mean is 220 W [A4, D-v3].
  - Not a true conflict, but don't mix TDP and measured figures.
- **C-3. TPUv3 chips per host.** 8 per [A2, A4]; 4 per [A5].
- **C-4. ICI link rates.**
  - v2: 496 Gb/s [A2] vs "500 Gb/s" and "60 GB/s" [A3].
  - v3: 656 Gb/s [A2] vs 650 Gbps [A3] vs "70 GB/s" [A4].
  - v4: 50 GB/s per link [A4] vs 4.5e10 B/s one-way [B1].
  - TPU7x: 200 GBps bidirectional per link [D-7x] vs 1.8e11 B/s [B1].
  - The v5p blog's "4,800 Gbps/chip" [G2] is 600 GB/s, half the docs' 1200 GBps bidirectional. It is probably the one-direction figure.
- **C-5. SparseCore generation numbering.**
  - A4 says SC has been in TPUs since v2. v3 has 2 and v4 has 4 per chip [A4 Tbl 4]; the v2 count is not stated.
  - Hot Chips 2023 [C2] calls v4's SC "3rd-generation".
  - The v5p blog [G2] says "second-generation"; the Trillium blog [G3] says "third-generation"; Ironwood is "4th-gen" [S3].
  - So the research papers and Cloud marketing count generations differently. Pick one convention for the course.
- **C-6. TPUv3 bf16 peak.** 123 TFLOPS [A2, A4, D-v3] vs 1.4e14 [B1]. 1.4e14 also disagrees with v3's int8 listing in B1.
- **C-7. Ironwood HBM units.**
  - 192 GiB [D-7x, B1] vs "192 GB" [G4, S3].
  - Bandwidth: 7380 GBps [D-7x], 7.37 TB/s [G4], 7.3 TB/s [S3], 7.4e12 [B1].
- **C-8. TPU 8i pod size.**
  - "Up to 1,152" chips [G6 Boardfly section; G5].
  - "Up to 1,024 active chips" [G6 pod description].
  - 36 groups × 8 boards × 4 chips = 1,152 physical. My reading is that 1,024 are active and the rest are spares, like Ironwood's 9,216 vs 8,192.
- **C-9. TPU 8 process node and GA date.** 2nm [S5] vs N3 [S6]. GA "later this year" (2026) [G5] vs "late 2027" [S5].
- **C-10. TPU 8i die count.** "One compute die" [S5] vs TensorCores "on core dies" [G6].
- **C-11. v4 power.** "Typical ~200W" [C2] vs a measured mean of 170 W [A4].
- **C-12. OCS cost share.** <5% [A4] vs <6% [A7].
- **C-13. v5e HBM bandwidth units.** "800 GiBps" [D-v5e] vs 8.2e11 B/s [B1]. The v6e docs vary between GBps and GiBps across language versions.
- **C-14. Ironwood MXU arithmetic.**
  - B1 says one bf16[8,256]@[256,256] every 4 cycles, about 5.8e14 FLOPs/s per MXU at 2.2 GHz.
  - That works out to 131,072 MACs per cycle on a 65,536-cell array, i.e. 2 MACs per cell per cycle.
  - This is unexplained in public sources. **Treat the clock and per-cell rate as UNVERIFIED.** Only the chip total of 2307 TFLOPS is official.

## 5. Gaps (not found in any public primary source)

- Clock frequency, process node and die size for v5e and later. B1's Ironwood clock is the only Google-authored number.
- VMEM sizes for v5e, v5p and v6e. Ironwood's 128 MiB is from B1 only; TPU 8's 128 MB / 384 MB are from G6.
- Whether CMEM exists after v4. It is not documented for v5 and later.
- TPU 8: BF16 and FP8 peaks, MXU count and size, TensorCores per 8t chip, absolute ICI bandwidth for 8t, SparseCore count for 8t, details of the "LLM Decoder Engine", TDP.
- v5p int8 (918 TOPS) appears only in secondary and semi-primary sources [B1].
- A full-text read of A6 (IEEE Micro 2026) is still needed and may fill several of these gaps.
