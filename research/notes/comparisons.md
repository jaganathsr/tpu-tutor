# Comparisons: TPU vs AWS Trainium vs NVIDIA GPU (research notes)

All sources accessed 2026-10-23 unless noted. All URLs were fetched and loaded on that date unless marked UNVERIFIED.
Conventions:
- "dense" means no structured sparsity. NVIDIA spec pages mostly quote **sparse** numbers. Dense is half of sparse unless noted.
- "derived" means I computed it from a primary number (rack total / GPU count, etc.). It is not a vendor-quoted figure.
- Interconnect BW: NVIDIA's "NVLink bandwidth" per GPU is the **bidirectional aggregate** (e.g. H100 = 900 GB/s = 18 links x 50 GB/s bidir). Google's ICI per-chip figure is also quoted as bidirectional. AWS NeuronLink per-chip figures do not state direction.

---

## 1. Spec tables (source for every number)

### 1a. NVIDIA (per GPU unless stated)

| Part | HBM | HBM BW | Dense BF16 | Dense FP8 | NVLink BW / GPU | NVLink domain | Source |
|---|---|---|---|---|---|---|---|
| H100 SXM | 80 GB HBM3 | 3.35 TB/s | 989 TFLOPS (page lists 1,979 sparse) | 1,979 TFLOPS (page lists 3,958 sparse) | 900 GB/s (NVLink 4) | 8 (HGX baseboard) | [N1], [N7] |
| H200 SXM | 141 GB HBM3e | 4.8 TB/s | 989 (1,979 sparse) | 1,979 (3,958 sparse) | 900 GB/s | 8 (HGX) | [N2] |
| B200 (HGX/DGX B200) | 180 GB (derived: 1,440 GB / 8) | 8 TB/s (derived: 64 TB/s / 8) | 2.25 PFLOPS (derived: 36 PF sparse / 8 / 2) | 4.5 PFLOPS (derived: 72 PF sparse / 8 / 2) | 1.8 TB/s (NVLink 5) | 8 | [N3], [N4] |
| GB200 (in NVL72) | ~186 GB (derived: 13.4 TB / 72) | 8 TB/s (derived: 576 TB/s / 72) | 2.5 PFLOPS (derived: 360 PF sparse / 72 / 2) | 5 PFLOPS (derived: 720 PF sparse / 72 / 2) | 1.8 TB/s | 72 (NVL72; 130 TB/s total NVLink) | [N5] |
| GB300 / Blackwell Ultra (in NVL72) | ~278 GB (derived: 20 TB / 72) | 8 TB/s (derived: 576 TB/s / 72) | 2.5 PFLOPS (derived: 360 PF sparse / 72 / 2) | 5 PFLOPS (derived: 720 PF sparse / 72 / 2) | 1.8 TB/s (derived: 130 TB/s / 72) | 72 | [N6] |
| Rubin GPU (in Vera Rubin NVL72) | 288 GB HBM4 | 19.2 TB/s (page) | 4 PFLOPS (page, "dense") | 17.5 PFLOPS FP8/FP6 training (page, "dense") | 3 TB/s (page; 216 TB/s per rack) | 72 | [N8] |

Rack-level, NVIDIA-quoted:
- GB200 NVL72: FP4 1,440 PF sparse / 720 dense; FP8 720 PF (sparse); BF16 360 PF (sparse); 13.4 TB HBM3e; 576 TB/s; NVLink 130 TB/s; 72 GPUs + 36 Grace. [N5]
- GB300 NVL72: FP4 1,440 PF sparse / 1,080 PF dense; FP8 720 PF; BF16 360 PF; 20 TB HBM; up to 576 TB/s; NVLink 130 TB/s; 37 TB "fast memory". [N6] (The fetched page text read "10802" for dense FP4. I read it as 1080 + footnote 2. Check the datasheet.)
- HGX B300: FP4 144 PF sparse / 108 dense; FP8 72 PF (sparse); BF16 36 PF (sparse); 2.1 TB total; 1.8 TB/s per GPU; 14.4 TB/s total NVLink. [N4]
- Vera Rubin NVL72: NVFP4 inference 3,600 PF (sparse); NVFP4 training 2,520 PF (dense); FP8 1,260 PF (dense); BF16 288 PF (dense); HBM4 20.7 TB at 1,400 TB/s; NVLink 6 216 TB/s; header says "Available Now". [N8]

Caveats:
- Rubin per-GPU HBM BW and NVLink differ across sources. At CES 2026 NVIDIA and the press quoted 22 TB/s and 3.6 TB/s NVLink 6 per GPU (260 TB/s per rack) ([Tom's Hardware CES 2026](https://www.tomshardware.com/pc-components/gpus/nvidia-launches-vera-rubin-nvl72-ai-supercomputer-at-ces-promises-up-to-5x-greater-inference-performance-and-10x-lower-cost-per-token-than-blackwell-coming-2h-2026)). The nvidia.com product page fetched 2026-10-23 says 19.2 TB/s and 3 TB/s. Use the product page and footnote the CES numbers.
- Rubin dense BF16 per GPU (4 PF) is 1.6x Blackwell's derived 2.5 PF, but its dense FP8 "training" figure (17.5 PF) is 3.5x GB300's derived 5 PF. Most of the gain is in low precision. Treat it as vendor-claimed until independently benchmarked.
- The earlier "Vera Rubin NVL144" naming (GTC 2025, counting dies) was replaced by "Vera Rubin NVL72" (counting packages) on the current product page. This is UNVERIFIED as an official rename statement; it is inferred from the page naming.

### 1b. AWS Trainium (per chip unless stated)

| Part | HBM | HBM BW | Dense BF16 | Dense FP8 / MXFP8 | Chip-to-chip | Scale-up domain | Source |
|---|---|---|---|---|---|---|---|
| Trainium2 (8x NeuronCore-v3) | 96 GiB (4 stacks) | 2.9 TB/s (arch page); "3 TB/s" (NKI guide) | 667 TFLOPS | 1,299 TFLOPS FP8 | NeuronLink-v3, 1.28 TB/s per chip (chip page); 1,024 GiB/s intra-instance + 256 GiB/s inter-instance (instance page) | 16 chips/instance (4x4 2D torus); 64 chips/UltraServer (4 instances, same-coordinate chips in rings) | [A1], [A2], [A3] |
| Trainium3 (8x NeuronCore-v4) | 144 GiB HBM3e | 4.9 TB/s (chip page, product page); 4.7 TB/s (NKI guide) | 671 TFLOPS | 2,517 TFLOPS MXFP8/MXFP4 (Neuron docs); 2.52 PFLOPS FP8 (AWS whats-new) | NeuronLink-v4, 2.56 TB/s per device (Neuron docs); "2 TB/s per chip" (product page) | Gen1 UltraServer 64 chips; Gen2 UltraServer 144 chips; NeuronSwitch-v1 all-to-all (replaces 2D torus) | [A4], [A5], [A6], [A7] |

Other Trainium per-chip specs:
- Trn2: sparse 2,563 TFLOPS (FP8/FP16/BF16/TF32); FP32 181 TFLOPS; SBUF 224 MiB (8 x 28 MiB); DMA 3.5 TB/s; CC-Cores: 16 on the chip page [A1] and 20 in the NKI guide [A2] (the docs conflict); up to 64-chip memory pool. [A1]
- Trn3: FP32 183 TFLOPS; sparse BF16/FP16/TF32 2,517; SBUF 256 MiB (8 x 32 MiB); DMA 4.9 TB/s; CC-Cores: 16 on the chip page and 20 in the NKI guide; stochastic rounding; 3 nm. [A4], [A5], [A6]
- Trn2 UltraServer (64 chips): 83.2 PF FP8 dense, 164 PF sparse, 42.8 PF BF16, 6,144 GiB HBM, 185.6 TB/s. [A3] (The Register labels 83.2 PF as *sparse*. The Neuron docs table lists it as dense, with sparse = 164. Follow the docs.)
- Trn3 Gen2 UltraServer (144 chips): 362,448 MXFP8 TFLOPS, 20,736 GiB HBM; Gen1 (64): 161,088 TFLOPS, 9,216 GiB. [A7] AWS marketing: "up to 362 FP8 PFLOPs", 20.7 TB, 706 TB/s aggregate. [A5]

Per-NeuronCore engine data (from the NKI architecture guides):

| | NeuronCore-v3 (Trn2) [A2] | NeuronCore-v4 (Trn3) [A6] |
|---|---|---|
| Tensor Engine | 128x128 PE systolic array, 2.4 GHz; 79 TF BF16, 158 TF FP8 dense; FP8 doubles the contraction dim to 256 | 2.4 GHz; ~315 TF MXFP8/MXFP4 (4x BF16; contraction dim effectively 512); 79 TF BF16; writes BF16 to PSUM with FP32 internal accumulation |
| Vector Engine | 0.96 GHz, 512 elem/cycle BF16 | 1.2 GHz, 512 elem/cycle BF16/FP16/FP8; MX quantize; fast exp |
| Scalar Engine | 1.2 GHz, 128 elem/cycle (activations: GeLU, sqrt) | 1.2 GHz, 256 elem/cycle BF16 |
| GpSimd | 8 programmable processors, 1.2 GHz, integrated DMA | 1.2 GHz, 128 elem/cycle |
| SBUF | 28 MiB = 128 partitions x 224 KiB | 32 MiB |
| PSUM | 2 MiB | 2 MiB |
| DMA | 16 engines/core; HW descriptor generation (DGE) | + SBUF read-add-write, traffic shaping (4 classes), indirect gather/scatter |

### 1c. Google TPU, for the same tables (per chip)

| Part | HBM | HBM BW | Dense BF16 | Dense FP8 | ICI / chip | Scale-up domain | On-chip SRAM | Source |
|---|---|---|---|---|---|---|---|---|
| TPU7x Ironwood | 192 GiB | 7,380 GB/s | 2,307 TFLOPS | 4,614 TFLOPS | 1,200 GB/s bidirectional | 9,216 chips/pod (144 cubes of 4x4x4), 3D torus + OCS | VMEM 64 MiB per TC x 2 TC [G4] (scaling book: 128 MiB/chip) | [G3], [G4], [G5] |
| TPU 8t | 216 GB | 6,528 GB/s | not published | not published (FP4 12.6 PF) | "2x" Ironwood ICI; 19.2 Tb/s per Google blog [G2] | 9,600-chip superpod, 3D torus | 128 MB "Vmem" (1 TC) | [G1], [G2], [G4] |
| TPU 8i | 288 GB | 8,601 GB/s | not published | not published (FP4 10.1 PF) | 19.2 Tb/s (= 2.4 TB/s, 2x Ironwood) | Boardfly: 4-chip ring boards, 8 boards/group fully connected, 36 groups via OCS = 1,152 chips (article also says "up to 1,024 active") | 384 MB (2 TC x 192 MiB) | [G1], [G2], [G4] |

Note: the PyTorch TorchTPU blog [G6] writes the ICI figure as "19.2 TBps". Google's own blog [G2] says Tb/s (bits). The PyTorch blog's unit is a typo.

---

## 2. Per-source summaries (with section pointers)

### NVIDIA
- **[N1] H100 product page**, https://www.nvidia.com/en-us/data-center/h100/ , "Specifications" table. SXM: FP8 3,958 / BF16 1,979 / TF32 989 TFLOPS, all with "* With sparsity"; 80 GB, 3.35 TB/s; NVLink 900 GB/s; 700 W. H100 NVL: 94 GB, 3.9 TB/s, NVLink 600 GB/s.
- **[N7] NVIDIA H100 Tensor Core GPU Architecture whitepaper v1.02 (GTC22)**. The official copy is registration-gated (https://www.nvidia.com/en-us/data-center/resources/hopper-architecture-whitepaper/ ; UNVERIFIED that this URL is ungated). I read a mirror: https://www.ece.lsu.edu/koppel/gp/refs/gtc22-whitepaper-hopper.pdf .
  - "NVIDIA H100 GPU Architecture In-Depth": full GH100 has 144 SMs. H100 SXM5 has 132 SMs, 128 FP32 cores/SM, **4 fourth-gen Tensor Cores per SM (528 per GPU)**, 50 MB L2, 80 GB HBM3 over 5 stacks.
  - "H100 SM Key Feature Summary": **256 KB combined shared memory + L1 per SM**; Tensor Cores give 2x A100 per-SM MMA rate on equal dtypes and 4x with FP8.
  - Table 1 (preliminary): BF16 TC 1000 | 2000 sparse; FP8 2000 | 4000. Shipping numbers on [N1] are 989/1,979.
  - Sections "Distributed Shared Memory" (DSMEM: direct SM-to-SM loads/stores/atomics within a thread block cluster), "Asynchronous Execution", "Tensor Memory Accelerator (TMA)", and "Asynchronous Transaction Barrier". Also NVLink 4 at 900 GB/s and NVLink Switch with SHARP in-network reductions.
- **[N9] NVIDIA Hopper Tuning Guide**, https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
  - §1.4.1.1 Occupancy: 64 warps/SM, 64K 32-bit registers/SM, max 255 regs/thread, 32 blocks/SM; **228 KB shared memory/SM, 227 KB max per block**.
  - §1.4.1.2: TMA moves 1D to 5D tensors between global and shared memory and between SMs in a cluster.
  - §1.4.1.3 Thread Block Clusters: portable max size 8, 16 non-portable; DSMEM.
  - §1.4.2.2: L2 cache 50 MB (A100: 40 MB).
  - §1.4.2.4: unified L1/tex/shared memory of 256 KB per SM; carveouts from 0 to 228 KB.
- **[N2] H200 page**, https://www.nvidia.com/en-us/data-center/h200/ : 141 GB, 4.8 TB/s, same TC FLOPS as H100 (sparse footnote), 900 GB/s NVLink.
- **[N3] DGX B200 page**, https://www.nvidia.com/en-us/data-center/dgx-b200/ : 8 GPUs, 1,440 GB, 64 TB/s; FP4 144 sparse | 72 dense PF; FP8 72 PF (sparse); NVLink 14.4 TB/s aggregate.
- **[N4] HGX page**, https://www.nvidia.com/en-us/data-center/hgx/ : HGX B200 and B300 tables, with footnote "Sparse. Dense is 1/2 sparse". 1.8 TB/s GPU-to-GPU NVLink.
- **[N5] GB200 NVL72 page**, https://www.nvidia.com/en-us/data-center/gb200-nvl72/ : spec table with "sparse | dense" footnotes. "1.8 TB/s of GPU-to-GPU interconnect" (Highlights); NVLink Switch System 130 TB/s.
- **[N6] GB300 NVL72 page**, https://www.nvidia.com/en-us/data-center/gb300-nvl72/ : rack table (see 1a).
- **[N8] Vera Rubin NVL72 page**, https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/ : three-column spec table (rack / superchip / GPU). Footnotes: 2 = sparse, 3 = dense, 4 = bidirectional networking.
- **[N10] CUDA Programming Guide, §4.2 CUDA Graphs**, https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
  - The §4.2 intro says launch overhead "can be paid once for the entire graph during instantiation", after which the graph is "launched repeatedly with very little overhead".
  - §4.2.2: three stages, definition, instantiation, and execution. Graphs come from stream capture (`cudaStreamBeginCapture`) or explicit construction.
  - §4.2.3: "Major changes to graph structure (such as topology or node types) require re-instantiation". Whole-graph update requires identical topology.
  - §4.2.4: conditional nodes (IF/WHILE/SWITCH) are evaluated on the device.
  - **Teaching contrast**: a CUDA Graph records *already-compiled kernels* and their launch DAG. It removes CPU launch overhead but does no cross-kernel optimization such as fusion, layout, or scheduling. XLA compiles the *whole program* (HLO/StableHLO) into one executable, choosing fusion, layouts, memory assignment and collective scheduling. A TPU executable is closer to "CUDA Graph + torch.compile/Inductor + a global memory planner" in one step. Both need static topology. XLA additionally needs static shapes, apart from bounded dynamism, which is a work in progress [G6].
- **[N11] CUTLASS tcgen05 MMA Programming Guide (Blackwell)**, https://docs.nvidia.com/cutlass/4.6.0/media/docs/pythonDSL/mma_docs/tcgen05_programming.html (in search results; not separately fetched, so UNVERIFIED load). Per the search summary: 5th-gen TC MMA issued by a single thread; accumulators live in a dedicated **Tensor Memory (TMEM)**; 2-CTA pairs; 2 to 4x Hopper WGMMA throughput. This moves Blackwell toward the TPU/Trainium model of a dedicated accumulator store (compare Trainium PSUM and the TPU MXU result buffers).

### Parallelism / collectives literature (GPU side)
- **[P1] Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"**, arXiv:1909.08053 (v1 2019-09-17, v4 2020-03-13), https://arxiv.org/abs/1909.08053 . Intra-layer (tensor) parallelism added with a few communication ops in native PyTorch. The MLP is split column-wise then row-wise; attention is split by heads. That gives 2 all-reduces in forward and 2 in backward per layer (Section 3). They trained 8.3B params on 512 GPUs at 15.1 PF sustained, with 76% scaling efficiency vs a 39 TF single-GPU baseline (~30% of peak).
- **[P2] Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models"**, arXiv:2205.05198 (2022-05-10), https://arxiv.org/abs/2205.05198 . Sequence parallelism shards LayerNorm/dropout along the sequence dimension, so TP all-reduce becomes reduce-scatter + all-gather at the same comms volume. Selective activation recomputation is the second technique. Results: 5x activation memory reduction, >90% less recompute overhead, 530B GPT on 2,240 A100s at 54.2% MFU vs 42.1%. This is the GPU origin of the "AG/RS around matmul" pattern that maps directly onto TPU collective matmul.
- **[P3] NCCL-tests PERFORMANCE.md**, https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md . algbw = S/t. busbw corrects algbw by the collective's factor so it can be compared to link peak:
  - AllReduce: x 2(n-1)/n
  - ReduceScatter and AllGather: x (n-1)/n, where S is the *total* size
  - Broadcast: x 1
  - AllToAll: x (n-1)/n
  The same ring-cost math applies on an ICI ring axis. The difference is that a TPU torus axis is bidirectional with wraparound, so a ring collective can use both directions.
- **[P4] NCCL env docs (NCCL 2.32.3)**, https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html , sections NCCL_ALGO and NCCL_PROTO. Algorithms: Ring, Tree (2.5+), CollnetChain/CollnetDirect (2.14+), NVLS (2.17+), NVLSTree (2.18+), PAT (2.23+). Protocols: LL, LL128, Simple. NVLS = NVLink SHARP, i.e. in-switch reduction. TPUs have no switch to offload reductions to: ICI is direct chip-to-chip, so reductions run on chip (and on SparseCore offload, and on the TPU 8i CAE).
- **[P5] NVIDIA blog, "Massively Scale Your Deep Learning Training with NCCL 2.4"** (S. Jeaugey, 2019-02-04), https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4 . Ring latency "scales linearly with the number of GPUs". Double binary trees give full bandwidth with logarithmic latency, up to 180x lower latency at 24k GPUs (Summit).
- **[P6] PyTorch "Introducing Async Tensor Parallelism in PyTorch"** (Yifu Wang et al., 2024-09-12), https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487 . The method decomposes AG+matmul and matmul+RS into P2P steps plus sub-matmuls. It uses **SymmetricMemory** and copy-engine `cudaMemcpyAsync`, avoiding NCCL send/recv SM contention and two-sided sync, and alternates streams to hide wave quantization. Enable via `torch._inductor.config._micro_pipeline_tp = True` + `enable_symm_mem_for_group`, or eager `torch.ops.symm_mem.fused_all_gather_matmul` / `fused_matmul_reduce_scatter`. Llama3 8B: ~29% forward / ~8% end-to-end. 70B: ~20% / ~8%. Tested on 64 H100s. Intra-node (NVSwitch) only.
- **[P7] PyTorch SymmetricMemory dev-discuss**, https://dev-discuss.pytorch.org/t/pytorch-symmetricmemory-harnessing-nvlink-programmability-with-ease/2798 (search result; not separately fetched). It says decomposition-based async-TP implements the **LoopedCollectiveEinsum** algorithm, i.e. Google's Wang et al. paper [P10], and notes GPU scheduling overhead at small sizes. Its follow-up is a single-kernel CUTLASS variant.
- **[P8] Chang et al., "FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion"**, arXiv:2406.06858 (2024-06-11, v5 2024-10-23), https://arxiv.org/abs/2406.06858 . Fuses fine-grained comm and GEMM tiles into one kernel. Claims: up to 96% of communication overlapped; up to 1.24x training vs Megatron-LM on 128 GPUs; 1.66x prefill and 1.30x decode vs vLLM on 8 GPUs.
- **[P9] NeMo / Transformer Engine TP comm overlap**, https://docs.nvidia.com/nemo-framework/user-guide/26.02/nemotoolkit/features/optimizations/communication_overlap.html . "Bulk" overlap (independent comms) is on by default. "Pipelined" overlap chunks TP comm and GEMM and overlaps them. AG becomes "multiple steps of input P2P ring exchanges"; RS uses GEMM-output P2P ring exchange + reduction. Enabled with `tp_comm_overlap=True`, `tp_comm_overlap_cfg` (TE userbuffers). Used with `sequence_parallel=True`; Megatron-Bridge docs say TP overlap silently disables without SP ([Megatron-Bridge skill page](https://docs.nvidia.com/nemo/megatron-bridge/latest/skills/nemo-mbridge-perf-tp-dp-comm-overlap/SKILL.html), from search result).
- **[P10] (TPU side) Wang et al., "Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models"**, ASPLOS 2023 (vol. 1, pp. 93–106). Authors: Shibo Wang, Jinliang Wei, Amit Sabne, Andy Davis, Berkin Ilbeyi, Blake Hechtman, Dehao Chen, Karthik Srinivasa Murthy, Marcello Maggioni, Qiao Zhang, Sameer Kumar, Tongfei Guo, Yuanzhong Xu, Zongwei Zhou. Bibliographic data from DBLP (https://dblp1.uni-trier.de/pid/124/4315.html). The ACM DL page returned 403 and the DOI is UNVERIFIED. This is the XLA "collective matmul / looped collective einsum" paper that PyTorch async-TP cites as its algorithm.
- **[P11] JAX shard_map notebook**, https://docs.jax.dev/en/latest/notebooks/shard_map.html , "Toy examples → Matrix multiplies → Example 1: all-gather on one side" and "Example 2: psum_scatter the result". These are hand-written collective matmuls via `ppermute`. The unidirectional version "uses only half the interconnect bandwidth"; the bidirectional version is needed to saturate TPU ICI. Use this as the TPU-side code to set next to [P6]/[P9].

### AWS Trainium / Neuron
- **[A1] Trainium2 chip architecture**, https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/_sources/about-neuron/arch/neuron-hardware/trainium2.rst : spec list (see 1b).
- **[A2] Trainium2 Architecture Guide for NKI**, https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/_sources/nki/guides/architecture/trainium2_arch.rst . Sections: "NeuronCore-v3 Compute Engine Updates" (Tensor: double FP8 matmul, built-in transpose; Vector perf mode; Scalar; GpSimd) and "Data Movement Updates" (DMA transpose, DGE). SBUF 28 MiB = 128 partitions x 224 KiB.
- **[A3] Trn2 architecture (instances/UltraServer)**, https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/about-neuron/arch/neuron-hardware/trn2-arch.rst . "16 Trainium2 chips are connected using a 4x4, 2D Torus topology". UltraServer = 4 x trn2u.48xlarge, and "chips with the same coordinates in each Trn2 instance are connected in a ring topology". Intra-instance 1,024 GiB/s/chip; inter-instance 256 GiB/s/chip; EFAv3 3.2 Tbps/instance.
- **[A4] Trainium3 chip architecture**, https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trainium3.html (Neuron 2.33.0 banner). Specs plus a Trn2→Trn3 comparison table: FP8 2x, BF16 1x, HBM 1.5x, BW 1.7x, ICI 2x.
- **[A5] EC2 Trn3 product page**, https://aws.amazon.com/ec2/instance-types/trn3/ , and the What's New post https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-ec2-trn3-ultraservers/ (GA 2025-12-02). Up to 144 chips, 362 MXFP8 PF, 20.7 TB, 706 TB/s, NeuronSwitch-v1, NeuronLink-v4 "2 TB/s per chip"; vs Trn2 UltraServer: 4.4x perf, 3.9x mem BW, 4x perf/W.
- **[A6] Trainium3 Architecture Guide for NKI**, https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.1/nki/guides/architecture/trainium3_arch.html . Tensor Engine "Quad-MXFP8/MXFP4" (32-element scaling groups, 8-bit shared scale; `nisa.nc_matmul_mx`, `nisa.quantize_mx`), BF16 results in PSUM, background transpose. Vector: MX quantize, fast exp, XORWOW PRNG. DMA: SBUF read-add-write, traffic shaping, indirect access.
- **[A7] Trn3 architecture (UltraServer)**, https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn3-arch.html . Gen1 is 4 servers x 16 devices = 64. Gen2 is 36 servers x 4 devices = 144, with first-level NeuronSwitch-v1 inside a server and two second-level switches across servers. "All-to-all connectivity design, replacing the previous 2D-torus architecture", targeted at MoE and autoregressive serving. (From search summary of the docs page; the page itself was not fetched separately.)
- **[A8] NKI Programming Model**, https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/programming_model.html , and the NKI Language Guide, https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.1/nki/get-started/nki-language-guide.html .
  - Kernels explicitly DMA HBM→SBUF (`nisa.dma_copy`), compute, and write back.
  - SBUF/PSUM tensors have a **partition dim at dim 0, max 128** (`nki.tile_size.pmax`); PSUM free dim ≤ 512; tiling is manual.
  - `nki.isa` maps 1:1 to engine instructions; `nki.language` is higher level.
  - Programs are sequential, but the compiler may reorder independent ops across engines.
  - The compiled kernel is linked into the graph compiled by the Neuron graph compiler (neuronx-cc), which emits NEFF.
  - Kernels are cached by arguments, so they must be pure.
  - Mentions NKI 0.6.0→0.7.0 migration.
- **[A9] PyTorch on Neuron / TorchNeuron**, https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/frameworks/torch/about/index.rst . "Neuron now supports native PyTorch on Trn2 and Trn3 platforms with TorchNeuron". Eager mode plus `torch.compile(backend="neuron")`; "TorchNeuron is currently in Beta". torch-neuronx (the older torch-xla/XLA-based path) is now "legacy … use only to maintain existing workloads". Per the docs search summary: TorchNeuron was announced at re:Invent Dec 2025 and first shipped in Neuron 2.27.0, using the PrivateUse1 backend + TorchDynamo.
- **[A10] JAX on Neuron**, https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/jax/setup/jax-setup.html . A JAX NeuronX plugin via **PJRT** (`libneuronxla`); `jax-neuronx` pins jax/jaxlib/libneuronxla/neuronx-cc and exposes NKI to JAX. (From search summary.)
- **[A11] AWS Neuron product page**, https://aws.amazon.com/machine-learning/neuron (from search summary, not fetched; UNVERIFIED). It says the NKI Compiler is open source and MLIR-based, along with the Neuron Kernel Library, NxD Inference, Neuron Explorer, and the PyTorch/JAX/vLLM integrations.
- **[A12] Trainium4 + NVLink Fusion**: NVIDIA blog https://developer.nvidia.com/blog/aws-integrates-ai-infrastructure-with-nvidia-nvlink-fusion-for-trainium4-deployment (search result) and The Register https://www.theregister.com/2025/12/02/amazon_nvidia_trainium/ . At re:Invent 2025 AWS said Trainium4 will integrate **NVLink 6 / NVLink Fusion and MGX racks**. Preview only; no date given.

### Google TPU (for the comparison columns)
- **[G1] "TPU 8t and TPU 8i technical deep dive"**, Google Cloud blog, 2026-04-22, https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive . Sections: "TPU 8t: The pre-training powerhouse", "TPU 8i: The sampling and serving specialist", "Deep dive: The Boardfly vs. torus math", "At a glance" table.
  - 8t: 3D torus, SparseCore + "LLM Decoder Engine", 216 GB, 6,528 GB/s, 128 MB VMEM, 12.6 FP4 PF, 9,600-chip superpod; Virgo network >134k chips at 47 Pb/s bisection; 4x DCN.
  - 8i: Boardfly with a 7-hop max diameter (vs 16 hops for an 8x8x16 torus), CAE (Collectives Acceleration Engine) replacing SparseCores, 288 GB, 8,601 GB/s, 384 MB SRAM, 10.1 FP4 PF.
  - No availability date in the article.
- **[G2] Google Keyword blog, "eighth generation TPU / agentic era"** (Cloud Next, April 2026), https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/ . Both chips "later this year". 8t: 121 EF per superpod, 2 PB HBM per superpod. 8i: ICI 19.2 Tb/s (2x prior gen). Up to 2x perf/W vs Ironwood.
- **[G3] Cloud TPU docs, TPU7x (Ironwood)**, https://docs.cloud.google.com/tpu/docs/tpu7x . Spec table (see 1c); 4x4x4 cubes; slices up to 8x16x16 = 2,048 without account-team approval; 9,216-chip pod; supports JAX and PyTorch, not TensorFlow.
- **[G4] Pallas TPU hardware reference**, https://docs.jax.dev/en/latest/pallas/tpu/hardware.html . Per-TensorCore VMEM/CMEM/SMEM: v4 has 16 MiB VMEM + ~64 MiB CMEM; v4i ~128 MiB CMEM; v5e 128 MiB VMEM; v5p 64 MiB; v6e 128 MiB; 7x 64 MiB x 2 TC; 8i 192 MiB x 2 TC; 8t 128 MiB x 1 TC. **CMEM exists only on v4/v4i** in this table. SMEM is 1 MiB from v4 onward.
- **[G5] Cloud TPU system architecture**, https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm . A TensorCore = MXUs + vector unit + scalar unit. MXU is 256x256 on v6e/7x and 128x128 before. SparseCores: 4 on v5p and 7x, 2 on v6e. Cubes connect over copper ICI and link to each other via OCS.
- **[G6] PyTorch blog, "Introducing TorchTPU"** (updated 2026-10-21), https://pytorch.org/blog/introducing-torchtpu/ ; repo https://github.com/google-pytorch/torch_tpu (link from blog; repo load UNVERIFIED).
  - "A TPU Primer": StableHLO and Mosaic are "equivalent to PTX"; the TPU ISA is proprietary.
  - Collectives are lowered as StableHLO ops "versus using a separate library such as NCCL".
  - "Streams and SparseCore": "a TPU TensorCore is fundamentally a single-threaded execution processor … On the contrary, a GPU includes many identical compute cores (SMs)".
  - Eager runs through a PrivateUse1 backend with modes Strict, DeferAndFuse, and Debug. torch.compile goes Dynamo → StableHLO → XLA, not Inductor.
  - Kernels: Pallas via `torch.tpu.pallas.jax_op`, Tokamax, and Helion.
  - Case Study 3: Qwen3-MoE-235B on 8,192 TPU v7 chips at 27.1% MFU, with weak scaling.
  - Case Study 2: a 30B model at 59% MFU prefill on one v7 chip.
  - A secondary-press claim of "52% MFU on 8,192 v7x" does not match the primary blog. Use the blog.
- **[G7] Google Developers blog, "TorchTPU: Running PyTorch Natively on TPUs at Google Scale"** (2026-04-07), https://developers.googleblog.com/en/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/ . First public description. Its roadmap lists a public GitHub repo, bounded dynamism, and Helion.
- **[G8] Jouppi et al., "TPU v4: An Optically Reconfigurable Supercomputer…"**, arXiv:2304.01433 (ISCA 2023), https://arxiv.org/abs/2304.01433 . OCS is under 5% of system cost and under 3% of power. SparseCore gives 5 to 7x on embeddings at ~5% of die. 1.2 to 1.7x faster than A100 at 1.3 to 1.9x less power (similar-size systems).
- **[G9] How to Scale Your Model**, "TPUs" and "GPUs" chapters, https://jax-ml.github.io/scaling-book/tpus/ and https://jax-ml.github.io/scaling-book/gpus/ .
  - TPU chapter: the MXU is a 256x256 systolic array (128x128 before v6e). VMEM is "much larger and programmer-controlled" than L1/L2, at 128 MiB on TPU7x. The 3D torus connects 6 nearest neighbours, and small slices have no wraparound.
  - GPU chapter: "Unlike a TPU, which has at most 2 independent 'Tensor Cores', a modern GPU has more than 100 SMs". H100 SMEM is 256 kB/SM. L2 is ~50 MB, and TPU VMEM is "2x larger and has much more bandwidth (around 40TB/s)". In-node AG/RS cost the same as on TPU, and AllReduce costs 2x unless in-network reduction (SHARP) is used.
  - See also research/notes/scaling_book_part{1,2}.md.
- **[G10] XLA:GPU docs**, https://openxla.org/xla/hlo_to_thunks and https://openxla.org/xla/gpu_architecture (search result). HLO goes through optimization passes (PriorityFusion with a cost model, multi-output fusion), autotuning, then emitters and thunks. Background: Snider & Liang, "Operator Fusion in XLA: Analysis and Evaluation", arXiv:2301.13062 (https://arxiv.org/pdf/2301.13062; not fetched).

---

## 3. TPU vs Trainium vs GPU: concrete differences

1. **Compute unit granularity.** GPU: 132 SMs (H100), each with 4 Tensor Cores, many warps, SIMT, hardware-scheduled [N7]. TPU: 1 or 2 TensorCores per chip, each a single-threaded VLIW-style core driving large MXUs (256x256 on v6e/7x) [G5][G6]. Trainium: 8 NeuronCores per chip, each with one 128x128 systolic Tensor Engine plus separate Vector, Scalar and GpSimd engines that run concurrently and sync via semaphores [A2]. **Trainium sits in between.** Like a TPU, it uses systolic arrays with software-managed SRAM. Like a GPU, it splits the chip into several independent cores (8 per chip, versus TPU's 1 or 2). With LNC it can merge them into logical cores.
2. **Systolic array size and dtype.** TPU MXU is 256x256 BF16 (v6e onward) [G5]. Trainium is 128x128, with FP8 doubling the contraction dim to 256 and MX formats quadrupling it to 512 [A2][A6]. GPU Tensor Cores are warp-group (Hopper wgmma) or single-thread-issued (Blackwell tcgen05) MMA units fed from SMEM, with accumulators in registers on Hopper and in TMEM on Blackwell [N7][N11].
3. **On-chip memory model.**
   - GPU: a hardware-managed L2 (50 MB on H100) plus a per-SM 228 KB programmer-managed SMEM, carved from 256 KB L1 [N9].
   - TPU: one large programmer/compiler-managed VMEM per TensorCore (128 MiB/chip on 7x, 384 MB on 8i) and no general cache hierarchy. CMEM appeared only on v4/v4i [G4].
   - Trainium: SBUF (28 to 32 MiB per core, 2-D with 128 partitions) plus a separate **PSUM** (2 MiB) accumulation buffer [A2][A6].
   - Partition-major layout constraints (Trainium P≤128; TPU (8,128) vreg tiling) matter in both. GPU SMEM has bank conflicts but no partition axis.
4. **Kernel languages.**
   - CUDA/Triton/CUTLASS: threads and blocks, with implicit caching.
   - **Pallas (TPU)**: BlockSpec/grid with Refs in VMEM/SMEM/ANY; the compiler pipelines HBM↔VMEM DMAs; lowers to Mosaic [G6][P11].
   - **NKI (Trainium)**: explicit `dma_copy` HBM→SBUF; tiles with a partition dim; direct engine ISA calls (`nki.isa`); the programmer picks the engine; linked into the neuronx-cc graph [A8].
   - NKI is lower-level than Pallas TPU (explicit engine choice and DMA). It is comparable to Pallas TPU with manual `make_async_copy` + semaphores.
   - Pallas also targets GPUs (Hopper/Blackwell) [Pallas index: https://docs.jax.dev/en/latest/pallas/index.html]; NKI is Trainium-only.
5. **Compiler and framework path.**
   - GPU: eager PyTorch with CUDA kernels; optional torch.compile/Inductor and CUDA Graphs. Graphs replay a fixed launch DAG but do no cross-kernel optimization [N10].
   - TPU: XLA whole-program compilation from StableHLO (JAX native; PyTorch via TorchTPU, which is PrivateUse1 → StableHLO → XLA, with an eager DeferAndFuse mode). Static shapes are required; bounded dynamism is in progress [G6].
   - Trainium: historically torch-neuronx (torch-xla lazy tensors → HLO → neuronx-cc → NEFF) and JAX via PJRT plugin (libneuronxla). Now TorchNeuron (PrivateUse1, eager + `torch.compile(backend="neuron")`, beta) [A9][A10].
   - **Both Google and AWS moved PyTorch off torch-xla lazy tensors to PrivateUse1 native backends in 2025–26.** Google kept XLA as the compiler; AWS kept neuronx-cc.
6. **Scale-up topology.**
   - GPU: switched, 1-hop all-to-all NVLink domain (8 on HGX, 72 on NVL72) with in-switch reduction (NVLS/SHARP) [N5][P4].
   - TPU: direct-connect 3D torus with wraparound and OCS-reconfigurable cubes, at 9,216 (7x) and 9,600 (8t) chips. That is a much larger scale-up domain, but multi-hop with no in-network reduce [G3][G1].
   - TPU 8i breaks the torus with Boardfly (max 7 hops, 1,152 chips) for latency-bound MoE/decoding [G1].
   - Trainium: Trn2 used a 4x4 2D torus per instance plus rings across 4 instances (64) [A3]. Trn3 moved to **switched all-to-all NeuronSwitch** (64 or 144) [A7], becoming more GPU-like. Trainium4 will adopt NVLink Fusion [A12].
   - Three-way trend: GPU and Trainium converge on switched scale-up of ~72 to 144. TPU keeps torus at ~10k for training and adds a low-diameter topology for inference.
7. **Collectives implementation.**
   - GPU: NCCL kernels run on SMs and compete with compute. Hence copy-engine/SymmetricMemory async-TP [P6], fused kernels (Flux [P8]), and TE userbuffers [P9].
   - TPU: collectives are XLA HLO ops scheduled by the compiler. Collective matmul (decomposed AG/RS + ppermute) is a compiler transform [P10][P11]. Can offload to SparseCore (7x) or CAE (8i) [G6][G1].
   - Trainium: dedicated **CC-Cores** (16 to 20 per chip) run collectives off the compute engines [A1][A2].
8. **Bandwidth accounting.** NCCL busbw factors [P3] apply unchanged to a single TPU torus axis treated as a ring. A bidirectional ring with wraparound doubles usable BW per axis. Multi-axis tori let the collective run over several axes at once. Note: on GPUs an in-node AllReduce costs 2x an AG unless NVLS is used, and the same 2x holds on TPU [G9].
9. **Numerics.** Blackwell, Rubin and TPU 8 headline FP4 (NVFP4 on NVIDIA, FP4 on TPU 8 [G1]). Trainium3 headlines MXFP8/MXFP4 (OCP MX, 32-element blocks) [A6], and its BF16 is flat vs Trn2 (667 → 671 TF) [A4]. Ironwood is the first TPU with published FP8 peak (4,614 TF) [G3]. Trainium offers hardware stochastic rounding [A1][A4].
10. **Memory per chip (latest generation).** Rubin 288 GB HBM4 at 19.2 to 22 TB/s [N8]. TPU 8i 288 GB at 8.6 TB/s; 8t 216 GB at 6.5 TB/s [G1]. Trn3 144 GiB at 4.9 TB/s [A4]. Ironwood 192 GiB at 7.4 TB/s [G3]. GB300 ~278 to 288 GB at 8 TB/s [N6].
11. **Partitioning/virtualization.** GPU MIG splits one GPU into isolated instances. TPU slices are sets of whole chips carved from the pod topology [G5]. Trainium LNC *merges* physical NeuronCores into larger logical ones [A4], the inverse of MIG.
12. **Availability model.** GPUs are sold everywhere. Trainium is AWS-only (including Bedrock). TPUs were GCP-only but now reach third-party DCs (Fluidstack) and possibly customer DCs (Meta talks); see §4.

---

## 4. Recent changes (Oct 2025 – Oct 2026), dated headlines

| Date | Headline | URL | Status |
|---|---|---|---|
| 2025-10-15 | JAX 0.8.0: `jax.shard_map` replaces experimental; `pmap` reimplemented on jit/shard_map | https://docs.jax.dev/en/latest/changelog.html | primary |
| 2025-10-16 | vLLM TPU rebuilt on **tpu-inference**: JAX and PyTorch (via Torchax) on one JAX lowering path; Ragged Paged Attention v3; v5e/v6e verified, v5p experimental | https://vllm.ai/blog/2025-10-16-vllm-tpu ; repo https://github.com/vllm-project/tpu-inference | primary |
| 2025-10-23 | Anthropic to use up to **1M Google Cloud TPUs**, "well over a gigawatt" in 2026, "tens of billions" | https://www.anthropic.com/news/expanding-our-use-of-google-cloud-tpus-and-services ; https://www.googlecloudpresscorner.com/2025-10-23-Anthropic-to-Expand-Use-of-Google-Cloud-TPUs-and-Services | primary |
| 2025-11-06 | Ironwood (TPU7x) GA announced ("coming weeks") with Axion VMs | https://www.constellationr.com/insights/news/google-clouds-ironwood-ready-general-availability ; docs https://docs.cloud.google.com/tpu/docs/tpu7x | secondary for date; docs show it available |
| 2025-11-25 (approx.) | Reports: Meta in talks to rent TPUs on GCP in 2026 and deploy TPUs in its own DCs from 2027; Google pitching TPUs for on-prem | https://www.trendforce.com/news/2025/11/25/news-meta-reportedly-weighs-google-tpu-deployment-in-2027-boosting-broadcom-taiwans-guc/ | secondary (orig. The Information). UNVERIFIED by either company |
| 2025-12-02 | **Trn3 UltraServers GA** (Trainium3, 3 nm, 144 chips, 362 MXFP8 PF) | https://press.aboutamazon.com/2025/12/trainium3-ultraservers-now-available-enabling-customers-to-train-and-deploy-ai-models-faster-at-lower-cost ; https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-ec2-trn3-ultraservers/ | primary |
| 2025-12-02 | AWS previews **Trainium4 with NVIDIA NVLink Fusion** (NVLink 6, MGX); no date | https://www.theregister.com/2025/12/02/amazon_nvidia_trainium/ ; https://developer.nvidia.com/blog/aws-integrates-ai-infrastructure-with-nvidia-nvlink-fusion-for-trainium4-deployment | secondary + NVIDIA blog (search result) |
| 2025-12 (Neuron 2.27.0) | AWS **TorchNeuron** (native PyTorch, PrivateUse1, eager + torch.compile); torch-neuronx marked legacy | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/frameworks/torch/about/index.rst | primary (beta) |
| 2026-01 (CES) | NVIDIA launches **Vera Rubin NVL72** (288 GB HBM4/GPU, NVLink 6), 2H 2026 | https://www.tomshardware.com/pc-components/gpus/nvidia-launches-vera-rubin-nvl72-ai-supercomputer-at-ces-promises-up-to-5x-greater-inference-performance-and-10x-lower-cost-per-token-than-blackwell-coming-2h-2026 ; product page https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/ ("Available Now" at access) | primary page + secondary |
| 2026-01-20 | JAX 0.9.0 (explicit sharding in `jax.export`) | https://docs.jax.dev/en/latest/changelog.html | primary |
| 2026-02 | Report (The Information): Meta signs multi-billion, multi-year deal to **rent Google TPUs** | https://storyboard18.com/digital/meta-signs-multi-billion-dollar-ai-chip-rental-deal-with-google-90907.htm | secondary single-source. UNVERIFIED |
| 2026 (various) | **Fluidstack** as first external TPU DC operator (TeraWulf NY site; Google lease backstops, ~14% equity stake); Google approached CoreWeave/Crusoe with little result | https://www.datacenterdynamics.com/en/news/google-offers-its-tpus-to-ai-cloud-providers-report ; https://forbes.it/2026/09/09/fluidstack-google-data-center-ai/ | secondary. UNVERIFIED |
| 2026-04-06 | Anthropic, Google and Broadcom: **multiple GW of next-gen TPU capacity from 2027** (later described as 5 GW) | https://www.anthropic.com/news/google-broadcom-partnership-compute | primary |
| 2026-04-07 | Google Developers blog first describes **TorchTPU** | https://developers.googleblog.com/en/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/ | primary |
| 2026-04-16 | JAX 0.10.0: C++ pmap removed; Pallas `pl.kernel` uses `out_type`/`scratch_types`; TPU semaphores move to `jax.experimental.pallas` | https://docs.jax.dev/en/latest/changelog.html ; https://docs.jax.dev/en/latest/pallas/CHANGELOG.html | primary |
| 2026-04-22 | **TPU 8t (training) and TPU 8i (inference)** announced at Cloud Next, the first split TPU generation. 8i uses Boardfly topology + CAE; Arm Axion hosts; GA "later this year" | https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive ; https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/ | primary |
| 2026-04 | Reports: Broadcom co-designs 8t, MediaTek leads 8i | https://heise.de/-11270454 | secondary. UNVERIFIED |
| 2026-05-20 | JAX 0.10.1: `with mesh:` deprecated → `jax.set_mesh` | https://docs.jax.dev/en/latest/changelog.html | primary |
| 2026-07-16 | JAX 0.11.0 drops Python 3.11 | same | primary |
| 2026-10-19 | JAX 0.12.0 (`sharding.no_collectives()`; Shardy always used) | same | primary |
| 2026-10-20/21 | **TorchTPU open-sourced** at PyTorch Conference 2026 (Google + Meta); repo github.com/google-pytorch/torch_tpu; eager + torch.compile→XLA; Pallas/Helion/Tokamax kernels | https://pytorch.org/blog/introducing-torchtpu/ | primary (repo load UNVERIFIED) |
| as of 2026-10-23 | TPU 8t/8i GA status: no GA announcement found | (none) | UNVERIFIED |

---

## 5. Open items / UNVERIFIED list
- ACM DOI for Wang et al. ASPLOS'23 [P10]: the ACM page returned 403.
- The official ungated URL for the Hopper whitepaper; a mirror was used.
- GB300 dense FP4 "1080" parse [N6]; per-GPU GB300 HBM (NVIDIA elsewhere markets 288 GB; the rack page says 20 TB/72 ≈ 278 GB).
- Rubin per-GPU HBM BW (19.2 vs 22 TB/s) and NVLink (3 vs 3.6 TB/s) [N8] vs CES coverage.
- Trn2/Trn3 CC-Core count (16 vs 20) and Trn3 HBM BW (4.7 vs 4.9 TB/s) and NeuronLink-v4 (2.0 vs 2.56 TB/s) differ between AWS pages.
- Meta–Google TPU rental and on-prem deals, Fluidstack financing details, and Broadcom/MediaTek TPU 8 roles: press only.
- TPU 8t/8i BF16/FP8 peak: not published by Google as of access date.
- [A7], [A10], [A11], [N11], [P7], [G10]: content taken from search-result summaries of the pages, not a direct fetch.
