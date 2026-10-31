# Teaching notes: TPUs for GPU engineers in 90 minutes

Audience: engineers fluent in CUDA, NCCL and H100/B200 performance. Goal: by the end they can predict whether a TPU workload is compute-, memory- or communication-bound, read a TPU profile, and say where TPU and GPU designs differ and why. Open the course site and the whole-course deck (`#/slides/all/0`) before starting; the demos are the interactives named below.

## Agenda

| Min | Segment | Make this point | Demo | Chapter |
|---|---|---|---|---|
| 0–10 | Why TPUs look the way they do | Each generation moved a ridge point. TPUv2 to Ironwood: FLOPs 50×, HBM bandwidth 10.5×, ridge 66 → 313 FLOP/B. TPU 8 split into training (8t) and inference (8i) chips. | Generation chart | 1 |
| 10–25 | The chip | A TensorCore is an SM, not a Tensor Core: 2 per Ironwood chip against 132 SMs on an H100. One instruction stream; the compiler hides latency. 256×256 weight-stationary MXU: rows per weight tile = token batch (1.5% utilization at 8 rows, 80% at 2,048). 128 MiB of VMEM is a scratchpad, not a cache. | Systolic array; TensorCore diagram | 2 |
| 25–40 | The network | No switch in the data path. A 3D torus of OCS-joined 4×4×4 cubes holds 9,216 chips at constant per-chip bandwidth. Ring AllGather costs V/W_ici whatever the ring size; AllToAll is ¼ of that in 1D. Small transfers are latency-bound at about 1 µs per hop, which is why 8i uses Boardfly (≤ 7 hops). | Torus explorer; ring collectives | 3, 5 |
| 40–55 | Training | Everything depends on α = C/W_ici. v5p 2,550; Ironwood 11,535. LLaMA-3-70B at a 4M-token batch is compute-bound on a full v5p pod (FSDP+TP needs 113 tokens per chip) but not on a full Ironwood pod (needs about 2,320). GPU recipes pipeline because NVLink stops at 8 or 72; TPU recipes rarely need to. | Mesh planner (switch chip, same batch) | 6 |
| 55–70 | Inference and serving | Decode step = KV read + max(weight read, FLOPs). LLaMA-3-70B on 8 Ironwood chips at 32k context: about 12 of 14 ms is KV. fp8 weights and KV double throughput; more FLOPs do nothing. vLLM's scheduler is unchanged on TPU; static shapes come from padded buckets and Ragged Paged Attention. New PyTorch ports go to vllm-torchtpu on TorchTPU. | Prefill/decode calculator; serving stack diagram | 7, 8 |
| 70–80 | Software and profiling | XLA compiles the whole step (fusion, layouts, VMEM placement, collective schedule), unlike CUDA Graphs, which replay compiled kernels. Pallas grids are sequential loops, not thread blocks. In a trace, synchronous collectives are bars alone on the XLA Ops row; overlapped ones are -start/-done pairs. | HLO decoder; trace reader; Chapter 9 listings | 9, 10 |
| 80–90 | Exercise and questions | Design review: LLaMA-3-70B on Ironwood, 30 ms per token at 32k. Answer: 4-chip replicas, fp8 weights and KV, batch about 140, about 1,250 tokens/s per chip, KV-bound; separate prefill servers. | Comparison explorer for TPU vs GPU follow-ups | 10, 11 |

## Numbers to have on the board

| | |
|---|---|
| BF16 ridge (FLOP/B) | v5e 241 · Ironwood 313 · H100 295 · GB200 312 · TPU 8i 128 (provisional) |
| Compute-bound batch | about the ridge for bf16; halves with 8-bit weights and bf16 math; unchanged with 8-bit math at 2× rate |
| Ironwood | 2,307 TFLOP/s BF16, 4,614 FP8, 192 GiB at 7.38 TB/s, 128 MiB VMEM, 2 chiplets = 2 JAX devices, 6 ICI links × 100 GB/s each way |
| Collectives (ring, per axis) | AllGather = ReduceScatter = V/W; AllReduce = 2V/W; AllToAll = V/(4W); latency-bound below about 100 kB per hop on Ironwood |
| α = C/W_ici | v5p 2,550 · Ironwood 11,535 · H100 across nodes 2,475 |
| KV cache | 2·L·K·H·bytes per token; LLaMA-3-70B bf16: 320 KB per token, 10.7 GB at 32k |

## Questions GPU engineers ask, and short answers

- **"Isn't XLA just CUDA Graphs?"** No. A CUDA Graph replays kernels that are already compiled. XLA chooses fusion, the layout of every array, what lives in VMEM and when collectives run, as one program. The price is static shapes.
- **"Where are the warps?"** There are none. One instruction stream per TensorCore; the compiler issues the next DMA before the current tile finishes.
- **"How do you do NCCL?"** Collectives are HLO ops that XLA schedules and DMA engines execute. Overlap with matmuls (collective matmul) is a compiler transform, not a library.
- **"Is the torus slower than NVLink?"** Inside 8 or 72 GPUs, often yes for AllToAll and small collectives: NVLink is one hop. Past that, the torus keeps full bandwidth to about 9,000 chips while GPUs drop to scale-out.
- **"Can I bring my PyTorch model?"** Yes. vllm-torchtpu (TorchTPU, `torch.compile` to XLA) for serving, TorchTPU for training. The torchax path in tpu-inference still works. PyTorch/XLA is deprecated.
- **"What is different about Ironwood?"** Five times v5p's FLOPs per chip with the same per-link ICI bandwidth, so training wants more tokens per chip. 8t doubles ICI.

## Pitfalls to warn about

- No wraparound on small slices (v5e below 16 per axis; Ironwood below a 4×4×4 cube): roughly 2× the collective time.
- FSDP in decode: weights over ICI every step, an order-of-magnitude regression.
- Dimensions not multiples of 256 (v6e, Ironwood), and head dimension 64, waste MXU capacity.
- Shapes outside precompiled buckets in serving trigger minutes-long compiles.
- Google's materials mix GB and GiB (192 "GB" against 192 GiB on Ironwood); check units before quoting capacity.

## After the session

Point people to the course: the Day 1 chapters, then the chapter matching their work (training: 5–6; serving: 7–8; kernels and profiling: 9–10). The Scaling Book (jax-ml.github.io/scaling-book) is the primary reference behind most of the numbers.
