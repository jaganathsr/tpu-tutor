# JAX Scaling Book ("How to Scale Your Model"), Part 2: research notes

Source: Austin et al., "How to Scale Your Model", Google DeepMind, 2025. https://jax-ml.github.io/scaling-book/
Accessed: 2026-10-23. Every URL below returned HTTP 200 on that date (each one redirects to a trailing-slash form).
Method: I downloaded the raw HTML with curl, converted it to text, and read it in full, including every "Click here for the answer" block.
Fidelity rule: numbers are copied from the book as written. If the book has an arithmetic quirk or an inconsistency, it is flagged **[NOTE]** and left as is. Anything not from the book is marked **[NOT IN BOOK]**.

## Table of contents (verified from the site's "Sections" dropdown)

| Part | Title | URL |
|---|---|---|
| 0 | Introduction | /scaling-book/index |
| 1 | Intro to Rooflines | /roofline |
| 2 | All About TPUs | /tpus |
| 3 | Sharded Matmuls | /sharding |
| 4 | Transformers | /transformers |
| 5 | Training | /training |
| 6 | Training LLaMA | /applied-training |
| **7** | **Inference** ("All About Transformer Inference") | https://jax-ml.github.io/scaling-book/inference/ |
| **8** | **Serving LLaMA** ("Serving LLaMA 3-70B on TPUs") | https://jax-ml.github.io/scaling-book/applied-inference/ |
| **9** | **Profiling** ("How to Profile TPU Programs") | https://jax-ml.github.io/scaling-book/profiling/ |
| **10** | **All About JAX** (page title "Programming TPUs in JAX") | https://jax-ml.github.io/scaling-book/jax-stuff/ |
| **11** | **Conclusions** ("Conclusions and Further Reading") | https://jax-ml.github.io/scaling-book/conclusion/ |
| **12** | **GPUs** ("How to Think About GPUs") | https://jax-ml.github.io/scaling-book/gpus/ |

The GPU chapter is numbered 12 and comes after the Conclusions. Its nav reads "Part 11: Conclusion | The End".
Publication dates: Parts 7–11 are dated Feb. 4, 2025. Part 12 (GPUs) is dated Aug. 18, 2025, by Jacob Austin, Swapnil Patil, Adam Paszke and Reiner Pope.
Appendices: no book-wide appendix exists. Some chapters have their own: Part 7 has Appendices A–D, and Part 12 has Appendices A–B.

---

# Part 7: All About Transformer Inference

URL: https://jax-ml.github.io/scaling-book/inference/ (accessed 2026-10-23)

## Section headings (anchors)
- `#the-basics-of-transformer-inference` The Basics of Transformer Inference
  - `#what-do-we-actually-want-to-optimize` What do we actually want to optimize?
  - `#a-more-granular-view-of-the-transformer` A more granular view of the Transformer
  - `#linear-operations-what-bottlenecks-us` Linear operations: what bottlenecks us?
  - `#what-about-attention` What about attention?
  - `#theoretical-estimates-for-llm-latency-and-throughput` Theoretical estimates for LLM latency and throughput
  - `#what-about-memory` What about memory?
  - `#modeling-throughput-and-latency-for-llama-2-13b` Modeling throughput and latency for LLaMA 2-13B
- `#tricks-for-improving-generation-throughput-and-latency` Tricks for Improving Generation Throughput and Latency
- `#distributing-inference-over-multiple-accelerators` Distributing Inference Over Multiple Accelerators
  - `#prefill` Prefill
  - `#generation` Generation
  - `#sharding-the-kv-cache` Sharding the KV cache
- `#designing-an-effective-inference-engine` Designing an Effective Inference Engine
  - `#continuous-batching` Continuous batching
  - `#prefix-caching` Prefix caching
  - `#lets-look-at-an-implementation-jetstream` Let's look at an implementation: JetStream
- `#worked-problems` Worked Problems
- `#appendix` Appendix
  - `#appendix-a-how-real-is-the-batch-size--240-rule` A: How real is the batch size > 240 rule?
  - `#appendix-b-2d-weight-stationary-sharding` B: 2D Weight Stationary sharding
  - `#appendix-c-latency-bound-communications` C: Latency bound communications
  - `#appendix-d-speculative-sampling` D: Speculative Sampling

## Core concepts
- Naive sampling reprocesses the whole prefix at every step. That costs O(n²) in the FFW and O(n³) in attention to generate n tokens. A KV cache brings this down to O(n) FFW and O(n²) attention.
- Inference splits into two phases:
  - **Prefill** processes the whole prompt at once and saves the KV cache plus the logits of the last token.
  - **Generation** samples one token per step, feeds it back in, and appends its K/V to the cache. It runs until `<EOS>` or the maximum length.
  - The book calls these "two tasks in disguise".
- Latency is a new objective that training does not have. It covers TTFT (time to first token) and per-token latency. The book names three regimes:
  - Offline batch inference cares only about cost.
  - Chat or streaming needs to be cheap at scale while keeping TTFT low and generating faster than human reading speed.
  - Edge inference (for example llama.cpp) serves one user at the lowest possible latency.
- Three groups of ops matter:
  - Linear ops: the MLP W_in and W_out, and the attention projections W_Q, W_K, W_V, W_O.
  - Dot-product attention.
  - Everything else (norms, activations, sampling, KV update, positional embeddings). These are dominated by, or fused into, the first two.
- The book wants the reader to take away the step-time lower bounds below above anything else in the chapter. Its words: "if there is one thing we want the reader to take away from this entire chapter".
- **Prefill rule:** matmuls are basically always compute-bound. Maximize MFU. Batching prompts adds latency for little throughput gain unless prompts are very short.
- **Generation rule:** the total token batch must exceed B_crit to be compute-bound on the linear layers. That means batching many concurrent requests. For dense models, a batch of 240 means 240 concurrent requests and 240 separate KV caches, which is hard to reach outside bulk inference.
- Attention during decode is always memory-bandwidth bound, and batching does not help: every sequence has its own KV cache. Once KV memory becomes comparable to parameter memory, a bigger batch gives diminishing returns. The relevant ratio is about 2DF / SHK, and since HK ≈ D this is roughly F vs S.
- **Throughput rule:** use the largest per-chip batch you can. Any per-chip batch above B_crit (usually 120 or 240) maximizes throughput, which may require a bigger topology. Smaller batches buy latency at the cost of throughput.
- Real-world caveats (the "nits"):
  - XLA does not overlap HBM reads perfectly.
  - For sharded models, XLA often fails to overlap ICI comms with FLOPs, so latency on linears starts to suffer above BS=32.
  - Batches above the theoretical roofline still gain a little throughput.

## Key formulas

**Linear layer roofline.** Matmul of bf16[B,D] by bf16[D,F]:

$$T_\text{math}=\frac{2BDF}{\text{FLOPs/s}},\qquad T_\text{comms}=\frac{2BD+2FD+2BF}{\text{HBM BW}}$$

$$\frac{2BDF}{2BD+2DF+2BF}\ge\frac{C}{W_\text{hbm}}\underset{\text{v5e}}{=}\frac{1.97\text{e}14}{8.20\text{e}11}=240$$

The matmul is compute-bound when its arithmetic intensity exceeds the chip's FLOPs-to-bandwidth ratio.

With B ≪ D, F (the book says batches are usually ≤ 500 while D, F > 10k), the bytes are ≈ 2DF, which gives $B \ge 240 = B_\text{crit}$.

**General critical batch.**

$$B_\text{crit}=\beta\,\alpha_\text{hbm},\quad \beta=\frac{\text{bits per param}}{\text{bits per activation}},\ \alpha_\text{hbm}=C/W_\text{hbm}$$

Weight quantization lowers B_crit. Low-precision compute raises it.

- int8 or fp8 weights with bf16 compute: B_crit halves to **120**.
- int8 or fp8 compute: B_crit doubles. With int8 weights × int8 activations on v5e it is back to **240**, because v5e does 400 TOPs/s of int8 (Part 8 gives 3.94e14).
- **v5e bf16 is 240 tokens. H100 is "about 280 tokens".** [NOTE] The GPU chapter computes 295 for H100 (990e12/3.35e12).

**Attention arithmetic intensity** (MHA, one head, Flash Attention). The kernel reads Q[B,T,D] and KV 2×[B,S,D], does 2BSTD FLOPs for QK and another 2BSTD for AV, and writes [B,T,D]:

$$\text{AI}=\frac{4BSTD}{4BSD+4BTD}=\frac{ST}{S+T}$$

FLOPs per byte of attention depends only on the query and KV lengths. B and D cancel.

- Prefill (S = T): AI = T/2. Compute-bound for any reasonable length, roughly > 480 tokens.
- Decode (T = 1, S ≫ 1): AI ≈ 1, so decode attention is always bandwidth-bound.

**Step-time and throughput bounds** (small-batch decode, memory-bound everywhere):

$$\text{Min step time}=\frac{B\cdot\text{KV size}+\text{Param size}}{\text{Total mem BW}}\qquad \text{Max tok/s}=\frac{B\cdot\text{Total mem BW}}{B\cdot\text{KV size}+\text{Param size}}$$

A decode step cannot be faster than reading every weight once plus each sequence's KV cache once.

**General step time:**

$$T_\text{step}=\underbrace{\frac{B\cdot\text{KV size}}{\text{Mem BW}}}_{\text{attention, always BW-bound}}+\underbrace{\max\!\Big(\frac{2B\cdot\text{Params}}{\text{FLOPs/s}},\frac{\text{Param size}}{\text{Mem BW}}\Big)}_{\text{MLP, can be compute-bound}}$$

Attention time is always the KV read. The MLP costs whichever is larger, weight loading or FLOPs.

**KV cache size:**

$$\text{KV bytes}=2\cdot\text{bytes/float}\cdot H\cdot K\cdot L\cdot T$$

H is the head dimension, K the number of KV heads, L the number of layers. The leading 2 counts K and V.

**Decode ICI vs HBM** (how far model sharding can go before ICI becomes the bottleneck):

$$T_\text{HBM}=\frac{2DF}{Y W_\text{hbm}},\quad T_\text{ICI}=\frac{2BD}{W_\text{ici}};\quad T_\text{ICI}>T_\text{HBM}\iff Y>\frac{F}{B\beta},\ \beta=W_\text{hbm}/W_\text{ici}$$

In memory-bound decode you can model-shard well past the training ICI limit, because each extra chip adds HBM bandwidth.

- β is "usually around 8 for TPU v5e and TPU v6e".
- Example: F = 16,384 and B = 32 give 16384/(32·8) = **64-way** model parallelism without a big throughput hit. This assumes the KV caches can also be sharded 64 ways.

**Prefill model-parallel bound:** about F/2200 for one axis, which is usually 4–8-way (from Section 5).

**Latency-bound collectives** (Appendix C):

$$T_\text{total}=\max\Big(\frac{T_\text{min}\cdot|X|}{2},\frac{B}{W_\text{ICI}}\Big)$$

When the per-hop data is tiny, the fixed per-hop latency dominates.

- A collective is latency-bound when (bytes/n_shards)/4.5e10 < 1e-6, i.e. less than 1 µs per hop at 4.5e10 B/s unidirectional ICI.
- For 8-way Megatron this happens when the buffer is under 360 kB.
- BS=16 with D=8192 in int8 is 131 kB, which is already latency-bound.
- Rule: latency-bound when total bytes < W_ICI × 1e-6. With Y-way model parallelism in int8, that is when Y > BD/45,000.

## Hardware numbers stated (Part 7)
- TPU v5e: 1.97e14 bf16 FLOPs/s; 8.20e11 B/s HBM; intensity 240; 400 TOPs/s int8; 16 GB HBM per chip.
- "8x TPU v5es gives us 128GiB of HBM, 6.5TiB/s of HBM bandwidth (0.82TiB/s each) and 1600TF/s of compute."
- An 8xTPUv5e server has 128 GiB of HBM and about 450 GiB of host DRAM.
- H100 B_crit is about 280.
- W_hbm/W_ici ≈ 8 on v5e and v6e. ICI is 4.5e10 B/s unidirectional (Appendix C).

## LLaMA 2-13B memory model
Hyperparameters: L=40, D=5,120, F=13,824, N=40, K=40, H=128, V=32,000.
- FFW params: 5,120×5,120×2.7×3×40 = 8.5e9.
- Vocab: 2×32,000×5,120 = 0.3e9.
- Attention: (2×5,120×40×128 + 2×5,120×40×128)×40 = 4.2e9.
- Total ≈ 13e9 params. That is 26 GB in bf16, with about 100 GB for training state (bf16 params plus fp32 optimizer).
- A prefill activation for 8k tokens is 8,192×5,120×2 B ≈ 80 MB, which is negligible.
- KV for one 8192-token sequence in bf16: 8192×40×128×40×2×2 = **6.7 GB**. Four sequences already exceed the parameter memory.

Table 1, MHA on 8×v5e (step times and throughput are theoretical):

| Batch | 1 | 8 | 16 | 32 | 64 | 240 |
|---|---|---|---|---|---|---|
| KV mem (GiB) | 6.7 | 53.6 | 107.2 | 214.4 | 428.8 | 1608 |
| Total mem (GiB) | 32.7 | 79.6 | 133.2 | 240.4 | 454.8 | 1634 |
| Step (ms) | 4.98 | 12.13 | 20.30 | 36.65 | 69.33 | 249.09 |
| tok/s | 200.61 | 659.30 | 787.99 | 873.21 | 923.13 | 963.53 |

The model OOMs beyond batch 16 and would need an order of magnitude more memory to get near 240.

Table 2, with a KV cache 5× smaller (1:5 GMQA, 8 KV heads):

| Batch | 1 | 8 | 16 | 32 | 64 | 240 |
|---|---|---|---|---|---|---|
| KV (GiB) | 1.34 | 10.72 | 21.44 | 42.88 | 85.76 | 321.6 |
| Total (GiB) | 27.34 | 36.72 | 47.44 | 68.88 | 111.76 | 347.6 |
| Step (ms) | 4.17 | 5.60 | 7.23 | 10.50 | 17.04 | 52.99 |
| tok/s | 239.94 | 1,429.19 | 2,212.48 | 3,047.62 | 3,756.62 | 4,529.34 |

Batch 64 now fits, and throughput keeps scaling up to 240. LLaMA-3 8B uses 32 Q heads and 8 KV heads.

## Tricks for generation
- **GQA/GMQA:** the KV cache shrinks by the Q:KV ratio, and attention arithmetic intensity rises.
- **Local attention layers:** cap the KV length on those layers by mixing local and global layers.
- **Cross-layer KV sharing:** a smaller cache, but the shared KV may be read from HBM several times, so step time does not necessarily improve.
- **Quantizing params and KV** (int8, int4, fp8): saves bandwidth, lowers the batch needed to reach the roofline, and frees memory for bigger batches. It can often be applied post-training.
- **Ragged HBM reads and Paged Attention:** only the non-padding part of the KV is read. Paged Attention stores KV in OS-style page tables. It is a runtime optimization that does not depend on the architecture.
- Big picture: together these can shrink the KV cache by more than 10× versus standard MHA, which can mean about a 10× improvement in total cost.

## Distributing inference
- **Prefill** behaves like training. Megatron model parallelism, sequence sharding, pipelining and even FSDP all work. The rule: do model parallelism up to the ICI bound (about F/2200, 4–8-way), then add sequence parallelism. Overlap comms using collective matmuls and ring attention.
- **Generation:**
  - **FSDP is impossible.** Weights would move over ICI, which is orders of magnitude slower than HBM. Leaving FSDP on after training is "an easy and common way to have order of magnitude regressions".
  - **No reason to use data parallelism.** Run separate model replicas instead.
  - **No sequence sharding,** because there is no sequence dimension.
  - That leaves model sharding. Shard W_Q and W_O over heads. The KV weights are small, and replicating them beyond K-way is often cheaper than sharding them.
- Takeaway: move activations, not KV or weights. With a large batch, do model parallelism up to the FLOPs-ICI bound (F/α). With a small batch, shard further to cut latency. Past K-way, shard the KV along batch.

**Sharding the KV cache.** Shard by head first (Megatron, at most K-way), then by batch: $\text{KV}[2, B_Z, S, K_Y, H]$.
- The cost is **two AllToAlls per attention layer**: one moves Q to batch sharding, the other moves the output back to model sharding.
- Full algorithm, with M = N/K:
  1. Q[B,N_YZ,H] = X·W_Q
  2. AllToAll_{Z→B} gives Q[B_Z,N_Y,H]
  3. Reshape to [B_Z,K_Y,M,H]
  4. Compute QK, softmax over S, then AV
  5. AllToAll_{Z→M} gives O[B,K_Y,M_Z,H]
  6. Reshape to O[B,N_YZ,H]
  7. X{U_YZ} = W_O · O, then AllReduce
- If the batch is small or the context long, you can sequence-shard the KV. This needs an AllGather of Q and Flash-Attention-style accumulation.

## Inference engine design
- **Naive batched prefill then generate:**
  - Poor TTFT, because all prefills must finish first.
  - Short generations block on long ones.
  - Padded prefills waste FLOPs ("historically XLA made it quite difficult to skip these FLOPs").
  - Prefill and generate are forced to share one sharding.
  - Recommended only for edge use or early iteration.
- **Interleaved:** prefill at batch 1 and batch the generation. Good for bulk evals. The drawback is that a prefill pauses all decodes, so token generation becomes jittery.
- **Disaggregated serving:** separate prefill servers and generate servers, with KV caches sent over the network.
  - Low latency at scale, because prefill and generate pools scale independently.
  - Specialization: generate wants more model parallelism. Prefill servers only hold the KV currently being processed, which frees memory for history caching.
  - The cost is moving the KV cache over the network.
  - Takeaway: for latency-sensitive, high-throughput serving, separate the two, with prefill at batch 1 and generate heavily batched.
- **Continuous batching:** a compiled prefill function handles variable lengths and inserts into a KV buffer (max batch, max length or number of pages). A compiled generate function steps all active requests. An orchestrator queues requests, picks prefill vs generate based on free slots, handles history caching, and streams tokens out.
- **Prefix caching:**
  - Reuse KV for shared prefixes. Example: after "I like dogs", the prompt "I like cats" needs only 1/3 of the compute.
  - Most useful for chat turns and few-shot or system prompts.
  - Use the idle HBM on prefill servers and the host DRAM (450 GiB vs 128 GiB HBM per 8×v5e). DRAM is too slow for generation but fast enough for cache reads.
  - Needs affinity routing, which complicates load balancing.
  - Store as a trie with LRU eviction (from the Character.ai blog).
- **JetStream:** Google's open-source engine. It has prefill engines and generate engines on different slices, with one controller and three threads (prefill, generate, transfer). The Engine interface has `prefill` (tokens to KV), `insert` (KV into the decode batch) and `generate` (one token per batch entry). A PyTorch version exists.

## Appendices
- **A:** empirical layer time for d_model 8192, d_ff 32768 with 2 matmuls per layer. Step time rises slowly until about batch 240, then linearly. The layer is about 600M params sharded 4 ways, so the expected minimum latency is about 365 µs. Throughput rises until about BS240 per data-parallel shard.
- **B:** 2D weight-stationary sharding (from the ESTI paper). Weights are sharded over both F and the reduction dimension E (= d_model). Activations are sharded over E. An AllGather over (yz) runs before the first layer and a ReduceScatter over (x) after. For attention, the n_heads dimension is sharded instead.
- **C:** latency-bound comms (formulas above).
- **D:** speculative sampling.
  - A cheap drafter proposes K tokens and the big model scores them all in one pass.
  - Greedy decoding accepts the longest correct prefix and adds one bonus token when all are correct.
  - It wins latency because decode is not compute-bound, so the extra tokens are scored "for free". It shares KV loads across steps, so it can also be a throughput win at long context.
  - The output distribution is unchanged.
  - Non-greedy decoding uses Metropolis-Hastings-style probabilistic rejection.
  - An embedded drafter head is an option.
  - Chinchilla 70B with a 4B drafter: the best lookahead is 3–4 tokens for XSum. HumanEval benefits from more aggressive speculation.
  - Takeaway: speculation trades throughput for per-token latency, and is win-win when batch is limited.

## Pop quiz (in text)
**Q:** Decode step with batch 4 on a 30B dense model. TPU v5e 4x4, int8 weights, bf16 FLOPs, 8192 context, 100 kB/token KV. What is the lower-bound latency? What about batch 256?
**A:**
- Params are 30e9 B. KV per sequence is 100e3×8192 = 819 MB. 16 chips, each 8.2e11 B/s and 1.97e14 FLOPs/s.
- Batch 4: (4×819e6 + 30e9)/(16×8.2e11) = **2.5 ms**.
- Batch 256: the MLP is compute-bound, so (256×819e6)/(16×8.2e11) + (2×256×30e9)/(16×1.97e14) ≈ **21 ms**.

## Worked problems (Part 7)
Model for these problems: L=64, D=4,096, F=16,384, N=32, K=8, H=256, V=32,128, with tied embeddings.

1. **How many params, and how big is the int8 KV cache per token?**
   - Params = L·D·(3F + 2H(N+K)) + D·V = 64·4096·(3·16384 + 2·256·40) + 4096·32128 ≈ **18.4e9**.
   - KV per token = 2·L·K·H = 2·64·8·256 = **262 kB**.
2. **Largest batch on v5e 4x4 with everything int8, 128k sequences, KV fully sharded? What with K=1?**
   - Each sequence is 262e3×128e3 = 33.5 GB.
   - (16×16e9 − 18.4e9)/33.5e9 = **7**.
   - With K=1 it is 8× more, **about 56**.
3. **Time to load all int8 params on a 4x4?** 18.4e9 B over 16 × 8.2e11 gives about **1.4 ms**. This is the lower bound on per-step latency.
4. **How to shard for prefill and decode on v5e 4x4 with int8 FLOPs, params and activations, and what is the per-step decode latency?** Hints: what ICI looks like on a 4x4, the TP roofline, KV sharding. **No answer is given in the book.**
5. **The same model as an MoE with E=16, k=2:**
   - (1) Total params: 64·4096·(3·16·16384 + 2·256·40) + 4096·32128 = **212e9** (about 12×). Activated: 64·4096·(3·2·16384 + 2·256·40) + 4096·32128 = **31.2e9** (less than 2×).
   - (2) The FLOPs-bound batch grows by E/k: 240×8 = **1920 tokens**.
   - (3) KV per token is unchanged.
   - (4) FLOPs = 2 · activated params · T = **2 × 31.2e9 × T**.
6. **Expert sharding with W_in[E_Z, D_X, F_Y] (X = FSDP, training only).** HBM weight-load time on v5e 8x16 with Y=8, Z=16, and free HBM per TPU? Smallest slice that fits? **No answer is given in the book.**
7. **2D weight-stationary math.**
   - Algorithm: AllGather_YZ the input, matmul W_in[D_X, F_YZ], AllReduce_X, matmul W_out, ReduceScatter_YZ.
   - T_math = 4BDF/(N·C).
   - $T_\text{2D comms}=\frac{2BD}{X W_\text{ici}}+\frac{4BF}{YZ W_\text{ici}}$. The AllReduce costs 2×.
   - With F = 4D, the optimum is X = √(N/8), YZ = √(8N), which gives $\frac{\sqrt{128}BD}{\sqrt N W_\text{ici}}\approx\frac{11.3BD}{\sqrt N W_\text{ici}}$.
   - 1D model parallelism costs 4BD/(3·W_ici). 2D wins when **N > 128·(3/4)² = 72** chips. The general condition is N > 32·(F/D)·(3/4)².
   - Insight: total comms keep falling as chips are added, so latency keeps dropping even when communication-bound.

---

# Part 8: Serving LLaMA 3-70B on TPUs

URL: https://jax-ml.github.io/scaling-book/applied-inference/ (accessed 2026-10-23)

## Section headings
- `#whats-the-llama-serving-story` What's the LLaMA Serving Story?
  - `#thinking-about-throughput` Thinking about throughput
  - `#what-about-prefill` What about prefill?
- `#visualizing-the-latency-throughput-tradeoff` Visualizing the Latency Throughput Tradeoff (interactive slider plots)
- `#worked-problems` Worked Problems

LLaMA 3-70B: L=80, D=8,192, F=28,672, N=64, K=8, H=128, V=128,256.

## Hardware and cost table (Google Cloud pricing as of Feb 2025)

| Chip | bf16 FLOPs/s | USD/hr | FLOPs/$ |
|---|---|---|---|
| H100 | 9.9e14 | $10.8 | 3.3e17 |
| v5p | 4.59e14 | $4.2 | 3.9e17 |
| v5e | 1.97e14 | $1.2 | 5.8e17 |

The book's heuristic is to serve on whatever is cheapest in FLOPs/$, which is v5e, "our current dedicated inference chip". The caveat is that sometimes HBM or ICI matters more. v5e has 16 GB HBM per chip. Its int8 rate is 3.94e14. ICI used in the TP check is 9e10 B/s.

## Q&A in the text (all answered)
1. **int8 KV per token for L3-70B?** 2·K·H·L = 2·8·128·80 = **160 kB**.
   - A 32k-token sequence is 160e3×32,768 = 5.3 GB.
   - BS=240 needs 1.3 TB, so about (70e9 + 1.3e12)/16e9 = **86 v5e chips** just to fit. Compare 70 GB of params.
2. **BS 32, 8192 length, everything int8: total memory and smallest slice?**
   - KV = 160e3×8192×32 = 41.9e9. Params = 70e9. Total **112 GB**.
   - 112e9/16e9 = 7 chips, so a **4x2**. That is tight, and a 4x4 may be needed in practice (or a smaller batch).
3. **Decode latency and throughput on a 4x2 (bf16 FLOPs, fully sharded)? On a 4x4?**
   - B_crit is about 120 (int8 weights, bf16 FLOPs), so the step is memory-bound.
   - Step = 112e9/(8×8.2e11) = **17 ms**. Throughput = 32/0.017 = 1882 tok/s, or **235 tok/s/chip**.
   - ICI check: the bound is Y > 2·F/2200 = 26 with two axes, so 8 chips is fine.
   - 4x4: **8.5 ms**, same per-chip throughput.
4. **Batch for compute-bound matmuls on v5e:**
   - bf16 weights and activations: **240**.
   - int8 weights with bf16 FLOPs: 2BDF/DF = 2B > 240, so **B > 120**.
   - int8 weights with int8 FLOPs (3.94e14): back to **about 240**.
   - int8 weights with bf16 FLOPs is common, because quantizing params losslessly is easier than low-precision arithmetic.
5. **Smallest v5e topology at 8k context, KV treated as negligible:**

| dtype | params | KV/token | min chips | min slice | HBM left for KV | # KV caches @8k |
|---|---|---|---|---|---|---|
| bf16 | 140GB | 324kB | 8.75 | 4x4 = 16 | 116 | 43 |
| int8 | 70GB | 162kB | 4.38 | 4x2 = 8 | 58 | 43 |
| int4 | 35GB | 81kB | 2.81 | 2x2 = 4 | 29 | 43 |

   The batch of 43 is far below 240, so FLOPs utilization is poor.

   [NOTE] This table has two self-inconsistencies, copied as is: KV per token here is 162 kB in int8 versus 160 kB earlier, and the "min chips" values for int8 and int4 do not equal 70/16 and 35/16 exactly.
6. **Latency at the largest batch that fits?** HBM is full, so step = 16 GB/8.2e11 = **19 ms**. With a 512-token median decode, that is about 9 s per decode. Params alone in int4 would give about 10 ms/step.
   - Takeaway: the decode lower bound is the time to load all params from HBM, often within 1.5×. At large batch, KV loading dominates. In the FLOPs-bound regime, the lower bound is 2·params·B/(N·C).
7. **Throughput per chip (QPS) with a 512-token median decode?** B/(step·steps·N) ≈ 43/(0.019·512·N) ≈ 4.42/N:
   - bf16 **0.27**
   - int8 **0.55**
   - int4 **1.11**

   This ignores working memory, so real numbers are probably about half.
8. **Doubled topology?** bf16 on a 4x8 leaves 372 GB for KV, so batch 140 and a throughput of 14.39/N:
   - bf16 on 4x8: **0.44**
   - int8 on 4x4: **0.90**
   - int4 on 2x4: **1.80**

   The smallest topology is not the most performant one when KV size is the limit.
9. **Sharding bf16 on a v5e 4x8 for generation?**
   - The model is comms-bound when Y > F·M_Y/2200. For 2 axes that is 28672·2/2200 = 26, so about 16 chips is fine. A 4x4 works and a 4x8 does not under pure model parallelism.
   - At small batch you can go further: Y = F/(8B). At B=64 that is 28,672/(8·64) = **56**.
   - Check on a 4x8 at B=64:
     - T_ici = 2·64·8192/9e10 = **11 µs**
     - T_hbm = 2·8192·28,672/(32·8.2e11) = **18 µs**
     - T_math = 2·64·8192·28,672/(32·1.97e14) = **4 µs**

     So the step is still HBM-bound. A 4x8 lowers latency but does not raise throughput.
   - int8 and int4 can use pure model parallelism on a 4x8. So quantization also allows a bigger batch before going comms-bound.
   - Tip: maximum useful model parallelism is usually 8–32, depending on d_ff and the number of axes.
10. **Prefill of 8192 tokens on 16 v5e at 40% MFU:** 2·70e9·8192/(16·1.97e14·0.4) = **0.91 s**.
11. **Median prefill 8192, median decode 4096, generate batch 32: evictions per step?**
    - 32/4096 sequences finish per step.
    - 32·(8192+4096)/4096 = **96 tokens evicted per step**. In general, B·(P+G)/G.
12. **Disaggregated prefill:generate server ratio** for 8192 prefill and 512 decode, using 910 ms prefill and 19 ms decode at batch 43 (rounded to 32): P/0.91 = 32·G/(0.019·512), so **P ≈ 3G**. You need about 3× more prefill servers.

## Latency-throughput visualization
Setup: L3-70B, int8 weights, 16-way TP, v5e 4x4.
- Doubling per-token latency can cut per-token cost by about 100×. Latency ranges from 5.5 ms at small batch to 20 ms at large batch.
- At 2k context, throughput plateaus around 1 token/ms/chip at the BS 120 roofline. Longer contexts cannot fit that batch.
- At every sequence length above 2048, KV loading takes longer than FLOPs.
- Takeaway: L3-70B is strongly KV-bandwidth-bound and HBM-bound in nearly every configuration.
- The book includes Python code for the roofline: kv_cache_size(bs) = 2·bs·128·8·80, hbm 8.2e11, flops 1.97e14, latency = max(flops, param load) + KV load.

## Worked problems (Part 8)
**None of these have published answers** (I checked the raw HTML: they contain no answer blocks).
1. FLOPs per token for a LLaMA 3-405B forward pass. Lower bound for one forward pass on N v5e chips if FLOPs-bound, and if comms-bound.
2. LLaMA 3-8B at BS240 with int8 weights and int8 KV. Bytes for params, KV, and peak working activations. Smallest topology.
3. Serving LLaMA 3-405B on v5e with int8 weights, bf16 FLOPs and a hard limit of 15 ms/token. Highest-throughput configuration and theoretical minimum step time.
4. Project. Load LLaMA 3 8B from HF (visualize with TreeScope, count MLP vs attention params). Implement the forward pass, then a KV cache, then separate prefill and decode servers, then Flash Attention in Pallas. Reference: jax-ml/jax-llm-examples/llama3. Watch positional embeddings and causal masking.

---

# Part 9: How to Profile TPU Programs

URL: https://jax-ml.github.io/scaling-book/profiling/ (accessed 2026-10-23)

## Section headings
- `#a-thousand-foot-view-of-the-tpu-software-stack` A Thousand-Foot View of the TPU Software Stack
- `#the-jax-profiler-a-multi-purpose-tpu-profiler` The JAX Profiler: A Multi-Purpose TPU Profiler
  - `#trace-viewer` Trace Viewer
  - `#how-to-read-an-xla-op` How to read an XLA op
  - `#graph-viewer` Graph Viewer
  - `#looking-at-a-realish-example-profile` Looking at a real(ish) example profile
  - `#memory-profile` Memory Profile
- `#worked-problems` Worked Problems

## Core concepts
- **The stack:**
  - JAX → (jax.jit traces) **StableHLO**, a platform-agnostic IR.
  - → XLA lowers to **HLO**, running passes for fusion and layout. This HLO is what you see in a profile.
  - → **LLO** (low-level optimizer IR), which programs the TPU directly: it schedules copies between memories, pushes arrays into the systolic array, and schedules DMAs.
  - → machine code loaded into TPU **IMEM**.
  - The escape hatch for low-level problems is Pallas kernels.
- To get the HLO text: `jax.jit(f).lower(*args, **kwargs).compile().as_text()`.
- Example HLO for `einsum('bf,fd->db')` with f32 × bf16 inputs:
  - `parameter(0/1)`, then `convert` (bf16 to f32), then `ROOT dot(...)` with `lhs_contracting_dims={0}, rhs_contracting_dims={1}`.
  - Mixed dtypes got upcast with a convert op.
- **Profiler usage:**
  - `with jax.profiler.trace("/tmp/tensorboard"): ...; y.block_until_ready()`.
  - View in TensorBoard after `pip install -U xprof` (also `protobuf`), then `%load_ext tensorboard` and `%tensorboard --logdir=/tmp/tensorboard`, or run `tensorboard --logdir=...`. A Perfetto link is provided.
- **Tabs:** Trace Viewer (timeline), Graph Viewer (HLO graph and shardings), Memory Profile and Memory Viewer.
- **Trace Viewer anatomy:**
  - One timeline per TPU core. Usually you only look at TPU:0, since all cores run the same instructions.
  - The **top row "XLA Ops" shows the real TPU ops, named by their HLO names.** The other rows are approximations built from `jax.named_scope`, `jax.named_call` and the Python stack.
  - Repeated blocks correspond to layers.
  - Clicking an op shows its source line and links to the Graph Viewer.
  - Navigation: **W/S zoom, A/D pan**.
- **Reading an XLA op.** Example: `%fusion.3 = bf16[32,32,4096]{2,1,0:T(8,128)(2,1)S(1)} fusion(bf16[32,32,8192]{...S(1)} %fusion.32), kind=kCustom, calls=%all-reduce-scatter.3`
  - Op name: `fusion.3`. A `dot` or `fusion` op holds **at most 1 matmul** plus related pointwise VPU ops.
  - Shape: the output dtype and shape, e.g. bf16[32,32,4096].
  - Layout `{2,1,0:T(8,128)(2,1)}`: the dimension order from most minor to most major (read it right to left), then the tiling.
  - Memory space: **S(1) = VMEM. S(0) (often omitted) = HBM. S(2) and S(3) are other spaces.**
  - Arguments: the input shapes and the producer op (here fusion.32).
- **Tiling:**
  - `f32[3,5]{1,0:T(2,2)}`: the physical layout is [3,5] in (2,2) tiles, row-major within each tile. It is padded to [4,6], about 1.6× the memory.
  - `T(8,128)(2,1)` means two levels of tiling: outer (8,128) and inner (2,1). The inner level is used for bf16 so that loads are always multiples of 4 bytes.
  - XLA may insert retile or re-layout copies at non-trivial cost.
  - Workaround: set input layouts to `AUTO` in `jax.jit`. jit then returns its preferred layout, and you load tensors in that layout to avoid the copies. The book calls this experimental.
- HLO op names seen in this chapter: `parameter`, `convert`, `dot`, `fusion` (`kind=kCustom` or `kind=kLoop`, `calls=%fused_computation.N` or `calls=%all-reduce-scatter.N`), `reduce`, `AllReduce` with `replica_groups={{0,16,32,...}}`, and `copy-done` (in Part 10).
  - The communication op shows up as a **fusion that calls `%all-reduce-scatter.N`**, i.e. a ReduceScatter.
  - **[NOT IN BOOK]** `all-gather`, `collective-permute`, `all-to-all`, `copy-start` and async start/done pairs are not shown or explained in this chapter. Get them from the XProf or XLA docs. From outside knowledge: `jax.lax.ppermute` lowers to `collective-permute`.
- **Worked profile** (fake Transformer, 8 TPU v2 cores, 4-way DP × 2-way MP):
  - FFW up-projection fusion: local bf16[8,1024,8192] × [8192,16384]. Globally that is X[32,1024,8192] × W_in[8192,32768].
  - Expected time: 2·32·1024·8192·32768/(23e12·8) = **95.6 ms**. Measured: **96 ms**, so near-perfect MFU.
  - The small ReduceScatter fusion after the 2nd matmul (`%fusion.1 ... calls=%all-reduce-scatter.1`) runs on a TPU v2 4x2 in one hop at 1.2e11 bidirectional bandwidth. The shard is 2·8·1024·8192 = 128 MB. Expected about **1.1 ms**, measured **1.13 ms**.
  - Attention: W_Q is [8192, 32 heads, 256], Megatron-sharded on heads. The time estimate is left as an exercise.
- **Memory Profile:** for debugging OOMs. In the example, about 7.5 GB holds params and about 8.5 GB is free.
- Practical notes:
  - Colab no longer gives out v2-8 slices. Use Kaggle or GCP.
  - Fake devices: `jax.config.update("jax_num_cpu_devices", 8)` (jax ≥ about 0.4.27). Only useful for toy problems.

## Hardware numbers (Part 9)
- TPU v2: **23e12** FLOPs/s per core, as used in the calculation.
- TPU v2 4x2 ICI: **1.2e11 bidirectional**.

## Worked problems (Part 9)
1. **Diagnose a profile.** **Answer:**
   - The profile is two matmuls: `einsum('wf,bf->bw', w2, einsum('fw,bw->bf', w1, x))`.
   - The profile shows a reduce, two big fusions, and an all-reduce.
   - The first fusion is `%fusion.1 = bf16[4096]{0:T(1024)(128)(2,1)} fusion(bf16[4096,8192]{1,0:T(8,128)(2,1)} %param.1, bf16[8192]{0:T(1024)(128)(2,1)} %reduce.6), kind=kLoop, calls=%fused_computation.1`. That gives the per-shard op bf16[8192] × bf16[4096,8192] → bf16[4096].
   - The final AllReduce has `replica_groups={{0,16,32,48,64,80,96,112},...}`, which means **8-way model parallelism**.
   - True shapes: **bf16[8,8192] × bf16[32768,8192] → bf16[8,32768]**.
2. **Benchmark the mock Transformer** with naive GSPMD partitioning and fix it with `jax.lax.with_sharding_constraint`. What is the best MFU? Then read from the profile: the sharding strategy, batch size, d_model and d_ff, the attention vs MLP time split, and the roofline split.
   - Reference: initially about 184 ms/layer, optimized 67 ms/layer.
   - Update note: the compiler has improved. The naive version is now about 90 ms/layer and the optimized one about 80 ms/layer. **No full answer is given.**

---

# Part 10: Programming TPUs in JAX

URL: https://jax-ml.github.io/scaling-book/jax-stuff/ (accessed 2026-10-23). The chapter says much of it is taken from the JAX docs on sharding.

## Section headings
- `#how-does-parallelism-work-in-jax` How Does Parallelism Work in JAX?
  - `#auto-sharding-mode` Auto sharding mode
  - `#explicit-sharding-mode` Explicit sharding mode
  - `#manual-sharding-mode-via-shard-map` Manual sharding mode via shard_map
- `#worked-problems` Worked Problems

## Core concepts
Three modes:

| Mode | View | Explicit sharding? | Explicit collectives? |
|---|---|---|---|
| Auto | Global | no | no |
| Explicit | Global | yes | no |
| Manual | Per-device | yes | yes |

- **Auto:** `jax.jit` with `AxisType.Auto` mesh axes. XLA's **Shardy** partitioner propagates shardings and inserts AllGather, ReduceScatter and AllReduce.
  - Mesh API: `jax.make_mesh(axis_sizes, axis_names, axis_types)` plus `jax.set_mesh(mesh)`. Arrays are created with `device=jax.NamedSharding(mesh, jax.P(...))`, or `jax.device_put(x, jax.P(...))`, or a jitted creator with `out_shardings`.
  - Example: In[B_X, D_Y] × W[D_Y, F] with `out_shardings=P('X', None)` yields Out{U_Y}, then an AllReduce.
  - The HLO shows a local `fusion` with bf16[2,1024] (local) × bf16[8192,1024], then `ROOT %AllReduce`. The weight arrives through `%copy-done`.
  - Fix bad propagation with `jax.lax.with_sharding_constraint(x, P(...))`. The book calls this "compiler tickling" and says it is "about 60% of JAX parallel programming in the automatic partitioning world". A typical symptom: a giant AllGather taking 80% of the profile.
- **Explicit ("sharding in types"):** `AxisType.Explicit`. JAX itself propagates shardings according to per-op rules. `jax.typeof(x)` shows e.g. `float32[8@X,2@Y]`.
  - When the result is ambiguous (einsum or reshape with sharded contracting dimensions), JAX raises a **trace-time error**: "Contracting dimensions are sharded and it is ambiguous how the output should be sharded. Please specify the output sharding via the `out_sharding` parameter."
  - You resolve it with `out_sharding=P('X','Y')`, which induces a ReduceScatter, or `P('X', None)`, which induces an AllReduce.
  - Auto and Explicit compose via `jax.sharding.auto_axes` and `explicit_axes`.
- **Manual: `jax.shard_map(f, in_specs=..., out_specs=..., mesh=...)`.**
  - It gives a **local per-device view**. All communication is written explicitly: `jax.lax.all_gather`, `psum` (AllReduce), `pmean`, `ppermute`, `all_to_all`, `axis_index`, `axis_size`.
  - Example: `slice_and_average` on a 512-element array over a (2,4) mesh, with `in_specs=P(('x','y'))` and `out_specs=P()`. Each device sees 64 elements and takes x[:4]. `pmean` over ('x','y') returns mean(x[:4], x[64:68], x[128:132], ...). This is hard to express under jit's global view.
  - `jax.lax.pcast(accum, ('X','Y'), to='varying')` is used to mark an accumulator as device-varying.
- **Collective matmul** (Wang et al. 2023): A[B_X, D_Y] ·_D W[D, F_Y] → Out[B_X, F_Y].
  - The naive version AllGathers A and cannot overlap that with compute.
  - The collective version loops over Y shards: matmul the local A chunk with the matching rhs chunk (`dynamic_slice_in_dim` at `(idx+i)%axis_size*chunk`), then `ppermute` A one step left and accumulate. It uses `fori_loop(0, axis_size-1, unroll=True)` plus a final chunk.
  - Results: default jit takes **311 µs** with a big blocking AllGather. The collective matmul takes **244 µs** and has no AllGather. Unsharded-contracting baseline: **224 µs**.

## Worked problems (Part 10)
1. A: f32[S_X, D_Y].
   - (a) Per-shard mean, returning an [X,Y] array, with both jit and shard_map. Profile for unwanted comms ("there shouldn't be [any], but sometimes XLA adds it anyway").
   - (b) Compute `roll(x, shift, axis=0) - x` within each X shard using shard_map.
   - **Answer:** shard_map `lambda x: x.mean(keepdims=True)` with `in_specs=out_specs=P('X','Y')`. The jit version is `x.reshape(X, S/X, Y, D/Y).mean(axis=(1,3))` with `out_shardings=P('X','Y')`. Part 2: shard_map `jnp.roll(x, shift, axis=0) - x`. The jit equivalent reshapes to (X, S/X, -1), rolls along axis=1, and reshapes back. Both are tested equal on an 8·64×8 array over a (4,2) mesh.
2. MoE: W f32[E_X, D, F], A f32[S_X, D], routing B int32[S_X]. Compute Out[i] = A[i] @ W[B[i]].
   - (1) A local version that never materializes [S,D,F]. Options: loop over experts with a mask; sort tokens into a padded [E,S,D] buffer (why size S? because one expert could get every token); or `jax.lax.ragged_dot`.
   - (2) Profile what plain jit does.
   - (3) jit will likely AllGather_X the activations. Rewrite with shard_map: first with all_gather, then ragged with `all_to_all` inside a `while_loop`, avoiding any [E,S,D] array.
   - (4) Extend to top-k with B int32[S_X, k].
   - **Partial answer:** a `lax.scan` over experts computing `A @ W[e]` masked by `(B==e)`. It wastes compute, which is why sort plus `ragged_dot` is better. For part 3, pseudocode only: sort x by assignment, then loop over chunks of 128 doing `all_to_all` and a local matmul, then concat.
3. Collective-matmul variants, **no answer given**:
   - (1) An AllReduce collective matmul A[B_X,D_Y]·W[D_Y,F] → Out[B_X,F]. Tile over the output dimension and use psum. Note that it may not beat the baseline because of how XLA handles it.
   - (2) A ReduceScatter collective matmul Tmp[B_X,F_Y]·W2[F_Y,D] → Out[B_X,D_Y], the down-projection. Hint: permute the result as you accumulate.
   - (3) Combine the AllGather and ReduceScatter versions into a full overlapped MLP block and compare with jit.
4. Make the AllReduce and ReduceScatter collective matmuls **bidirectional**. How much faster are they? **No answer given.**

---

# Part 11: Conclusions and Further Reading

URL: https://jax-ml.github.io/scaling-book/conclusion/ (accessed 2026-10-23)

Headings: `#acknowledgments`, `#further-reading`, `#feedback`. The chapter has no technical content and no problems.
- Credits:
  - James Bradbury, Reiner Pope, Noam Shazeer and Blake Hechtman originated many of the ideas.
  - Sholto Douglas wrote the first version and the narrative.
  - Jacob Austin led the polish and release.
  - Anselm Levskaya and Charlie Chen made the figures.
  - Charlie Chen wrote the inference section.
  - Roy Frostig handled publication and editing.
- Further reading list, useful for the course bibliography:
  - "TPU Deep Dive" (blog)
  - "Domain specific architectures for AI inference"
  - "A Domain-Specific Supercomputer for Training Deep Neural Networks" (TPU paper)
  - "Making Deep Learning Go Brrrr From First Principles"
  - "Writing TPU Kernels with Pallas"
  - "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance"
  - JAX "Distributed arrays and automatic parallelization"
  - Rafi Witten's High Performance LLMs 2024 class
  - **Efficiently Scaling Transformer Inference (arXiv 2211.05102)**, the basis of the inference chapter
  - HF Ultra-Scale Playbook
  - "Transformer Inference Arithmetic"
  - Stanford CS336
  - Stas Bekman's ML Engineering Handbook
  - ezyang's blog
  - "The Anatomy of Collective Communication", which has a better write-up of N-D and GPU collectives than the book
- Feedback goes to Jacob Austin (jacobaustin123 [at] gmail) or via GitHub issues and PRs.

---

# Part 12: How to Think About GPUs (most important for this course)

URL: https://jax-ml.github.io/scaling-book/gpus/ (accessed 2026-10-23). Published Aug. 18, 2025. The chapter covers NVIDIA only.

## Section headings
- `#what-is-a-gpu` What Is a GPU?
  - `#memory` Memory
  - `#summary-of-gpu-specs` Summary of GPU specs
  - `#gpus-vs-tpus-at-the-chip-level` GPUs vs. TPUs at the chip level
  - `#quiz-1-gpu-hardware` Quiz 1: GPU hardware
- `#networking` Networking
  - `#at-the-node-level` At the node level
  - `#quiz-2-gpu-nodes` Quiz 2: GPU nodes
- `#beyond-the-node-level` Beyond the node level
  - `#quiz-3-beyond-the-node-level` Quiz 3
- `#how-do-collectives-work-on-gpus` How Do Collectives Work on GPUs?
  - `#intra-node-collectives`, `#cross-node-collectives`, `#quiz-4-collectives`
- `#rooflines-for-llm-scaling-on-gpus` Rooflines for LLM Scaling on GPUs
  - `#data-parallelism`, `#tensor-parallelism`, `#expert-parallelism`, `#pipeline-parallelism`, `#examples`, `#tldr-of-llm-scaling-on-gpus`, `#quiz-5-llm-rooflines`
- `#acknowledgements-and-further-reading`
- `#appendix-a-how-does-this-change-with-gb200` Appendix A: GB200
- `#appendix-b-more-networking-details` Appendix B: More networking details

## Chip-level concepts
- A GPU is many matmul-specialized **SMs** connected to HBM: **H100 has 132 SMs, B200 has 148.** "Note how much this looks like a TPU!"
- Naming collision:
  - On a GPU, the **Tensor Core** is the matmul sub-unit of an SM, equivalent to the MXU.
  - On a TPU, the **TensorCore** is the umbrella unit containing the MXU, VPU, and so on, equivalent to the SM.
- A TPU has at most 2 TensorCores. A GPU has more than 100 SMs, each much weaker but independent, so a GPU can run hundreds of separate tasks. They still have to coordinate in practice because they share a limited L2.
- An SM has 4 **subpartitions**. Each one holds a Tensor Core, 16k 32-bit registers, and a SIMD/SIMT **Warp Scheduler** whose lanes are the **CUDA cores** (32 fp32 cores per subpartition, plus fewer int32 and fp64 cores).
- **H100: 990 bf16 TC TFLOPs/s vs 66 TFLOPs/s from CUDA cores.**
- FLOPs per Tensor Core per cycle: 990e12 at 1.76 GHz over 132 SMs gives 7.5e12/1.76e9/4 ≈ 1024 bf16 FLOPs/cycle per TC, roughly an 8×8×8 matmul. The book says this is a guess.
  - V100: 256, A100: 512, H100: 1024.
  - B200 is probably about 2048: 2250e12/(148·4·1.86e9).
- H100 has 2× the fp8 FLOPs of fp16.
- Tensor Cores have grown every generation since Volta:
  - Ampere TCs are fed by one warp, Hopper by a full SM (warpgroup), Blackwell by 2 SMs.
  - Blackwell adds **TMEM** (256 kB per SM) because the accumulator no longer fits in registers or SMEM.
- **SIMT vs SIMD:**
  - GPU CUDA cores (since V100) are SIMT. Each thread has its own instruction pointer, but a warp still executes one op per cycle. Divergence is handled by masking, which silently degrades performance.
  - CUDA cores can access individual floats and keep per-thread state. The VPU works only on contiguous blocks.
  - SMs schedule up to **64 resident warps** and switch between them to hide memory latency. **"TPUs are generally single threaded by comparison."**
- **Memory hierarchy:**
  - **Registers:** 16,384 32-bit words per subpartition, so 256 KiB per SM. A thread can use at most 256 registers. At 256 registers per thread only 8 warps fit (256·1024/(4·32·256)).
  - **SMEM/L1:** 256 kB per SM, usable as programmer "shared memory" or as a hardware cache. It holds activations and TC inputs.
  - **L2:** about 50 MB shared on H100, split into two halves of 25 MB joined by a slower link. It is not programmer-controlled ("spooky action at a distance"). Measured bandwidth is about **5.5 TB/s** (NVIDIA does not publish it). That is about 1.6× HBM, full-duplex, so about 3× effective bidirectional. **TPU VMEM is about 2× larger with about 40 TB/s.**
  - **HBM:** from 32 GB (Volta) to 192 GB (B200). Bandwidth is **about 3.35 TB/s on H100 and 9 TB/s on B200** in the text. [NOTE] The spec table lists B200 at 8.0e12, and the quizzes use 8e12.

## GPU spec tables (copied exactly)

| GPU | Gen | Clock | SMs/chip | SMEM/SM | L2/chip | HBM/chip |
|---|---|---|---|---|---|---|
| V100 | Volta | 1.25GHz/1.38GHz | 80 | 96kB | 6MB | 32GB |
| A100 | Ampere | 1.10GHz/1.41GHz | 108 | 192kB | 40MB | 80GB |
| H100 | Hopper | 1.59GHz/1.98GHz | 132 | 256kB | 50MB | 80GB |
| H200 | Hopper | 1.59GHz/1.98GHz | 132 | 256kB | 50MB | 141GB |
| B200 | Blackwell | ? | 148 | 256kB | 126MB | 192GB |

All generations have 256 kB of registers per SM. Blackwell adds 256 kB of TMEM per SM.

| GPU | HBM BW/chip | bf16/fp16 FLOPs/s | fp8/int8 FLOPs/s | fp4 FLOPs/s |
|---|---|---|---|---|
| V100 | 9.0e11 | — | — | — |
| A100 | 2.0e12 | 3.1e14 | 6.2e14 | — |
| H100 | 3.4e12 | 9.9e14 | 2.0e15 | — |
| H200 | 4.8e12 | 9.9e14 | 2.0e15 | — |
| B200 | 8.0e12 | 2.3e15 | 4.5e15 | 9.0e15 |

B100 is excluded because it was not mass-produced (reportedly throttled on heat and power).

**GPU↔TPU cheat sheet:**

| GPU | TPU | What |
|---|---|---|
| SM | Tensor Core | core "cell" |
| Warp Scheduler | VPU | SIMD vector unit |
| CUDA Core | VPU ALU | SIMD ALU |
| SMEM (L1) | VMEM | fast on-chip memory |
| Tensor Core | MXU | matmul unit |
| HBM (GMEM) | HBM | high-capacity memory |

**1:1 counts, H100 vs TPU v5p:**

| GPU | TPU | H100 # | TPU v5p # |
|---|---|---|---|
| SM | Tensor Core | 132 | 2 |
| Warp Scheduler | VPU slots | 528 | 8 |
| SMEM (L1) | VMEM | 32MB | 128MB |
| Registers | VRegs | 32MB | 256kB |
| Tensor Core | MXU | 528 | 8 |

## GPU vs TPU at the chip level: the book's conclusions
- GPUs began as graphics hardware and have become more TPU-like over time. They are general-purpose, so they "just work" on new tasks more often and lean far less on the compiler. The flip side is that roofline performance is harder to reason about and harder to reach.
- **Modularity:**
  - A TPU TensorCore has one big VPU made of 4 independently programmable 8×128 units (4096 ALUs).
  - An H100 has 132×4 = 528 independent 32-wide SIMD units (about 16k ALUs).
  - TPUs are cheaper to build and simpler to understand, but put the burden on the compiler. With a single thread of control, the compiler must pipeline every memory load and every MXU and VPU operation by hand.
  - GPUs launch independent kernels per SM but can suffer L2 thrashing or uncoalesced loads that are hard to see.
  - **"TPUs can often get closer to peak roofline performance with less work."**
- **Per-chip power and price:** an H200 has close to 2× the FLOPs/s of a TPU v5p and 1.5× the HBM. On Google Cloud it costs about **$10/hr vs about $4/hr for v5p**. TPUs rely more on networking many chips together.
- **Cache:** TPUs have much more VMEM than GPUs have SMEM plus TMEM. This can make TPUs faster for LLM inference if weights can be stored in or prefetched into VMEM.

## Quiz 1: GPU hardware (all answered)
1. **fp32 CUDA cores?**
   - H100: 132·4·32 = **16,896**. B200: 148·4·32 = **18,944**.
   - TPU v5p: 2 TCs × 4 ALUs per lane × (8×128) = **8192**, about half, at a similar clock.
2. **Vector fp32 FLOPs/s on H100** (132 SMs, 1.59 GHz, 1 op per ALU per cycle): 132·4·32·1.59e9 = **26.9 TFLOPs/s**, **33.5** with boost. That is half the spec-sheet number because FMA counts as 2. Tensor Cores give about **30×** more (990 TFLOPs/s).
3. **Peak matmul intensity:**
   - H100 fp16: 990e12/3.35e12 = **295**. B200: 2250e12/8e12 = **281**. Both are close to v5e's 240, so you need a batch of about 280 to be compute-bound.
   - fp8 doubles these to **590 and 562**, effectively constant again if weights are also fp8.
4. **B200 matmul times:**
   - fp16[64,4096]×[4096,8192] is bandwidth-bound (below 281). Bytes = 2·64·4096 + 2·4096·8192 + 2·64·8192 = 69e6, over 8e12, gives **8.6 µs**, likely 10–12 µs in practice.
   - fp16[512,4096]×[4096,8192] is compute-bound: 2·512·4096·8192/2.3e15 = **15 µs**, likely about 20 µs in practice.
5. **Total L1/SMEM and registers on H100?** 132 × 256 kB ≈ **33 MB** of each, **66 MB** together.
   - Compare the "about half the 120MB of a modern TPU's VMEM". [NOTE] The cheat-sheet table says v5p VMEM is 128 MB.
   - TPU has only 256 kB of registers in total. VMEM latency is lower than SMEM latency, so spills and fills on TPU are cheap.
6. **B200 clock from 80 TFLOPs/s fp32 vector** (NVIDIA figure): 18,944 cores × 2 (FMA) = 37,888 FLOPs/cycle, so 80e12/37,888 ≈ **2.1 GHz**. Plausible, because B200s are liquid-cooled.
7. **Adding two fp32[N] vectors on H100:**
   - N FLOPs over 12N bytes gives AI = **1/12**.
   - Vector hardware intensity is 33.5e12/3.35e12 = **10**. The book notes this stays roughly constant across generations: H100 is 33.5/3.5 and B200 is 80/8.
   - At N=1024 only 1024 cores (8 SMs) are used, about 16× slower if compute-bound.
   - T = 12N/3.35e12 = N/2.8e11.
   - N = 65,536 gives a roofline of **0.23 µs**, measured about **1.5 µs** in JAX (latency-bound). N = 1024³ gives a roofline of **3.84 ms**, measured **4.1 ms**.
   - [NOTE] The question asks about N=1024, but the answer computes N=65,536.

## Networking: GPU vs TPU
- **TPU:** a 2D or 3D torus with nearest-neighbor links only. Messages pass through every intermediate chip, so only uniform communication patterns work well. The upside is a constant number of links per chip, so pods scale "arbitrarily large" without losing bandwidth.
- **GPU:** a hierarchical, tree-based switched network.
  - **Nodes** of 8 GPUs (72 for GB200 NVL72) are all-to-all within one hop over **NVLink** via **NVSwitches**.
  - Nodes connect into **Scalable Units (SUs)** over InfiniBand or Ethernet through per-GPU NICs, and then through higher-level switches.
  - "Node" is ambiguous: it can mean the NVLink domain or the set of GPUs on one CPU host. In NVL72 there are 72 GPUs per NVLink domain but still 8 per host. The book uses "node" to mean the NVLink domain.
- The H100 diagram shows about **450 GB/s egress per GPU inside the NVLink domain** and **400 GB/s egress per node into IB**.
- How the book contrasts the two link types: NVLink is "something like a souped-up PCIe connection, with low latency and protocol overhead but not designed for scalability/fault tolerance", while InfiniBand is "more like Ethernet, designed for larger lossy networks".
- **H100 node:** 4 NVSwitches, with GPUs attached in a 5+4+4+5 link pattern. NVLink 4.0 runs **25 GB/s full-duplex per link** (25 GB/s each way). 18 links × 25 = **450 GB/s per GPU**. An NVSwitch has up to 64 ports, so an 8×H100 node has 64·25e9·4 = **6.4 TB/s** of switch capacity.

| NVLink Gen | NVSwitch Gen | GPU Gen | NVLink BW (GB/s, FD) | Ports/GPU | Node GPU↔GPU BW (GB/s FD) | Node size | NVSwitches/node |
|---|---|---|---|---|---|---|---|
| 3.0 | 2.0 | Ampere | 25 | 12 | 300 | 8 | 6 |
| 4.0 | 3.0 | Hopper | 25 | 18 | 450 | 8 | 4 |
| 5.0 | 4.0 | Blackwell | 50 | 18 | 900 | 8/72 | 2/18 |

### Quiz 2: GPU nodes (answered)
1. **Total bandwidth of an 8×H100 node.** The switches offer 4 × (64×25e9 = 1.6 TB/s) = 6.4 TB/s, but each GPU is capped at 450 GB/s, so 8×450 = **3.6 TB/s** is the peak.
2. **Bisection bandwidth of an 8×H100 node.** 4 GPUs per half, counting both directions: 8×450 = **3.6 TB/s**, matching NVIDIA's figure.
3. **AllGather of B bytes on an 8×H100 node**, with bf16[D_X, F], D = 4096, F = 65,536.
   - Cost: (N−1)·B/(N·W_uni) ≈ B/450e9.
   - B = 4096·65536·2 = 536e6 bytes.
   - Exact: 536e6·7/(8·450e9) = **1.04 ms**. Approximation: **1.19 ms**. In practice about **1.5 ms**, because it may be latency-bound.

## Beyond the node: the DGX SuperPod
- Reference 1024-GPU H100 SuperPod:
  - 128 nodes (sometimes 127), each with 8 GPUs, 8 × 400 Gbps CX7 NICs (one per GPU) and 4 NVSwitches.
  - **SU = 32 nodes (256 GPUs) under 8 leaf IB switches.**
  - All cabling is IB NDR at 50 GB/s full-duplex. Switches are 64-port NDR at 50 GB/s per port. The IB switches have 2× the bandwidth of the NVSwitches (64 ports at 400 Gbps).
  - **SuperPod = 4 SUs plus 16 spine switches.** Totals: 512 NVSwitches + 32 leaf + 16 spine = **560 switches**.
  - The topology is a **fat tree with full bisection bandwidth above the node (400 GB/s per node)**, so AllReduce bandwidth is roughly constant across the scale-out network.
  - Meta trained LLaMA-3 on a different design: Ethernet, a 3-layer fabric, and an oversubscribed top level.

| Level | GPUs | Switches/unit | Type | BW/unit (TB/s FD) | GPU↔GPU (GB/s FD) | Fat-tree BW (GB/s FD) |
|---|---|---|---|---|---|---|
| Node | 8 | 4 | NVL | 3.6 | 450 | 450 |
| Leaf | 256 | 8 | IB | 12.8 | 50 | 400 |
| Spine | 1024 | 16 | IB | 51.2 | 50 | 400 |

- **TPU comparison:** a TPU v5p has **about 90 GB/s egress per link, or 540 GB/s egress across all axes of its 3D torus**. These are not point-to-point links, so only uniform patterns get that bandwidth. Even so, TPU-to-TPU bandwidth is much higher and scales up to 8960 TPUs.
- The GPU fabric can grow arbitrarily large by adding switches or layers, at the cost of latency and expensive switches.
- Takeaway: inside an H100 node, each GPU has 450 GB/s. Beyond the node, it drops to 400 GB/s node-to-node.
- **GB200 NVL72:**
  - 72 GPUs in one NVLink domain at 900 GB/s GPU-to-GPU.
  - Node egress to leaf is 4·18·400/8 = **3.6 TB/s**, 9× the H100 node (matching 9× the GPUs).
  - Cross-node collective bandwidth can now exceed what is available within the node.

| Node type | GPUs/node | GPU egress | Node egress |
|---|---|---|---|
| H100 | 8 | 450e9 | 400e9 |
| B200 | 8 | 900e9 | 400e9 |
| GB200 NVL72 | 72 | 900e9 | 3600e9 |

### Quiz 3 (answered)
1. **Bisection bandwidth of the 1024-GPU pod.**
   - Each node has 8×400 Gbps = 400 GB/s to the leaf.
   - A leaf switch is 3.2 TB/s (64 × 400 Gbps), but only 32 ports face down. Across the SU that is 32·400/8 = 12.8 TB/s for 32 nodes, which is 400 GB/s per node.
   - SU to spine: 8·16·2 × 400 Gbps = 12.8 TB/s per SU, which is 400 GB/s per node.
   - 16 spine switches × 3.2 TB/s = 51.2 TB/s over 128 nodes, again 400 GB/s per node.
   - Every component is sized exactly for a full fat tree.
2. **Scaling to 2048 GPUs?** Use 8 SUs and 32 spine switches. Leaf ports are already exhausted, so run 1 NDR cable per spine instead of 2, which gives the same bandwidth with fewer ports. For **4096 GPUs** you run out of ports and need another level: NVIDIA's "core switches", with **128 spine and 64 core switches**.

## Collectives on GPUs
They are implemented in **NCCL** (open source) and **NVSHMEM**. NCCL picks among several algorithms. The book models an ideal switched tree.

- **Intra-node AllGather and ReduceScatter:** a ring at full GPU-to-GPU bandwidth, the same as on TPU.

  $$T_\text{AG/RS}=\frac{\text{bytes}\,(N-1)}{N\,W_\text{GPU egress}}\to\frac{\text{bytes}}{W_\text{GPU egress}}$$

  Time is total bytes over per-GPU egress, the same form as on TPU.

  - H100: about B/450e9. B200: about B/900e9.
  - AllReduce costs 2× (RS + AG) unless in-network reduction is used.
  - A tree reduction takes log N hops instead of N−1, for latency, at the same total cost.
- **Pop Quiz 1:** AllGather(bf16[B_X, F]) with B = 1024, F = 16,384 on 8×H100: 2BF/450e9 ≈ **75 µs**, or more precisely **65 µs** with the (N−1)/N factor.
- **Intra-node AllToAll:**

  $$T=\frac{B(N-1)}{W N^2}\approx\frac{B}{WN}$$

  Every GPU sends directly to every other GPU, so per-GPU egress is about B/N.

  - **TPU: B/(4W). GPU within a node: B/(8W). "2X theoretical speedup".**
  - Ragged or top-k AllToAll (MoE): scale by min(k/N, 1). Random routing gives about (N−1)/N·min(k/N,1)·B/(WN). The exact form is (1 − ((Z−1)/Z)^K)·(Z−1)/Z.
  - Takeaway: ≈ B/(8·W). Top-k: (B·k)/(64·W).
- **Pop Quiz 2:** AllToAll_{X→N}(bf16[B_X, N]) on 8×H100, with V = 2BN bytes. Dense ≈ V/(W·N). With only 4 of 8 entries nonzero (k/N = ½), the cost is **about half**.
- **Empirical:** an 8×H100 AllReduce (SHARP off) peaks around **370 GB/s** bus bandwidth, and only for messages around 10 GB per device.
  - Bus BW is defined as 2·W·(N−1)/N, and the blue curve as 2·bytes·(N−1)/(N·runtime).
  - LLaMA-3 70B's MLP AllReduce (bf16[8192,3584] = 58 MB under 8-way sharding) reaches only about **150 GB/s**.
  - **TPUs reach peak bandwidth at much smaller message sizes** (Appendix B plots).
  - Takeaway: plan for about 370 GB/s, not 450.
- **SHARP** (in-network reduction, Hopper and later):
  - The theory is T = bytes/W_egress, halving AllReduce cost (exactly B ingress, with no 1/N factor).
  - **In practice it gives about 30% more bandwidth** versus the 75% predicted (NCCL 2.27.5), reaching about **480 GB/s**.
  - "Since pure AllReduces are fairly rare in LLMs, this is not especially useful."
- **Cross-node:**

  $$T_\text{AG/RS}\approx\frac{\text{bytes}}{W_\text{node egress}}\underset{H100}{=}\frac{\text{bytes}}{400\text{e9}}$$

  Across nodes, the node's IB egress (400 GB/s) is the bottleneck. Picture one big ring through all nodes.

  - In general, T = max(bytes/W_GPU, bytes/W_node).
  - Precise version: bytes · max_i[(D_i−1)/(D_i·W_link,i)].
    - Node: 450e9·8/7 = **514 GB/s**.
    - Leaf: 400e9·32/31 = **413 GB/s**.
    - Spine: 12.8e12·4/3 = **17.1 TB/s**.
    - So **413 GB/s** at the leaf is the binding level. With SHARP it is about 400 GB/s.
  - AllReduce costs 2× unless SHARP. Not every provider has SHARP-enabled IB switches.
  - **Cross-node AllToAll** is not hierarchical: an N-way AllToAll spanning M = N/8 nodes costs B(M−1)/(M²·W_node) ≈ B/(M·W_node). That is effectively 50 GB/s rather than 400 GB/s. B/(8·450e9) inside one node becomes B/(2·400e9) across 2 nodes, **a more than 4× degradation**.

SuperPod summary:

| Level | GPUs | Degree | Switch BW (TB/s FD) | Cable BW (TB/s FD) | Collective BW (GB/s) |
|---|---|---|---|---|---|
| Node | 8 | 8 | 6.4 | 3.6 | 450 |
| Leaf (SU) | 256 | 32 | 25.6 | 12.8 | 400 |
| Spine | 1024 | 4 | 51.2 | 51.2 | 400 |

"Collective bandwidth" here means bisection bandwidth × 2/N.

- **Reduction over an array sharded on another axis**, AllReduce_X(A[I_Y, J]{U_X}):
  - On TPU, the cost falls by 1/Y.
  - On GPU it depends on which axis is inner and whether Y spans nodes:
    - T_node = bytes/W_GPU · 1/min(Y, D_node)
    - T_scale-out = bytes/W_node · D_node/max(D_node, Y)
    - T = the max of the two.
  - Y < 8 helps only at the node level, with no net gain. Y > 8 speeds things up in proportion to the number of nodes spanned.
  - Precise form: bytes · max_i[(D_i−1)/(D_i·max(Y, S_{i−1})·W_link,i)].
- **Pop Quiz 3:** AllGather_X(bf16[D_X, F_Y]) over one SU (256 GPUs), with Y inner.
  - Y ≤ 8: 2DF·31/(32·400e9).
  - Y > 8: 2DF·256/(Y·12.8e12) = 2DF/(Y·50 GB/s).
  - Exactly 8-way model parallelism cuts the node-level cost by 8 but leaves the total unchanged.
  - Takeaway: the outer reduction is cheaper by the number of nodes the inner axis spans.

### Quiz 4: Collectives (answered)
1. **Node-switch and top-switch traffic for an AllGather in one SU with M nodes of N GPUs.**
   - Node switch: ingress B/M from its GPUs, egress B/M to the spine, ingress B(M−1)/M from the spine, egress N·(B − B/MN) = NB − B/M to its GPUs. Totals: B in, BN out. Egress-bound: T = BN/W_node = **B/450e9**.
   - Spine: ingress B, egress B(M−1). T = B(M−1)/(M·400e9).
2. **Single-node SHARP AllReduce, switch traffic:** ingress B(N−1), egress B (partial shards), ingress B again (residuals), egress B(N−1) (multicast). Total BN in and BN out, so T = **B/W_egress** exactly.
3. **SHARP AllReduce(bf16[D, F_Y]{U_X}) in one node of N GPUs:** each GPU egresses B(X−1)/XY, and so on. Total NB/Y, so T = **2DF/(Y·W_link)**, which falls with Y. Across more than one node, leaving the node switch requires all B bytes, not B/Y, because the shards must stay separate.
4. **The same with Y = 256 (AllReduce at the spine):** 51.2 TB/s over 4 SUs is 12.8 TB/s per SU, so as little as **2DF/12.8e12** with SHARP.
5. **Exact AllGather of B bytes over exactly 2 nodes:**
   - Node level: 7B/(8·450e9) = **B/514e9**.
   - Cross-node: B·(2−1)/(2·400e9) = **B/800e9**.
   - **The node level is the bottleneck**, which is why DeepSeek v3 uses 2-way data parallelism.

## Rooflines for LLM scaling on GPUs
The analysis covers only the MLP: x[B,D]·W_in[D,F]·W_out[F,D], where B is global tokens. W_collective means GPU egress inside a node and node egress across nodes.
- **DP / ZeRO (backward pass):**

  $$T_\text{math}=\frac{8BDF}{XC},\quad T_\text{comms}=\frac{8DF}{W_\text{collective}}\ \Rightarrow\ \frac{B}{X}>\frac{C}{W_\text{collective}}$$

  Per-GPU token batch must exceed FLOPs/s over collective bandwidth.

  - Within a node: 990e12/450e9 = **2200**. SU or spine: 990e12/400e9 = **2475**.
  - "Quite a bit higher than on a TPU, where the number is 850 with all three axes."
  - LLaMA-3 on 16k H100s would need at least 40M tokens per batch; it used 16M.
  - DeepSeek v3 on 2048 H800s (300 GB/s): 990e12/300e9 = 3300 per GPU, so about 6.7M tokens; it used 4M.
  - SHARP plus pure DP halves this in theory. In practice it is about 30%, which mostly offsets not reaching the spec bandwidth.
  - **MoE:** multiply by E/k, so B/X > (E/k)·C/W. OpenAI OSS (k=4, E=128): 32·2475 = **79,200** across nodes.
  - **Small X** (2-node DP): the (X−1)/X factor gives B/N > α(X−1)/X, about **1237**, which is why 2-way DP shows up often.
  - Takeaway: about 2500 tokens per GPU on H100 or B200, ×E/k for MoE.
- **Tensor parallelism** (AllGather plus ReduceScatter on activations, forward):

  $$T_\text{math}=\frac{4BDF}{YC},\ T_\text{comms}=\frac{4BD}{W}\Rightarrow Y<\frac{F\,W_\text{collective}}{C}$$

  The maximum TP degree is F divided by the critical intensity.

  - F/2200 within a node, F/2475 across nodes.
  - F ≈ 28000 (LLaMA-3) gives about 11-way, which rounds to 8, one node.
  - Spanning exactly 2 nodes doubles the bandwidth, so 16-way is possible, with F > 2475·(Y−8), up to about 19-way in theory.
  - Takeaway: Y > F/2475 makes TP comms-bound, so TP stays intra-node or at most 2 nodes.
- **Expert parallelism** (W_in[E_Z, D, F], 2× AllToAll):

  $$T_\text{math}=\frac{4BkDF}{ZC},\quad T_\text{comms}=\frac{4BD(Z-8)}{WZ}\min\Big(\frac{8k}{Z},1\Big)$$

  - Two viable regimes:
    - k > Z/8 with F > α(Z−8)/k: small EP (about 2 nodes) with small F.
    - Z ≫ k with F > 8α: large F, and EP can go arbitrarily large (up to E-way).
  - Examples: DeepSeek v3 (small F, restricted cross-node EP), or large-F models doing heavy cross-node EP alongside TP.
  - Takeaway: if F < 8·C/W_node, EP spans 1–2 nodes. Otherwise EP can span up to E nodes cheaply.
- **Pipeline parallelism:**

  $$T_\text{total PP}=\frac{2BD}{W N_\text{MB}}(N_\text{MB}+N_\text{stages}-2),\quad T_\text{per-layer}\approx1.5\frac{2BD}{W N_\text{layers}}$$

  Pipelining's communication is basically free.

  - Downsides:
    - Code complexity. It does not fit GSPMD-style auto-parallelism, and zero-bubble schedules interleave forward and backward.
    - It interacts badly with FSDP and DP. ZeRO-3 must AllGather weights for every microbatch, and gradient reductions wait for the last microbatch.
    - Bubbles and stage imbalance, plus activation transfers on the critical path.
  - Zero-bubble schedules can mostly avoid bubbles.
- **Latency caveat:** GPUs struggle to reach full AllReduce bandwidth even with large messages. Multi-node EP AllToAlls may not reach even 50%, so keep TP and EP within few nodes.
- **DeepSeek V3** (2048 H800s): 64-way EP across 8 nodes, 16-way PP, 2-way ZeRO-1 DP.
  - Batch 4096·15360 = 62,914,560 tokens, about 30k per GPU. Sparse model: k=8, E=256.
  - 64 × 16 = 1024-way model parallelism, so the AllReduce happens at the spine level and is only 2-way, giving 2/(2−1) = 2× bandwidth.
- **LLaMA-3** (16k GPUs, 16M batch, about 1k per GPU): 8-way TP within a node, 16-way PP, 128-way ZeRO-1 DP. The 16-way PP cuts the DP AllReduce cost by 16×.
- **TLDR:**
  - DP and FSDP need about 2500 tokens per GPU.
  - TP is limited to about 8-way, i.e. one NVLink domain, or up to 72 with NVL72.
  - Multi-node model parallelism (PP + EP + TP) cuts the FSDP cost.
  - PP needs zero-bubble scheduling and usually means ZeRO-1 rather than ZeRO-3.
  - Recipes:
    - Small dense models: FSDP.
    - Large dense models: 1–2-node TP + many-node PP + DP.
    - MoE: prefer EP over TP. If F > 8C/W_node, use lots of multi-node EP; otherwise about 2-node EP.

### Quiz 5: LLM rooflines (answered)
1. **B200 DGX** (900 GB/s inside the node, 400 GB/s scale-out, 990 → 2250 TFLOPs, 2.25×):
   - Inside a node, rooflines stay about the same. TP: 2250e12/900e9 = **2500**, so Y < F/2500.
   - Across nodes it gets worse. The DP critical batch becomes 2250e12/400e9 = **5625**.
   - GB200's larger egress fixes this.
2. **LLaMA-3 70B training** (bf16, fp32 Adam):
   - (a) 2 B weights + 8 B optimizer per param = 700 GB. With 80 GB GPUs that is at least **9 GPUs**, i.e. 2 nodes.
   - (b) 6·70e9·15e12 = 6.3e24 FLOPs. 4096 GPUs × 990e12 × 0.45 MFU ≈ 1.8e18 FLOPs/s, so 3.5e6 s ≈ **40 days**.
   - (c) TP limit is about F/1995 = 28672/1995 = 14.37, i.e. 8-way in practice. [NOTE] The book uses 1995 here, not 2200.
     - 8-way TP + 512-way DP: 700 GB/8 = 87.5 GB per GPU, which **does not fit**.
     - ZeRO-3 + 8-way TP: the per-GPU batch is 4e6/4096 = 976, below the limit, and 2× worse because weights move. **No.**
     - 8-way PP: each model-parallel shard spans 8 nodes, giving 8 × 400 = 3200 GB/s. The roofline is 990e12/3200e9 = **309**, so **this works**.
3. **Megatron-LM MFU figure** (sequence length 4096):
   - Per-GPU tokens: 16B: 192·4096/192 = **4096**. 70B: 384·4096/768 = **2048**. 314B: 1536·4096/3072 = **2048**.
   - These sit around the FSDP threshold of **2,472** (spine).
   - The 16-way and 64-way model sharding (PP + TP) of the 70B and 314B runs gives 2× and 8× spine bandwidth. They become compute-bound at about **1k** and **300** tokens per step respectively.

## Appendix A: GB200
- NVLink 5 runs at 900 GB/s.
- B200 still uses 8-GPU nodes. GB200 (B200 + Grace) offers NVL72 and, in theory, up to 576 GPUs per NVLink domain.
- Inside the node, doubling the bandwidth does not change much, because FLOPs also doubled. Expert parallelism gets easier.
- Node egress is 4·18·400/8 = **3.6 TB/s** (vs 400 GB/s). Cross-node rooflines improve by about 4×, given FLOPs per chip also doubled. Now the node level can become the bottleneck.
- **GH200** pairs 1 H200 with 1 Grace CPU. A **GB200 system** pairs 2 B200s with 1 Grace. The CPU attaches over full-bandwidth **NVLink C2C**, so host memory is as reachable as another GPU's HBM, which helps when offloading params.

## Appendix B: More networking details
- An NVLink4 switch has 64 ports (2 physical lanes each) and a large crossbar. **TPUs use dynamically reconfigurable optical switches with mirrors.**
- Node level: 4 × 1.6 TB/s = 6.4 TB/s of switch capacity, but GPUs cap it at 8 × 450 = 3.6 TB/s.
- SU level: 8·32·400/8 = 12.8 TB/s of node egress, which matches 8 × 1.6 TB/s of switch capacity.
- Spine: 32·16·400·2/8 = 51.2 TB/s, which matches 16 × 3.2 TB/s. All 64 spine ports face down.
- Per GPU: 450 GB/s at the node level, 50 GB/s at the SU and spine levels.
- Empirical plots: AllReduce on 8×H100 vs TPU v5p 4x4x4 (one axis), and AllGather on 8×H100 vs TPU v5e 8x16 (one axis). They show TPUs saturating at smaller message sizes. These are figures only, with no numbers in the text.
- Ragged AllToAll: the approximation min(K/Z)·(Z−1)/Z vs the exact (1−((Z−1)/Z)^K)·(Z−1)/Z. They agree except at small Z.

---

# Cross-chapter TPU vs GPU comparisons (for the course)

| Topic | TPU (book) | GPU (book) |
|---|---|---|
| Compute cells | 1–2 TensorCores per chip (v5p: 2) | 132 SMs (H100), 148 (B200) |
| Matmul units | v5p: 8 MXUs | H100: 528 Tensor Cores |
| Vector units | v5p: 8 VPU slots, 8192 ALUs per chip | H100: 528 warp schedulers, 16,896 fp32 cores |
| Fast on-chip memory | VMEM, v5p 128 MB (also "120MB"), about 40 TB/s | SMEM 33 MB + L2 50 MB (about 5.5 TB/s) |
| Registers | 256 kB in total | 32 MB (256 kB per SM) |
| Programming model | single-threaded, SIMD, compiler-pipelined | SIMT, up to 64 resident warps per SM, hardware-scheduled |
| bf16 matmul intensity | v5e 240 | H100 295 (Part 7 says about 280), B200 281 |
| Decode B_crit | 240 bf16, 120 int8-weights | about 280 (H100) |
| Interconnect | 2D/3D torus, nearest neighbor, about 90 GB/s per link and 540 GB/s total egress (v5p), up to 8960 chips, optical reconfigurable switches | NVLink switched node (450 GB/s H100, 900 GB/s B200) + IB fat tree at 400 GB/s per node (H100/B200), 3.6 TB/s (NVL72) |
| AllToAll | B/(4W) | B/(8W) inside a node (2× better); cross-node ≈ B/(M·W_node), more than 4× worse |
| Reduction over an already-sharded array | cost × 1/Y | only helps if the inner axis spans more than one node |
| Saturating message size | small | about 370 GB/s only at about 10 GB messages; 58 MB gets about 150 GB/s |
| DP critical batch per chip | 850 ("with all three axes") | 2200 (node) / 2475 (scale-out); B200 5625 |
| Price (GCP, Feb 2025) | v5e $1.2/hr, 5.8e17 FLOPs/$; v5p $4.2/hr | H100 $10.8/hr, 3.3e17 FLOPs/$; H200 about $10/hr |
| Book's verdict | closer to roofline with less work; more VMEM, which helps inference | "just works" more often and leans less on the compiler, but harder to reason about and to reach roofline |

# Gaps and failures
- Part 7 Questions 4 and 6 have **no published answers**.
- Part 8 Worked Problems 1–3 have **no published answers** (verified in the raw HTML), and Problem 4 is an implementation project.
- Part 9 Problem 2 and Part 10 Problems 3–4 have no answers. Part 10 Problem 2 has only a partial answer.
- Figures and interactive plots (trace screenshots, Pareto plots, empirical bandwidth curves) were not readable as text. I captured only the numbers stated in the prose and captions.
- The profiling chapter does **not** cover `all-gather`, `collective-permute`, `all-to-all` or async `-start`/`-done` HLO ops. Only `fusion` calling `all-reduce-scatter`, `AllReduce`, `reduce`, `dot`, `convert`, `parameter` and `copy-done` appear. Source those from the XProf or XLA docs.
- Internal inconsistencies, flagged inline and kept as written:
  - H100 B_crit: 280 (Part 7) vs 295 (Part 12).
  - B200 HBM bandwidth: 9 TB/s (prose) vs 8.0e12 (table).
  - TPU VMEM: 120 MB vs 128 MB.
  - L3-70B int8 KV per token: 160 kB vs 162 kB.
  - The TP divisor 1995 in Part 12 Quiz 5 Question 2.
  - Part 12 Quiz 1 Question 7 asks about N=1024 but the answer computes N=65,536.
