"""Numbers used in Chapter 8 (the TPU serving stack).

    python3 scripts/ch08_serving.py
"""
from specs import CHIPS
from model import LLAMA3_70B, LLAMA3_405B

KB = 1e3


def step(m, chip, n, B, S, w=1, kvb=1, math="bf16"):
    c = CHIPS[chip]
    C = getattr(c, f"flops_{math}") * n
    bw = c.hbm_bw * n
    t_kv = B * S * m.kv_bytes_per_token(kvb) / bw
    return t_kv + max(2 * B * m.active_params / C, m.params * w / bw)


if __name__ == "__main__":
    v5e = CHIPS["v5e"]
    m = LLAMA3_70B
    # SB8 Q1-Q3 (book uses 70e9 params, 160 kB/token int8 KV).
    kv = 2 * 8 * 128 * 80
    print(f"SB8 Q1: int8 KV/token {kv/KB:.0f} kB; 32k seq {kv*32768/1e9:.1f} GB; BS240 at 32k {240*kv*32768/1e12:.2f} TB "
          f"-> {(70e9 + 240*kv*32768)/16e9:.0f} v5e chips to fit")
    tot = kv * 8192 * 32 + 70e9
    print(f"SB8 Q2: BS32 8k int8: {tot/1e9:.0f} GB -> {tot/16e9:.1f} chips (4x2)")
    for n in (8, 16):
        t = tot / (n * 8.2e11)
        print(f"SB8 Q3: {n} v5e: step {t*1e3:.1f} ms, {32/t:.0f} tok/s, {32/t/n:.0f} tok/s/chip")
    # SB8 Q9: bf16 on 4x8, B=64.
    D, F = 8192, 28672
    print(f"SB8 Q9: T_ici {2*64*D/9e10*1e6:.0f} us, T_hbm {2*D*F/(32*8.2e11)*1e6:.0f} us, T_math {2*64*D*F/(32*1.97e14)*1e6:.0f} us")
    print(f"SB8 cost table: v5e {1.97e14*3600/1.2:.2g} FLOPs/$, v5p {4.59e14*3600/4.2:.2g}, H100 {9.9e14*3600/10.8:.2g}")

    # SB8 Q3 (no book answer): 405B, int8 weights, bf16 math, v5e, 15 ms/token limit, 8k context, int8 KV.
    m = LLAMA3_405B
    print(f"\n405B: {m.params/1e9:.0f}B params, int8 KV/token {m.kv_bytes_per_token(1)/KB:.0f} kB")
    for n in (36, 64, 128, 256):
        floor = m.params / (n * v5e.hbm_bw)
        free = n * v5e.hbm_bytes - m.params
        bmax = int(free / (8192 * m.kv_bytes_per_token(1)))
        best = max((b for b in range(1, bmax + 1) if step(m, "v5e", n, b, 8192) <= 0.015), default=0)
        tput = best / step(m, "v5e", n, best, 8192) / n if best else 0
        beta_ici = v5e.hbm_bw / v5e.W_axis
        print(f"  {n:3} v5e: floor {floor*1e3:5.1f} ms, max batch by memory {bmax:4}, best batch under 15 ms {best:4}, "
              f"{tput:5.0f} tok/s/chip; ICI ok while n < F/(B*beta) = {m.F/(max(best,1)*beta_ici):.0f}")
    for n in (8, 16):
        floor = m.params / (n * CHIPS['7x'].hbm_bw)
        print(f"  {n} Ironwood, fp8 weights: floor {floor*1e3:.1f} ms")

    # tpu-inference blog numbers (B-vllmtpu).
    print("\nvLLM TPU blog: 3.6x (Llama-3.1-8B, v6e-1) and 2.1x (70B, v6e-8) over the PyTorch/XLA backend; "
          "~20% from JAX lowering; ~10% from RPA v3 over v2")
    # KV packing.
    for bits in (16, 8, 4):
        print(f"KV packing for {bits}-bit: {32 // bits} values per 32-bit sublane")
