"""Numbers used in Chapter 7 (inference rooflines). Mirrored by src/lib/inference.ts.

Decode step model (Scaling Book ch. 7):
  T_step = B * KV_bytes / BW  +  max(2 * B * active_params / C, param_bytes / BW)
with weights and KV caches sharded evenly over N chips (BW and C summed over chips).

    python3 scripts/ch07_inference.py
"""
from specs import CHIPS, GB
from model import LLAMA3_70B, LLAMA3_8B, LLAMA3_405B, SB7_DENSE, SB7_MOE


def decode_step(m, chip, n, B, S, w_bytes=2, kv_bytes=2, math="bf16"):
    c = CHIPS[chip]
    C = getattr(c, f"flops_{math}") * n
    bw = c.hbm_bw * n
    p_bytes = m.params * w_bytes
    kv = B * S * m.kv_bytes_per_token(kv_bytes)
    t_kv = kv / bw
    t_flops = 2 * B * m.active_params / C
    t_w = p_bytes / bw
    fits = p_bytes + kv <= c.hbm_bytes * n
    return dict(t=t_kv + max(t_flops, t_w), t_kv=t_kv, t_flops=t_flops, t_w=t_w, fits=fits, mem=(p_bytes + kv) / n)


if __name__ == "__main__":
    print("B_crit = (C_math / W_hbm) * bytes_per_weight / 2:")
    for k in ("v5e", "v6e", "7x", "8i", "H100"):
        c = CHIPS[k]
        r = c.flops_bf16 / c.hbm_bw
        r8 = (c.flops_fp8 / c.hbm_bw / 2) if c.flops_fp8 else None
        print(f"  {c.name:16} bf16 {r:5.0f} | 8-bit weights, bf16 math {r/2:5.0f} | 8-bit math {r8 if r8 is None else round(r8)}")

    # SB7 pop quiz: 30B dense, int8 weights, bf16 math, 8192 ctx, 100 kB/token KV, v5e 4x4.
    P, kvtok, S = 30e9, 100e3, 8192
    for chip, n in (("v5e", 16), ("7x", 4)):
        c = CHIPS[chip]
        for B in (4, 256):
            kv = B * kvtok * S
            t = kv / (n * c.hbm_bw) + max(2 * B * P / (n * c.flops_bf16), P / (n * c.hbm_bw))
            print(f"  30B on {n} x {c.name}, B={B}: {t*1e3:.2f} ms (fits {P + kv <= n*c.hbm_bytes})")

    # SB7 worked problems 1-3.
    m = SB7_DENSE
    print(f"\nSB7 model: {m.params/1e9:.1f}B params, int8 KV/token {m.kv_bytes_per_token(1)/1e3:.0f} kB")
    seq = m.kv_bytes_per_token(1) * 128_000
    print(f"  128k sequence {seq/1e9:.1f} GB; max batch on v5e 4x4 int8: {(16*16e9 - m.params)/seq:.1f}; with K=1: {(16*16e9 - m.params)/(seq/8):.0f}")
    print(f"  param load on v5e 4x4: {m.params/(16*CHIPS['v5e'].hbm_bw)*1e3:.2f} ms")
    print(f"  MoE (E=16,k=2): total {SB7_MOE.params/1e9:.0f}B active {SB7_MOE.active_params/1e9:.1f}B; bf16 B_crit x E/k = {240*8}")

    # LLaMA-3-70B decode on Ironwood: 8 chips, bf16 / fp8 weights, 32k context.
    print("\nLLaMA-3-70B decode on 8 Ironwood chips, 32k context:")
    for w, kvb, label in ((2, 2, "bf16 weights, bf16 KV"), (1, 1, "fp8 weights, fp8 KV")):
        for B in (16, 64, 128):
            r = decode_step(LLAMA3_70B, "7x", 8, B, 32768, w, kvb)
            print(f"  {label}, B={B:3}: step {r['t']*1e3:5.2f} ms (KV {r['t_kv']*1e3:.2f}, weights {r['t_w']*1e3:.2f}, flops {r['t_flops']*1e3:.2f}), "
                  f"{B/r['t']/8:6.0f} tok/s/chip, mem/chip {r['mem']/GB:.0f} GB fits={r['fits']}")

    # Decode model-parallel limit before ICI binds: Y > F / (B * beta), beta = W_hbm / W_ici.
    for k in ("v5e", "7x"):
        c = CHIPS[k]
        beta = c.hbm_bw / c.W_axis
        print(f"{c.name}: beta = W_hbm/W_ici = {beta:.1f}; F=28672, B=32: model-parallel up to {28672/(32*beta):.0f}-way before ICI binds")

    # Latency-bound TP collectives (per hop chunk < link one-way * 1us).
    for k, w1 in (("v5e", 45e9), ("7x", 100e9)):
        print(f"{k}: 8-way TP collective latency-bound below {8*w1*1e-6/1e3:.0f} kB total; "
              f"BS=16, D=8192 int8 activation = {16*8192/1e3:.0f} kB")

    # Prefill TTFT at 40% MFU.
    for chip, n in (("v5e", 16), ("7x", 8)):
        c = CHIPS[chip]
        t = 2 * LLAMA3_70B.params * 8192 / (n * c.flops_bf16 * 0.4)
        print(f"Prefill 8192 tokens, 70B on {n} x {c.name} at 40% MFU: {t*1e3:.0f} ms")

    # Disaggregation ratio (SB8 Q12): prefill 0.91 s per 8192, decode 19 ms/step at batch 32, 512 tokens.
    # Prefill servers finish P/0.91 requests/s; decode servers finish 32 G / (0.019 * 512) requests/s.
    print(f"SB8 Q12 prefill:decode servers = {0.91*32/(0.019*512):.1f}")
