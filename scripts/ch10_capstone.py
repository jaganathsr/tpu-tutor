"""Model answer for the Chapter 10 design review.

Brief: serve LLaMA-3-70B on Ironwood; 8k-token prompts, conversations up to 32k
tokens; TTFT <= 500 ms; time per output token <= 30 ms; maximise tokens/s per chip.

    python3 scripts/ch10_capstone.py
"""
from specs import CHIPS
from model import LLAMA3_70B as M
from ch07_inference import decode_step

c = CHIPS["7x"]
for n in (4, 8):
    for w, kv, math, label in ((2, 2, "bf16", "bf16 / bf16 KV"), (1, 1, "fp8", "fp8 / fp8 KV")):
        free = n * c.hbm_bytes - M.params * w
        bmax = int(free / (32768 * M.kv_bytes_per_token(kv)))
        best = None
        for b in range(1, bmax + 1):
            r = decode_step(M, "7x", n, b, 32768, w, kv, math)
            if r["t"] <= 0.030:
                best = (b, r)
        if best:
            b, r = best
            tp_lat = 4 * M.L * (n / 2) * 1e-6
            print(f"{n} chips, {label}: max batch by memory {bmax}; best under 30 ms: B={b}, step {r['t']*1e3:.1f} ms "
                  f"(KV {r['t_kv']*1e3:.1f}, weights {r['t_w']*1e3:.1f}, flops {r['t_flops']*1e3:.1f}); "
                  f"{b/r['t']/n:.0f} tok/s/chip; TP collective latency floor {tp_lat*1e3:.2f} ms")
        else:
            print(f"{n} chips, {label}: nothing fits under 30 ms (max batch by memory {bmax})")

# Prefill on a separate 8-chip server at 40% MFU, bf16 math.
for n in (4, 8):
    t = (2 * M.params * 8192 + 4 * 8192 * M.N * M.H * M.L * 8192 * 0.5) / (n * c.flops_bf16 * 0.4)
    print(f"TTFT, 8k prompt on {n} Ironwood chips at 40% MFU: {t*1e3:.0f} ms")
