"""Numbers used in Chapter 4 (transformer accounting).

    python3 scripts/ch04_transformer.py
"""
from model import MODELS, LLAMA3_70B, SB7_MOE, Model
from specs import CHIPS, GiB

if __name__ == "__main__":
    for m in MODELS:
        print(f"{m.name}: total {m.params/1e9:.2f}B, active {m.active_params/1e9:.2f}B; "
              f"MLP {m.L*m.mlp_params/1e9:.2f}B, attn {m.L*m.attn_params/1e9:.2f}B, vocab {m.vocab_params/1e9:.2f}B; "
              f"KV/token bf16 {m.kv_bytes_per_token(2)/1e3:.1f} kB, int8 {m.kv_bytes_per_token(1)/1e3:.1f} kB")

    m = LLAMA3_70B
    print(f"\nLLaMA-3-70B training FLOPs per token 6N = {6*m.params:.3g}; 15T tokens = {6*m.params*15e12:.3g}")
    for chip, mfu in (("v5p", 0.4), ("7x", 0.4)):
        c = CHIPS[chip]
        pods = 8960 if chip == "v5p" else 9216
        days = 6 * m.params * 15e12 / (pods * c.flops_bf16 * mfu) / 86400
        print(f"  one {c.name} pod ({pods} chips) at {mfu:.0%} MFU: {days:.0f} days")

    for S in (8192, 32768, 131072):
        kv = m.kv_bytes_per_token(2) * S
        print(f"  KV for one {S}-token sequence, bf16: {kv/1e9:.2f} GB")
    free = CHIPS["7x"].hbm_bytes - m.params * 2 / 8   # weights bf16 over 8 chips
    print(f"  Ironwood: 70B bf16 over 8 chips leaves {free/1e9:.0f} GB/chip; 32k-token bf16 sequences per chip: "
          f"{free/(m.kv_bytes_per_token(2)*32768):.1f}")

    # Attention share of FLOPs (F=4D, MHA): T/(8D)
    print("\nAttention FLOPs equal MLP+projection FLOPs when T = 8D (F=4D, MHA): D=8192 -> T=", 8 * 8192)
    print("Self-attention equals QKVO projections when T = 2D: D=4096 ->", 2 * 4096)

    # GQA decode attention intensity -> G (bf16 Q and KV).
    for G in (1, 8):
        S = 8192
        ai = (S * G) / (S + G)
        print(f"Decode attention intensity, G={G}, S={S}: {ai:.2f} FLOP/B")

    # MoE critical batch: B > (C/W) * beta * E/k.
    for chip, beta, label in (("v5e", 0.5, "int8 weights, bf16 math"), ("7x", 0.5, "fp8 weights, bf16 math"),
                              ("7x", 1.0, "fp8 weights, fp8 math")):
        c = CHIPS[chip]
        C = c.flops_fp8 if "fp8 math" in label else c.flops_bf16
        ridge = C / c.hbm_bw
        for E, k, name in ((256, 8, "DeepSeek-V3 shape"), (16, 2, "ch.7 MoE")):
            print(f"  {c.name}, {label}, {name} (E={E}, k={k}): B > {ridge*beta*E/k:.0f}")

    # Training memory.
    print(f"\nAdam training state for 70B: {10*m.params/1e12:.2f} TB (10 B/param)")
    acts = 2 * m.D * 4e6 * 2 * m.L       # SB6: 2 checkpoints per layer, bf16, 4M tokens
    total = 2 * m.params + 8 * m.params + acts
    print(f"  activations at 4M tokens, 2 checkpoints/layer: {acts/1e12:.1f} TB; total {total/1e12:.1f} TB")
    for chip in ("v5p", "7x"):
        c = CHIPS[chip]
        print(f"  chips to hold it on {c.name}: {total/c.hbm_bytes:.0f}")
    remat = 2 * 20 * 4e6 * 64 * 8192
    print(f"SB4 remat example (BT=4M, L=64, D=8192, 20 saves/layer): {remat/1e12:.0f} TB")

    # SB4 Q1 model: D=4096, F=4D, V=32000, L=64, MHA, int8 KV.
    q1 = Model("SB4 Q1", L=64, D=4096, F=16384, N=32, K=32, H=128, V=32000)
    print(f"\nSB4 Q1: params {q1.params/1e9:.1f}B, attention fraction {q1.L*q1.attn_params/q1.params:.2f}, "
          f"KV/token int8 {q1.kv_bytes_per_token(1)/2**10:.0f} KiB")
    # SB4 Q7: DeepSeek-V3 MFU.
    avail = 1.513e15 * 2.79e6 * 3600
    need = 6 * 37e9 * 14.8e12
    print(f"SB4 Q7 DeepSeek-V3 MFU: {need/avail:.1%}")
    print(f"SB7 MoE: total {SB7_MOE.params/1e9:.0f}B, active {SB7_MOE.active_params/1e9:.1f}B")
