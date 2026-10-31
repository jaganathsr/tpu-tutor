"""Numbers used in Chapter 1 (roofline section). Re-run after editing specs.py.

    python3 scripts/ch01_roofline.py
"""
from specs import CHIPS

BYTES = {"bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5, "fp4": 0.5}


def matmul(chip, B, D, F, w="bf16", a="bf16", math="bf16"):
    c = CHIPS[chip]
    C = getattr(c, f"flops_{math}")
    flops = 2 * B * D * F
    nbytes = BYTES[a] * B * D + BYTES[w] * D * F + BYTES[a] * B * F
    denom = 2 * D * F * c.hbm_bw - C * BYTES[a] * (D + F)
    b_star = C * BYTES[w] * D * F / denom if denom > 0 else float("inf")
    return dict(flops=flops, bytes=nbytes, ai=flops / nbytes, ridge=C / c.hbm_bw,
                t_math=flops / C, t_hbm=nbytes / c.hbm_bw, b_star=b_star)


def show(label, r):
    print(f"{label}: AI {r['ai']:.1f} FLOP/B, ridge {r['ridge']:.1f}, "
          f"T_math {r['t_math']*1e6:.1f} us, T_hbm {r['t_hbm']*1e6:.1f} us, "
          f"bytes {r['bytes']:.4g}, flops {r['flops']:.4g}, compute-bound above B = {r['b_star']:.0f}")


if __name__ == "__main__":
    # LLaMA-3-70B MLP up-projection shape [SB6]: D = 8192, F = 28672.
    D, F = 8192, 28672
    show("7x bf16, B=64", matmul("7x", 64, D, F))
    show("7x int8 weights, bf16 math, B=64", matmul("7x", 64, D, F, w="int8"))
    show("7x fp8 everything, B=64", matmul("7x", 64, D, F, w="fp8", a="fp8", math="fp8"))
    show("v5e bf16, B=64", matmul("v5e", 64, D, F))
    show("H100 bf16, B=64", matmul("H100", 64, D, F))
    show("8i fp8 everything, B=64", matmul("8i", 64, D, F, w="fp8", a="fp8", math="fp8"))
    # Ratio of compute-bound batch to the simple ridge (B << D, F approximation).
    for k in ("v5e", "7x", "H100"):
        c = CHIPS[k]
        print(f"{c.name}: ridge {c.flops_bf16 / c.hbm_bw:.1f}")
