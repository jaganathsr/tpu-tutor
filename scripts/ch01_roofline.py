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


# ---- Generation growth (flagship training chips), indexed to TPUv2 ----
# bf16 dense TFLOP/s and HBM GB/s per chip. Sources: P-v23 Tbl 3 (v2, v3),
# P-v4 / D-v4 (v4), D-v5p, D-7x, J-hw (8t bf16, provisional), G-8dd (8t HBM).
GENERATIONS = [
    ("TPUv2", 2017, 46, 700),
    ("TPUv3", 2018, 123, 900),
    ("TPU v4", 2020, 275, 1200),
    ("TPU v5p", 2023, 459, 2765),
    ("TPU7x", 2025, 2307, 7380),
]


def generation_growth():
    f0, b0 = GENERATIONS[0][2], GENERATIONS[0][3]
    rows = []
    for name, year, tf, bw in GENERATIONS:
        rows.append((name, year, tf / f0, bw / b0, tf * 1e12 / (bw * 1e9)))
    return rows


def network_roofline(C, link_oneway):
    """SB1 #network-communication-rooflines: X[B,D] @ Y[D,F] with D split over
    2 chips. Each chip does BDF FLOPs then exchanges 2BF bytes of partial sums.
    Compute-bound iff D > 2C / W."""
    return 2 * C / link_oneway


if __name__ == "__main__":
    print("\nGrowth since TPUv2 (FLOPs x, HBM BW x, ridge FLOP/B):")
    for r in generation_growth():
        print(f"  {r[0]:8} {r[1]}  {r[2]:6.1f}x  {r[3]:5.1f}x  ridge {r[4]:.0f}")
    print(f"TPUv1 ridge (int8 ops per weight byte, P-v1 §4): 92e12 / 34e9 = {92e12/34e9:.0f}")
    print(f"Network roofline, v5e (book, 4.5e10 one-way): D > {network_roofline(1.97e14, 4.5e10):,.0f}")
    print(f"Network roofline, Ironwood (100 GB/s one-way, derived from D-7x): D > {network_roofline(2.307e15, 100e9):,.0f}")
    print(f"Ironwood fp8 peak / v2 bf16 peak: {4614/46:.0f}x ; HBM {7380/700:.1f}x  (P-Micro26: ~100x, ~10x)")
