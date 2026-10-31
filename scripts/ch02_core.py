"""Numbers used in Chapter 2 (anatomy of a TensorCore).

    python3 scripts/ch02_core.py
"""
from specs import CHIPS


def systolic_utilization(M, K, N, size):
    """Weight-stationary array of size x size, one tile W[K, N] (K, N <= size)
    preloaded, M activation rows streamed. Activation x[m, k] reaches cell (k, n)
    at cycle m + k + n, so the last output leaves after M + K + N - 2 cycles
    (excluding the K-cycle weight preload). Utilization = useful MACs over
    cells x cycles."""
    cycles = M + K + N - 2
    return (M * K * N) / (size * size * cycles), cycles


if __name__ == "__main__":
    c = CHIPS["7x"]
    per_mxu = c.flops_bf16 / 4
    print(f"Ironwood per MXU (2 TC x 2 MXU): {per_mxu/1e12:.0f} TFLOP/s")
    implied_clock = per_mxu / (256 * 256 * 2)
    print(f"  if 1 MAC/cell/cycle: implied clock {implied_clock/1e9:.2f} GHz (SB2 says ~2.2 GHz -> 2 MACs/cell/cycle, UNVERIFIED)")
    v5e = CHIPS["v5e"]
    print(f"v5e implied MXU clock (4 x 128x128, 1 MAC/cell/cycle): {v5e.flops_bf16/(4*128*128*2)/1e9:.2f} GHz (derived)")

    vpu = 8 * 128 * 4 * 2.2e9
    print(f"Ironwood VPU per core f32: {vpu:.2e} FLOP/s; bf16 packed {2*vpu:.2e}; MXU/VPU(bf16) = {c.flops_bf16/2/(2*vpu):.0f}x")

    # VMEM-resident roofline, Ironwood: VMEM ~7x HBM (SB2 prose).
    print(f"Ironwood ridge vs HBM {c.flops_bf16/c.hbm_bw:.0f}; vs VMEM (~7x HBM BW) ~{c.flops_bf16/(7*c.hbm_bw):.0f} FLOP/B")

    # SB2 Q4: int8 [16384, 4096] weights x int8 [B, 4096] on v5e.
    D, F = 4096, 16384
    C8 = 3.94e14
    for name, bw in (("HBM", 8.2e11), ("VMEM (22x HBM, SB2 Q4)", 22 * 8.2e11)):
        # 2*B*D*F >= C8/bw * (D*F + B*D + B*F)  -> solve for B
        r = C8 / bw
        b = r * D * F / (2 * D * F - r * (D + F))
        print(f"SB2 Q4 {name}: compute-bound above B = {b:.0f}")

    # SB2 Q1: 200B bf16 model, time to read all params.
    params_bytes = 400e9
    for k, n in (("v6e", 32), ("7x", 32), ("7x", 4)):
        ch = CHIPS[k]
        fits = params_bytes / n <= ch.hbm_bytes
        print(f"200B bf16 on {n} x {ch.name}: {params_bytes/n/ch.hbm_bw*1e3:.2f} ms per full weight read (fits: {fits})")

    print("\nSystolic utilization, one tile, weight-stationary:")
    for M, K, N, S in ((8, 256, 256, 256), (256, 256, 256, 256), (2048, 256, 256, 256),
                       (2048, 64, 256, 256), (2048, 128, 128, 256), (2048, 128, 128, 128)):
        u, cyc = systolic_utilization(M, K, N, S)
        print(f"  M={M:5} K={K:3} N={N:3} array={S}: {cyc:5} cycles, utilization {u:.1%}")

    # Padding waste: a dimension of 100 on a 128- or 256-wide array.
    for d, s in ((100, 128), (100, 256), (320, 256), (64, 256)):
        padded = -(-d // s) * s
        print(f"  dim {d} padded to {padded} on {s}-wide array: {d/padded:.0%} useful")
