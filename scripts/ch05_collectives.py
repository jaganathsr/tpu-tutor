"""Numbers used in Chapter 5 (sharded matmuls and collectives).

    python3 scripts/ch05_collectives.py
"""
from specs import CHIPS


def ag(V, W, axes=1):
    """Bidirectional-ring AllGather or ReduceScatter: V / (W_ici * axes)."""
    return V / (W * axes)


if __name__ == "__main__":
    W5e, W5e_uni = 9e10, 4.5e10        # book, v5e
    W7 = CHIPS["7x"].W_axis             # 200 GB/s, docs-derived

    # SB3 pop quiz: AllGather_Y([E_Y, F]) on v5e mesh {X:8, Y:4}; Y=4 has no wraparound on v5e.
    E, F = 2048, 8192
    per_dev = 2 * (E // 4) * F
    print(f"SB3 pop quiz (a): per device {per_dev/1e6:.1f} MB; naive V/W = {4*per_dev/W5e*1e6:.0f} us; "
          f"no wraparound, 3 hops of one shard each way-less: 3*{per_dev/1e6:.1f}MB/45GB/s = {3*per_dev/W5e_uni*1e6:.0f} us (measured 680)")
    small = 2 * (256 // 4) * 256
    print(f"SB3 pop quiz (b): shard {small/1e3:.0f} kB -> {small/W5e_uni*1e6:.2f} us bandwidth per hop; 3 hops ~3 us (measured ~8)")

    # SB3 Q2 on v4 4x4x4 (W = 9e10 bidi), B=1024, D=4096 bf16.
    B, D = 1024, 4096
    W4 = 9e10
    print(f"SB3 Q2 (a) AllGather_X([B_X, D_Y]): {2*B*D/(W4*4)*1e6:.1f} us")
    print(f"SB3 Q2 (b) AllGather_XY: {2*B*D/(2*W4)*1e6:.1f} us")
    print(f"SB3 Q2 (c) AllReduce_Z([B_X, D_Y]{{U_Z}}): {4*B*D/(16*W4)*1e6:.1f} us")

    # Latency threshold per hop.
    for name, w1 in (("v5e", 4.5e10), ("Ironwood", 1e11)):
        print(f"Latency-bound below {w1*1e-6/1e3:.0f} kB per hop on {name}")

    # ICI intensity for strategy comparison (SB3 Q4), v5p book vs Ironwood docs-derived.
    for name, C, W in (("v5p (book)", 4.59e14, 1.8e11), ("v5p (docs)", 4.59e14, 2e11), ("Ironwood", CHIPS['7x'].flops_bf16, W7)):
        a = C / W
        print(f"{name}: C/W_ici = {a:,.0f}; AllGather-first (S1) compute-bound if B > {a:,.0f}; "
              f"matmul-then-AllReduce (S2) beats S1 when D > {2*a:,.0f} (for B > C/W)")

    # One LLaMA-3-70B MLP layer's weights gathered (FSDP) over Ironwood axes.
    w = 3 * 8192 * 28672 * 2
    for axes in (1, 2, 3):
        print(f"FSDP AllGather of one 70B MLP layer ({w/1e9:.2f} GB) over {axes} Ironwood axis/axes: {ag(w, W7, axes)*1e3:.2f} ms")

    # GPU comparison numbers (SB12).
    print("\nH100 node AllReduce: peaks ~370 GB/s busbw only near 10 GB/device; 58 MB -> ~150 GB/s (SB12)")
    print(f"8xH100 AllGather of bf16[1024, 16384]: {2*1024*16384*7/(8*450e9)*1e6:.0f} us (SB12 pop quiz 1: 65 us)")
    print(f"Same array, Ironwood one axis of 8 (bidi ring): {2*1024*16384/W7*1e6:.0f} us")
    print(f"SB10 collective matmul: 311 -> 244 us vs 224 us unsharded baseline: overlap recovers {(311-244)/(311-224):.0%} of the gap")
