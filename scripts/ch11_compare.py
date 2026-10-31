"""Numbers used in Chapter 11 (TPU and GPU side by side). Mirrored by
src/interactives/CompareExplorer.tsx.

Collective models (Scaling Book ch. 3 and ch. 12), V = full array bytes:
  TPU torus, N chips on `axes` axes: AllGather V / (W_axis * axes_used)
                                     AllToAll  V * side / (4 * N * W_axis), side = N^(1/axes_used)
  GPU switched, N <= domain:         AllGather V * (N-1)/N / link
                                     AllToAll  V / (N * link)
  GPU across nodes (M = N / node):   AllGather max(V / link, V / egress)
                                     AllToAll  V / (M * egress)

    python3 scripts/ch11_compare.py
"""
from specs import CHIPS


def allgather(c, V, N):
    n = c.net
    if n.get("kind") == "torus":
        axes = 1 if N <= 16 else n["axes"]
        return V / (c.W_axis * axes)
    if N <= n["domain"]:
        return V * (N - 1) / N / n["link"]
    return max(V / n["link"], V / n["egress"]) if "egress" in n else float("nan")


def alltoall(c, V, N):
    n = c.net
    if n.get("kind") == "torus":
        axes = 1 if N <= 16 else n["axes"]
        side = N ** (1 / axes)
        return V * side / (4 * N * c.W_axis)
    if N <= n["domain"]:
        return V / (N * n["link"])
    if "egress" not in n:
        return float("nan")
    M = N / n["node"]
    return V / (M * n["egress"])


if __name__ == "__main__":
    keys = ["7x", "8t", "8i", "H100", "B200", "GB200", "Rubin", "Trn2", "Trn3"]
    print(f"{'chip':18} {'BF16 TF':>8} {'FP8 TF':>8} {'HBM GB':>7} {'HBM TB/s':>8} {'ridge':>6}")
    for k in keys:
        c = CHIPS[k]
        print(f"{c.name:18} {c.flops_bf16/1e12:8.0f} {(c.flops_fp8 or 0)/1e12:8.0f} {c.hbm_bytes/1e9:7.0f} "
              f"{c.hbm_bw/1e12:8.2f} {c.flops_bf16/c.hbm_bw:6.0f}")

    V = 1e9
    for N in (8, 64, 512):
        print(f"\n1 GB collectives over {N} chips:")
        for k in ("7x", "H100", "GB200", "Trn3"):
            c = CHIPS[k]
            ag, a2a = allgather(c, V, N), alltoall(c, V, N)
            print(f"  {c.name:16} AllGather {ag*1e3:7.3f} ms   AllToAll {a2a*1e3:7.3f} ms")

    # SB12 quiz numbers reproduced.
    print(f"\nSB12 Quiz 1: H100 CUDA cores 132*4*32 = {132*4*32}; v5p VPU ALUs 2*4*8*128 = {2*4*8*128}")
    print(f"SB12 Quiz 2: 8xH100 AllGather of bf16[4096,65536]: {4096*65536*2*7/(8*450e9)*1e3:.2f} ms")
    print(f"SB12 Quiz 4 Q5 (2 nodes): node level B/{450e9*8/7/1e9:.0f}e9, cross-node B/{2*400e9/1e9:.0f}e9 -> node-bound")
    print(f"SB12 Quiz 5 Q1: B200 TP limit F/{2.25e15/900e9:.0f}; cross-node DP {2.25e15/400e9:.0f} tokens/GPU")
