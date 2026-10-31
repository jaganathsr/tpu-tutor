"""Numbers used in Chapter 3 (ICI, OCS, slices and pods).

    python3 scripts/ch03_ici.py
"""
from specs import CHIPS, GiB

# Cloud docs quote ICI per chip, summed over links and both directions.
# (chip, per-chip bidirectional GB/s, links, Scaling Book one-way per link)
ICI_DOCS = [
    ("v5e", 400, 4, 45),
    ("v6e", 800, 4, 90),
    ("v5p", 1200, 6, 90),
    ("7x", 1200, 6, 90),
]


def diameter(shape, wrap=True):
    return sum((d // 2) if wrap else (d - 1) for d in shape)


def ring_avg_distance(n, wrap=True):
    import itertools
    pts = range(n)
    dist = (min(abs(a - b), n - abs(a - b)) if wrap else abs(a - b) for a, b in itertools.product(pts, pts))
    return sum(dist) / (n * n)


if __name__ == "__main__":
    print("ICI conversion: per-chip bidi total -> per-link bidi -> per-link one way (book value)")
    for k, total, links, book in ICI_DOCS:
        per_link_bidi = total / links
        print(f"  {k:4}: {total} GB/s / {links} links = {per_link_bidi:.0f} GB/s per link bidi = {per_link_bidi/2:.0f} one way (book {book})")
    print(f"  8i: 19.2 Tb/s = {19.2e12/8/1e9:.0f} GB/s per chip; 8t: '2x Ironwood' = {2*1200} GB/s per chip (derived)")
    print("  H100 NVLink 4: 18 links x 25 GB/s each way = 450 GB/s per direction, 900 GB/s bidi")

    c = CHIPS["7x"]
    print("\nIronwood bandwidth ladder, per chip, one direction:")
    ladder = [("HBM", c.hbm_bw), ("ICI, all 6 links", 600e9), ("ICI, one link", 100e9),
              ("PCIe (book, v6e)", 32e9), ("DCN", 12.5e9)]
    for name, bw in ladder:
        print(f"  {name:18} {bw/1e9:8.1f} GB/s   HBM/x = {c.hbm_bw/bw:6.0f}")

    print("\nPod accounting (SB2 Q2, plus Ironwood):")
    for name, chips, per_host, cores_per_chip, flops, hbm in (
        ("v5e pod", 256, 8, 1, 1.97e14, 16 * GiB),
        ("v5p pod", 8960, 4, 2, 4.59e14, 96 * GiB),
        ("Ironwood pod", 9216, 4, 2, 2.307e15, 192 * GiB),
    ):
        print(f"  {name:13}: {chips//per_host:5} hosts, {chips*cores_per_chip:6} TensorCores, "
              f"{chips*flops:.2e} bf16 FLOP/s, HBM {chips*hbm/2**40:,.0f} TiB = {chips*hbm/1e15:.2f} PB")
    print(f"  Ironwood fp8: {9216*4.614e15:.3e} FLOP/s (G-7x: 42.5 EF); 9216 x 192 GB = {9216*192e9/1e15:.2f} PB (G-7x-ga: 1.77 PB)")

    print("\nDiameters (hops):")
    for shape, wrap in (((16, 16), True), ((4, 4), False), ((4, 4, 4), True), ((2, 2, 2), False),
                        ((8, 8, 16), True), ((8, 16, 16), True), ((16, 24, 24), True)):
        print(f"  {'x'.join(map(str, shape)):9} wrap={wrap!s:5}: {diameter(shape, wrap)}")
    for n in (4, 8, 16):
        print(f"  ring {n}: avg distance wrap {ring_avg_distance(n):.2f}, no wrap {ring_avg_distance(n, False):.2f}")

    # SB2 Q5: v5e 4x4 (no wraparound), bf16[8,128,8192] from (0,0) to (3,3).
    nbytes = 2 * 8 * 128 * 8192
    t_bw = nbytes / (2 * 45e9)
    print(f"\nSB2 Q5: {nbytes:.3g} bytes over 2 links at 45 GB/s = {t_bw*1e6:.0f} us + 6 hops x 1 us = {(t_bw+6e-6)*1e6:.0f} us")
    # Ironwood 4x4x4 cube with wraparound, (0,0,0) to (2,2,2): 6 hops, 3 outgoing directions used.
    t_bw7 = nbytes / (3 * 100e9)
    print(f"Ironwood cube, same bytes over 3 links at 100 GB/s: {t_bw7*1e6:.0f} us + 6 us = {(t_bw7+6e-6)*1e6:.0f} us (derived)")

    # SB2 Q6 pieces: 16 GB int8 matrix from host DRAM onto a v5e 4x4, gathered at one chip.
    print(f"\nSB2 Q6: PCIe 16e9/(16*1.6e10) = {16e9/(16*1.6e10)*1e3:.0f} ms; ICI 15e9/(2*4.5e10) = {15e9/(2*4.5e10)*1e3:.0f} ms; "
          f"HBM read 16e9/8.2e11 = {16e9/8.2e11*1e3:.0f} ms")
