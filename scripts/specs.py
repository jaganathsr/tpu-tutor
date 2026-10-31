"""Per-chip TPU and GPU specs used by every numeric example in the course.

Each value carries the source ID from research/sources.md. To add a chip, add an
entry to CHIPS and re-run `python3 scripts/specs.py` and the chapter scripts.

Conventions (match the Scaling Book, SB1/SB3/SB5):
  flops_*      dense peak per chip, FLOPs/s (OPs/s for int formats)
  hbm_bw       bytes/s per chip
  ici_link_bidi  bytes/s per ICI link, both directions summed
  W_axis       bandwidth one chip can push into one torus axis during a
               bidirectional ring collective = 2 links x one-way link rate
               = ici_link_bidi. This is the book's W_ici.
"""

from dataclasses import dataclass, field

GB = 1e9
TB = 1e12
GiB = 2**30


@dataclass
class Chip:
    name: str
    flops_bf16: float | None
    hbm_bw: float
    hbm_bytes: float
    flops_fp8: float | None = None
    flops_int8: float | None = None
    flops_fp4: float | None = None
    flops_int4: float | None = None
    ici_link_bidi: float | None = None   # per link, both directions
    ici_links: int | None = None
    vmem_bytes: float | None = None      # per chip
    tensorcores: int | None = None
    src: dict = field(default_factory=dict)
    provisional: tuple = ()             # fields not in an official Google spec
    # Network model for Chapter 11 comparisons:
    #   kind 'torus': W_axis per axis, `axes` torus dimensions, `domain` chips per ICI domain
    #   kind 'switch': `link` bytes/s per chip per direction inside a one-hop domain of `domain`
    #                  chips; `egress` bytes/s per node of `node` chips beyond it
    net: dict = field(default_factory=dict)

    @property
    def W_axis(self):
        return self.ici_link_bidi


CHIPS = {
    "v5e": Chip(
        "TPU v5e", flops_bf16=197e12, flops_int8=394e12, flops_int4=788e12, hbm_bw=819e9, hbm_bytes=16 * GB,
        ici_link_bidi=100e9, ici_links=4, vmem_bytes=128 * 2**20, tensorcores=1,
        net={"kind": "torus", "axes": 2, "domain": 256},
        src={"flops": "D-v5e (int4: J-hw)", "hbm": "819 GB/s: J-hw/SB1 (8.2e11); D-v5e prints '800 GiBps', see DECISIONS",
             "ici": "D-v5e 400 GB/s bidi over 4 ports", "vmem": "J-hw"},
    ),
    "v5p": Chip(
        "TPU v5p", flops_bf16=459e12, flops_fp8=459e12, flops_int8=918e12, flops_int4=1840e12, hbm_bw=2765e9, hbm_bytes=95 * GiB,
        ici_link_bidi=200e9, ici_links=6, vmem_bytes=2 * 64 * 2**20, tensorcores=2,
        net={"kind": "torus", "axes": 3, "domain": 8960},
        src={"flops": "D-v5p (fp8 listed equal to bf16; int8: SB2, J-hw; int4: J-hw)", "hbm": "D-v5p", "ici": "D-v5p 1200 GB/s bidi / 6",
             "vmem": "J-hw"},
    ),
    "v6e": Chip(
        "TPU v6e", flops_bf16=918e12, flops_fp8=920e12, flops_int8=1836e12, flops_int4=3680e12, hbm_bw=1638e9, hbm_bytes=32 * GB,
        ici_link_bidi=200e9, ici_links=4, vmem_bytes=128 * 2**20, tensorcores=1,
        net={"kind": "torus", "axes": 2, "domain": 256},
        src={"flops": "D-v6e (fp8, int4: J-hw; fp8 equals bf16)", "hbm": "D-v6e", "ici": "D-v6e 800 GB/s bidi / 4", "vmem": "J-hw"},
    ),
    "7x": Chip(
        "TPU7x Ironwood", flops_bf16=2307e12, flops_fp8=4614e12, hbm_bw=7380e9, hbm_bytes=192 * GiB,
        ici_link_bidi=200e9, ici_links=6, vmem_bytes=2 * 64 * 2**20, tensorcores=2,
        net={"kind": "torus", "axes": 3, "domain": 9216},
        src={"flops": "D-7x", "hbm": "D-7x", "ici": "D-7x 6 x 200 GB/s bidi", "vmem": "J-hw, SB2"},
    ),
    "8t": Chip(
        "TPU 8t", flops_bf16=996e12, flops_fp8=5977e12, flops_int8=996e12, flops_int4=11954e12, flops_fp4=12.6e15, hbm_bw=6528e9,
        hbm_bytes=216 * GiB, ici_link_bidi=400e9, ici_links=6, vmem_bytes=128 * 2**20, tensorcores=1,
        net={"kind": "torus", "axes": 3, "domain": 9600},
        src={"fp4": "G-8dd", "bf16/fp8/int8/int4": "J-hw (provisional)", "hbm": "G-8dd",
             "ici": "G-8 '2x Ironwood' -> derived 400 GB/s bidi/link", "vmem": "G-8dd, J-hw"},
        provisional=("flops_bf16", "flops_fp8", "flops_int8", "flops_int4", "ici_link_bidi"),
    ),
    "8i": Chip(
        "TPU 8i", flops_bf16=2 * 550e12, flops_fp8=2 * 4404e12, flops_fp4=10.1e15, hbm_bw=8601e9,
        hbm_bytes=288 * GiB, vmem_bytes=384 * 2**20, tensorcores=2,
        src={"fp4": "G-8dd", "bf16/fp8": "J-hw (provisional)", "hbm": "G-8dd",
             "ici": "G-8: 19.2 Tb/s per chip; Boardfly, per-link split not published"},
        provisional=("flops_bf16", "flops_fp8"),
    ),
    "H100": Chip("H100 SXM", flops_bf16=989e12, flops_fp8=1979e12, hbm_bw=3.35e12, hbm_bytes=80 * GB,
                 net={"kind": "switch", "link": 450e9, "domain": 8, "egress": 400e9, "node": 8},
                 src={"all": "N-h100 (dense = sparse/2)", "net": "SB12: NVLink 450 GB/s per direction, 400 GB/s node egress"}),
    "B200": Chip("B200 (HGX)", flops_bf16=2.25e15, flops_fp8=4.5e15, hbm_bw=8e12, hbm_bytes=180 * GB,
                 net={"kind": "switch", "link": 900e9, "domain": 8, "egress": 400e9, "node": 8},
                 src={"all": "N-b200 (derived per GPU)", "net": "SB12 Appendix A"}),
    "GB200": Chip("GB200 (NVL72)", flops_bf16=2.5e15, flops_fp8=5e15, hbm_bw=8e12, hbm_bytes=186 * GB,
                  net={"kind": "switch", "link": 900e9, "domain": 72, "egress": 3.6e12, "node": 72},
                  src={"all": "N-gb200 (derived per GPU from rack totals)", "net": "SB12 Appendix A: 3.6 TB/s node egress"}),
    "Rubin": Chip("Rubin (VR NVL72)", flops_bf16=4e15, flops_fp8=17.5e15, hbm_bw=19.2e12, hbm_bytes=288 * GB,
                  net={"kind": "switch", "link": 1.5e12, "domain": 72},
                  src={"all": "N-rubin product page (vendor claims; CES coverage differs)", "net": "3 TB/s bidirectional per GPU"},
                  provisional=("flops_fp8",)),
    "Trn2": Chip("Trainium2", flops_bf16=667e12, flops_fp8=1299e12, hbm_bw=2.9e12, hbm_bytes=96 * GiB,
                 net={"kind": "torus", "axes": 2, "domain": 16, "w_axis_note": "NeuronLink-v3 1.28 TB/s per chip, direction unstated"},
                 ici_link_bidi=1.28e12 / 4,
                 src={"all": "A-trn2, A-trn2-sys"}, provisional=("ici_link_bidi",)),
    "Trn3": Chip("Trainium3", flops_bf16=671e12, flops_fp8=2517e12, hbm_bw=4.9e12, hbm_bytes=144 * GiB,
                 net={"kind": "switch", "link": 1.0e12, "domain": 144},
                 src={"all": "A-trn3 (FP8 is MXFP8)", "net": "NeuronLink-v4 2.0-2.56 TB/s per chip; per-direction split assumed"},
                 provisional=("net",)),
}


def hbm_intensity(c: Chip, flops):
    """Critical arithmetic intensity against HBM, FLOPs per byte."""
    return None if flops is None else flops / c.hbm_bw


def ici_intensity(c: Chip, flops=None):
    """C / W_ici for one torus axis (book convention). Training DP/FSDP needs
    per-chip token batch above this / M_X; TP needs F above Y * this / M_Y."""
    flops = flops or c.flops_bf16
    return None if (c.W_axis is None or flops is None) else flops / c.W_axis


def fmt(x, nd=0):
    return "n/a" if x is None else f"{x:,.{nd}f}"


def export_json(path="src/generated/chips.json"):
    """Write the chip table for the in-browser interactives."""
    import json
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {}
    for k, c in CHIPS.items():
        out[k] = {
            "name": c.name,
            "flops": {"bf16": c.flops_bf16, "fp8": c.flops_fp8, "int8": c.flops_int8, "fp4": c.flops_fp4, "int4": c.flops_int4},
            "hbm_bw": c.hbm_bw, "hbm_bytes": c.hbm_bytes, "W_axis": c.W_axis,
            "vmem_bytes": c.vmem_bytes, "tensorcores": c.tensorcores,
            "src": c.src, "provisional": list(c.provisional), "net": c.net,
        }
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    import sys
    if "--json" in sys.argv:
        export_json()
    print(f"{'chip':16} {'bf16 HBM AI':>11} {'fp8 HBM AI':>10} {'B_crit int8w/bf16':>17} {'C/W_ici bf16':>12}  provisional")
    for k, c in CHIPS.items():
        ai = hbm_intensity(c, c.flops_bf16)
        print(f"{c.name:16} {fmt(ai):>11} {fmt(hbm_intensity(c, c.flops_fp8)):>10} "
              f"{fmt(ai / 2 if ai else None):>17} {fmt(ici_intensity(c)):>12}  {','.join(c.provisional)}")
    print("\nSB book values for comparison: v5e 240 (8.2e11 HBM), v5p C/W_ici 2550 (W = 1.8e11), H100 295.")
